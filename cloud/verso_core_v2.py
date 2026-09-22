"""VERSO region inference on one local GPU, v2: ONE pass, EVERY head the checkpoint has.

v1 (`verso_core.py` on the pod) runs the unified student at rung 2 with the radial vector negated and
keeps `sigmoid(y[:, 0])` -- one probability plane. A Phase-B checkpoint (docs/unified_design.md section
29) also carries a distance channel, a thickness, a normal triple and a heteroscedastic log-variance, and
`usrm2 export-tracer` used to fetch them with one sliding-window pass EACH. They are all pointwise
functions of the same raw output and the Gaussian blend is linear, so one pass gives exactly what five
give (section 30, `usrm2.predict.probs(head=[...])`); this module is that pass, in the pod's own
region-at-a-time arrangement.

**It is a no-op for a cout-2 checkpoint.** `Net.planes` is empty unless the checkpoint's `channels` name
a distance channel, and then `Net.__call__` and `run_region` execute v1's code, kernel for kernel, in the
same order -- which is what makes the v1/v2 comparison on the live checkpoint BIT-IDENTICAL rather than
merely close (the bf16+cuDNN run-to-run floor is dice 0.994, so "close" would prove nothing).

Everything else -- the one context super-cube per rung, the GPU z-scores, the radial vector, the fp16
accumulators, the all-zero-window skip -- is v1's, unchanged.
"""
import os
import numpy as np
import torch

from usrm2 import data, model as M, predict as P
from usrm2.train import autocast

# ---------------------------------------------------------------------------- compat with an old usrm2
# The pod runs a SNAPSHOT of usrm2 taken before Phase B existed (no `targets.py`, no `losses.py`, a
# 349-line `predict.py`), and refreshing that tree under a live production loop is exactly the kind of
# change that breaks a 70-hour job on its next supervisor restart. So everything this module needs from
# the new package is imported when it is there and defined here when it is not. The definitions are
# copies of `usrm2/predict.py` and `usrm2/losses.py`, and the test that keeps them honest lives in the
# repo (`tests/test_wave3.py::test_the_pod_v2_compat_shims_match_usrm2`), where BOTH are importable.
TRACER_UNIT, TRACER_OFF, TRACER_CAP, NORMAL_SCALE = 0.25, 128, 31.75, 127.0
TMIN = 3.0


def _head_names(args):
    """`predict.head_names`: every head a checkpoint can serve, in the multi-pass order."""
    a = args or {}
    ch = [str(c) for c in (a.get("channels") or ("recto", "verso"))]
    np_ = int(a.get("cout_p", a.get("cout_t", a.get("cout", 1))) or 1)
    out = list(ch[:np_])
    dch = next((c for c in ("sdist", "midline") if c in ch), None)
    if dch is not None:
        out.append(dch)
        if "thickness" in ch:
            out.append("thickness")
        out.append("normals")
        if "logvar" in ch:
            out.append("conf")
    return out


def _grad3(d):
    """`losses.grad3`: central differences of (B,1,Z,Y,X) along z, y, x -> (B,3,Z,Y,X)."""
    import torch.nn.functional as F
    g = []
    for a in (2, 3, 4):
        pad = [0] * 6
        i = 2 * (4 - a)
        pad[i] = pad[i + 1] = 1
        p = F.pad(d, pad, mode="replicate")
        n = p.shape[a]
        g.append(0.5 * (p.narrow(a, 2, n - 2) - p.narrow(a, 0, n - 2)))
    return torch.cat(g, 1)


def _normals_from(d, eps=1e-4):
    g = _grad3(d)
    return g / g.norm(dim=1, keepdim=True).clamp_min(eps)


def _soft_thickness(raw, tmin=TMIN):
    import torch.nn.functional as F
    return float(tmin) + F.softplus(raw)


def _u8(prob):
    return np.clip(np.rint(np.asarray(prob, np.float32) * 255.0), 0, 255).astype(np.uint8)


def _enc_signed(d, valid):
    c = np.rint(np.clip(d, -TRACER_CAP, TRACER_CAP) / TRACER_UNIT) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def _enc_normal(n, valid):
    c = np.rint(np.clip(n, -1.0, 1.0) * NORMAL_SCALE) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def _scharr3(d):
    import scipy.ndimage as ndi
    sm = np.array([3.0, 10.0, 3.0]) / 16.0
    dv = np.array([-1.0, 0.0, 1.0]) / 2.0
    out = []
    for a in range(3):
        g = np.asarray(d, np.float32)
        for b in range(3):
            g = ndi.correlate1d(g, dv if b == a else sm, axis=b, mode="nearest")
        out.append(g)
    return np.stack(out)


def _tracer_fields(sd, thick=None, conf=None, valid=None):
    d = np.asarray(sd, np.float32)
    if thick is not None:
        d = d - 0.5 * np.asarray(thick, np.float32)
    g = scharr3(d)
    mag = np.linalg.norm(g, axis=0)
    n = g / np.maximum(mag, 1e-4)
    v = np.ones(d.shape, bool) if valid is None else np.asarray(valid, bool)
    return d, n, mag, v


head_names = getattr(P, "head_names", _head_names)
u8 = getattr(P, "u8", _u8)
enc_signed = getattr(P, "enc_signed", _enc_signed)
enc_normal = getattr(P, "enc_normal", _enc_normal)
scharr3 = getattr(P, "scharr3", _scharr3)
tracer_fields = getattr(P, "tracer_fields", _tracer_fields)
try:
    from usrm2.losses import normals_from, soft_thickness, TMIN as TMIN_
    TMIN = TMIN_
except Exception:  # noqa: BLE001  (the pod's snapshot has no losses.py)
    normals_from, soft_thickness = _normals_from, _soft_thickness

VOL = os.environ.get("USRM2_CT", data.CT)
CTX = tuple(range(1, 10))
RUNG = 2


def starts(n, w, stride):
    s = list(range(0, max(n - w, 0) + 1, stride))
    if s[-1] != n - w:
        s.append(n - w)
    return s


def gauss_t(w, dev, dtype=torch.float32):
    g = torch.exp(-0.5 * ((torch.arange(w, device=dev, dtype=torch.float64) - (w - 1) / 2) / (w / 6)) ** 2)
    return (g[:, None, None] * g[None, :, None] * g[None, None, :]).to(dtype)


class Net:
    """The unified student, loaded once for the whole run.

    `planes` is the list of output plane names this checkpoint can serve BEYOND the one verso probability:
    empty for the cout-2 production checkpoint (and then this class is v1), otherwise the distance, the
    thickness, the three normal components and the confidence, in `usrm2.predict.head_names` order."""

    def __init__(self, ckpt, dev, compile_mode=None, channels_last=False, window=256, bf16_input=False,
                 head=0, multi=True):
        st = torch.load(ckpt, map_location=dev)
        a = st["args"]
        M.CHANNELS_LAST = bool(channels_last)
        net = M.build(a["size"], verbose=False, cout=a.get("cout", 1), cin=a.get("cin", 4),
                      add_skip=a.get("add_skip", 0), deep=a.get("deep", 0)).to(dev)
        net.load_state_dict(st["ema"])
        net.eval()
        self.args, self.step, self.dev, self.window = a, int(st.get("step", 0)), dev, window
        self.ctx = tuple(a.get("ctx") or ())
        self.scale_plane = bool(a.get("scale_plane"))
        self.norm = tuple(a["norm_stats"]) if a.get("norm") == "global" else None
        self.cin = int(a.get("cin", 4))
        self.bf16_input = bool(bf16_input)
        self.head = int(head)
        self.raw = net
        self.net = torch.compile(net, mode=compile_mode) if compile_mode else net
        ch = [str(c) for c in (a.get("channels") or ())]
        dch = next((c for c in ("sdist", "midline") if c in ch), None)
        # the FIELD heads, and only when the caller wants them: a checkpoint without a distance channel
        # is v1 by construction, and `multi=False` forces v1 for the comparison run
        self.dist_channel = dch if multi else None
        # A checkpoint WITHOUT a distance channel is v1, full stop: `planes` stays empty and every line
        # below takes v1's branch. With one, every head goes in -- the probability channels AND the
        # fields -- because the tracer contract wants recto and verso beside the distance anyway.
        self.planes = list(head_names(a)) if self.dist_channel is not None else []
        self.nplanes = len(self.planes)
        self._pieces = None
        if self.nplanes:
            nh = [j for j, c in enumerate(ch) if c in ("nz", "ny", "nx")]
            ci = ch.index(dch)
            pieces, names = [], []
            for nm in self.planes:
                if nm in ch and nm not in ("thickness",):       # another probability channel (recto/verso)
                    j = ch.index(nm)
                    pieces.append(lambda y, j=j: torch.sigmoid(y[:, j:j + 1].float()))
                    names.append(nm)
                elif nm == "thickness":
                    j = ch.index("thickness")
                    pieces.append(lambda y, j=j: soft_thickness(y[:, j:j + 1].float(), TMIN))
                    names.append("thickness")
                elif nm == "conf":
                    j = ch.index("logvar")
                    pieces.append(lambda y, j=j: 1.0 / (1.0 + torch.exp(0.5 * y[:, j:j + 1].float().clamp(-8, 8))))
                    names.append("conf")
                elif nm == "normals":
                    if len(nh) == 3:
                        pieces.append(lambda y, j=nh[0]: (lambda v: v / v.norm(dim=1, keepdim=True).clamp_min(1e-4))(
                            y[:, j:j + 3].float()))
                    else:
                        pieces.append(lambda y, j=ci: normals_from(y[:, j:j + 1].float()))
                    names += ["nz", "ny", "nx"]
                else:                                            # the distance channel itself
                    j = ch.index(nm)
                    pieces.append(lambda y, j=j: y[:, j:j + 1].float())
                    names.append(nm)
            self._pieces, self.plane_names = pieces, names
        else:
            self.plane_names = []

    def __call__(self, x):
        with torch.no_grad(), autocast(self.dev):
            y = self.net(x.contiguous(memory_format=M.memfmt()))
        if not self.nplanes:
            return torch.sigmoid(y[:, self.head].float())        # v1, kernel for kernel
        return torch.cat([f(y) for f in self._pieces], 1)


class RegionInputs:
    """Everything a 1024^3 region needs, read once: the CT at rung 2 and one context super-cube per rung."""

    def __init__(self, lo, size, window, halo, dev, sign=-1.0, pyr=None, ax=None):
        self.lo, self.size, self.window, self.halo, self.dev = tuple(lo), tuple(size), window, halo, dev
        pyr = pyr or data.rungs(VOL)
        stride = window - 2 * halo
        self.offs = [(z, y, x)
                     for z in starts(max(size[0], window), window, stride)
                     for y in starts(max(size[1], window), window, stride)
                     for x in starts(max(size[2], window), window, stride)]
        roi = data.read_rung(pyr, RUNG, lo, size, dtype=np.uint8)
        if any(s < window for s in size):  # pad to a full window, exactly as predict.slide does
            roi = np.pad(roi, [(0, max(window - s, 0)) for s in size])
        self.roi = torch.from_numpy(np.ascontiguousarray(roi)).to(dev)
        self.shape = tuple(self.roi.shape)
        cs = [sorted({lo[a] + o[a] + window // 2 for o in self.offs}) for a in range(3)]
        self.ctx = {}
        for d in CTX:
            lo_d = [(min(cs[a]) >> d) - window // 2 for a in range(3)]
            sz_d = [(max(cs[a]) >> d) - window // 2 + window - lo_d[a] for a in range(3)]
            cube = data.read_rung(pyr, RUNG + d, lo_d, sz_d, dtype=np.uint8)
            self.ctx[d] = (lo_d, torch.from_numpy(np.ascontiguousarray(cube)).to(dev))
        ax = data.axis_at(ax if ax is not None else data.axis(), RUNG)
        z = np.arange(self.shape[0]) + lo[0]
        self.cy = torch.from_numpy(np.interp(z, ax[0], ax[1]).astype(np.float32)).to(dev)
        self.cx = torch.from_numpy(np.interp(z, ax[0], ax[2]).astype(np.float32)).to(dev)
        self.sign = float(sign)
        self.ay = torch.arange(self.shape[1], device=dev, dtype=torch.float32) + lo[1]
        self.ax_ = torch.arange(self.shape[2], device=dev, dtype=torch.float32) + lo[2]
        self.scale = float(RUNG - 2) / 9.0

    def window_ct(self, o):
        w = self.window
        return self.roi[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w]

    def prep(self, o, out_dtype=torch.float32, lut=None, fn=None):
        """(14, w, w, w): z-scored CT, nine z-scored context cubes, the scale plane, the radial vector."""
        w = self.window
        ch = []
        c = self.window_ct(o)
        c = (lut[c.long()] if lut is not None else c.float())
        if fn is not None:
            c = fn(c)
        ch.append((c - c.mean()) / (c.std(unbiased=False) + 1e-3))
        for d in CTX:
            lo_d, cube = self.ctx[d]
            i = [((self.lo[a] + o[a] + w // 2) >> d) - w // 2 - lo_d[a] for a in range(3)]
            q = cube[i[0]:i[0] + w, i[1]:i[1] + w, i[2]:i[2] + w]
            q = (lut[q.long()] if lut is not None else q.float())
            if fn is not None:
                q = fn(q)
            ch.append((q - q.mean()) / (q.std(unbiased=False) + 1e-3))
        ch.append(torch.full((w, w, w), self.scale, device=self.dev, dtype=torch.float32))
        dy = self.ay[o[1]:o[1] + w][None, :, None] - self.cy[o[0]:o[0] + w][:, None, None]
        dx = self.ax_[o[2]:o[2] + w][None, None, :] - self.cx[o[0]:o[0] + w][:, None, None]
        n = torch.sqrt(dy * dy + dx * dx) + 1e-6
        ch += [torch.zeros((w, w, w), device=self.dev, dtype=torch.float32),
               (dy / n) * self.sign, (dx / n) * self.sign]
        return torch.stack(ch).to(out_dtype)


def shifted_offs(shape, window, halo, shift):
    """Window starts with the tiling grid moved by `shift`."""
    st = window - 2 * halo
    out = []
    for a in range(3):
        n = shape[a]
        s = sorted({0, n - window} | {v for v in range(int(shift[a]), max(n - window, 0) + 1, st)
                                      if 0 <= v <= n - window})
        out.append(s)
    return [(z, y, x) for z in out[0] for y in out[1] for x in out[2]]


def run_region(net, R, batch=1, acc_dtype=torch.float32, prep_dtype=torch.float32, offs=None, lut=None, fn=None):
    """The blended output of one region.

    With a v1 (single-plane) net this is v1's function, statement for statement, returning
    ((Z,Y,X) float32, windows). With field heads it returns ((P,Z,Y,X) float32, windows), the planes in
    `net.plane_names` order -- the same Gaussian blend, one accumulator per plane."""
    w, dev = R.window, R.dev
    g = gauss_t(w, dev, acc_dtype)
    todo = [o for o in (R.offs if offs is None else offs) if R.window_ct(o).any()]
    if not net.nplanes:
        acc = torch.zeros(R.shape, dtype=acc_dtype, device=dev)
        wsum = torch.zeros(R.shape, dtype=acc_dtype, device=dev)
        for i in range(0, len(todo), batch):
            offs_b = todo[i:i + batch]
            offs = offs_b
            x = torch.stack([R.prep(o, prep_dtype, lut=lut, fn=fn) for o in offs])
            p = net(x).to(acc_dtype)
            del x
            for o, pj in zip(offs, p):
                acc[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += pj * g
                wsum[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += g
            del p
        out = torch.where((wsum > 0) & (R.roi > 0), acc.float() / wsum.float().clamp_min(1e-6),
                          torch.zeros((), device=dev))
        del acc, wsum
        Z, Y, X = R.size
        return out[:Z, :Y, :X], len(todo)
    acc = torch.zeros((len(net.plane_names),) + R.shape, dtype=acc_dtype, device=dev)
    wsum = torch.zeros(R.shape, dtype=acc_dtype, device=dev)
    for i in range(0, len(todo), batch):
        offs_b = todo[i:i + batch]
        offs = offs_b
        x = torch.stack([R.prep(o, prep_dtype, lut=lut, fn=fn) for o in offs])
        p = net(x).to(acc_dtype)
        del x
        for o, pj in zip(offs, p):
            acc[:, o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += pj * g
            wsum[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] += g
        del p
    keep = ((wsum > 0) & (R.roi > 0))[None]
    out = torch.where(keep, acc.float() / wsum.float().clamp_min(1e-6)[None], torch.zeros((), device=dev))
    del acc, wsum
    Z, Y, X = R.size
    return out[:, :Z, :Y, :X], len(todo)
