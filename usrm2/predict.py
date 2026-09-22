"""Sliding-window inference -> uint8 recto probability * 255 zarr (volcomp q=8)."""
import numpy as np
import torch

from usrm2 import data, model as M
from usrm2.train import autocast


def gauss(w):
    g = np.exp(-0.5 * ((np.arange(w) - (w - 1) / 2) / (w / 6)) ** 2).astype(np.float32)
    return g[:, None, None] * g[None, :, None] * g[None, None, :]


def starts(n, w, stride):
    s = list(range(0, max(n - w, 0) + 1, stride))
    if s[-1] != n - w:
        s.append(n - w)
    return s


def slide(fn, roi, window, halo, dev, prep=None):
    """Gaussian-blended sliding window over a uint8 ROI. fn: normalized (B,C,w,w,w) tensor -> prob (B,w,w,w).
    prep(ct_window, (z,y,x) window offset) -> (C,w,w,w) float input; default is z-scored CT alone."""
    prep = prep or (lambda c, o: data.zscore(c)[None])
    if any(s < window for s in roi.shape):  # thinner than a window: pad with air, crop the result
        S = roi.shape
        roi = np.pad(roi, [(0, max(window - s, 0)) for s in S])
        return slide(fn, roi, window, halo, dev, prep)[:S[0], :S[1], :S[2]]
    Z, Y, X = roi.shape
    acc, wsum, g = None, np.zeros(roi.shape, np.float32), gauss(window)
    stride = window - 2 * halo
    with torch.no_grad():
        for z in starts(Z, window, stride):
            for y in starts(Y, window, stride):
                for x in starts(X, window, stride):
                    c = roi[z:z + window, y:y + window, x:x + window]
                    if not c.any():
                        continue
                    t = torch.from_numpy(prep(c, (z, y, x)))[None].to(dev).to(memory_format=M.memfmt())
                    with autocast(dev):
                        p = fn(t)[0].float().cpu().numpy()  # (w,w,w) or (C,w,w,w): fn may return every head
                    if acc is None:
                        acc = np.zeros(p.shape[:-3] + roi.shape, np.float32)
                    acc[..., z:z + window, y:y + window, x:x + window] += p * g
                    wsum[z:z + window, y:y + window, x:x + window] += g
    if acc is None:
        return np.zeros(roi.shape, np.float32)
    return np.where((wsum > 0) & (roi > 0), acc / np.maximum(wsum, 1e-6), 0)  # masked CT (0) -> no surface


def slide_gpu(fn, roi, window, halo, dev, prep, batch=1, streams=1):
    """slide() with the ROI, the normalization and the (fp16) accumulators all on the GPU, `batch` windows per
    forward and `streams` CUDA streams (one thread + its own accumulators each, summed at the end) so the card
    never idles between kernel launches. A 384x2048x2048 box needs ~8 GB per stream + the model + batch x the
    activations. prep(ct_window uint8 tensor, (z,y,x)) -> (C,w,w,w) float tensor; masked CT (0) -> no surface."""
    import threading
    if any(s < window for s in roi.shape):
        S = roi.shape
        roi = np.pad(roi, [(0, max(window - s, 0)) for s in S])
        return slide_gpu(fn, roi, window, halo, dev, prep, batch, streams)[:S[0], :S[1], :S[2]]
    R = torch.from_numpy(np.ascontiguousarray(roi)).to(dev)
    g = torch.from_numpy(gauss(window)).to(dev).half()
    Z, Y, X = R.shape
    stride = window - 2 * halo
    todo = [(z, y, x) for z in starts(Z, window, stride) for y in starts(Y, window, stride) for x in starts(X, window, stride)
            if R[z:z + window, y:y + window, x:x + window].any()]
    accs, wsums = [], []

    def work(k):
        acc, wsum = None, torch.zeros(R.shape, dtype=torch.float16, device=dev)
        wsums.append(wsum)
        s = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        with torch.no_grad(), (torch.cuda.stream(s) if s else contextlib.nullcontext()):
            mine = todo[k::streams]
            for i in range(0, len(mine), batch):
                offs = mine[i:i + batch]
                t = torch.stack([prep(R[z:z + window, y:y + window, x:x + window], o) for o in offs for z, y, x in [o]])
                with autocast(dev):
                    p = fn(t.contiguous(memory_format=M.memfmt()))  # (B,w,w,w) or (B,C,w,w,w)
                if acc is None:
                    acc = torch.zeros(tuple(p.shape[1:-3]) + tuple(R.shape), dtype=torch.float16, device=dev)
                    accs.append(acc)
                for (z, y, x), pj in zip(offs, p):
                    acc[..., z:z + window, y:y + window, x:x + window] += (pj.float() * g.float()).half()
                    wsum[z:z + window, y:y + window, x:x + window] += g
            if s:
                s.synchronize()
    import contextlib
    if streams > 1:
        ths = [threading.Thread(target=work, args=(k,)) for k in range(streams)]
        [t.start() for t in ths], [t.join() for t in ths]
    else:
        work(0)
    if not accs:  # nothing but air
        return np.zeros(R.shape, np.float32)
    acc, wsum = accs[0], wsums[0]
    for a, w in zip(accs[1:], wsums[1:]):
        acc += a
        wsum += w
    del accs, wsums
    with torch.no_grad():
        out = np.empty(tuple(acc.shape), np.float32)
        for z in range(0, Z, 64):  # a slab at a time keeps the float32 temporaries small
            a, w, r = acc[..., z:z + 64, :, :].float(), wsum[z:z + 64].float(), R[z:z + 64]
            out[..., z:z + 64, :, :] = torch.where((w > 0) & (r > 0), a / w.clamp_min(1e-6), torch.zeros_like(a)).cpu().numpy()
    del R, acc, wsum
    return out


def zscore_t(c):
    x = c.float()
    return (x - x.mean()) / (x.std() + 1e-3)


def shard_shape(shape, chunk=128, cap=1024):
    """Shard shape for a store: one shard per `cap`^3 box, a multiple of the chunk and covering the array
    (so a store smaller than `cap` on an axis is a single shard, i.e. ONE data file on disk)."""
    return tuple(min(cap, -(-int(s) // chunk) * chunk) for s in shape)


def out_array(path, shape, origin, volcomp=True, volume=None, umbilicus=None, rung=2, channels=("recto",),
              q=8):
    import zarr
    try:
        from volcomp_zarr import VolcompCodec
    except Exception:
        VolcompCodec = None
    # ONE data file per ~1024^3 of store: a zarr v3 shard holds the 128^3 inner chunks. 512 chunk files per
    # store cost ~590 sftp operations to publish to the mirror; a shard costs one put. Every store usrm2
    # writes is sharded this way, and its writers fill a shard region in one write (see teacher.run).
    sh = shard_shape(shape)
    kw = dict(shape=shape, chunks=(128, 128, 128), shards=sh, dtype="uint8", fill_value=0, overwrite=True)
    if volcomp and VolcompCodec is not None:  # volcomp requires exactly 128^3 chunks, hence 3D
        assert all(s % 128 == 0 for s in shape), "volcomp output needs box sizes that are multiples of 128"
        # compressors=None: the inner codec chain is exactly [volcomp]. zarr-python otherwise appends its
        # default zstd AFTER the serializer, and zstd on volcomp output is worthless -- measured 0.2% on a
        # real region store, for a decode step on every chunk read. The C-tool exports and the CT volumes
        # are volcomp-only, so this also makes our stores byte-comparable with theirs.
        # `q` is the codec's quantisation step. q=8 is the probability stores' setting (section 9's
        # table by physical voxel size) and it is LOSSY: a stored 0 can read back as a small non-zero.
        # That is harmless for a probability and FATAL for a field whose code 0 means "no data" and
        # whose codes are a distance in 0.25-voxel steps, so every store of section 29 passes q=0
        # (lossless). A lossy codec also compounds under partial-chunk writes, because filling a 128^3
        # chunk in pieces decodes and re-encodes it once per piece.
        z = zarr.create_array(path, serializer=VolcompCodec(q=int(q)), compressors=None, **kw)
    else:
        kw["shape"], kw["chunks"], kw["shards"] = (1,) + tuple(shape), (1, 128, 128, 128), (1,) + sh
        z = zarr.create_array(path, **kw)
    z.attrs.update({"channels": [str(c) for c in channels], "voxel_um": data.rung_um(rung), "rung": int(rung),
                    **({"volcomp_q": int(q)} if int(q) != 8 else {}),
                    "origin_zyx": [int(v) for v in origin], "scale": 1.0,
                    "volume": volume or data.CT, "umbilicus": umbilicus or data.UMBILICUS})  # so loaders know the scroll
    return z


def put(arr, u8, z=0, y=0, x=0):
    """Write a uint8 block into a store at a local offset, whatever its layout ((Z,Y,X) or (1,Z,Y,X))."""
    s = (slice(z, z + u8.shape[0]), slice(y, y + u8.shape[1]), slice(x, x + u8.shape[2]))
    arr[(0,) + s if arr.ndim == 4 else s] = u8


def u8(prob):
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def write(path, prob, origin, volcomp=True, volume=None, rung=2, channels=("recto",), q=8):
    a = out_array(path, prob.shape, origin, volcomp=volcomp, volume=volume, rung=rung, channels=channels, q=q)
    p = u8(prob)
    a[:] = p[None] if a.ndim == 4 else p


def flips_vec(fn, n=8):
    """Flip TTA for the student: flipping spatial axis d also negates radial component d (channel 1+d)."""
    fl = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)][:n]

    def go(t):
        out = 0
        for f in fl:
            x = torch.flip(t, [2 + d for d in f]).clone()
            ni = x.shape[1] - 3
            for d in f:
                x[:, ni + d] = -x[:, ni + d]
            out = out + torch.flip(fn(x), [1 + d for d in f])
        return out / len(fl)
    return go


def flips_chan(fn, n=8, vec=()):
    """`flips_vec` for a function whose output KEEPS a channel axis, (B, C, Z, Y, X).

    The input side is identical (flipping spatial axis d negates radial component d). The output side
    differs twice: the spatial axes are 2 + d, not 1 + d, and a VECTOR output has to be flipped as a
    vector -- flipping the world along axis d negates component d of every normal. `vec` lists the
    (cz, cy, cx) output-plane triples that are ZYX vectors; every other plane is a scalar field and is
    only reordered. Without this a `--head normals` TTA average silently cancels the normal field.
    """
    fl = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)][:n]

    def go(t):
        out = 0
        for f in fl:
            x = torch.flip(t, [2 + d for d in f]).clone()
            ni = x.shape[1] - 3
            for d in f:
                x[:, ni + d] = -x[:, ni + d]
            y = torch.flip(fn(x), [2 + d for d in f]).clone()
            for tri in vec:
                for d in f:
                    y[:, tri[d]] = -y[:, tri[d]]
            out = out + y
        return out / len(fl)
    return go


HEADS = {"mean": lambda p: p.mean(1), "prod": lambda p: p.prod(1) ** (1 / p.shape[1]), "max": lambda p: p.max(1).values}

# PHASE B (docs/unified_design.md section 29). Fields a checkpoint trained with `--sdist` can emit that
# are NOT probabilities, so they never see a sigmoid and are not in the checkpoint's channel list under
# these names: `--head sdist` (the signed distance, in voxels at the rung), `--head thickness` (voxels),
# `--head normals` ((3,Z,Y,X), ZYX, unit, sign verso -> recto), `--head conf` (0..1 from the
# heteroscedastic log-variance). `--head sdist` is also spelled by the store's channel name, which for a
# midline run is "midline"; both resolve here.
FIELDS = ("sdist", "midline", "thickness", "normals", "conf")   # "midline" is an alias of "sdist":
# a midline checkpoint's distance channel is NAMED midline, and both spellings return that field

CHANNEL_DEFAULT = ("recto", "verso")  # the unified model's output channel order (section 23)

def head_names(args):
    """Every head a checkpoint can serve, in the canonical order the multi-head pass stacks them in:

        recto, verso, ...        the probability channels (`cout_p` of them, by their own names)
        sdist | midline          the distance channel, if it has one
        thickness                if it has one
        normals                  if it has a distance channel (from the normals head, else derived)
        conf                     if it was trained --sdist-hetero

    This is what `probs(head=head_names(args))` and `export-tracer` walk, so ONE sliding-window pass
    produces every field the tracer contract asks for instead of five or six passes over the same box
    (docs/unified_design.md section 30)."""
    a = args or {}
    ch = [str(c) for c in (a.get("channels") or CHANNEL_DEFAULT)]
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


def field_head(head, args):
    """`--head sdist|thickness|normals|conf` -> (kind, head-channel index) for a Phase-B checkpoint,
    or None when `head` names an ordinary probability channel."""
    h = str(head).strip()
    if h not in FIELDS:
        return None
    ch = [str(c) for c in ((args or {}).get("channels") or ())]
    np_ = int((args or {}).get("cout_p", 0) or 0)
    if h == "conf":
        assert "logvar" in ch, "--head conf: this checkpoint was not trained with --sdist-hetero"
        return h, ch.index("logvar")
    if h == "thickness":
        assert "thickness" in ch, "--head thickness: this checkpoint was not trained with --thickness"
        return h, ch.index("thickness")
    assert np_ and np_ < len(ch) and ch[np_] in ("sdist", "midline"), \
        f"--head {h}: this checkpoint has no distance channel (channels {ch})"
    return h, np_


def resolve_head(head, args=None):
    """`--head` -> what `probs` selects: an output channel INDEX, "all", or a `HEADS` reducer name.

    A channel NAME is looked up in the checkpoint's own channel list (`args["channels"]`, the order the
    unified model's head was trained in), falling back to recto = 0, verso = 1. `--head verso` on a cout=1
    checkpoint is an error and says so: the old way to get a verso band out of a recto-only student is
    `--radial-sign -1` (verso.py), which still works and is a different thing -- a recto model looking at a
    mirrored world, not a trained verso output."""
    if not isinstance(head, str):
        return int(head)
    h = head.strip()
    if h in HEADS or h == "all":
        return h
    if h.lstrip("+-").isdigit():
        return int(h)
    if h in FIELDS:
        return h
    ch = [str(c) for c in ((args or {}).get("channels") or CHANNEL_DEFAULT)]
    i = ch.index(h) if h in ch else (CHANNEL_DEFAULT.index(h) if h in CHANNEL_DEFAULT else None)
    assert i is not None, f"--head {head}: an output channel of {ch}, an index, 'all', or one of {sorted(HEADS)}"
    n = int((args or {}).get("cout", len(ch)))
    assert i < n, (f"--head {head} is output channel {i} but the checkpoint has cout={n}. A single-output "
                   "recto checkpoint has no verso channel; --radial-sign -1 is the old flip trick.")
    return i


CASCADE_HALO = 16  # rung-(k+1) voxels of margin around a coarse prediction's footprint


def crop_pad(a, off, shape):
    """`a[off : off + shape]` zero-padded where it runs past the end: `slide` pads an ROI thinner than one
    window and then asks for windows the cascade array does not cover."""
    out = np.zeros(tuple(int(v) for v in shape), np.float32)
    if a is None:
        return out
    s = tuple(slice(int(o), min(int(o) + int(n), int(d))) for o, n, d in zip(off, shape, a.shape))
    blk = a[s]
    out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
    return out


def up2x_np(a):
    """2x trilinear upsample of a (Z,Y,X) float32 array, the same `model.up2x` training uses."""
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    return M.up2x(t, tuple(2 * int(q) for q in a.shape))[0, 0].numpy()


def probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=128, halo=16, device=None, tta=0, luts=(), head=0, radial_sign=1.0, batch=1, rung=None,
          cascade=None, cascade_depth=3, calib=True):
    """Sliding-window recto probability (float32) over a box; returns (prob, checkpoint state).
    tta: number of axis flips to average; luts: intensity LUTs (uint8->float) whose predictions are averaged in;
    head: which OUTPUT CHANNEL to write -- an index, a channel name of the checkpoint ("recto" / "verso",
    from `args["channels"]`), "mean" / "prod" / "max" over all of them, or "all" for a (channels, Z, Y, X)
    result. radial_sign=-1 negates the radial vector: a recto-trained student then places its band on the
    other face of the sheet (the verso; see verso.py) -- the old flip trick, still the only way to get a
    verso band out of a cout=1 checkpoint, and unrelated to `--head verso`. batch > 1: windows batched on the GPU
    (slide_gpu; CT + radial inputs only, no context channels / luts / tta).

    head may also be a LIST of head names (`head_names(args)` is every one the checkpoint can serve): the
    result is then one array with a leading plane axis holding them in that order -- ONE sliding-window
    pass, one forward per window, for every field. `probs_multi` splits it into a dict. The arithmetic per
    plane is exactly what the single-head pass computes, because every head is a pointwise function of the
    SAME raw net output and the Gaussian blend is linear in it.
    cascade: None = whatever the checkpoint was trained with (`args["cascade"]`), False / "off" = force the
    channel to zero. cascade_depth: how many rungs above k are predicted top-down to fill it (default 3;
    the cost is geometric, 1/8 per level, so three levels add ~14 %). A checkpoint trained without the
    cascade channel ignores both and runs on 14 channels exactly as before."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.load(ckpt, map_location=dev)
    net = M.build(st["args"]["size"], verbose=False, cout=st["args"].get("cout", 1), cin=st["args"].get("cin", 4), add_skip=st["args"].get("add_skip", 0), deep=st["args"].get("deep", 0)).to(dev)
    net.load_state_dict(st["ema"])
    net.eval()
    scale = bool(st["args"].get("scale_plane"))
    k = int(rung) if rung is not None else (data.base_rung(volume) if scale else 2)
    ax0 = data.axis()
    if rung is None:
        roi = data.open_zarr(volume)[z0:z0 + Z, y0:y0 + Y, x0:x0 + X]
    else:  # read the CT at that rung of the pyramid (a rung above its top is pooled from the top)
        roi = data.read_rung(data.rungs(volume), k, (z0, y0, x0), (Z, Y, X)).astype(np.uint8)
    ax = data.axis_at(ax0, k)
    r = 0.0 if st["args"].get("no_radial") else float(radial_sign)  # training zeroed the radial channels
    data.NORM = tuple(st["args"]["norm_stats"]) if st["args"].get("norm") == "global" else None  # as trained
    ctx = tuple(st["args"].get("ctx") or ())
    kk = k if scale else None  # the scale plane only exists in checkpoints trained with it
    kr = k if (rung is not None or scale) else None  # legacy: let context() take the rung from the level path
    # CASCADE (docs/unified_design.md section 22): a checkpoint trained with --cascade takes a 15th channel,
    # the rung-(k+1) prediction over the same field of view. Inference is TOP-DOWN and recursive: the box's
    # footprint is predicted at rung k+1 first (half the size on every axis, so 1/8 of the work, plus a halo),
    # upsampled 2x and fed in. `cascade_depth` rungs above that the channel is zero, which is exactly the
    # `--cascade-drop` case the model was trained on. A checkpoint trained without it is unaffected.
    cmode = str(st["args"].get("cascade", "off") or "off")
    use_cas = cmode != "off"                       # the checkpoint HAS the channel: it must always be fed
    off = cascade is not None and (cascade is False or str(cascade) == "off")
    depth0 = 0 if (off or not use_cas) else max(int(cascade_depth), 0)  # 0 = the channel is fed as zeros
    multi = isinstance(head, (list, tuple))
    heads = [str(q) for q in head] if multi else [head]
    assert not multi or heads, "probs(head=[]): name at least one head"
    fld = None if multi else field_head(head, st["args"])
    if not multi:
        head = 0 if fld is not None else resolve_head(head, st["args"])
        pick = (lambda p: p) if head == "all" else HEADS[head] if isinstance(head, str) else (lambda p: p[:, int(head)])
    # A run with `--affinity` carries extra output channels that exist only for the training loss
    # (docs/unified_design.md section 26): inference never reads them, so the head is cut to `cout_t`
    # before anything selects or reduces over channels.
    ct_n = int(st["args"].get("cout_p", st["args"].get("cout_t", st["args"].get("cout", 1))))
    # the METADATA / RADIUS planes (section 29): built here exactly as `prep.fill_planes_` builds them
    # while training -- the radius from this volume's own axis and r_max, the five scan values from the
    # volume's own metadata.json (or `args["scan_meta"]`), zero for anything it does not say.
    pls = data.parse_planes(st["args"].get("planes") or ())
    pmeta = data.scan_planes(__import__("usrm2.scanmeta", fromlist=["load"]).load(
        st["args"].get("scan_meta") or data.pyramid_base(volume))) if "meta" in pls else None
    prmax = {}

    def planes_at(kk_, o, shape):
        if not pls:
            return None
        parts = []
        if "radius" in pls:
            if kk_ not in prmax:
                pyr = data.rungs(volume)
                kn = min(pyr)
                prmax[kk_] = data.rmax_vox(data.axis_at(ax0, kn), data.rung_shape(pyr, kn)) / (2.0 ** (kk_ - kn))
            parts.append(data.radius(data.axis_at(ax0, kk_), o, shape, prmax[kk_]))
        if pmeta is not None:
            parts.append(np.broadcast_to(pmeta[:, None, None, None], (len(pmeta),) + tuple(shape)))
        return np.concatenate(parts)
    # PER-RUNG TEMPERATURE (usrm2/calib.py): a no-op unless `usrm2 calibrate` has written `args["temps"]`
    from usrm2 import calib as CAL
    temp = lambda kk_: CAL.temp_for(st["args"], kk_, use=calib)  # noqa: E731
    raw = lambda t: net(t)  # noqa: E731
    logits = lambda t, kk_: net(t)[:, :ct_n] / temp(kk_)  # noqa: E731
    fn = lambda t: pick(torch.sigmoid(logits(t, k)))  # (B,C,...) -> (B,...)  (or (B,C,...) for "all")
    if fld is not None:   # a FIELD, not a probability: no sigmoid, no temperature
        hname, ci = fld
        # `field_head` returns the NAME asked for, and "midline" is the distance channel's own name in a
        # midline checkpoint -- not a fourth kind. Mapping it here is what makes `--head midline` the
        # distance and not (as the name test used to fall through to) the normals.
        kind = "sdist" if hname in ("sdist", "midline") else hname
        from usrm2 import losses as L
        if kind == "sdist":
            fn = lambda t: raw(t)[:, ci].float()
        elif kind == "thickness":
            fn = lambda t: L.soft_thickness(raw(t)[:, ci].float(), L.TMIN)
        elif kind == "conf":
            fn = lambda t: 1.0 / (1.0 + torch.exp(0.5 * raw(t)[:, ci].float().clamp(-8, 8)))
        else:  # normals: (B,3,w,w,w), ZYX, unit, sign verso -> recto (dot(n, radial) > 0)
            nh = [j for j, c in enumerate(st["args"].get("channels") or []) if c in ("nz", "ny", "nx")]
            fn = ((lambda t: L.normals_from(raw(t)[:, ci:ci + 1].float()))
                  if len(nh) != 3 else
                  (lambda t: (lambda v: v / v.norm(dim=1, keepdim=True).clamp_min(1e-4))(
                      raw(t)[:, nh[0]:nh[0] + 3].float())))
    plane_names, vec_tri = None, []
    if multi:   # ---- THE MULTI-HEAD PASS (section 30): one forward per window, every head off that output
        from usrm2 import losses as L
        nh = [j for j, c in enumerate(st["args"].get("channels") or []) if c in ("nz", "ny", "nx")]

        def piece(nm):
            """`nm` -> (f(raw output, rung) -> (B, C, ...), the plane names it contributes)."""
            f2 = field_head(nm, st["args"])
            if f2 is None:
                h2 = resolve_head(nm, st["args"])
                assert not isinstance(h2, str) or h2 == "all", \
                    f"probs(head=[...]): {nm!r} reduces over channels; name the channels instead"
                if h2 == "all":
                    return (lambda o, kk_: torch.sigmoid(o[:, :ct_n] / temp(kk_))), \
                        [str(c) for c in (st["args"].get("channels") or CHANNEL_DEFAULT)][:ct_n]
                return (lambda o, kk_, i=int(h2): torch.sigmoid(o[:, :ct_n] / temp(kk_))[:, i:i + 1]), [str(nm)]
            hn2, ci = f2
            kind = "sdist" if hn2 in ("sdist", "midline") else hn2
            if kind == "sdist":
                return (lambda o, kk_, i=ci: o[:, i:i + 1].float()), [str(nm)]
            if kind == "thickness":
                return (lambda o, kk_, i=ci: L.soft_thickness(o[:, i:i + 1].float(), L.TMIN)), ["thickness"]
            if kind == "conf":
                return (lambda o, kk_, i=ci: 1.0 / (1.0 + torch.exp(0.5 * o[:, i:i + 1].float().clamp(-8, 8)))), \
                    ["conf"]
            if len(nh) != 3:   # derived from the predicted distance field, exactly as the single pass does
                return (lambda o, kk_, i=ci: L.normals_from(o[:, i:i + 1].float())), ["nz", "ny", "nx"]
            return ((lambda o, kk_, i=nh[0]: (lambda v: v / v.norm(dim=1, keepdim=True).clamp_min(1e-4))(
                o[:, i:i + 3].float())), ["nz", "ny", "nx"])
        parts, plane_names = [], []
        for nm in heads:
            f2, nms = piece(nm)
            if nm == "normals":
                vec_tri.append(tuple(len(plane_names) + j for j in range(3)))
            parts.append(f2)
            plane_names += nms
        def fn(t):   # one forward per window, every head read off that one output

            o = net(t)
            return torch.cat([f2(o, k) for f2 in parts], 1)
    if tta > 1:
        assert head != "all", "tta and head=all do not combine"
        # A normal is a VECTOR: a flip of the world negates the component along the flipped axis, so the
        # multi pass and `--head normals` must average with `flips_chan`, not `flips_vec` (which would
        # flip the channel axis as if it were z and cancel the field).
        fn = flips_chan(fn, tta, vec=vec_tri) if multi else \
            (flips_chan(fn, tta, vec=[(0, 1, 2)]) if (fld is not None and fld[0] == "normals") else
             flips_vec(fn, tta))

    def cascade_for(kk_, o, s, depth):
        """The rung-(kk_+1) prediction over the footprint of the box (o, s), upsampled 2x onto that box's
        own grid; None when nothing above may be predicted (depth exhausted, or the top of the ladder)."""
        if depth <= 0 or kk_ + 1 >= data.NRUNGS:
            return None
        m = CASCADE_HALO
        o1 = [max(int(v) // 2 - m, 0) for v in o]
        s1 = [(int(v) + 1) // 2 + 2 * m for v in s]
        p1 = at_rung(kk_ + 1, o1, s1, depth - 1)
        up = up2x_np(p1)
        a = [int(o[i]) - 2 * o1[i] for i in range(3)]
        return np.ascontiguousarray(up[a[0]:a[0] + int(s[0]), a[1]:a[1] + int(s[1]), a[2]:a[2] + int(s[2])])

    def make_prep(kk_, o, casc, lut=None):
        axk = data.axis_at(ax0, kk_)

        def pr(c, off):
            g = (int(o[0]) + off[0], int(o[1]) + off[1], int(o[2]) + off[2])
            cc = crop_pad(casc, off, c.shape) if use_cas else None
            cxs = data.context(volume, g, c.shape, ctx, rung=kk_) if ctx else ()
            return data.inputs(c if lut is None else lut[c], data.radial(axk, g, c.shape) * r, cxs,
                               rung=(kk_ if scale else None), cascade=cc, planes=planes_at(kk_, g, c.shape))
        return pr

    def at_rung(kk_, o, s, depth):
        """Head-0 probability over a box at rung kk_ (rung-kk_ voxels), cascading `depth` rungs above it."""
        sub = data.read_rung(data.rungs(volume), kk_, o, s).astype(np.uint8)
        fn0 = lambda t: torch.sigmoid(logits(t, kk_))[:, 0]  # the coarse passes only ever need head 0
        return slide(fn0, sub, window, halo, dev, make_prep(kk_, o, cascade_for(kk_, o, s, depth)))

    if multi:   # what the caller needs to split the result: the plane name of every output plane
        st["head_planes"] = list(plane_names)
    casc = cascade_for(k, (z0, y0, x0), (Z, Y, X), depth0) if use_cas else None
    rad = lambda c, o: data.radial(ax, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape) * r
    cx = (lambda c, o: data.context(volume, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape, ctx, rung=kr)) if ctx else (lambda c, o: ())
    cas_at = (lambda c, o: crop_pad(casc, o, c.shape)) if use_cas else (lambda c, o: None)
    pl = lambda c, o: planes_at(k, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape)  # noqa: E731
    preps = [lambda c, o: data.inputs(c, rad(c, o), cx(c, o), rung=kk, cascade=cas_at(c, o), planes=pl(c, o))] + \
            [(lambda c, o, l=l: data.inputs(l[c], rad(c, o), cx(c, o), rung=kk, cascade=cas_at(c, o), planes=pl(c, o))) for l in luts]
    if batch > 1:
        assert not ctx and not luts and tta <= 1, "batched inference: CT + radial inputs only"
        assert fld is None, "batched inference (--batch > 1) does not serve the Phase B field heads"
        # the MULTI pass is fine batched: `slide_gpu` already accumulates a (B, C, w, w, w) output, and
        # the ladder of fields (distances are +-32 voxels) fits its fp16 accumulators with room to spare
        R = torch.from_numpy(data.radial(ax, (z0, y0, x0), roi.shape) * r).to(dev)
        C = None if casc is None else torch.from_numpy(casc).to(dev)
        norm = data.NORM

        def prep_t(c, o):
            x = c.float()
            x = (x - norm[0]) / norm[1] if norm else (x - x.mean()) / (x.std() + 1e-3)
            z, y, xx = o
            sl = (slice(z, z + window), slice(y, y + window), slice(xx, xx + window))
            parts = [x[None]]
            if use_cas:  # the cascade channel sits before the scale plane; the compiled net never sees the recursion
                parts.append((torch.zeros_like(x) if C is None else C[sl]).float()[None])
            if pls:
                parts.append(torch.from_numpy(np.ascontiguousarray(
                    planes_at(k, (z + z0, y + y0, xx + x0), tuple(x.shape)))).to(x.device).float())
            if scale:
                parts.append(torch.full_like(x, (k - 2) / 9.0)[None])
            parts.append(R[(slice(None),) + sl])
            return torch.cat(parts)
        return slide_gpu(fn, roi, window, halo, dev, prep_t, batch=batch), st
    return sum(slide(fn, roi, window, halo, dev, pr) for pr in preps) / len(preps), st


def probs_multi(ckpt, volume, z0, y0, x0, Z, Y, X, heads=None, **kw):
    """Every head of a checkpoint from ONE sliding-window pass: ({name: (Z,Y,X) float32}, state).

    `heads` defaults to `head_names(args)` -- the probability channels plus, when the checkpoint has a
    distance channel, the distance, the thickness, the normals and the confidence. The normal head
    contributes three planes, `nz`, `ny`, `nx`.

    Why one pass is correct and not merely cheaper: every head is a POINTWISE function of the same raw
    net output, and `slide`'s Gaussian blend is a per-voxel weighted mean of per-window values, so
    stacking the heads and blending once gives exactly what blending each separately gives. What it
    saves is the forward pass: `export-tracer` used to run five or six of them over the same box
    (docs/unified_design.md section 30)."""
    import torch as _t
    st0 = _t.load(ckpt, map_location="cpu")
    hs = list(heads) if heads else head_names(st0["args"])
    v, st = probs(ckpt, volume, z0, y0, x0, Z, Y, X, head=hs, **kw)
    names = st.get("head_planes") or hs
    assert v.shape[0] == len(names), f"multi-head pass: {v.shape[0]} planes for {names}"
    return {n: np.ascontiguousarray(v[i]) for i, n in enumerate(names)}, st


def predict(ckpt, volume, z0, y0, x0, Z, Y, X, out, window=128, halo=16, device=None, volcomp=True, tta=0, luts=(), head=0, radial_sign=1.0, rung=None,
            cascade=None, cascade_depth=3, calib=True):
    prob, st = probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=window, halo=halo, device=device, tta=tta, luts=luts, head=head, radial_sign=radial_sign, rung=rung,
                     cascade=cascade, cascade_depth=cascade_depth, calib=calib)
    h = resolve_head(head, st["args"])
    ch = [str(c) for c in (st["args"].get("channels") or CHANNEL_DEFAULT)]
    # the store says WHICH output it holds: the region loaders key the verso stores on it
    names = ch[:int(st["args"].get("cout", 1))] if h == "all" else \
        [ch[h] if isinstance(h, int) and h < len(ch) else str(h)]
    if radial_sign < 0 and names == ["recto"]:
        names = ["verso"]  # the flip trick: a recto head pointed at the other face
    write(out, prob, (z0, y0, x0), volcomp=volcomp, volume=volume, rung=2 if rung is None else int(rung),
          channels=names)
    return out


# =========================================================== the tracer contract (section 29)
# docs/research/synthesis_v2_with_literature.md section 2. The consumer (`fit_spiral.py` via
# `lasagna_data.py`) reads dense fields on the shard grid, in ZYX, in WORKING VOXELS of the store's own
# rung -- so `voxel_um` is recorded on every store and no field is ever pooled across rungs.
#
#     recto / verso   uint8   probability * 255                              (as today)
#     surf_sdist      uint8   d = (v - 128) * 0.25 voxels, cap +-31.75, v=0 = NO DATA,
#                             POSITIVE on the recto side (radially outward), i.e. outside the sheet body
#     nz / ny / nx    uint8   component = (v - 128) / 127, ZYX, unit vector, sign VERSO -> RECTO
#                             (equivalently dot(n, radial) > 0); v=0 = no data
#     gmag            uint8   |grad d| * 127, so 127 is the Eikonal ideal |grad d| = 1
#     conf            uint8   confidence * 255, from the heteroscedastic log-variance
#
# A `midline` checkpoint stores the RECTO-FACE convention: the export is the subtraction m - t/2, done
# here, so the training-time representation (which is what buys non-crossing) never reaches the tracer.

TRACER_UNIT = 0.25
TRACER_OFF = 128
TRACER_CAP = 31.75
NORMAL_SCALE = 127.0


def enc_signed(d, valid):
    """voxels -> uint8, code 0 = no data (`usrm2.targets.encode_signed`, in numpy here)."""
    c = np.rint(np.clip(d, -TRACER_CAP, TRACER_CAP) / TRACER_UNIT) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def enc_normal(n, valid):
    """a normal COMPONENT in -1..1 -> uint8, code 0 = no data."""
    c = np.rint(np.clip(n, -1.0, 1.0) * NORMAL_SCALE) + TRACER_OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def dec_normal(u):
    return (np.asarray(u, np.float32) - TRACER_OFF) / NORMAL_SCALE


def scharr3(d):
    """Scharr gradient of a (Z,Y,X) field, (3,Z,Y,X) in ZYX order.

    The contract is explicit that the normal is derived from the STORED distance field, never from the
    decoder's own autograd gradient: a ReLU/SiLU conv decoder has piecewise-constant gradients, so its
    analytic gradient is noisier than a finite difference of its output
    (`lit_implicit_surfaces_manifold.md`, the SIREN pitfall). Scharr is the rotationally best-behaved
    3x3 first derivative; the 3D kernel is the 1D derivative [-1, 0, 1] along the axis times the 1D
    smoother [3, 10, 3] on the other two, normalised so a unit ramp gives 1."""
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


def tracer_fields(sd, thick=None, conf=None, valid=None):
    """(recto-face sdist, normals, |grad|, valid) from a predicted distance field.

    `sd` is the model's distance field in voxels. With `thick` it is a MIDLINE distance and the
    recto-face field is `m - t/2` -- the subtraction the contract asks for, so a midline-trained model
    and a face-trained one export byte-comparable stores. The normal is the Scharr gradient of the
    EXPORTED field, normalised; because d grows outward the normal points verso -> recto."""
    d = np.asarray(sd, np.float32)
    if thick is not None:
        d = d - 0.5 * np.asarray(thick, np.float32)
    g = scharr3(d)
    mag = np.linalg.norm(g, axis=0)
    n = g / np.maximum(mag, 1e-4)
    v = np.ones(d.shape, bool) if valid is None else np.asarray(valid, bool)
    return d, n, mag, v


def export_tracer(ckpt, volume, z0, y0, x0, Z, Y, X, out, window=128, halo=16, device=None, rung=None,
                  cascade=None, cascade_depth=3, tta=0, marching_cubes=False, mc_level=0.0,
                  volcomp=True, log=print):
    """Write the tracer contract's stores over a box. Returns {name: path}.

    ONE `probs` pass for every field the checkpoint can produce -- recto, verso (when it has one), the
    distance, the thickness (when it has one) and the confidence (when it was trained `--sdist-hetero`)
    -- through `probs_multi`: the heads are pointwise functions of the same net output, so stacking them
    and blending once gives exactly what a pass each gives, at a fifth of the forward passes
    (docs/unified_design.md section 30; wave 2 shipped it as 5-6 passes over the same box). The normal
    field and the gradient magnitude are still derived HERE, from the exported distance field, with a
    Scharr kernel -- not read out of the net -- so what the tracer reads is exactly the gradient of what
    it reads.

    Every store is zarr v3 sharded, 128^3 inner chunks, volcomp q8, `compressors=None`
    (docs/unified_design.md 18.1b and 24), with `origin_zyx`, `voxel_um`, `rung`, `volume`, `umbilicus`
    and the encoding written into its attrs.

    `marching_cubes`: also run skimage's marching cubes on the ZERO LEVEL of the distance field, one
    SHARD at a time with a one-voxel halo for stitching, and write `<out>/mesh/shard_<z>_<y>_<x>.obj`
    with vertices in GLOBAL ZYX voxels of this rung. This is the step that replaces `make_surf_sdt.py`'s
    threshold-and-EDT round trip."""
    import os
    import torch
    st = torch.load(ckpt, map_location="cpu")
    ch = [str(c) for c in (st["args"].get("channels") or CHANNEL_DEFAULT)]
    npb = int(st["args"].get("cout_p", st["args"].get("cout_t", st["args"].get("cout", 1))))
    dch = next((c for c in ("sdist", "midline") if c in ch), None)
    assert dch is not None, f"{ckpt}: no distance channel (channels {ch}); train it with --sdist"
    k = int(rung) if rung is not None else 2
    kw = dict(window=window, halo=halo, device=device, rung=rung, cascade=cascade,
              cascade_depth=cascade_depth, tta=tta)
    os.makedirs(out, exist_ok=True)
    got = {}

    want = [c for c in ch[:npb]] + [dch] + (["thickness"] if "thickness" in ch else []) \
        + (["conf"] if "logvar" in ch else [])   # no "normals": the export derives them from the FIELD
    log(f"export-tracer {out}: rung {k}, channels {ch}, one pass for {want}")
    got_f, _ = probs_multi(ckpt, volume, z0, y0, x0, Z, Y, X, heads=want, **kw)
    rec = got_f["recto"] if "recto" in got_f else got_f[ch[0]]
    sd = got_f[dch]              # the FIELD, whatever the channel is called in this checkpoint
    th = got_f.get("thickness")
    cf = got_f.get("conf")
    ver = got_f.get("verso") if npb >= 2 else None
    d, n, mag, valid = tracer_fields(sd, th)
    valid = valid & (rec > 0)     # CT==0 is masked by both sides: `slide` already zeroes it

    def w(name, u8, enc, q=0):
        p = os.path.join(str(out), f"{name}.zarr")
        # q=0 (lossless) for every field: their code 0 means NO DATA and their other codes are a
        # distance or a normal component, not a probability the codec may round. The recto/verso
        # probability stores keep q=8, so they stay byte-comparable with every other prediction store.
        a = out_array(p, u8.shape, (z0, y0, x0), volcomp=volcomp,
                      volume=volume, rung=k, channels=(name,), q=q)
        a.attrs.update({"encoding": enc, "unit": "voxels_of_this_rung", "no_data": 0,
                        "axis_order": "ZYX", "sign_convention":
                        "d > 0 and n pointing from the VERSO face towards the RECTO face, i.e. "
                        "radially OUTWARD from the scroll axis (dot(n, radial) > 0)"})
        a[:] = u8[None] if a.ndim == 4 else u8
        got[name] = p
        return p

    w("recto", u8(rec), "prob_u8", q=8)
    if ver is not None:
        w("verso", u8(ver), "prob_u8", q=8)
    w("surf_sdist", enc_signed(d, valid), "signed_u8_off128_q0.25")
    for j, nm in enumerate(("nz", "ny", "nx")):
        w(nm, enc_normal(n[j], valid & (mag > 1e-3)), "normal_u8_off128_div127")
    w("gmag", np.clip(np.rint(mag * NORMAL_SCALE), 0, 255).astype(np.uint8), "gradmag_u8_x127")
    if cf is not None:
        w("conf", u8(cf), "conf_u8")
    if th is not None:
        w("thickness", np.clip(np.rint(th / TRACER_UNIT), 0, 255).astype(np.uint8), "unsigned_u8_q0.25")
    if marching_cubes:
        got["mesh"] = mesh_shards(d, valid, (z0, y0, x0), os.path.join(str(out), "mesh"),
                                  level=mc_level, log=log)
    log(f"export-tracer {out}: wrote {sorted(got)}")
    return got


def mesh_shards(d, valid, origin, out, level=0.0, shard=1024, log=print):
    """Marching cubes on the ZERO LEVEL of a distance field, one shard at a time, one .obj per shard.

    The shard grid is the store's own (`shard_shape`), and each shard is extracted with a ONE-VOXEL halo
    on its high faces so neighbouring shards' triangles meet: skimage's marching cubes places a vertex
    between two samples, so without that overlap there is a missing cell between shards. Vertices are
    written in GLOBAL ZYX voxels of this rung, which is the frame the rest of the contract uses (`v z y x`
    in the .obj, so an .obj reader's x is our z -- stated here and in the store attrs rather than silently
    swapped, because a swap is exactly the bug this contract exists to prevent).

    Invalid voxels are pushed to +cap, so the surface never closes over a no-data region.
    """
    import os
    from skimage import measure
    os.makedirs(out, exist_ok=True)
    f = np.where(valid, np.asarray(d, np.float32), TRACER_CAP)
    Z, Y, X = f.shape
    paths = []
    for z in range(0, Z, shard):
        for y in range(0, Y, shard):
            for x in range(0, X, shard):
                blk = f[z:min(z + shard + 1, Z), y:min(y + shard + 1, Y), x:min(x + shard + 1, X)]
                if blk.min() > level or blk.max() < level or min(blk.shape) < 2:
                    continue
                try:
                    v, tri, _, _ = measure.marching_cubes(blk, level=float(level))
                except (ValueError, RuntimeError) as e:
                    log(f"  shard {z},{y},{x}: marching cubes failed ({e!r})")
                    continue
                v = v + np.array([z + origin[0], y + origin[1], x + origin[2]], np.float32)
                p = os.path.join(out, f"shard_{z + origin[0]}_{y + origin[1]}_{x + origin[2]}.obj")
                with open(p, "w") as fh:
                    fh.write("# usrm2 export-tracer: vertices are GLOBAL ZYX voxels of this rung\n")
                    for q in v:
                        fh.write(f"v {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}\n")
                    for q in tri:
                        fh.write(f"f {q[0] + 1} {q[1] + 1} {q[2] + 1}\n")
                paths.append(p)
                log(f"  shard {z},{y},{x}: {len(v)} vertices, {len(tri)} triangles -> {os.path.basename(p)}")
    return out if paths else None
