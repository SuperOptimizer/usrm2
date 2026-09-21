"""Compact uint8 loader samples -> the model input, built on the GPU.

In rung mode the loader worker only READS (data.rung_item): the CT cube and the nine context cubes stay
uint8, the target and the weight stay uint8 (255 = 1.0), and the per-voxel float work -- the z-score of
every cube, the scale plane, the radial unit vector and the cube symmetry -- happens here, on the device,
once the batch has arrived. At 256^3 that turns ~1 GB of float32 per sample in the worker (14 channels
z-scored and stacked, then permuted and flipped) into ~200 MB of uint8, and moves ~13 s of CPU per sample
onto the card. `aug.apply` (the GPU intensity/spatial augs) runs after, unchanged.

tests/test_prep.py checks this path against the CPU one (data.inputs + data.sym_apply) for all 48 cube
symmetries, and predict.py still builds its inputs with data.inputs.
"""
import torch

from usrm2 import data


def sym_apply_t(sym, x, tg):
    """`data.sym_apply` on tensors: the cube symmetry `sym` (0..47) applied to a (B,C,Z,Y,X) input and a
    (B,T,Z,Y,X) target. The last 3 channels of x are the radial VECTOR and are permuted and negated with
    the axes, exactly as the CPU path does; every other channel -- the image cubes, the CASCADE channel and
    the scale plane -- is a plain spatial field and is only permuted and flipped. The channel count is read
    off the tensor, so 14- and 15-channel stacks both work."""
    perm, flip = data.sym_decode(sym)
    d = x.dim() - 3  # 2 with a batch dimension, 1 without
    ax = tuple(range(d)) + tuple(d + int(q) for q in perm)
    x, tg = x.permute(ax), tg.permute(ax)
    dims = [d + i for i, f in enumerate(flip) if f]
    if dims:
        x, tg = x.flip(dims), tg.flip(dims)
    ni = x.shape[d - 1] - 3
    idx = torch.as_tensor([ni + int(q) for q in perm], device=x.device)
    sh = [1] * x.dim()
    sh[d - 1] = 3
    sgn = torch.tensor([-1.0 if f else 1.0 for f in flip], device=x.device, dtype=x.dtype).view(sh)
    v = torch.index_select(x, d - 1, idx) * sgn
    return torch.cat([x.narrow(d - 1, 0, ni), v], d - 1).contiguous(), tg.contiguous()


def radial_t(cyx, lo, shape, dtype=torch.float32, out=None):
    """`data.radial` on the device: (B,3,Z,Y,X) unit vectors pointing away from the scroll axis in the xy
    plane (z component 0). cyx (B,2,Z) is the axis (y, x) at each z of the cube and lo (B,3) its corner;
    the differences are formed in float64 as they are on the CPU, only the normalization is float32.
    `out` (B,3,Z,Y,X), when given, is written in place: at 256^3 that is 200 MB of card not allocated
    twice. Only one full-size temporary (the norm) is ever made."""
    Z, Y, X = (int(v) for v in shape)
    dev = cyx.device
    ay = torch.arange(Y, device=dev, dtype=torch.float64) + lo[:, 1, None]
    ax = torch.arange(X, device=dev, dtype=torch.float64) + lo[:, 2, None]
    dy = (ay[:, None, :] - cyx[:, 0][:, :, None]).to(dtype)[..., None]   # (B,Z,Y,1)
    dx = (ax[:, None, :] - cyx[:, 1][:, :, None]).to(dtype)[:, :, None]  # (B,Z,1,X)
    n = (dy * dy + dx * dx).sqrt_().add_(1e-6)
    if out is None:
        out = torch.empty((cyx.shape[0], 3, Z, Y, X), device=dev, dtype=dtype)
    out[:, 0] = 0
    torch.div(dy, n, out=out[:, 1])
    torch.div(dx, n, out=out[:, 2])
    return out


def zscore_cubes_(img, norm, dtype):
    """z-score (B,C,Z,Y,X) image cubes in place with the sample's (mean, std): std 0 = the per-patch
    z-score, the semantics of data.zscore with NORM unset."""
    B = img.shape[0]
    glob = (norm[:, 1] > 0).view(B, 1, 1, 1, 1)
    m = torch.where(glob, norm[:, 0].view(B, 1, 1, 1, 1).to(dtype), img.mean((2, 3, 4), keepdim=True))
    s = torch.where(glob, norm[:, 1].view(B, 1, 1, 1, 1).to(dtype),
                    img.std((2, 3, 4), keepdim=True, correction=0) + 1e-3)
    return img.sub_(m).div_(s)


def prepare(b, dev, dtype=torch.float32, norad=False, non_blocking=True, cascade=None):
    """A collated batch of compact samples (data.rung_item) -> (x, target, weight) on `dev`.

    x is (B, 1 + len(ctx) + [1] + 1 + 3, Z, Y, X): every cube z-scored with the sample's norm (std 0 = the
    per-patch z-score, the semantics of data.zscore with NORM unset), the CASCADE channel (only when the
    sample carries the `cm` block: a probability in 0..1, not z-scored), the constant scale plane
    (k - 2) / 9 and the radial unit vector; then the worker's cube symmetry (every sample of a batch has the
    same patch shape, and `data.draw_sym` only draws permutations that keep it) applied to the whole stack,
    the cascade channel included -- it is a spatial channel with no vector part. target and weight come back
    as floats in 0..1. norad=True zeroes the radial channels (the --no-radial runs), as aug.apply's "norad"
    does. `cascade`: a `Cascade` (below) that says where the channel's values come from; None with a `cm`
    block present means the mask source with no noise and no dropout (what validation and the tests want)."""
    to = lambda t: t.to(dev, non_blocking=non_blocking)  # noqa: E731
    ct, tgt, w = to(b["ct"]), to(b["tgt"]), to(b["w"])
    lo, cyx, norm, rung = to(b["lo"]), to(b["cyx"]), to(b["norm"]), to(b["rung"])
    B, C, S = ct.shape[0], ct.shape[1], ct.shape[2:]
    casc = 1 if "cm" in b else 0
    x = torch.empty((B, C + 4 + casc) + tuple(S), dtype=dtype, device=ct.device)  # filled in place: no second copy
    img = x[:, :C]
    img.copy_(ct)
    zscore_cubes_(img, norm, dtype)
    if casc:
        cascade = cascade if cascade is not None else Cascade("mask", drop=0.0, noise=False)
        x[:, C:C + 1] = cascade.channel(b, x, norm, dtype, norad=norad, non_blocking=non_blocking)
    x[:, C + casc] = ((rung.to(dtype) - 2) / 9.0).view(B, 1, 1, 1)
    if norad:
        x[:, C + casc + 1:] = 0
    else:
        radial_t(cyx, lo, S, dtype, out=x[:, C + casc + 1:])
    tgt, w = tgt.to(dtype) / 255.0, w.to(dtype) / 255.0
    sym = [int(v) for v in b["sym"].reshape(-1).tolist()]
    if any(sym):  # one symmetry per sample, so the batch is done one sample at a time (B is 1 or 2)
        nt = tgt.shape[1]
        xs, ts = zip(*[sym_apply_t(v, x[i:i + 1], torch.cat([tgt[i:i + 1], w[i:i + 1]], 1))
                       for i, v in enumerate(sym)])
        x, tw = torch.cat(xs), torch.cat(ts)
        tgt, w = tw[:, :nt], tw[:, nt:]
    return x.contiguous(), tgt, w


class Cascade:
    """The source of the CASCADE input channel (docs/unified_design.md section 22).

    mode:
      "off"   nothing (the channel does not exist; the model is 14-channel).
      "mask"  the rung-(k+1) TARGET block over the patch footprint (`cm`, read by the worker), upsampled
              2x. Without noise that is a blurred copy of the rung-k target -- a leak -- so `noise` is on
              by default: a one-voxel erosion or dilation (prob 0.3) and 32^3 block dropout (prob 0.2)
              on top of the blur the 2x pool + 2x upsample already applies.
      "self"  the model's OWN rung-(k+1) prediction, computed here with the EMA weights (no grad,
              autocast bf16) from the cubes the sample already carries: its CT cube is ctx_1, its contexts
              are ctx_2..ctx_9 plus the tenth cube `cx`, its scale plane is (k+1-2)/9, its radial vector is
              recomputed at rung k+1, and its own cascade channel is ZERO (one-level truncation). The
              central half of the 256^3 output is the patch footprint; upsampled 2x it is the channel.
      "mix"   per sample, "self" with probability `self_p`, else "mask" (+ noise). The production mode:
              "mask" alone leaks the target, "self" alone never shows the model a coarse prediction better
              than its own, and the mixture brackets what inference actually feeds it.

    drop: probability that a sample's channel is zeroed altogether, so inference WITHOUT a coarse
    prediction (rung 11, or `--cascade-depth 0`) stays in distribution. Rung 11 is always zero: it is the
    top of the ladder and has no rung 12."""

    def __init__(self, mode="off", self_p=0.5, drop=0.1, noise=True, net=None, seed=None):
        self.mode = str(mode or "off")
        assert self.mode in data.CASCADE_MODES, f"cascade {mode}: one of {data.CASCADE_MODES}"
        self.self_p, self.drop, self.noise, self.net = float(self_p), float(drop), bool(noise), net
        self.gen = None if seed is None else torch.Generator().manual_seed(int(seed))

    @property
    def on(self):
        return self.mode != "off"

    def sync(self, ema):
        """Point the self-mode net at the current EMA weights (one pair of foreach kernels, like
        `train.ema_update`, not a 300-tensor python copy loop)."""
        if self.net is None or self.mode not in ("self", "mix"):
            return
        own = self.net.state_dict()
        es, vs = [], []
        for k, v in own.items():
            if k in ema:
                es.append(v), vs.append(ema[k])
        if es:
            torch._foreach_copy_(es, vs)

    def _rand(self, n=()):
        return torch.rand(n, generator=self.gen) if self.gen is not None else torch.rand(n)

    def _mask(self, b, i, dev, dtype, S):
        """The `mask` source for sample i: the coarse target block, optionally roughened, upsampled 2x."""
        from usrm2 import model as M
        c = b["cm"][i:i + 1].to(dev).to(dtype)[:, None] / 255.0
        if self.noise:
            r = float(self._rand())
            if r < 0.15:
                c = -torch.nn.functional.max_pool3d(-c, 3, stride=1, padding=1)   # erosion by one voxel
            elif r < 0.30:
                c = torch.nn.functional.max_pool3d(c, 3, stride=1, padding=1)     # dilation by one voxel
        u = M.up2x(c, tuple(S))
        if self.noise and float(self._rand()) < 0.20:  # block dropout: the model must not lean on it
            bs = [min(32, max(int(s) // 2, 1)) for s in S]
            for _ in range(4):
                o = [int(self._rand() * max(int(s) - q, 1)) for s, q in zip(S, bs)]
                u[..., o[0]:o[0] + bs[0], o[1]:o[1] + bs[1], o[2]:o[2] + bs[2]] = 0
        return u

    def coarse_input(self, b, i, dev, dtype, norad=False):
        """The rung-(k+1) model input of sample i, (1, C, Z, Y, X): exactly what `prepare` would build for
        a sample at rung k+1 over the same centre, with a ZERO cascade channel."""
        cubes = torch.cat([b["ct"][i, 1:], b["cx"][i]]).to(dev)[None]
        S = cubes.shape[2:]
        C = cubes.shape[1]
        x = torch.empty((1, C + 5) + tuple(S), dtype=dtype, device=dev)
        x[:, :C].copy_(cubes)
        zscore_cubes_(x[:, :C], b["norm"][i:i + 1].to(dev), dtype)
        x[:, C] = 0                                                   # its own cascade channel: truncated
        x[:, C + 1] = (float(int(b["rung"][i]) + 1) - 2) / 9.0
        if norad:
            x[:, C + 2:] = 0
        else:
            radial_t(b["cyx1"][i:i + 1].to(dev), b["lo1"][i:i + 1].to(dev), S, dtype, out=x[:, C + 2:])
        return x

    @torch.no_grad()
    def _self(self, b, i, dev, dtype, S):
        """The `self` source: one extra forward of the EMA net at rung k+1, its central half upsampled 2x."""
        from usrm2 import model as M
        from usrm2.train import autocast
        x = self.coarse_input(b, i, dev, dtype)
        was = self.net.training
        self.net.eval()
        with autocast(dev):
            y = self.net(x.to(memory_format=M.memfmt()))
        self.net.train(was)
        y = y[0] if isinstance(y, (list, tuple)) else y
        p = torch.sigmoid(y.float())[:, :1].to(dtype)
        sl = tuple(slice(int(s) // 4, int(s) // 4 + max(int(s) // 2, 1)) for s in S)
        return M.up2x(p[(slice(None), slice(None)) + sl], tuple(S))

    def channel(self, b, x, norm, dtype=torch.float32, norad=False, non_blocking=True):
        """(B,1,Z,Y,X) cascade channel for the batch."""
        dev, B, S = x.device, x.shape[0], x.shape[2:]
        out = torch.zeros((B, 1) + tuple(S), dtype=dtype, device=dev)
        if not self.on:
            return out
        rung = b["rung"].reshape(-1).tolist()
        for i in range(B):
            if int(rung[i]) + 1 >= data.NRUNGS:      # rung 11: no rung above it, so no coarse prediction
                continue
            if self.drop > 0 and float(self._rand()) < self.drop:
                continue
            if self.mode == "self" or (self.mode == "mix" and float(self._rand()) < self.self_p):
                assert self.net is not None and "cx" in b, "cascade self mode needs a net and the tenth context cube"
                out[i:i + 1] = self._self(b, i, dev, dtype, S)
            else:
                out[i:i + 1] = self._mask(b, i, dev, dtype, S)
        return out


def batch1(item):
    """One compact sample -> a batch of one (what the validation grid needs)."""
    return {k: v[None] for k, v in item.items()}


def shapes(item):
    """(input channels, output channels) of a compact sample: CT + context (+ the cascade channel, when the
    sample carries a `cm` block) + the scale plane + radial."""
    return int(item["ct"].shape[0]) + 4 + (1 if "cm" in item else 0), int(item["tgt"].shape[0])
