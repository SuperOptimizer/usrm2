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
    the axes, exactly as the CPU path does."""
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


def prepare(b, dev, dtype=torch.float32, norad=False, non_blocking=True):
    """A collated batch of compact samples (data.rung_item) -> (x, target, weight) on `dev`.

    x is (B, 1 + len(ctx) + 1 + 3, Z, Y, X): every cube z-scored with the sample's norm (std 0 = the
    per-patch z-score, the semantics of data.zscore with NORM unset), the constant scale plane (k - 2) / 9
    and the radial unit vector; then the worker's cube symmetry (every sample of a batch has the same patch
    shape, and `data.draw_sym` only draws permutations that keep it). target and weight come back as floats in
    0..1. norad=True zeroes the radial channels (the --no-radial runs), as aug.apply's "norad" does."""
    to = lambda t: t.to(dev, non_blocking=non_blocking)  # noqa: E731
    ct, tgt, w = to(b["ct"]), to(b["tgt"]), to(b["w"])
    lo, cyx, norm, rung = to(b["lo"]), to(b["cyx"]), to(b["norm"]), to(b["rung"])
    B, C, S = ct.shape[0], ct.shape[1], ct.shape[2:]
    x = torch.empty((B, C + 4) + tuple(S), dtype=dtype, device=ct.device)  # filled in place: no second copy
    img = x[:, :C]
    img.copy_(ct)
    glob = (norm[:, 1] > 0).view(B, 1, 1, 1, 1)
    m = torch.where(glob, norm[:, 0].view(B, 1, 1, 1, 1).to(dtype), img.mean((2, 3, 4), keepdim=True))
    s = torch.where(glob, norm[:, 1].view(B, 1, 1, 1, 1).to(dtype),
                    img.std((2, 3, 4), keepdim=True, correction=0) + 1e-3)
    img.sub_(m).div_(s)
    x[:, C] = ((rung.to(dtype) - 2) / 9.0).view(B, 1, 1, 1)
    if norad:
        x[:, C + 1:] = 0
    else:
        radial_t(cyx, lo, S, dtype, out=x[:, C + 1:])
    tgt, w = tgt.to(dtype) / 255.0, w.to(dtype) / 255.0
    sym = [int(v) for v in b["sym"].reshape(-1).tolist()]
    if any(sym):  # one symmetry per sample, so the batch is done one sample at a time (B is 1 or 2)
        nt = tgt.shape[1]
        xs, ts = zip(*[sym_apply_t(v, x[i:i + 1], torch.cat([tgt[i:i + 1], w[i:i + 1]], 1))
                       for i, v in enumerate(sym)])
        x, tw = torch.cat(xs), torch.cat(ts)
        tgt, w = tw[:, :nt], tw[:, nt:]
    return x.contiguous(), tgt, w


def batch1(item):
    """One compact sample -> a batch of one (what the validation grid needs)."""
    return {k: v[None] for k, v in item.items()}


def shapes(item):
    """(input channels, output channels) of a compact sample: CT + context + the scale plane + radial."""
    return int(item["ct"].shape[0]) + 4, int(item["tgt"].shape[0])
