"""3D augmentation, batched on the GPU (train.py calls `apply` once the batch has arrived).

Spatial augs (rot/scale/shear/elastic) resample CT + target trilinearly and transform the radial
unit-vector channels 1..3 by the same map: a direction transforms as M^-1 d (M is the out->in
affine of grid_sample), then it is renormalized. The elastic field deliberately does NOT rotate the
vectors by its local Jacobian: the displacements are small and smooth and the global rotation part
is already covered by `rot`. Intensity augs touch channel 0 only; "air" is the per-sample minimum
of the z-scored CT (CT 0 = air, so the minimum is the z-scored air value).
The 48 cube symmetries stay in the worker (data.augment), switched by cfg["sym"].
"""
import math

import torch
import torch.nn.functional as F


def _p(c, lo, hi, n=1):  # per-sample scalars shaped to broadcast over (B,C,Z,Y,X)
    return torch.empty(c.shape[0], n, 1, 1, 1, device=c.device, dtype=c.dtype).uniform_(lo, hi)


def _m(b, dev, p):
    return (torch.rand(b, device=dev) < p).float().view(b, 1, 1, 1, 1)


def _eye(b, dev):
    return torch.eye(3, device=dev).expand(b, 3, 3).clone()


def _rot(b, dev, k, m):
    a = torch.randn(b, 3, device=dev)
    a = a / a.norm(dim=1, keepdim=True)
    th = (torch.empty(b, device=dev).uniform_(-1, 1) * k["max_deg"] * math.pi / 180 * m.view(b)).view(b, 1, 1)
    z = torch.zeros(b, device=dev)
    K = torch.stack([z, -a[:, 2], a[:, 1], a[:, 2], z, -a[:, 0], -a[:, 1], a[:, 0], z], 1).view(b, 3, 3)
    return _eye(b, dev) + th.sin() * K + (1 - th.cos()) * (K @ K)


def _scale(b, dev, k, m):
    s = torch.exp(torch.empty(b, 3, device=dev).uniform_(math.log(k["lo"]), math.log(k["hi"])))
    s = torch.where(torch.rand(b, 1, device=dev) < k["iso"], s[:, :1], s)  # isotropic or per-axis
    return torch.diag_embed(1 / (1 + (s - 1) * m.view(b, 1)))  # out->in is the inverse zoom


def _shear(b, dev, k, m):
    o = torch.empty(b, 3, 3, device=dev).uniform_(-k["max"], k["max"]) * (1 - torch.eye(3, device=dev))
    return _eye(b, dev) + o * m.view(b, 1, 1)


def _elastic(grid, k, m):
    b, S = grid.shape[0], grid.shape[1:4]
    c = max(2, min(S) // k["grid"])
    d = torch.randn(b, 3, c, c, c, device=grid.device)
    d = F.interpolate(d, size=S, mode="trilinear", align_corners=False)
    return grid + (d * (2 * k["sigma"] / min(S)) * m).permute(0, 2, 3, 4, 1)


def warp(x, tg, M, el=None):
    """Resample (B,4,Z,Y,X) input + (B,1,Z,Y,X) target by the out->in matrix M (B,3,3), xyz order."""
    b, dev, S = x.shape[0], x.device, x.shape[2:]
    grid = F.affine_grid(torch.cat([M, torch.zeros(b, 3, 1, device=dev)], 2), (b, 1, *S), align_corners=False)
    if el is not None:
        grid = _elastic(grid, *el)
    g = dict(mode="bilinear", padding_mode="reflection", align_corners=False)
    o, tg = F.grid_sample(x.float(), grid, **g), F.grid_sample(tg.float(), grid, **g)
    v = torch.einsum("bij,bjzyx->bizyx", M.inverse(), o[:, [3, 2, 1]])[:, [2, 1, 0]]  # zyx <-> xyz
    n = v.norm(dim=1, keepdim=True)
    return torch.cat([o[:, :1], v / n.clamp_min(1e-6) * (n > 1e-3)], 1), tg.clamp(0, 1)


def spatial(x, tg, cfg):
    if not any(k in cfg for k in ("rot", "scale", "shear", "elastic")):
        return x, tg
    b, dev = x.shape[0], x.device
    M = _eye(b, dev)
    for name, f in (("rot", _rot), ("scale", _scale), ("shear", _shear)):
        if name in cfg:
            M = M @ f(b, dev, cfg[name], _m(b, dev, cfg[name]["p"]))
    el = cfg.get("elastic")
    return warp(x, tg, M, (el, _m(b, dev, el["p"])) if el else None)


def _gamma(c, k):
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    n = ((c - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return n.pow(torch.exp(_p(c, -1, 1) * math.log(k["max"]))) * (hi - lo) + lo


def _contrast(c, k):
    m = c.mean((2, 3, 4), True)
    return m + (c - m) * torch.exp(_p(c, -1, 1) * math.log(k["max"]))


def _bright(c, k):
    return c + _p(c, -k["max"], k["max"])


def _noise(c, k):
    return c + torch.randn_like(c) * _p(c, 0, k["max"])


def _mulnoise(c, k):
    return c * (1 + torch.randn_like(c) * _p(c, 0, k["max"]))


def _blur1(c, s):  # separable gaussian, one sigma for the whole batch
    r = max(1, int(2 * s))
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=c.device, dtype=c.dtype) / s) ** 2)
    g = g / g.sum()
    for d in range(3):
        pad = [0] * 6
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        sh = [1, 1, 1, 1, 1]
        sh[2 + d] = -1
        c = F.conv3d(F.pad(c, pad, mode="replicate"), g.view(sh))
    return c


def _blur(c, k):
    return _blur1(c, float(torch.empty(1).uniform_(k["lo"], k["hi"])))


def _sharpen(c, k):
    return c + float(torch.empty(1).uniform_(0, k["max"])) * (c - _blur1(c, k["sigma"]))


def _lowres(c, k):
    f = float(torch.empty(1).uniform_(k["lo"], k["hi"]))
    S = c.shape[2:]
    d = F.interpolate(c, size=[max(2, int(s / f)) for s in S], mode="nearest")
    return F.interpolate(d, size=S, mode="trilinear", align_corners=False)


def _bias(c, k):  # low-frequency multiplicative gain (beam hardening / cupping)
    f = torch.randn(c.shape[0], 1, k["grid"], k["grid"], k["grid"], device=c.device, dtype=c.dtype)
    return c * (1 + F.interpolate(f, size=c.shape[2:], mode="trilinear", align_corners=False) * _p(c, 0, k["max"]))


def _clip(c, k):
    m, s = c.mean((2, 3, 4), True), c.std((2, 3, 4), True)
    a = _p(c, k["lo"], k["hi"]) * s
    return torch.maximum(torch.minimum(c, m + a), m - a)


def _boxes(c, k):  # set k["n"] random boxes to the air value
    b, dev, S = c.shape[0], c.device, c.shape[2:]
    hit = torch.zeros_like(c, dtype=torch.bool)
    for _ in range(k["n"]):
        sel = torch.ones(b, 1, 1, 1, 1, dtype=torch.bool, device=dev)
        for d, s in enumerate(S):
            n = torch.randint(max(1, int(k["lo"] * s)), max(2, int(k["hi"] * s)), (b, 1), device=dev)
            o = (torch.rand(b, 1, device=dev) * (s - n)).long()
            i = torch.arange(s, device=dev)
            sh = [b, 1, 1, 1, 1]
            sh[2 + d] = s
            sel = sel & ((i >= o) & (i < o + n)).view(sh)
        hit |= sel
    return torch.where(hit, c.amin((2, 3, 4), True), c)


INTENS = [("bias", _bias), ("gamma", _gamma), ("contrast", _contrast), ("bright", _bright),
          ("blur", _blur), ("sharpen", _sharpen), ("lowres", _lowres), ("noise", _noise),
          ("mulnoise", _mulnoise), ("clip", _clip), ("airbox", _boxes), ("cutout", _boxes)]


def intensity(c, cfg):
    for name, f in INTENS:
        k = cfg.get(name)
        if k:
            c = torch.where(_m(c.shape[0], c.device, k["p"]).bool(), f(c, k), c)
    return c


def apply(x, tg, cfg):
    """(B,4,Z,Y,X) input + (B,1,Z,Y,X) target -> augmented pair (float32)."""
    if not cfg:
        return x, tg
    x, tg = spatial(x.float(), tg.float(), cfg)
    x = torch.cat([intensity(x[:, :1], cfg), x[:, 1:]], 1)
    if cfg.get("norad"):
        x[:, 1:] = 0
    return x, tg


SPATIAL = {"rot": {"p": 0.5, "max_deg": 30}, "scale": {"p": 0.3, "lo": 0.8, "hi": 1.25, "iso": 0.5},
           "shear": {"p": 0.2, "max": 0.1}, "elastic": {"p": 0.3, "sigma": 4.0, "grid": 12}}
INTENSITY = {"gamma": {"p": 0.3, "max": 1.8}, "contrast": {"p": 0.3, "max": 1.5},
             "bright": {"p": 0.3, "max": 0.3}, "noise": {"p": 0.3, "max": 0.2},
             "mulnoise": {"p": 0.2, "max": 0.15}, "blur": {"p": 0.2, "lo": 0.5, "hi": 1.5},
             "sharpen": {"p": 0.2, "max": 1.0, "sigma": 1.0}, "lowres": {"p": 0.25, "lo": 1.5, "hi": 4.0},
             "bias": {"p": 0.3, "max": 0.3, "grid": 4}, "clip": {"p": 0.15, "lo": 1.5, "hi": 3.0},
             "airbox": {"p": 0.15, "n": 1, "lo": 0.1, "hi": 0.4}}
CUTOUT = {"cutout": {"p": 0.25, "n": 3, "lo": 0.05, "hi": 0.25}}


def _pre(*ds):
    return {"sym": True, **{k: v for d in ds for k, v in d.items()}}


PRESETS = {
    "none": {"sym": False},
    "geo": _pre(),  # the 48 cube symmetries only (current behaviour)
    "geo+intensity": _pre(INTENSITY),
    "geo+lowres": _pre({k: INTENSITY[k] for k in ["lowres"]}),
    "geo+cutout": _pre(CUTOUT),
    "geo+elastic": _pre({k: SPATIAL[k] for k in ["elastic"]}),
    "geo+rot": _pre({k: SPATIAL[k] for k in ["rot"]}),
    "geo+scale": _pre({k: SPATIAL[k] for k in ["scale"]}),
    "geo+bias": _pre({k: INTENSITY[k] for k in ["bias"]}),
    "all": _pre(SPATIAL, INTENSITY, CUTOUT),
    "all_norad": _pre(SPATIAL, INTENSITY, CUTOUT, {"norad": True}),
}


def get(name):
    return PRESETS[name]
