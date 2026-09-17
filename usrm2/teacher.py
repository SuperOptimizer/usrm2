"""Run the upstream teacher (scrollprize/surface_recto_3dunet, villa 3D U-Net, 256^3 windows,
per-window z-score, 2-way softmax) over a box -> uint8 recto prob * 255 volcomp zarr."""
import numpy as np
import torch

from usrm2 import data
from usrm2.predict import out_array, slide

CKPT = "/vesuvius/tsm/models/surface_recto_3dunet.pth"


def load(ckpt=CKPT, dev="cuda"):
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = sd["model_config"]
    mgr = type("Mgr", (), dict(model_config=mc, targets=mc["targets"], in_channels=mc["in_channels"],
                               train_patch_size=mc["train_patch_size"], train_batch_size=mc["train_batch_size"],
                               model_name=mc["model_name"], autoconfigure=False, spacing=(1, 1, 1),
                               enable_deep_supervision=False))()
    net = NetworkFromConfig(mgr)
    net.load_state_dict(sd["model"], strict=True)
    return net.to(dev).eval()


def flips(fn, n=8):
    """Test-time augmentation: average fn over the first n of the 8 axis flips (exact inverses)."""
    fl = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)][:n]
    return lambda t: sum(torch.flip(fn(torch.flip(t, f)), [d - 2 for d in f]) for f in fl) / len(fl)


def lut_to(volume, ref=data.CT, n=30, seed=0):
    """uint8 LUT that histogram-matches `volume` to `ref` (non-air voxels of n random 128^3 patches each)."""
    def hist(v):
        a, rng, h, k = data.open_zarr(v), np.random.default_rng(seed), np.zeros(256), 0
        while k < n:
            o = rng.integers(0, np.array(a.shape) - 128); c = a[o[0]:o[0] + 128, o[1]:o[1] + 128, o[2]:o[2] + 128]
            if (c > 0).mean() >= 0.7 and c.mean() >= 30:
                h += np.bincount(c[c > 0].ravel(), minlength=256); k += 1
        return np.cumsum(h) / h.sum()
    lut = np.interp(hist(volume), hist(ref), np.arange(256)).astype(np.float32); lut[0] = 0
    return lut


def run(out, z0, y0, x0, Z, Y, X, volume=data.CT, window=256, halo=32, tile=2048, margin=128, ckpt=CKPT, device=None,
        tta=0, luts=(), backend="torch"):
    """tta: number of flips to average (0/1 = none, 8 = all). luts: extra intensity LUTs (uint8->float) whose
    predictions are averaged with the plain one (intensity TTA, e.g. lut_to(volume, other_scroll))."""
    """Tiles over y/x so RAM stays bounded. Each tile is read with a `margin` (>= half a window: the teacher
    is poor within ~32 voxels of a window edge, and the crop boundary must be covered by an interior window)."""
    dev = torch.device(device or "cuda")
    torch.backends.cudnn.benchmark = True  # one window shape all run long
    if backend == "trt":  # tsm's fp16 engine for this GPU (usrm2/trt.py)
        from usrm2 import trt
        net = trt.Engine(trt.plan("recto", window), dev)
        fn = lambda t: torch.softmax(net(t), 1)[0, 1]
    else:
        net = load(ckpt, dev)
        fn = lambda t: torch.softmax(net(t)["surface"].float(), 1)[0, 1]
    if tta > 1:
        fn = flips(fn, tta)
    preps = [None] + [(lambda c, _, l=l: data.zscore(l[c])[None]) for l in luts]
    ct, arr = data.open_zarr(volume), out_array(out, (Z, Y, X), (z0, y0, x0), volume=volume)
    for y in range(0, Y, tile):
        for x in range(0, X, tile):
            ya, yb, xa, xb = max(y - margin, 0), min(y + tile + margin, Y), max(x - margin, 0), min(x + tile + margin, X)
            roi = ct[z0:z0 + Z, y0 + ya:y0 + yb, x0 + xa:x0 + xb]
            prob = sum(slide(fn, roi, window, halo, dev, pr) for pr in preps) / len(preps) if roi.any() else np.zeros(roi.shape, np.float32)
            prob = prob[:, y - ya:y - ya + tile, x - xa:x - xa + tile]
            arr[:, y:y + prob.shape[1], x:x + prob.shape[2]] = np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)
            print(f"tile y={y} x={x} done", flush=True)
    return out


def boxes(out_dir, n=50, size=(384, 2048, 2048), seed=0, volume=data.CT, exclude=data.VAL, min_mean=30, runner=None, **kw):
    """Run the teacher over `n` random non-air boxes spread over the scroll -> out_dir/box_Z_Y_X.zarr each.
    Air test uses level 2 of the volume (1/4 pitch); boxes touching the val box are skipped."""
    from pathlib import Path
    rng = np.random.default_rng(seed)
    ct, lo = data.open_zarr(volume), data.open_zarr(volume.rstrip("/").rsplit("/", 1)[0] + "/2")
    ex = data.box(data.open_zarr(exclude)) if exclude else None
    size, shape, done = np.array(size), np.array(ct.shape), 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    while done < n:
        o = (rng.integers(0, shape - size) // 128) * 128
        if ex is not None and np.all(o < ex[0] + ex[1]) and np.all(o + size > ex[0]):
            continue
        s = lo[o[0] // 4:(o[0] + size[0]) // 4, o[1] // 4:(o[1] + size[1]) // 4, o[2] // 4:(o[2] + size[2]) // 4]
        if s.mean() < min_mean or (s > 0).mean() < 0.7:
            continue
        out = f"{out_dir}/box_{o[0]}_{o[1]}_{o[2]}.zarr"
        if not Path(out).exists():
            (runner or run)(out, *o, *size, volume=volume, **kw)
        done += 1
        print(f"box {done}/{n} {out}", flush=True)
