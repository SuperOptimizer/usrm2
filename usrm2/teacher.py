"""Run the upstream teacher (scrollprize/surface_recto_3dunet, villa 3D U-Net, 256^3 windows,
per-window z-score, 2-way softmax) over a box -> uint8 recto prob * 255 volcomp zarr."""
import numpy as np
import torch

from usrm2 import data
from usrm2.predict import out_array, put, slide, slide_gpu, zscore_t

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
    return lambda t: sum(torch.flip(fn(torch.flip(t, f)), [d - 1 for d in f]) for f in fl) / len(fl)  # fn: (B,C,..)->(B,..)


def lut_to(volume, ref=data.CT, n=30, seed=0):
    """uint8 LUT that histogram-matches `volume` to `ref` (non-air voxels of n random 128^3 patches each)."""
    def hist(v):
        a, rng, h, k, t = data.open_zarr(v), np.random.default_rng(seed), np.zeros(256), 0, 0
        while k < n:
            t += 1
            assert t < 100 * n, f"{v}: too few bright non-air patches for a histogram"
            o = rng.integers(0, np.array(a.shape) - 128); c = a[o[0]:o[0] + 128, o[1]:o[1] + 128, o[2]:o[2] + 128]
            if (c > 0).mean() >= 0.7 and c.mean() >= 30:
                h += np.bincount(c[c > 0].ravel(), minlength=256); k += 1
        return np.cumsum(h) / h.sum()
    lut = np.interp(hist(volume), hist(ref), np.arange(256)).astype(np.float32); lut[0] = 0
    return lut


def run(out, z0, y0, x0, Z, Y, X, volume=data.CT, window=256, halo=32, tile=2048, margin=128, ckpt=CKPT, device=None,
        tta=0, luts=(), backend="torch", gpu_acc=False, batch=1, streams=1):
    """tta: number of flips to average (0/1 = none, 8 = all). luts: extra intensity LUTs (uint8->float) whose
    predictions are averaged with the plain one (intensity TTA, e.g. lut_to(volume, other_scroll))."""
    """Tiles over y/x so RAM stays bounded. Each tile is read with a `margin` (>= half a window: the teacher
    is poor within ~32 voxels of a window edge, and the crop boundary must be covered by an interior window)."""
    dev = torch.device(device or "cuda")
    torch.backends.cudnn.benchmark = True  # one window shape all run long
    if backend == "trt":  # tsm's fp16 engine for this GPU (usrm2/trt.py)
        from usrm2 import trt
        net = trt.Engine(trt.plan("recto", window), dev)
        fn = lambda t: torch.softmax(net(t), 1)[:, 1]
    else:
        net = load(ckpt, dev)
        fn = lambda t: torch.softmax(net(t)["surface"].float(), 1)[:, 1]
    if tta > 1:
        fn = flips(fn, tta)
    preps = [None] + [(lambda c, _, l=l: data.zscore(l[c])[None]) for l in luts]
    if gpu_acc:  # everything on the card (predict.slide_gpu); intensity LUTs stay on the CPU path
        assert not luts, "gpu_acc has no LUT support"
        preps = [lambda c, _: zscore_t(c)[None]]
    ct, arr = data.open_zarr(volume), out_array(out, (Z, Y, X), (z0, y0, x0), volume=volume)
    import time
    for y in range(0, Y, tile):
        for x in range(0, X, tile):
            ya, yb, xa, xb = max(y - margin, 0), min(y + tile + margin, Y), max(x - margin, 0), min(x + tile + margin, X)
            t0 = time.time()
            roi = ct[z0:z0 + Z, y0 + ya:y0 + yb, x0 + xa:x0 + xb]
            t1 = time.time()
            sl = (lambda *a: slide_gpu(*a, batch=batch, streams=streams)) if gpu_acc else slide
            prob = sum(sl(fn, roi, window, halo, dev, pr) for pr in preps) / len(preps) if roi.any() else np.zeros(roi.shape, np.float32)
            prob = prob[:, y - ya:y - ya + tile, x - xa:x - xa + tile]
            t2 = time.time()
            u8 = np.empty(prob.shape, np.uint8)
            for z in range(0, prob.shape[0], 32):  # slab-wise: no full-size float temporaries (a box is 1.6 G voxels)
                u8[z:z + 32] = np.clip(np.rint(prob[z:z + 32] * 255), 0, 255)
            del prob
            put(arr, u8, 0, y, x)
            print(f"tile y={y} x={x} done: read {t1 - t0:.0f}s slide {t2 - t1:.0f}s write {time.time() - t2:.0f}s", flush=True)
    arr.attrs["done"] = True  # boxes() regenerates stores without it (interrupted runs)
    return out


def boxes(out_dir, n=50, size=(384, 2048, 2048), seed=0, volume=data.CT, exclude=data.VAL, min_mean=30, runner=None,
          shard=(0, 1), **kw):
    """Run the teacher over `n` random non-air boxes spread over the scroll -> out_dir/box_Z_Y_X.zarr each.
    Air test uses level 2 of the volume (1/4 pitch); boxes touching the val box are skipped. shard=(i, k): this
    process takes every k-th accepted box starting at i (the accepted sequence is deterministic in the seed, so k
    processes on one GPU cover the set exactly once; see `usrm2 teacher-boxes --procs`)."""
    from pathlib import Path
    rng = np.random.default_rng(seed)
    ct, lo = data.open_zarr(volume), data.open_zarr(volume.rstrip("/").rsplit("/", 1)[0] + "/2")
    ex = data.box(data.open_zarr(exclude)) if exclude else None
    size, shape, done, seen, tries = np.array(size), np.array(ct.shape), 0, set(), 0
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    while done < n:
        tries += 1
        assert tries < 200 * n, f"only {done} acceptable distinct boxes in {tries} draws (volume mostly air/masked?)"
        o = (rng.integers(0, shape - size) // 128) * 128
        if tuple(o) in seen:
            continue
        if ex is not None and np.all(o < ex[0] + ex[1]) and np.all(o + size > ex[0]):
            continue
        s = lo[o[0] // 4:(o[0] + size[0]) // 4, o[1] // 4:(o[1] + size[1]) // 4, o[2] // 4:(o[2] + size[2]) // 4]
        if s.mean() < min_mean or (s > 0).mean() < 0.7:
            continue
        out = f"{out_dir}/box_{o[0]}_{o[1]}_{o[2]}.zarr"
        seen.add(tuple(o))
        done += 1
        if (done - 1) % shard[1] != shard[0]:
            continue
        if not (Path(out).exists() and data.open_zarr(out).attrs.get("done")):  # absent or interrupted
            (runner or run)(out, *o, *size, volume=volume, **kw)
        print(f"box {done}/{n} {out}", flush=True)
