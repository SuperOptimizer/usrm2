#!/usr/bin/env python3
"""Build a pyramid level of the local volcomp mirror OFFLINE by 2x mean pooling, at a chosen volcomp q, into
<base>/<dst>.new and swap it in atomically when done (readers keep using the old level meanwhile).
Quality per level (user): 0 q=8, 1 q=4, 2 q=2, 3-5 q=1.
    python cloud/make_levels.py BASE SRC DST Q [--threads N]
"""
import os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, zarr
from volcomp_zarr import VolcompCodec

base, src, dst, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 8
a = zarr.open(f"{base}/{src}", mode="r")
S = np.array(a.shape); T = (S + 1) // 2
tmp = f"{base}/{dst}.new"
resume = os.path.exists(f"{tmp}/zarr.json") and "--fresh" not in sys.argv  # an interrupted build continues
if resume:
    out = zarr.open(tmp, mode="r+")
else:
    shutil.rmtree(tmp, ignore_errors=True)
    out = zarr.create_array(tmp, shape=tuple(int(v) for v in T), chunks=(128, 128, 128), dtype="uint8", fill_value=0,
                            overwrite=True, serializer=VolcompCodec(q=q))
out.attrs.update(dict(a.attrs)); out.attrs.update({"level": dst, "pooled_from": src, "volcomp_q": q})
jobs = [(z, y, x) for z in range(0, T[0], 128) for y in range(0, T[1], 128) for x in range(0, T[2], 128)]


def one(j):
    z, y, x = j
    if resume and os.path.exists(f"{tmp}/c/{z // 128}/{y // 128}/{x // 128}"):
        return 0  # written by the interrupted run (all-air chunks are recomputed, they are cheap)
    lo = np.array([z, y, x]) * 2; hi = np.minimum(lo + 256, S)
    blk = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
    if not blk.any():
        return 0
    pad = [(0, (2 - s % 2) % 2) for s in blk.shape]
    blk = np.pad(blk, pad)
    p = blk.reshape(blk.shape[0] // 2, 2, blk.shape[1] // 2, 2, blk.shape[2] // 2, 2).mean((1, 3, 5))
    out[z:z + p.shape[0], y:y + p.shape[1], x:x + p.shape[2]] = np.rint(p).astype(np.uint8)
    return 1


t0 = time.time()
with ThreadPoolExecutor(threads) as ex:
    n = sum(ex.map(one, jobs))
old = f"{base}/{dst}"
if os.path.exists(old):
    shutil.rmtree(old + ".old", ignore_errors=True); os.rename(old, old + ".old")
os.rename(tmp, old)
shutil.rmtree(old + ".old", ignore_errors=True)
print(f"level {dst} <- {src} at q={q}: shape {tuple(int(v) for v in T)}, {n} non-air chunks, {time.time() - t0:.0f} s", flush=True)
