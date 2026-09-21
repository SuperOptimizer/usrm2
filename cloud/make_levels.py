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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usrm2.predict import shard_shape  # noqa: E402  (every store usrm2 writes is zarr v3 sharded)
SHARD = shard_shape(T)
tmp = f"{base}/{dst}.new"
resume = os.path.exists(f"{tmp}/zarr.json") and "--fresh" not in sys.argv  # an interrupted build continues
if resume and tuple(getattr(zarr.open(tmp, mode="r"), "shards", None) or ()) != tuple(SHARD):
    resume = False  # a level left over from the unsharded layout: rebuild it rather than mix layouts
if resume:
    out = zarr.open(tmp, mode="r+")
else:
    shutil.rmtree(tmp, ignore_errors=True)
    out = zarr.create_array(tmp, shape=tuple(int(v) for v in T), chunks=(128, 128, 128), shards=SHARD,
                            dtype="uint8", fill_value=0, overwrite=True, serializer=VolcompCodec(q=q))
out.attrs.update(dict(a.attrs)); out.attrs.update({"level": dst, "pooled_from": src, "volcomp_q": q})
# One job = one SHARD (up to 1024^3): the level is written one whole shard at a time, so a shard file is
# never rewritten and resume is per shard file (c/<sz>/<sy>/<sx>) instead of per 128^3 chunk.
jobs = [(z, y, x) for z in range(0, T[0], SHARD[0]) for y in range(0, T[1], SHARD[1]) for x in range(0, T[2], SHARD[2])]


def one(j):
    z, y, x = j
    if resume and os.path.exists(f"{tmp}/c/{z // SHARD[0]}/{y // SHARD[1]}/{x // SHARD[2]}"):
        return 0  # written by the interrupted run (all-air shards are recomputed, they are cheap)
    ez, ey, ex = (int(min(z + SHARD[0], T[0])), int(min(y + SHARD[1], T[1])), int(min(x + SHARD[2], T[2])))
    buf, any_ = np.zeros((ez - z, ey - y, ex - x), np.uint8), False
    for bz in range(z, ez, 128):  # pooled 128^3 at a time: no 2048^3 float temporaries
        for by in range(y, ey, 128):
            for bx in range(x, ex, 128):
                lo = np.array([bz, by, bx]) * 2; hi = np.minimum(lo + 256, S)
                blk = np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], np.float32)
                if not blk.any():
                    continue
                blk = np.pad(blk, [(0, (2 - s % 2) % 2) for s in blk.shape])
                pl = blk.reshape(blk.shape[0] // 2, 2, blk.shape[1] // 2, 2, blk.shape[2] // 2, 2).mean((1, 3, 5))
                pl = np.rint(pl).astype(np.uint8)[:ez - bz, :ey - by, :ex - bx]
                buf[bz - z:bz - z + pl.shape[0], by - y:by - y + pl.shape[1], bx - x:bx - x + pl.shape[2]] = pl
                any_ = True
    if not any_:
        return 0
    out[z:ez, y:ey, x:ex] = buf  # one whole-shard write
    return 1


t0 = time.time()
with ThreadPoolExecutor(threads) as ex:
    n = sum(ex.map(one, jobs))
old = f"{base}/{dst}"
if os.path.exists(old):
    shutil.rmtree(old + ".old", ignore_errors=True); os.rename(old, old + ".old")
os.rename(tmp, old)
shutil.rmtree(old + ".old", ignore_errors=True)
print(f"level {dst} <- {src} at q={q}: shape {tuple(int(v) for v in T)}, {n} non-air shards, {time.time() - t0:.0f} s", flush=True)
