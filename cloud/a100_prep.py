#!/usr/bin/env python3
"""Where the 270 ms of `prep.prepare` goes: the host->device copy, the z-score, the radial vector and
the cube symmetry, timed separately on synthetic samples of the training shape."""
import time

import numpy as np
import torch

from usrm2 import data, prep


def T():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, n=6, warm=2):
    for _ in range(warm):
        fn()
    t = T()
    for _ in range(n):
        fn()
    return (T() - t) / n


dev = torch.device("cuda")
B, C, S = 2, 10, 256
rng = np.random.default_rng(0)
ct = rng.integers(0, 255, (B, C, S, S, S), dtype=np.uint8)
tg = rng.integers(0, 255, (B, 1, S, S, S), dtype=np.uint8)
b = {"ct": torch.from_numpy(ct).pin_memory(), "tgt": torch.from_numpy(tg).pin_memory(),
     "w": torch.from_numpy(tg).pin_memory(),
     "lo": torch.zeros(B, 3, dtype=torch.int64).pin_memory(),
     "cyx": torch.zeros(B, 2, S, dtype=torch.float64).pin_memory(),
     "norm": torch.zeros(B, 2).pin_memory(), "rung": torch.full((B,), 2), "sym": torch.zeros(B, dtype=torch.int64)}
bn = sum(v.numel() * v.element_size() for v in b.values())
print(f"batch {bn / 2**20:.0f} MiB (pinned)", flush=True)

dt = bench(lambda: {k: v.to(dev, non_blocking=True) for k, v in b.items()} and torch.cuda.synchronize())
print(f"H2D whole batch          : {dt*1e3:7.1f} ms  ({bn / dt / 2**30:5.2f} GiB/s)", flush=True)

d = {k: v.to(dev) for k, v in b.items()}
x = torch.empty((B, C + 4, S, S, S), dtype=torch.float32, device=dev)
img = x[:, :C]
print(f"x is {x.numel()*4 / 2**30:.2f} GiB", flush=True)


def zscore():
    img.copy_(d["ct"])
    m = img.mean((2, 3, 4), keepdim=True)
    s = img.std((2, 3, 4), keepdim=True, correction=0) + 1e-3
    img.sub_(m).div_(s)


print(f"uint8 -> fp32 copy       : {bench(lambda: img.copy_(d['ct']))*1e3:7.1f} ms", flush=True)
print(f"z-score (copy+mean+std+2): {bench(zscore)*1e3:7.1f} ms", flush=True)
print(f"radial_t into x          : "
      f"{bench(lambda: prep.radial_t(d['cyx'], d['lo'], (S, S, S), torch.float32, out=x[:, C+1:]))*1e3:7.1f} ms",
      flush=True)
t1 = x[:, :1].contiguous()
for sym in (0, 5, 23):
    print(f"sym_apply_t sym={sym:2d} (1 sample): "
          f"{bench(lambda: prep.sym_apply_t(sym, x[:1], t1[:1]), 4)*1e3:7.1f} ms", flush=True)
print(f"\nwhole prepare (sym 0)    : {bench(lambda: prep.prepare(b, dev), 4)*1e3:7.1f} ms", flush=True)
b["sym"] = torch.tensor([5, 23])
print(f"whole prepare (sym 5,23) : {bench(lambda: prep.prepare(b, dev), 4)*1e3:7.1f} ms", flush=True)
print(f"peak allocated {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
