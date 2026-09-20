#!/usr/bin/env python3
"""Micro-benchmarks that separate "the kernels are slow" from "the card is starved".

On a virtualised / proxied GPU (Thunder Compute) every CUDA API call pays a network round trip, so a
model made of many small kernels can sit at a few per cent of peak while each kernel itself runs at full
speed. These probes say which it is:

  matmul   achieved bf16 TFLOPS on one big GEMM (pure kernel speed)
  launch   wall time per trivial kernel launch (API/proxy latency)
  h2d      host->device bandwidth, pinned and pageable
  conv3d   the U-Net's hot convolutions, measured against their own FLOP count
  graph    the same conv chain replayed from a CUDA graph (no per-kernel API calls)
"""
import sys
import time

import torch
import torch.nn.functional as F


def T():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    t = T()
    for _ in range(n):
        fn()
    return (T() - t) / n


dev = torch.device("cuda")
torch.backends.cudnn.benchmark = "nobench" not in sys.argv
print(torch.cuda.get_device_name(0), "cudnn.benchmark", torch.backends.cudnn.benchmark, flush=True)

# --- pure GEMM
for n in (4096, 8192):
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    dt = bench(lambda: a @ b, 20)
    print(f"matmul bf16 {n}^3: {dt * 1e3:7.2f} ms  {2 * n ** 3 / dt / 1e12:6.1f} TFLOPS", flush=True)

# --- launch latency
x = torch.zeros(1024, device=dev)
dt = bench(lambda: [x.add_(1.0) for _ in range(100)] and None, 20)
print(f"launch: {dt / 100 * 1e6:7.1f} us per trivial kernel (100 per iteration)", flush=True)
t = time.perf_counter()
for _ in range(100):
    x.add_(1.0)
print(f"launch (queue only, no sync): {(time.perf_counter() - t) / 100 * 1e6:7.1f} us", flush=True)

# --- H2D
for pin in (True, False):
    h = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, pin_memory=pin)
    d = torch.empty_like(h, device=dev)
    dt = bench(lambda: d.copy_(h, non_blocking=pin), 10)
    print(f"h2d {'pinned  ' if pin else 'pageable'} 256 MB: {dt * 1e3:7.1f} ms  {h.numel() / dt / 2 ** 30:6.2f} GiB/s", flush=True)
    del h, d
torch.cuda.empty_cache()

# --- the U-Net's hot convolutions, channels_last_3d, bf16 autocast
for (b, c, s) in ((2, 32, 256), (2, 64, 128), (2, 128, 64), (2, 256, 32)):
    x = torch.randn(b, c, s, s, s, device=dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d)
    w = torch.randn(c, c, 3, 3, 3, device=dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d)
    fl = 2 * 27 * c * c * b * s ** 3
    dt = bench(lambda: F.conv3d(x, w, padding=1), 10)
    print(f"conv3d {c:3d}->{c:3d} b{b} {s:3d}^3: {dt * 1e3:8.2f} ms  {fl / dt / 1e12:6.1f} TFLOPS "
          f"({x.numel() / 1e9:.2f}e9 elements)", flush=True)
    del x, w
torch.cuda.empty_cache()

# --- GroupNorm + SiLU at the widest level (the memory-bound part)
for dt_ in (torch.bfloat16, torch.float32):
    x = torch.randn(2, 32, 256, 256, 256, device=dev, dtype=dt_).to(memory_format=torch.channels_last_3d)
    g = torch.nn.GroupNorm(8, 32).to(dev).to(dt_)
    dt = bench(lambda: F.silu(g(x)), 5)
    nb = x.numel() * x.element_size()
    print(f"groupnorm+silu 32ch 2x256^3 {dt_}: {dt * 1e3:8.2f} ms  ({3 * nb / dt / 2 ** 30:.0f} GiB/s eff.)", flush=True)
    dt = bench(lambda: x + x, 5)
    print(f"  plain add  same shape {dt_}: {dt * 1e3:8.2f} ms  ({3 * nb / dt / 2 ** 30:.0f} GiB/s)", flush=True)
    del x, g
    torch.cuda.empty_cache()
torch.cuda.empty_cache()

# --- trilinear upsample (the decoder's every stage) and its backward
for (c, s) in ((64, 128), (128, 64)):
    x = torch.randn(2, c, s, s, s, device=dev, dtype=torch.bfloat16, requires_grad=True).to(memory_format=torch.channels_last_3d)
    f = lambda: F.interpolate(x, size=(2 * s,) * 3, mode="trilinear", align_corners=False)  # noqa: E731
    dt = bench(f, 5)
    y = f()
    go = torch.ones_like(y)
    dtb = bench(lambda: torch.autograd.grad(f(), x, go, retain_graph=False)[0], 5)
    print(f"interpolate trilinear {c}ch {s}->{2 * s}: fwd {dt * 1e3:7.1f} ms  fwd+bwd {dtb * 1e3:7.1f} ms", flush=True)
    del x, y, go
    torch.cuda.empty_cache()
