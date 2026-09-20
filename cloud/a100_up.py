#!/usr/bin/env python3
"""Is the decoder's trilinear 2x upsample replaceable by a cheaper, EXACTLY equal operation?

`F.interpolate(mode="trilinear", align_corners=False)` at an exact factor of 2 is, per axis,
    out[2m]   = 0.75 in[m] + 0.25 in[m-1]
    out[2m+1] = 0.75 in[m] + 0.25 in[m+1]
with the out-of-range neighbour clamped to the edge, i.e. exactly a nearest 2x upsample followed by
a separable [1/4, 1/2, 1/4] filter with replicate padding. The nearest+filter form has a gather
backward; the interpolate form's backward is a scatter-add (`_unsafe_index_put`), which at 2x64x256^3
takes over a second on an A100.

Prints max |difference| (must be 0 in fp32) and the forward / forward+backward times of both.
"""
import time

import torch
import torch.nn.functional as F


def T():
    torch.cuda.synchronize()
    return time.perf_counter()


def bench(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    t = T()
    for _ in range(n):
        fn()
    return (T() - t) / n


def up2x(x):
    """The exact trilinear 2x upsample, as nearest 2x + a separable [1/4,1/2,1/4] blur."""
    C = x.shape[1]
    y = F.interpolate(x, scale_factor=2, mode="nearest")
    k = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype)
    w = (k.view(3, 1, 1) * k.view(1, 3, 1) * k.view(1, 1, 3)).expand(C, 1, 3, 3, 3).contiguous()
    return F.conv3d(F.pad(y, (1,) * 6, mode="replicate"), w, groups=C)


def up2x_sep(x):
    """Same, with the blur as three 1D depthwise convolutions instead of one 27-tap one."""
    C = x.shape[1]
    y = F.interpolate(x, scale_factor=2, mode="nearest")
    k = torch.tensor([0.25, 0.5, 0.25], device=x.device, dtype=x.dtype)
    for d in range(3):
        sh = [1, 1, 1, 1, 1]
        sh[2 + d] = 3
        pad = [0] * 6
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = 1
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.view(sh).expand(C, 1, *sh[2:]).contiguous(), groups=C)
    return y


def up2x_shift(x):
    """The exact trilinear 2x upsample as three separable interleaves: per axis, the two output
    planes of input plane m are 0.75 in[m] + 0.25 in[m-1] and 0.75 in[m] + 0.25 in[m+1] (the
    out-of-range neighbour clamped), interleaved with stack+flatten. Every operation is a gather,
    so the backward is a gather too -- no scatter-add over 2e9 elements."""
    for d in range(3):
        n = x.shape[2 + d]
        prev = torch.cat([x.narrow(2 + d, 0, 1), x.narrow(2 + d, 0, n - 1)], 2 + d)
        nxt = torch.cat([x.narrow(2 + d, 1, n - 1), x.narrow(2 + d, n - 1, 1)], 2 + d)
        x = torch.stack([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt], 3 + d).flatten(2 + d, 3 + d)
    return x


dev = torch.device("cuda")
for dt in (torch.float32, torch.bfloat16):
    for (c, s) in ((64, 128), (128, 64), (8, 16)):
        x = torch.randn(2, c, s, s, s, device=dev, dtype=dt).to(memory_format=torch.channels_last_3d)
        a = F.interpolate(x, size=(2 * s,) * 3, mode="trilinear", align_corners=False)
        for name, f in (("blur27", up2x), ("blur1d", up2x_sep), ("shift", up2x_shift)):
            b = f(x)
            print(f"{dt} {c}ch {s}->{2*s} {name}: max abs diff {float((a - b).abs().max()):.3e} "
                  f"rel {float((a - b).abs().max() / a.abs().max()):.3e}", flush=True)
        del x, a, b
        torch.cuda.empty_cache()

print()
for (c, s) in ((64, 128), (128, 64)):
    for name, f in (("interpolate", lambda x: F.interpolate(x, size=(2 * x.shape[-1],) * 3, mode="trilinear", align_corners=False)),
                    ("shift", up2x_shift), ("shift-compiled", torch.compile(up2x_shift))):
        x = torch.randn(2, c, s, s, s, device=dev, dtype=torch.bfloat16,
                        requires_grad=True).to(memory_format=torch.channels_last_3d)
        go = torch.ones(2, c, 2 * s, 2 * s, 2 * s, device=dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last_3d)
        torch.cuda.reset_peak_memory_stats()
        fw = bench(lambda: f(x))
        fb = bench(lambda: torch.autograd.grad(f(x), x, go)[0])
        print(f"{c}ch {s}->{2*s} {name:12s}: fwd {fw*1e3:8.1f} ms  fwd+bwd {fb*1e3:8.1f} ms  "
              f"peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB", flush=True)
        del x, go
        torch.cuda.empty_cache()
