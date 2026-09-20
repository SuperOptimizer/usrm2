#!/usr/bin/env python3
"""How much of the step is GroupNorm, and can it be made cheaper?

Part 1 -- the operator: GroupNorm + SiLU at each level shape of the 30m6 net, eager vs torch.compile,
channels_last_3d vs contiguous, bf16 vs float32, forward and forward+backward, against the bandwidth
a plain `x + x` reaches on the same tensor.

Part 2 -- the net: the real UNet (compiled, bf16 autocast, channels_last_3d, deep heads, the same
ckpt_act / add_skip), forward+backward, with the GroupNorms as they are, replaced by Identity (a lower
bound: what the step would cost if normalisation were free) and replaced by a functional group_norm
written out in torch ops so Inductor fuses it itself.
"""
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from usrm2 import model as M


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


def gn_silu(x, w, b, G, eps=1e-5):
    """`F.silu(F.group_norm(x, G, w, b, eps))` written in ops: statistics in float32 over each group,
    then the affine and the activation. Splitting C into (G, C/G) is a plain view in either layout."""
    B, C = x.shape[0], x.shape[1]
    g = x.float().view(B, G, C // G, *x.shape[2:])
    m = g.mean((2, 3, 4, 5), keepdim=True)
    v = g.var((2, 3, 4, 5), unbiased=False, keepdim=True)
    y = ((g - m) * torch.rsqrt(v + eps)).reshape(B, C, *x.shape[2:])
    return F.silu(y * w.view(1, C, 1, 1, 1) + b.view(1, C, 1, 1, 1)).to(x.dtype)


class GNSiLU(nn.Module):
    """A drop-in for nn.Sequential(GroupNorm(G, C), SiLU()) with the same parameters and eps."""

    def __init__(self, G, C, eps=1e-5):
        super().__init__()
        self.G, self.eps = G, eps
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))

    def forward(self, x):
        return gn_silu(x, self.weight, self.bias, self.G, self.eps)


dev = torch.device("cuda")
LEVELS = [(32, 256), (64, 128), (128, 64), (256, 32), (384, 16)]

if "net" not in sys.argv:
    print("=== GroupNorm(8 groups) + SiLU, batch 2 ===", flush=True)
    for C, S in LEVELS:
        for dt, layout in ((torch.bfloat16, "channels_last"), (torch.bfloat16, "contiguous"),
                           (torch.float32, "channels_last")):
            if True:
                x = torch.randn(2, C, S, S, S, device=dev, dtype=dt, requires_grad=True)
                if layout == "channels_last":
                    x = x.detach().to(memory_format=torch.channels_last_3d).requires_grad_()
                g = nn.GroupNorm(min(8, C), C).to(dev).to(dt)
                go = torch.ones_like(x)
                nb = x.numel() * x.element_size()
                f = lambda: F.silu(g(x))  # noqa: E731
                fc = torch.compile(lambda t: F.silu(g(t)))
                r = {}
                for name, fn in (("eager", f), ("compiled", lambda: fc(x))):
                    try:
                        r[name + "_f"] = bench(fn, 4)
                        r[name + "_fb"] = bench(lambda: torch.autograd.grad(fn(), x, go)[0], 4)
                    except Exception as e:
                        r[name + "_f"] = r[name + "_fb"] = float("nan")
                        print("   ", name, repr(e)[:90], flush=True)
                add = bench(lambda: x + x, 4)
                print(f"{C:4d}ch {S:3d}^3 {str(dt).split('.')[-1]:9s} {layout:13s}: "
                      f"eager f {r['eager_f']*1e3:7.1f} f+b {r['eager_fb']*1e3:7.1f} | "
                      f"compiled f {r['compiled_f']*1e3:7.1f} f+b {r['compiled_fb']*1e3:7.1f} | "
                      f"add {add*1e3:6.2f} ms ({3*nb/add/2**30:5.0f} GiB/s)", flush=True)
                del x, g, go
                torch.cuda.empty_cache()

print("\n=== the 30m6 net, batch 2, 256^3, bf16 autocast, compiled ===", flush=True)


def swap(net, make, fuse):
    """Replace every nn.GroupNorm by `make(G, C)`; with `fuse`, also drop the SiLU that follows it
    (the replacement does the activation itself)."""
    dev_ = next(net.parameters()).device
    for mod in net.modules():
        if isinstance(mod, nn.Sequential):
            for i in [j for j, m in enumerate(mod) if isinstance(m, nn.GroupNorm)]:
                g = mod[i]
                mod[i] = make(g.num_groups, g.num_channels).to(dev_)
                if fuse and i + 1 < len(mod) and isinstance(mod[i + 1], nn.SiLU):
                    mod[i + 1] = nn.Identity()
    return net


CASES = [("as is (GroupNorm+SiLU)", None, False, True),
         ("GroupNorm -> Identity, SiLU kept", lambda G, C: nn.Identity(), False, True),
         ("fused gn_silu in torch ops", lambda G, C: GNSiLU(G, C), True, True),
         ("as is, CONTIGUOUS layout end to end", None, False, False),
         ("GroupNorm -> Identity, CONTIGUOUS", lambda G, C: nn.Identity(), False, False)]
if "clast" in sys.argv:
    CASES = CASES[3:]
for ck in (0,):
    for tag, make, fuse, clast in CASES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = M.build("30m6", verbose=False, cout=1, cin=14, ckpt_act=ck, add_skip=1, deep=3).to(dev)
        if not clast:
            net = net.to(memory_format=torch.contiguous_format)
        if make is not None:
            swap(net, make, fuse)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
        cm = torch.compile(net)
        x = torch.randn(2, 14, 256, 256, 256, device=dev)
        x = x.to(memory_format=torch.channels_last_3d) if clast else x.contiguous()
        tg = torch.rand(2, 1, 256, 256, 256, device=dev)

        def step():
            with torch.autocast("cuda", torch.bfloat16):
                out = cm(x)
            loss = sum(F.binary_cross_entropy_with_logits(o.float(), F.avg_pool3d(tg, 2 ** k) if k else tg)
                       for k, o in enumerate(out if isinstance(out, (list, tuple)) else [out]))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        try:
            dt = bench(step, 6, 4)
            print(f"ckpt_act {ck}  {tag:38s}: {dt*1e3:8.0f} ms/step  "
                  f"{2*256**3/dt/1e6:6.1f} Mvox/s  peak {torch.cuda.max_memory_allocated()/2**30:5.1f} GiB", flush=True)
        except Exception as e:
            print(f"ckpt_act {ck}  {tag:38s}: {repr(e)[:110]}", flush=True)
        del net, cm, x, tg, opt
        torch.cuda.empty_cache()
