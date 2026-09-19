#!/usr/bin/env python3
"""On the A100: memory and step time of the deeper presets at large patches (fwd+bwd, bf16, batch 1..2,
with and without activation checkpointing). Prints one line per config; OOM configs are reported, not fatal.
    python cloud/a100_bench.py 5m,26m,45m,30m6 256,384,512
"""
import sys, time, torch
from usrm2 import model as M

sizes = (sys.argv[1] if len(sys.argv) > 1 else "5m,26m,45m,30m6").split(",")
patches = [int(p) for p in (sys.argv[2] if len(sys.argv) > 2 else "256,384,512").split(",")]
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def T():
    torch.cuda.synchronize(); return time.time()


def step(net, x, tg, n=4):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    for i in range(2 + n):
        if i == 2:
            t = T()
        with torch.autocast("cuda", torch.bfloat16):
            out = net(x).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, tg)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return (T() - t) / n


print(f"{torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
for s in sizes:
    for p in patches:
        for ck in (0, 2, -1):  # no checkpointing / the two full-res levels / every level
            for b in (1, 2):
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    net = M.build(s, verbose=False, cout=4, cin=7, ckpt_act=ck).to(dev)
                    x = torch.randn(b, 7, p, p, p, device=dev).contiguous(memory_format=torch.channels_last_3d)
                    tg = torch.rand(b, 4, p, p, p, device=dev)
                    dt = step(net, x, tg)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    vox = b * p**3 / dt
                    print(f"{s:5s} patch {p:3d} ckpt={ck:2d} batch {b}: {dt*1000:7.0f} ms/step  {vox/1e6:6.1f} Mvox/s  peak {peak:5.1f} GiB", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{s:5s} patch {p:3d} ckpt={ck:2d} batch {b}: OOM", flush=True)
                finally:
                    for v in ("net", "x", "tg"):
                        if v in dir(): pass
                    net = x = tg = None; torch.cuda.empty_cache()
