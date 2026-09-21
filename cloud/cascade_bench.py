#!/usr/bin/env python3
"""Per-step cost of the CASCADE input channel (docs/unified_design.md section 22): `off` vs `mask` vs `self`.

Synthetic samples (random uint8 cubes in the shape `data.rung_item` yields), so it needs no volumes and can
run anywhere; what it measures is the cascade work itself -- the extra channel in the stem, the 2x upsample
of the coarse block, and, in `self` mode, the one extra no-grad forward of the EMA net per sample.

    python cloud/cascade_bench.py --size 5m --patch 128 --batch 1 --steps 30
"""
import argparse
import time

import numpy as np
import torch

from usrm2 import data, model as M, prep, train as T


def item(patch, nctx, rung, cascade, rng):
    p = np.array([patch] * 3)
    ct = rng.integers(0, 255, (1 + nctx, *p), dtype=np.uint8)
    tg = rng.integers(0, 255, (1, *p), dtype=np.uint8)
    ax = np.array([[0.0, 4 * patch], [patch / 2.0] * 2, [patch / 2.0] * 2])
    kw = {}
    if cascade != "off":
        kw["cm"] = rng.integers(0, 255, tuple(p // 2), dtype=np.uint8)
    if cascade in ("self", "mix"):
        kw["cx"] = rng.integers(0, 255, (1, *p), dtype=np.uint8)
        kw["lo1"] = np.zeros(3, np.int64)
    return data.rung_item(ct, tg, np.full_like(tg, 255), rung, np.zeros(3, np.int64), ax, **kw)


def run(mode, size, patch, batch, nctx, steps, dev, deep, ckpt_act, add_skip, self_p=0.5):
    rng = np.random.default_rng(0)
    cin = 1 + nctx + (1 if mode != "off" else 0) + 1 + 3
    net = M.build(size, verbose=False, cin=cin, cout=1, deep=deep, ckpt_act=ckpt_act, add_skip=add_skip).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    cas = None
    if mode != "off":
        cnet = None
        if mode in ("self", "mix"):
            cnet = M.build(size, verbose=False, cin=cin, cout=1, deep=deep, add_skip=add_skip).to(dev).eval()
        cas = prep.Cascade(mode, self_p=self_p, drop=0.0, noise=True, net=cnet)
    b = {k: torch.stack([v] * batch) for k, v in
         {k: v for k, v in item(patch, nctx, 4, mode, rng).items()}.items()}
    ts = []
    for i in range(steps):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        x, tg, w = prep.prepare(b, dev, cascade=cas)
        x = x.to(memory_format=M.memfmt())
        with T.autocast(dev):
            pred = net(x)
            bce, dice = T.deep_losses([o.float() for o in pred] if isinstance(pred, (list, tuple)) else pred.float(), tg, w=w)
        (bce + dice).backward()
        opt.step(), opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t0)
    del net, opt, cas
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    return float(np.median(ts[max(len(ts) // 3, 1):]))  # drop the warm-up third


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="5m")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=9)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--deep", type=int, default=3)
    ap.add_argument("--ckpt-act", type=int, default=2)
    ap.add_argument("--add-skip", type=int, default=1)
    ap.add_argument("--modes", nargs="+", default=["off", "mask", "self"])
    ap.add_argument("--self-p", type=float, default=0.5, help="--cascade-self-p for the mix mode")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vox = a.batch * a.patch ** 3
    base = None
    for m in a.modes:
        t = run(m, a.size, a.patch, a.batch, a.ctx, a.steps, dev, a.deep, a.ckpt_act, a.add_skip, a.self_p)
        base = t if base is None else base
        print(f"{m:5s} {t * 1000:8.1f} ms/step  {vox / t / 1e6:7.2f} Mvox/s  x{t / base:.3f}", flush=True)


if __name__ == "__main__":
    main()
