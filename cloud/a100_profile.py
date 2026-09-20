#!/usr/bin/env python3
"""Profile the REAL training step of a rung run: the real loader, prep, aug, model, loss and optimizer.

Run it ON the training host with the training process stopped (it needs the whole card).

    python cloud/a100_profile.py ~/runs/u1_30m6_p4 --steps 20 --ckpt-act 2 [--trace out.json] [--periodic]

Reads the run's ckpt.pt for `args` so the config is exactly the live one; overrides may be passed.
Prints (a) a per-phase wall clock table (loader wait / H2D+prepare / aug / forward / loss / backward /
clip+opt / ema / log), measured with cuda synchronisation around each phase, (b) the torch.profiler
key-averages table (CUDA time, top kernels and top modules), (c) memory (allocated vs reserved), and
optionally (d) the periodic costs: evaluate, val_png, checkpoint save.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from usrm2 import aug as A, data, model as M, prep, train as T


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--ckpt-act", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--compile", default=None, help="0/1/mode string")
    ap.add_argument("--channels-last-prep", action="store_true", help="prepare() emits channels_last_3d")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--periodic", action="store_true", help="also time evaluate / val_png / save")
    ap.add_argument("--no-profiler", action="store_true")
    ap.add_argument("--cudnn-benchmark", type=int, default=0)
    ap.add_argument("--fused-adam", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true", help="skip aug.apply (measure its cost)")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--prefetch", type=int, default=0, help="overlap the H2D copy on a side stream")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = bool(a.cudnn_benchmark)

    run = Path(a.run)
    st = torch.load(run / "ckpt.pt", map_location="cpu")
    args = st["args"]
    dev = torch.device("cuda")
    patch = args["patch"]
    batch = a.batch or args["batch"]
    workers = args.get("workers", 6) if a.workers is None else a.workers
    ckpt_act = args.get("ckpt_act", 0) if a.ckpt_act is None else a.ckpt_act
    cfg = args.get("aug_cfg", {"sym": True})
    cout, cin = args["cout"], args["cin"]
    deep, add_skip = args.get("deep", 0), args.get("add_skip", 0)
    norad = bool(args.get("no_radial"))
    rungs = args["rungs"]
    print(f"[cfg{a.tag}] size={args['size']} patch={patch} batch={batch} ckpt_act={ckpt_act} add_skip={add_skip} "
          f"deep={deep} cin={cin} cout={cout} workers={workers} aug={args.get("aug", "geo")} cfg={cfg} step={st['step']} "
          f"cudnn_bench={torch.backends.cudnn.benchmark} fused_adam={a.fused_adam} no_aug={a.no_aug} prefetch={a.prefetch}",
          flush=True)

    net = M.build(args["size"], cout=cout, cin=cin, ckpt_act=ckpt_act, add_skip=add_skip, deep=deep).to(dev)
    net.load_state_dict(st["model"])
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01, **({"fused": True} if a.fused_adam else {}))
    opt.load_state_dict(st["opt"])
    ema = {k: v.to(dev) for k, v in st["ema"].items()}

    lines = data.Patches(patch=patch, stores_file=args.get("stores_file"), rungs=rungs).paths
    lines = [",".join(q) for q in lines]
    dl = data.loader(patch, batch, workers, stores_file=args.get("stores_file"),
                     exclude=args.get("val", data.VAL), seed=12345, sym=cfg.get("sym", True), aug=cfg,
                     dense_pow=args.get("dense_pow", 0.0), ctx=tuple(args.get("ctx", ())),
                     rungs=rungs, rung_boost=args.get("rung_boost"), channels=args.get("channels"),
                     require_targets=args.get("require_targets", False))

    model = net
    cmode = a.compile if a.compile is not None else ("1" if args.get("compile") else "0")
    if cmode not in ("0", ""):
        model = torch.compile(net) if cmode == "1" else torch.compile(net, mode=cmode, dynamic=False)
        print(f"[compile] {cmode}", flush=True)

    ridge_w, wtgt = args.get("ridge_w", 0.0), tuple(args.get("wtgt", ()))
    ph = {k: [] for k in ("wait", "prep", "aug", "clast", "fwd", "loss", "bwd", "opt", "ema", "total")}
    it = iter(dl)
    prof = None
    n = a.warmup + a.steps

    def one(i):
        t0 = sync()
        item = next(it)
        t1 = sync()
        ct, tg, wt = prep.prepare(item, dev, norad=norad)
        tg = torch.cat([tg, wt], 1)
        t2 = sync()
        if not a.no_aug:
            ct, tg = A.apply(ct, tg, cfg)
        tg, wt = tg[:, :cout], tg[:, cout:]
        t3 = sync()
        ct = ct.to(memory_format=torch.channels_last_3d)
        t4 = sync()
        with T.autocast(dev):
            pred = model(ct)
        t5 = sync()
        with T.autocast(dev):
            bce, dice = T.deep_losses([o.float() for o in pred] if isinstance(pred, (list, tuple)) else pred.float(),
                                      tg, ridge_w, wtgt, wt)
        loss = bce + dice
        t6 = sync()
        loss.backward()
        t7 = sync()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        t8 = sync()
        if not a.no_ema:
            T.ema_update(ema, net, 0.999)
        t9 = sync()
        if i >= a.warmup:
            for k, v in zip(("wait", "prep", "aug", "clast", "fwd", "loss", "bwd", "opt", "ema"),
                            (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t9 - t8)):
                ph[k].append(v)
            ph["total"].append(t9 - t0)
        return float(loss.detach())

    torch.cuda.reset_peak_memory_stats()
    for i in range(n):
        if i == a.warmup and not a.no_profiler:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False, profile_memory=True,
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=min(4, a.steps), repeat=1))
            prof.start()
        v = one(i)
        if prof is not None:
            prof.step()
        if prof is not None and i == a.warmup + min(4, a.steps):
            prof.stop()
            prof2, prof = prof, None
        print(f"  step {i} loss {v:.4f} {ph['total'][-1] * 1000:.0f} ms" if i >= a.warmup else f"  warm {i} loss {v:.4f}",
              flush=True)

    tot = sum(ph["total"])
    vox = a.steps * batch * int(np.prod(data.shape3(patch)))
    print(f"\n=== PHASE TABLE{a.tag} ({a.steps} steps, batch {batch}, patch {patch}) ===")
    print(f"{'phase':12s} {'ms/step':>9s} {'%':>6s}")
    for k in ("wait", "prep", "aug", "clast", "fwd", "loss", "bwd", "opt", "ema", "total"):
        v = sum(ph[k]) / a.steps * 1000
        print(f"{k:12s} {v:9.1f} {100 * sum(ph[k]) / tot:6.1f}")
    print(f"throughput {vox / tot / 1e6:.2f} Mvox/s   {tot / a.steps * 1000:.0f} ms/step")
    print(f"peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  "
          f"reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB  "
          f"current alloc {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    if not a.no_profiler:
        ka = prof2.key_averages()
        print(f"\n=== TOP CUDA KERNELS{a.tag} ===")
        print(ka.table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=70))
        print(f"\n=== TOP CPU OPS{a.tag} ===")
        print(ka.table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=70))
        if a.trace:
            prof2.export_chrome_trace(a.trace)
            print("trace ->", a.trace)

    if a.periodic:
        grid = data.val_grid_rungs(patch, lines, args.get("val", data.VAL), rungs=args.get("val_rungs", data.VAL_RUNGS),
                                   limit=args.get("val_patches", 32), ctx=tuple(args.get("ctx", ())))
        evnet = M.build(args["size"], verbose=False, cout=cout, cin=cin, add_skip=add_skip, deep=deep).to(dev)
        t = sync()
        evnet.load_state_dict(ema)
        t0 = sync()
        out = T.evaluate(evnet, grid, dev, wtgt, norad=norad)
        t1 = sync()
        T.val_png(run / "profile_val.png", evnet, grid, dev, norad=norad)
        t2 = sync()
        torch.save({"model": net.state_dict(), "ema": ema, "opt": opt.state_dict(), "step": 0, "args": args},
                   run / "profile_ckpt.tmp")
        t3 = time.perf_counter()
        os.remove(run / "profile_ckpt.tmp")
        print(f"\n=== PERIODIC{a.tag} === grid {len(grid)} patches")
        print(f"ema->evnet {t0 - t:.2f}s  evaluate {t1 - t0:.2f}s  val_png {t2 - t1:.2f}s  save {t3 - t2:.2f}s")
        print(f"amortised over {args.get('eval_every', 500)} steps: "
              f"{(t3 - t) / args.get('eval_every', 500) * 1000:.0f} ms/step")
        print("eval", json.dumps(out))


if __name__ == "__main__":
    main()
