import argparse
import os


def model_presets():
    from usrm2.model import PRESETS
    return PRESETS


def main(argv=None):
    ap = argparse.ArgumentParser("usrm2")
    ap.add_argument("--umbilicus", default=None, help="scroll axis json (default: PHerc Paris 4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("out_dir")
    t.add_argument("--size", default="1m", choices=list(model_presets()))
    t.add_argument("--add-skip", type=int, default=0, help="additive (projected) skips at the first N levels instead of concat (big patches)")
    t.add_argument("--deep", type=int, default=0, help="also predict at decoder levels 1..N (2x/4x/8x coarser), deep supervision")
    t.add_argument("--ckpt-act", type=int, default=0, help="activation checkpointing of the first N levels (-1 = all; the full-res levels hold most memory)")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--patch", type=int, nargs="+", default=[128], help="patch size: one int (cube) or Z Y X (e.g. 384 512 512)")
    t.add_argument("--batch", type=int, default=1)  # 128^3 batch 2 needs >3.5 GiB
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--val-patches", type=int, default=32)
    t.add_argument("--accum", type=int, default=1, help="gradient accumulation: micro-batches per optimizer step")
    t.add_argument("--ema", type=float, default=0.999, help="EMA decay of the evaluated weights (0.9995 for 100k+ steps)")
    t.add_argument("--lr-floor", type=float, default=0.0, help="cosine decays to this fraction of --lr instead of 0")
    t.add_argument("--ridge-w", type=float, default=0.0, help="extra BCE weight on the band core (target >= 0.9)")
    t.add_argument("--dense-pow", type=float, default=0.0, help="bias sampling towards sheet-dense patches (1-2)")
    t.add_argument("--norm", default="patch", choices=["patch", "global"], help="per-patch z-score or fixed scan mean/std")
    t.add_argument("--ctx", nargs="*", default=(), help="coarse context channels: pyramid levels, e.g. 1 2 3 "
                   "(= 4.8/9.6/19.2um cubes) or the range 1..9")
    t.add_argument("--init-from", default=None, help="warm start from this checkpoint's EMA weights (new input channels start at zero, new heads copy old ones)")
    t.add_argument("--wtgt", type=int, nargs="*", default=(), help="target channels that carry loss weights (verso targets), e.g. 2 3")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stores", nargs="+", default=None, help="teacher stores to train on (default: data.TRAIN); "
                   "'a.zarr,a_m7.zarr' = several teachers over one box, one head each")
    t.add_argument("--compile", action="store_true", help="torch.compile the training step (~1.4x on the 5m)")
    t.add_argument("--stores-file", default=None, help="text file of store groups (one per line), re-read while training as it grows")
    t.add_argument("--val", nargs="+", default=None, help="validation box(es) (default: data.VAL); each a comma-joined teacher group; all are excluded from sampling")
    t.add_argument("--rungs", default=None, help="train the unified multi-resolution model on the rung ladder "
                   "(0.6 * 2^k um): 'all', a range '2-11' or a list '2,3,4'. The stores are then "
                   "'ct_base,target_group[,...]' lines of whole-scroll pyramids")
    t.add_argument("--rung-boost", nargs="*", default=(), metavar="K=M", help="per-rung sampling multipliers, e.g. 2=2 11=0.5")
    t.add_argument("--val-rungs", default="2,3,4,6", help="rungs the held-out box is scored at")
    t.add_argument("--require-targets", action="store_true", help="only draw windows whose target chunks are on disk (a partially pulled export)")
    t.add_argument("--aug", default="geo", help="augmentation preset (see aug.PRESETS)")
    t.add_argument("--no-radial", action="store_true", help="zero the radial channels 1..3")
    t.add_argument("--region", type=int, default=0, help="region mode: visit one REGION^3 region of one "
                   "source at one rung, take --windows-per-region windows inside it, then move on")
    t.add_argument("--windows-per-region", type=int, default=64)
    t.add_argument("--teacher-regions", default=None, help="prefer a region's teacher probability store "
                   "(usrm2.data.teacher_region_path under this root) over the exported mask as the rung-2/3 "
                   "target; rung 3 is its 2x mean pool")
    t.add_argument("--stream", default=None, help="replay a `usrm2 stream-plan` queue directory instead of "
                   "sampling: the windows come from the rolling local buffer the planner fills")
    sp = sub.add_parser("stream-plan", help="plan and stream the training windows into a rolling disk buffer "
                        "(usrm2/stream.py): the planner IS the sampler")
    sp.add_argument("stores_file")
    sp.add_argument("--queue", required=True, help="the queue directory (queue.jsonl, meta.json, state.json)")
    sp.add_argument("--ahead", type=int, default=400, help="windows kept in front of the trainer")
    sp.add_argument("--cache-gb", type=float, default=20.0, help="disk budget of the chunk buffer")
    sp.add_argument("--seed", type=int, default=0, help="must match the training run's (train uses the step it starts at)")
    sp.add_argument("--workers", type=int, default=4, help="the training run's --workers: entry i is worker i %% W's")
    sp.add_argument("--patch", type=int, nargs="+", default=[256])
    sp.add_argument("--rungs", default="all")
    sp.add_argument("--rung-boost", nargs="*", default=(), metavar="K=M")
    sp.add_argument("--ctx", nargs="*", default=(), help="context offsets, e.g. 1..9")
    sp.add_argument("--aug", default="geo")
    sp.add_argument("--dense-pow", type=float, default=0.0)
    sp.add_argument("--require-targets", action="store_true")
    sp.add_argument("--val", nargs="+", default=None, help="held-out box(es), excluded exactly as in training")
    sp.add_argument("--jobs", type=int, default=48, help="concurrent HTTP requests")
    sp.add_argument("--report", type=float, default=30.0, help="seconds between plan.jsonl reports")
    sp.add_argument("--region", type=int, default=0, help="region mode (see `train --region`)")
    sp.add_argument("--windows-per-region", type=int, default=64)
    sp.add_argument("--walk", default=None, choices=["once", "mix"], help="NO-REPEAT walk: enumerate every "
                    "region of every source at every usable rung, weight them so the source and rung mixes "
                    "are honoured in expectation, and visit each exactly ONCE; `epoch_done` then stops the "
                    "trainer. 'mix' gives a region round(w * regions) visits instead of one (see "
                    "data.region_visits), so the coarse rungs -- a handful of regions carrying a large "
                    "--rung-boost share -- are spread over the whole epoch instead of being used up in its "
                    "first percent; the fine rungs still get one visit each")
    sp.add_argument("--visits-max", type=int, default=64, help="most visits of one region under --walk mix")
    sp.add_argument("--active-regions", type=int, default=4, help="regions kept open at once; their windows "
                    "are emitted round robin, so consecutive queue entries come from different regions")
    sp.add_argument("--epochs", type=int, default=1, help="walk the region list this many times (a fresh "
                    "permutation each time)")
    sp.add_argument("--teacher-regions", default=None, help="see `train --teacher-regions`")
    sp.add_argument("--region-fails", type=int, default=0, help="consecutive rejected draws that abandon a "
                    "region (0 = 8 x --windows-per-region)")
    sp.add_argument("--val-rungs", default="2,3,4,6", help="rungs the held-out box is scored at (prefetched and pinned)")
    sp.add_argument("--val-patches", type=int, default=32)
    sp.add_argument("--limit", type=int, default=0, help="stop after this many queued windows (0 = forever)")
    b = sub.add_parser("ablate", help="train one run per augmentation preset, sequentially")
    b.add_argument("out_dir")
    b.add_argument("--presets", default="geo,all")
    b.add_argument("--size", default="1m", choices=list(model_presets()))
    b.add_argument("--steps", type=int, default=3000)
    b.add_argument("--patch", type=int, default=96)
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--lr", type=float, default=3e-4)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--eval-every", type=int, default=500)
    b.add_argument("--val-patches", type=int, default=32)
    b.add_argument("--stores", nargs="+", default=None)
    b.add_argument("--val", default=None)
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--patch", type=int, default=128)
    e.add_argument("--val-patches", type=int, default=32)
    e.add_argument("--val", default=None, help="teacher store to score (default: the checkpoint's own)")
    e.add_argument("--ct", default=None)
    p = sub.add_parser("predict")
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--volume", default=None)
    p.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    p.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--rung", type=int, default=None, help="predict at rung k of the volume's pyramid (0.6 * 2^k um); "
                   "--origin/--size are then rung-k voxels (default: the level the volume names, rung 2)")
    p.add_argument("--plain", action="store_true", help="plain zarr (1,Z,Y,X) instead of volcomp")
    p.add_argument("--tta", type=int, default=0, help="average over this many axis flips (8 = all)")
    p.add_argument("--head", default="0", help="head index of a multi-teacher student, or mean / prod / max")
    p.add_argument("--radial-sign", type=float, default=1.0, help="-1 negates the radial vector (the student then predicts the verso face)")
    p.add_argument("--lut-to", nargs="*", default=(), metavar="REF", help="also average with the input histogram-matched to REF volumes")
    ub = sub.add_parser("umbilicus", help="put a scroll's axis where the loader looks for it "
                        "(a published file if there is one, otherwise derived from the scroll's own CT)")
    ub.add_argument("ct_base", nargs="+", help="CT pyramid group(s), or a stores file with --stores-file")
    ub.add_argument("--stores-file", action="store_true", help="the arguments are stores files")
    ub.add_argument("--rung", type=int, default=7, help="the rung the axis is derived from (76.8 um)")
    ub.add_argument("--force", action="store_true")
    rm = sub.add_parser("rung-mix", help="print the rung sampling mix of a stores file and the local CT coverage")
    rm.add_argument("stores_file")
    rm.add_argument("--patch", type=int, nargs="+", default=[256])
    rm.add_argument("--rungs", default="all")
    rm.add_argument("--rung-boost", nargs="*", default=(), metavar="K=M")
    s = sub.add_parser("evalsurf", help="score a checkpoint or store against the published surfaces")
    s.add_argument("--box", type=int, nargs=6, default=None, metavar=("Z0", "Y0", "X0", "Z", "Y", "X"))
    s.add_argument("--ckpt")
    s.add_argument("--store")
    s.add_argument("--teacher", help="a second store, scored on the same points")
    s.add_argument("--tifxyz", default=None)
    s.add_argument("--volume", default=None)
    s.add_argument("--png", default=None)
    s.add_argument("--window", type=int, default=128)
    s.add_argument("--tta", type=int, default=0)
    s.add_argument("--head", default="0", help="head index of a multi-teacher student, or mean / prod / max")
    s.add_argument("--lut-to", nargs="*", default=(), metavar="REF")
    s.add_argument("--halo", type=int, default=16)
    s.add_argument("--device", default=None)
    t = sub.add_parser("teacher", help="run the upstream teacher over a box")
    t.add_argument("out")
    t.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    t.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    t.add_argument("--volume", default=None)
    t.add_argument("--tta", type=int, default=0, help="average over this many axis flips (8 = all)")
    t.add_argument("--lut-to", nargs="*", default=(), metavar="REF", help="also average with the input histogram-matched to each REF volume")
    t.add_argument("--model", default="recto", choices=["recto", "m7"], help="m7 = the 8um nnU-Net on level 2, upsampled")
    t.add_argument("--backend", default="torch", choices=["torch", "trt"], help="trt = tsm's fp16 TensorRT engine")
    t.add_argument("--gpu-acc", action="store_true", help="box, normalization and accumulators on the GPU (needs ~8 GB spare)")
    t.add_argument("--window", type=int, default=None, help="recto 256 / m7 192 by default; bigger = less halo overlap")
    t.add_argument("--batch", type=int, default=1, help="windows per forward (with --gpu-acc)")
    t.add_argument("--streams", type=int, default=1, help="concurrent CUDA streams (with --gpu-acc)")
    r = sub.add_parser("refine", help="move a tifxyz surface onto the probability peaks of a store (prediction-guided)")
    r.add_argument("out", help="output root: one refined tifxyz directory per surface goes under it")
    r.add_argument("surfaces", nargs="*", help="tifxyz directories (z.tif y.tif x.tif meta.json)")
    r.add_argument("--tifxyz", default=None, help="refine every surface of this tifxyz root that crosses the store's box")
    r.add_argument("--store", required=True, help="recto probability store to refine on")
    r.add_argument("--eval-store", default=None, help="a different store to report before/after metrics on")
    r.add_argument("--far", type=int, default=12, help="search range along the normal (voxels), shrinks per iteration")
    r.add_argument("--sigma", type=float, default=2.0, help="grid smoothing of the displacement field (grid cells)")
    r.add_argument("--iters", type=int, default=3)
    r.add_argument("--thr", type=float, default=0.5, help="minimum peak probability to count as evidence")
    r.add_argument("--volume", default=None)
    r.add_argument("--png", default=None, help="also write before/after slice images into this directory")
    r.add_argument("--up", type=int, default=1, help="resample the grids this many times denser before refining (published = 1/20 voxel)")
    v = sub.add_parser("verso", help="write verso targets (flipped-student skin anchored to CT edges) for teacher store groups")
    v.add_argument("ckpt", help="student checkpoint (one head per store of a group)")
    v.add_argument("--stores", nargs="+", required=True, help="teacher store groups, 'a.zarr,a_m7.zarr' each")
    v.add_argument("--window", type=int, default=128)
    v.add_argument("--halo", type=int, default=16)
    v.add_argument("--tile", type=int, default=512)
    v.add_argument("--margin", type=int, default=32)
    v.add_argument("--batch", type=int, default=1, help="windows per forward on the GPU-accumulated path (1 = unbatched; batching was slower on the 5080)")
    v.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "K"), help="process every K-th group starting at I")
    v.add_argument("--force", action="store_true", help="rewrite outputs marked done")
    v.add_argument("--reverse", action="store_true", help="take the groups from the end (a second machine working towards the first)")
    v.add_argument("--modes", nargs="+", default=["skin", "raw"], choices=["skin", "raw"], help="skin: anchored outer skin (_v); raw: flipped probability as is (_vraw)")
    b = sub.add_parser("teacher-boxes", help="run the teacher over many random non-air boxes")
    b.add_argument("out_dir")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--size", type=int, nargs=3, default=(384, 2048, 2048), metavar=("Z", "Y", "X"))
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--volume", default=None, help="CT zarr array (default Paris 4); pair with --umbilicus")
    b.add_argument("--exclude", default="default", help="val store to avoid (default: data.VAL; 'none' for other scrolls)")
    b.add_argument("--tta", type=int, default=0, help="axis-flip TTA (4 = identity + 3 single flips)")
    b.add_argument("--model", default="recto", choices=["recto", "m7"], help="m7 = the 8um nnU-Net on level 2, upsampled")
    b.add_argument("--backend", default="torch", choices=["torch", "trt"], help="trt = tsm's fp16 TensorRT engine")
    b.add_argument("--gpu-acc", action="store_true", help="box, normalization and accumulators on the GPU (needs ~8 GB spare)")
    b.add_argument("--window", type=int, default=None, help="recto 256 / m7 192 by default; bigger = less halo overlap")
    b.add_argument("--batch", type=int, default=1, help="windows per forward (with --gpu-acc)")
    b.add_argument("--streams", type=int, default=1, help="concurrent CUDA streams (with --gpu-acc)")
    b.add_argument("--procs", type=int, default=1, help="worker processes sharing the GPU (each takes every k-th box)")
    b.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("I", "K"), help="(internal) this worker's share")
    a = ap.parse_args(argv)
    from usrm2 import data, model, predict as P, train as T

    def parse_rungs(v):
        """'all' -> True, '2-11' -> {2..11}, '2,3,4' -> {2,3,4}."""
        if v is None or str(v).lower() in ("all", "true"):
            return True
        if "-" in str(v):
            lo, hi = str(v).split("-")
            return set(range(int(lo), int(hi) + 1))
        return {int(q) for q in str(v).replace(" ", ",").split(",") if q}

    def parse_ctx(vs):
        """'1 2 3' or '1..9' -> (1, 2, 3) / (1, ..., 9)."""
        out = []
        for v in ([vs] if isinstance(vs, str) else vs):
            if ".." in str(v):
                lo, hi = str(v).split("..")
                out += list(range(int(lo), int(hi) + 1))
            else:
                out.append(int(v))
        return tuple(out)

    def parse_boost(vs):
        return {int(q.split("=")[0]): float(q.split("=")[1]) for q in vs}
    if a.umbilicus:
        data.UMBILICUS = a.umbilicus
    if a.cmd == "rung-mix":
        lines = [l.strip() for l in open(a.stores_file) if l.strip() and not l.startswith("#")]
        rs = parse_rungs(a.rungs)
        rows = data.rung_mix(lines, patch=a.patch if len(a.patch) > 1 else a.patch[0],
                             allowed=None if rs is True else rs, boost=parse_boost(a.rung_boost))
        print(data.format_rung_mix(rows))
    elif a.cmd == "train":
        T.train(a.out_dir, accum=a.accum, ema_decay=a.ema, lr_floor=a.lr_floor, ridge_w=a.ridge_w, dense_pow=a.dense_pow,
                norm=a.norm, ctx=parse_ctx(a.ctx), stream=a.stream, init_from=a.init_from, wtgt=tuple(a.wtgt), compile=a.compile, ckpt_act=a.ckpt_act, add_skip=a.add_skip, deep=a.deep, size=a.size, steps=a.steps, patch=a.patch if len(a.patch) > 1 else a.patch[0], batch=a.batch, lr=a.lr,
                workers=a.workers, eval_every=a.eval_every, val_patches=a.val_patches, resume=a.resume,
                aug=a.aug, no_radial=a.no_radial,
                **({"rungs": parse_rungs(a.rungs), "rung_boost": parse_boost(a.rung_boost),
                    "val_rungs": [int(q) for q in a.val_rungs.split(",")], "require_targets": a.require_targets,
                    "region": a.region, "windows_per_region": a.windows_per_region,
                    "teacher_regions": a.teacher_regions} if a.rungs else {}),
                **{k: v for k, v in dict(stores=a.stores, stores_file=a.stores_file, val=a.val).items() if v})
    elif a.cmd == "umbilicus":
        from usrm2 import umbilicus as U
        bases = []
        for q in a.ct_base:
            if a.stores_file:
                bases += [l.split(",")[0].strip() for l in open(q) if l.strip() and not l.startswith("#")]
            else:
                bases.append(q)
        for b in dict.fromkeys(bases):
            print(b, "->", U.ensure(b, urls=U.published_urls(b), rung=a.rung, force=a.force), flush=True)
    elif a.cmd == "stream-plan":
        from usrm2 import stream as S
        S.plan(stores_file=a.stores_file, queue=a.queue, patch=a.patch if len(a.patch) > 1 else a.patch[0],
               rungs=parse_rungs(a.rungs), rung_boost=parse_boost(a.rung_boost), seed=a.seed, workers=a.workers,
               ahead=a.ahead, cache_gb=a.cache_gb, ctx=parse_ctx(a.ctx), aug=a.aug, dense_pow=a.dense_pow,
               require_targets=a.require_targets, val=a.val, jobs=a.jobs, report=a.report, limit=a.limit,
               val_rungs=[int(q) for q in a.val_rungs.split(",")], val_patches=a.val_patches,
               region=a.region, windows_per_region=a.windows_per_region, walk=a.walk,
               active_regions=a.active_regions, epochs=a.epochs, region_fails=a.region_fails,
               teacher_regions=a.teacher_regions, visits_max=a.visits_max)
    elif a.cmd == "ablate":
        from usrm2 import ablate
        ablate.sweep(a.out_dir, a.presets.split(","), size=a.size, steps=a.steps, patch=a.patch,
                     batch=a.batch, lr=a.lr, workers=a.workers, eval_every=a.eval_every,
                     val_patches=a.val_patches,
                     **{k: v for k, v in dict(stores=a.stores, val=a.val).items() if v})
    elif a.cmd == "eval":
        import torch
        st = torch.load(a.ckpt, map_location="cpu")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model.build(st["args"]["size"], cout=st["args"].get("cout", 1), cin=st["args"].get("cin", 4)).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in st["ema"].items()})
        sa = st["args"]  # the run's own validation set unless overridden
        grid = data.val_grid(patch=a.patch, limit=a.val_patches, ct=a.ct or sa.get("ct", data.CT), store=a.val or sa.get("val", data.VAL), ctx=tuple(sa.get("ctx") or ()))
        print("val", a.val or sa.get("val", data.VAL))
        if st["args"].get("no_radial"):
            for x, _ in grid:
                x[-3:] = 0
        print(st["step"], T.evaluate(net, grid, dev))
    elif a.cmd == "evalsurf":
        from usrm2 import evalsurf as E
        assert a.ckpt or a.store, "need --ckpt or --store"
        b = a.box or (*E.VAL_BOX[0], *E.VAL_BOX[1])
        from usrm2 import teacher
        luts = [teacher.lut_to(a.volume or data.CT, r) for r in a.lut_to]
        E.run(b[:3], b[3:], ckpt=a.ckpt, store=a.store, teacher=a.teacher, tifxyz=a.tifxyz or E.TIFXYZ,
              volume=a.volume, window=a.window, halo=a.halo, device=a.device, png_path=a.png, tta=a.tta, luts=luts,
              head=a.head if a.head in P.HEADS else int(a.head))
    elif a.cmd == "refine":
        from usrm2 import refine
        outs = refine.run(a.surfaces, a.store, a.out, eval_store=a.eval_store, far=a.far, sigma=a.sigma, iters=a.iters, thr=a.thr,
                          volume=a.volume, tifxyz=a.tifxyz, up=a.up)
        if a.png:
            refine.compare(a.store, a.tifxyz or os.path.dirname(a.surfaces[0].rstrip("/")), a.out, a.png, volume=a.volume)
    elif a.cmd == "teacher-boxes" and a.procs > 1:  # k workers on one GPU: a virtualized GPU only fills up this way
        import subprocess, sys
        argv, skip = [], False
        for x in sys.argv[1:]:  # drop "--procs K" / "--procs=K"
            if skip or x.startswith("--procs"):
                skip = (x == "--procs")
                continue
            argv.append(x)
        ps = [subprocess.Popen([sys.executable, "-m", "usrm2.cli"] + argv + ["--shard", str(i), str(a.procs)]) for i in range(a.procs)]
        sys.exit(max(p.wait() for p in ps))
    elif a.cmd == "teacher-boxes":
        from usrm2 import teacher
        ex = data.VAL if a.exclude == "default" else (None if a.exclude == "none" else a.exclude)
        runner = __import__("usrm2.m7", fromlist=["run"]).run if a.model == "m7" else None
        teacher.boxes(a.out_dir, n=a.n, size=tuple(a.size), seed=a.seed, volume=a.volume or data.CT, exclude=ex, tta=a.tta, runner=runner, backend=a.backend, shard=tuple(a.shard), **({"window": a.window} if a.window else {}),
                      **({"gpu_acc": True, "batch": a.batch, "streams": a.streams} if a.gpu_acc and a.model == "recto" else {}))
    elif a.cmd == "teacher":
        from usrm2 import teacher
        vol = a.volume or data.CT
        if a.model == "m7":
            from usrm2 import m7
            m7.run(a.out, *a.origin, *a.size, volume=vol, tta=a.tta, backend=a.backend, **({"window": a.window} if a.window else {}))
        else:
            teacher.run(a.out, *a.origin, *a.size, volume=vol, tta=a.tta, luts=[teacher.lut_to(vol, r) for r in a.lut_to], backend=a.backend,
                        gpu_acc=a.gpu_acc, batch=a.batch, streams=a.streams, **({"window": a.window} if a.window else {}))
    elif a.cmd == "verso":
        import time
        from usrm2 import verso
        groups = a.stores[::-1] if a.reverse else a.stores
        groups = groups[a.shard[0]::a.shard[1]] if a.shard else groups
        import torch
        failed = 0
        for i, g in enumerate(groups):
            t0 = time.time()
            try:
                outs = verso.run(g, a.ckpt, window=a.window, halo=a.halo, tile=a.tile, margin=a.margin, force=a.force, batch=a.batch, modes=tuple(a.modes))
            except torch.OutOfMemoryError as e:  # one oversized group must not kill the shard: it is left without `done` (a later pass redoes it)
                failed += 1
                print(f"verso {i + 1}/{len(groups)} {g} FAILED (OOM: {str(e)[:80]}) ({time.time() - t0:.0f} s)", flush=True)
                torch.cuda.empty_cache()
                continue
            print(f"verso {i + 1}/{len(groups)} {g} -> {outs} ({time.time() - t0:.0f} s)", flush=True)
        if failed:
            print(f"verso: {failed} groups failed (OOM); rerun with a smaller --tile to fill them", flush=True)
    else:
        from usrm2 import teacher
        vol = a.volume or data.CT
        P.predict(a.ckpt, vol, *a.origin, *a.size, a.out, window=a.window, halo=a.halo, volcomp=not a.plain, rung=a.rung,
                  tta=a.tta, luts=[teacher.lut_to(vol, r) for r in a.lut_to], head=a.head if a.head in P.HEADS or a.head == "all" else int(a.head),
                  radial_sign=a.radial_sign)


if __name__ == "__main__":
    main()
