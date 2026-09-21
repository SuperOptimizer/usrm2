#!/usr/bin/env python3
"""Desk: run the recto teacher over the 1024^3 regions of Paris 4 (rung 2) that the published recto mask occupies,
in a seed-0 shuffled order, writing one probability store per region for the streaming trainer to use as a soft
target (the planner prefers a region's teacher store over the mask when it exists). A region = one shard of the
exported mask level; its origin is the shard index * 1024. Resumable: a region with a `done` store is skipped.
    CUDA_VISIBLE_DEVICES=1 python cloud/teacher_regions.py [--shard I K] [--out DIR] [--limit N]

One region = one teacher.run call over the whole 1024^3 box (--tile 1024): the tile margin is clipped to the box,
so a single tile computes exactly the same windows as the 2x2 tiling did, with none of the overlap - 1.4x fewer
windows. --backend trt (usrm2/trt.py, fp16 engine for this GPU) is another 2x on the desk; on virtualized cloud
GPUs the untimed engine is slower, so pass --backend torch there.
"""
import argparse, os, sys, time
import numpy as np
from usrm2 import data, teacher

MASK = "/vesuvius/usrm/volcomp/PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr/2.4"
R = 1024


def regions(seed=0):
    """Occupied 1024^3 regions (origins, rung-2 voxels) in a seeded shuffled order, excluding any touching the val box."""
    keys = []
    for root, _, files in os.walk(os.path.join(MASK, "c")):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), os.path.join(MASK, "c")).split("/")
            if len(rel) == 3 and all(q.isdigit() for q in rel):
                keys.append(tuple(int(q) * R for q in rel))
    keys.sort()
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    vo, vs = data.box(data.open_zarr(data.VAL))
    out = []
    for o in keys:
        o = np.array(o)
        if np.all(o < vo + vs) and np.all(o + R > vo):
            continue  # touches the held-out box
        out.append(tuple(int(v) for v in o))
    return out


def walk_regions(stores, seed, boost, walk="mix", rungs=range(2, 12)):
    """The rung-2 regions of the Paris 4 recto in the ORDER the stream planner will visit them (the same stores,
    rungs, boosts, seed and walk mode as the training run), so finished stores are found by the run as soon as
    possible. Regions touching the val box are excluded by the walk itself."""
    from usrm2 import stream
    out = []
    for line, k, lo in stream.region_walk(stores, rungs=set(rungs), seed=seed, patch=256, region=R, boost=boost,
                                          exclude=data.VAL, walk=walk):
        if k == 2 and MASK.rsplit("/", 1)[0] in line:
            out.append(tuple(int(v) for v in lo))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vesuvius/usrm2/teacher_regions/recto")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1))
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--margin", type=int, default=128)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--halo", type=int, default=32)
    ap.add_argument("--backend", default="trt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", choices=["shuffle", "walk"], default="walk",
                    help="walk = the stream planner's visit order for --stores/--boost/--seed (default); shuffle = a plain seeded shuffle")
    ap.add_argument("--stores", default=os.path.expanduser("~/u1_stores.txt"))
    ap.add_argument("--boost", nargs="*", default=["2=2", "8=4", "9=16", "10=40", "11=80"], metavar="K=M")
    ap.add_argument("--walk", default="mix")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.order == "walk":
        boost = {int(q.split("=")[0]): float(q.split("=")[1]) for q in a.boost}
        regs = walk_regions(a.stores, a.seed, boost, a.walk)
    else:
        regs = regions(a.seed)
    print(f"{len(regs)} occupied regions; this process takes every {a.shard[1]}th from {a.shard[0]}", flush=True)
    ct = data.open_zarr(data.CT)
    n = 0
    for i, (z, y, x) in enumerate(regs):
        if i % a.shard[1] != a.shard[0]:
            continue
        out = os.path.join(a.out, f"region_{z}_{y}_{x}.zarr")
        if os.path.exists(out) and data.open_zarr(out).attrs.get("done"):
            continue
        Z, Y, X = (int(min(R, s - o)) for s, o in zip(ct.shape[-3:], (z, y, x)))
        t0 = time.time()
        try:
            teacher.run(out, z, y, x, Z, Y, X, window=a.window, halo=a.halo, tile=a.tile, margin=a.margin,
                        gpu_acc=True, backend=a.backend)
            import zarr
            arr = zarr.open(out, mode="r+"); arr.attrs["done"] = True; arr.attrs["region"] = R
        except Exception as e:
            print(f"FAILED {out}: {e!r}", flush=True)
            continue
        n += 1
        print(f"{i + 1}/{len(regs)} region {z} {y} {x}: {time.time() - t0:.0f} s", flush=True)
        if a.limit and n >= a.limit:
            break
    print("TEACHERREGIONSDONE", flush=True)


if __name__ == "__main__":
    main()
