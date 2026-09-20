#!/usr/bin/env python3
"""Desk: run the recto teacher over the 1024^3 regions of Paris 4 (rung 2) that the published recto mask occupies,
in a seed-0 shuffled order, writing one probability store per region for the streaming trainer to use as a soft
target (the planner prefers a region's teacher store over the mask when it exists). A region = one shard of the
exported mask level; its origin is the shard index * 1024. Resumable: a region with a `done` store is skipped.
    CUDA_VISIBLE_DEVICES=1 python cloud/teacher_regions.py [--shard I K] [--out DIR] [--tile 512] [--limit N]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/vesuvius/usrm2/teacher_regions/recto")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1))
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
            teacher.run(out, z, y, x, Z, Y, X, window=256, halo=32, tile=a.tile, margin=128, gpu_acc=True)
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
