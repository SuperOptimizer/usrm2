# usrm2 — Ultra Small Recto Model 2

A ~1.5M-parameter 3D U-Net distilled from the upstream `surface_recto_3dunet` teacher:
input is a z-scored CT patch, output is one recto-probability logit per voxel.

## Conventions
- All arrays are indexed `(z, y, x)` in level-0 (2.4 um) voxels.
- Probability maps are `uint8 = p * 255`; `attrs = {channels: ["recto"], voxel_um: 2.4,
  origin_zyx: [z, y, x], scale: 1.0}` where `origin_zyx` is the level-0 index of element 0.
- CT is the volcomp zarr `.../PHercParis4/...-masked.zarr/0` (uint8, 0 = air/masked);
  reading needs `volcomp_zarr` (set `VOLCOMP_LIB`, see `desk.sh`).
- Teacher stores are `(1, Z, Y, X)`; predictions written with the volcomp codec are `(Z, Y, X)`
  (the codec only accepts 128^3 chunks); `--plain` writes `(1, Z, Y, X)` plain zarr instead.

- Every store usrm2 writes is a zarr v3 SHARDED array: 128^3 inner chunks inside one shard per
  1024^3 box, so a 1024^3 region store is a single data file (`c/0/0/0`) instead of 512.
- `evalsurf` scores a store/checkpoint at the published tifxyz surface points (recall along the
  normal, offset bias, a precision proxy, merged-sheet runs).

- Augmentation (`aug.py`): the 48 cube symmetries and the raw-uint8 stage (`window`, `volcomp`,
  `blank`; `data.raw` / `data.Patches`) run in the dataloader worker, everything else
  (rot/scale/shear/elastic + intensity) runs batched on the GPU in `train.py`; spatial augs rotate
  the radial vector channels by the same map. Pick a preset with `--aug` (see `aug.PRESETS`).
  `all2` is `all` + the cross-scroll families: the scan-domain set ported from tsm (`scan`, ranges
  calibrated on PHercParis4 vs PHerc1667), `tone`, `thick`, `volcomp`, `blank` and the ESRF/nabu
  recon set (`haze`, `unsharp`, `quant`, `cor`); `all2_light` halves every `p`.

## The rung ladder (unified multi-resolution model, `docs/unified_design.md`)
- Rung `k` has voxel size `0.6 * 2^k` um: 2.4 um is rung 2, 1228.8 um rung 11 (12 rungs). Level `l` of a
  2.4 um CT mirror is rung `l + 2`; the exported prediction pyramids name their levels by the exact voxel
  size in um (`2.4/`, `4.8/`, ... `1228.8/`) and state it in the group's OME `multiscales`. `data.rungs(base)`
  reads either scheme, always from the local mirror (never over HTTP).
- A training source is one line `ct_base,target_group[,target_group...]`: a CT pyramid plus one whole-scroll
  target pyramid per output channel (group attrs: `channel`, `weight`, `box`, `umbilicus`). A sample is
  (source, rung, corner); the loader yields `(x, target, weight, rung)` with
  `x = [CT at rung k, 9 context cubes at rungs k+1..k+9, scale plane (k-2)/9, radial vector]` = 14 channels.
  The weight is 1 inside the target box where CT > 0, times the source weight, and 0 for a channel a source
  does not provide. The desk's partially mirrored CT level 0 is handled by a per-level chunk index: corners
  whose CT chunks are not on disk are never drawn.

- `--cascade` adds a 15th channel, the CASCADE channel (`docs/unified_design.md` section 22): the model's own
  rung-(k+1) prediction over the same field of view, upsampled 2x, inserted between the context cubes and the
  scale plane -- `x = [CT, ctx_1..ctx_9, CASCADE, scale, radial]`. Sources: `mask` (the rung-(k+1) target block,
  roughened), `self` (an extra no-grad forward of the EMA net at rung k+1), `mix` (self with probability
  `--cascade-self-p`, else mask; the production mode). `--cascade-drop` zeroes the channel for a fraction of
  samples, and rung 11 always gets zero, so a missing coarse prediction stays in distribution. Inference is
  top-down and recursive (`--cascade-depth`, default 3 rungs, 1/8 of the work per level). A 14-channel
  checkpoint warm-starts into it exactly (the cascade weights start at zero).

    usrm2 train RUN --rungs 2-11 --ctx 1 2 3 4 5 6 7 8 9 --patch 256 --stores-file stores.txt \
        --rung-boost 2=2 --val-rungs 2,3,4,6
    usrm2 train RUN --rungs 2-11 --ctx 1..9 --cascade mix --init-from OLD/ckpt.pt   # 14 -> 15 channels
    usrm2 rung-mix stores.txt --patch 256      # the sampling mix and the local CT coverage per rung
    usrm2 predict RUN/ckpt.pt out.zarr --rung 4 --origin ... --size ...   # origin/size in rung-4 voxels

## Commands
    usrm2 train /vesuvius/usrm2/runs/p4_1m --size 1m --steps 20000 --patch 128 --batch 1
    usrm2 eval  /vesuvius/usrm2/runs/p4_1m/ckpt.pt
    usrm2 predict RUN/ckpt.pt out.zarr --origin 34432 15104 18432 --size 256 256 256
    usrm2 evalsurf --ckpt RUN/ckpt.pt --teacher /vesuvius/usrm2/teacher/eval.zarr
    usrm2 ablate /vesuvius/usrm2/runs/ablate1 --presets geo,all --steps 3000 --patch 96 --batch 4
