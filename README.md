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
  normal, offset bias, a precision proxy, merged-sheet runs). **Evaluation v2**
  (`docs/unified_design.md` section 25) adds, all additive and all CPU-only: `--ceiling [STORE]`, the
  same suite run on the published recto mask pyramid so every number prints as `value (ceiling)`
  (cached per box+store next to the eval box); expected run length in micrometres along each mesh
  (`erl_um`, `erl_break_um`, `erl_merge_um`, `lost_break_frac`, `lost_merge_frac`); Betti-0/1 error of
  the thresholded band vs the mesh on the box interior (`--betti-margin 8 --betti-band 6`, `usrm2/topo.py`);
  `offset_hd95`/`offset_p99`; and 95% bootstrap CIs over surfaces (`--bootstrap 200 --seed 0`).
  `--json OUT` dumps per-surface rows, pooled metrics, CIs, Betti and the ceiling.
  `usrm2 evalsurf-curve RUN_DIR --metric dice` fits the plateau (asymptote, the step reaching 95% of the
  remaining gain, slope per 10k steps). Quote a number as `value (ceiling) [lo, hi]`; a change inside the
  CI is not evidence.

- Augmentation (`aug.py`): the 48 cube symmetries and the raw-uint8 stage (`window`, `volcomp`,
  `blank`; `data.raw` / `data.Patches`) run in the dataloader worker, everything else
  (rot/scale/shear/elastic + intensity) runs batched on the GPU in `train.py`; spatial augs rotate
  the radial vector channels by the same map. Pick a preset with `--aug` (see `aug.PRESETS`).
  `all2` is `all` + the cross-scroll families: the scan-domain set ported from tsm (`scan`, ranges
  calibrated on PHercParis4 vs PHerc1667), `tone`, `thick`, `volcomp`, `blank` and the ESRF/nabu
  recon set (`haze`, `unsharp`, `quant`, `cor`); `all2_light` halves every `p`.
  `full2` is `full` + physics augmentation v2 (`docs/unified_design.md` section 27): `paganin`, an FFT
  op that re-filters the cube as if nabu's Paganin delta/beta and unsharp `(coeff, sigma)` had been
  different (the RATIO of the two transfer functions, so the scan's own parameters are the exact
  identity), and `shuffle`, a per-sample order for the intensity/artefact ops (SinoSynth). Sigmas are
  defined in MICRONS: `aug.apply(..., rung=k)` converts them to voxels for the rung
  (`sigma_vox = sigma_um / rung_um(k)`), and rung 2 is bit-identical to the old voxel numbers.
  `usrm2/scanmeta.py` loads the upstream `metadata.json` next to a volume (local path or bucket URL)
  into a flat dict with documented defaults, and `scanmeta.ranges_for(meta)` centres the ranges on that
  scan: `aug.get("full2", meta=scanmeta.load(vol), rung=k)`.

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

- `--verso` adds the VERSO OUTPUT CHANNEL (`docs/unified_design.md` section 23): ONE head, `cout 2` --
  channel 0 recto, channel 1 verso -- and the deep heads follow. The verso target has no pyramid; it comes
  from region stores `<--verso-regions>/verso/region_<z>_<y>_<x>.zarr` (1024^3 uint8 probability, rung 2, its
  2x pool at rung 3), and every voxel no finished store covers gets WEIGHT 0 in that channel, so a sample
  without verso is still a valid recto sample and rungs >= 4 are verso-free for now. `stream-plan --verso
  --verso-regions-url URL` fetches the stores from the published tree as the pod writes them. A `cout 1`
  checkpoint warm-starts into it by copying the recto filter into the verso channel (the recto output is
  unchanged to a float32 ulp), and composes with the cascade 14 -> 15 warm start in one restart.
  `predict`/`evalsurf --head recto|verso` picks the channel; `--radial-sign -1` (the old flip trick) still
  works for `cout 1` checkpoints. There is no published verso surface, so `evalsurf --head verso` only runs
  and reads out the band offset -- it is not a score.

    usrm2 train RUN --rungs 2-11 --ctx 1 2 3 4 5 6 7 8 9 --patch 256 --stores-file stores.txt \
        --rung-boost 2=2 --val-rungs 2,3,4,6
    usrm2 train RUN --rungs 2-11 --ctx 1..9 --cascade mix --init-from OLD/ckpt.pt   # 14 -> 15 channels
    usrm2 train RUN --rungs 2-11 --ctx 1..9 --cascade mix --verso --cout 2 \
        --teacher-regions ~/teacher_regions --init-from OLD/ckpt.pt   # 14 -> 15 in, 1 -> 2 out
    usrm2 stream-plan stores.txt --queue Q --verso --teacher-regions ~/teacher_regions \
        --verso-regions-url https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/representations/predictions/teacher_regions/verso-2.4um
    usrm2 pretrain PRE --rungs 0-4 --ctx 1..9 --patch 256 --stores-file stores.txt   # masked-cube (MAE)
    usrm2 train RUN --rungs 2-11 --ctx 1..9 --init-from PRE/ckpt.pt   # ... warm-starts the same trunk
    usrm2 rung-mix stores.txt --patch 256      # the sampling mix and the local CT coverage per rung
    usrm2 predict RUN/ckpt.pt out.zarr --rung 4 --origin ... --size ...   # origin/size in rung-4 voxels

## Commands
    usrm2 train /vesuvius/usrm2/runs/p4_1m --size 1m --steps 20000 --patch 128 --batch 1
    usrm2 eval  /vesuvius/usrm2/runs/p4_1m/ckpt.pt
    usrm2 predict RUN/ckpt.pt out.zarr --origin 34432 15104 18432 --size 256 256 256
    usrm2 evalsurf --ckpt RUN/ckpt.pt --teacher /vesuvius/usrm2/teacher/eval.zarr
    usrm2 evalsurf --ckpt RUN/ckpt.pt --ceiling --json /vesuvius/usrm2/eval/run.json   # value (ceiling) [CI]
    usrm2 evalsurf-curve RUN --metric dice        # fitted asymptote / step95 / slope per 10k steps
    usrm2 ablate /vesuvius/usrm2/runs/ablate1 --presets geo,all --steps 3000 --patch 96 --batch 4
