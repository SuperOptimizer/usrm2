# Unified multi-resolution surface model: design (2026-09-19)

One model, one recto head (a verso head is added later), 256^3 patches, any input voxel size on a
power-of-2 ladder from 0.6 um to 1.23 mm. Bootstrapped from the published binary surface predictions so
that training starts without new inference.

## 1. The ladder

Rung k has voxel size 0.6 * 2^k um. Every scan is snapped ONCE to its nearest rung (nearest in log2) by a
single trilinear resample, then pooled by exact 2x means for the rungs above it. Scans at 2.4 um are on the
ladder already (rung 2, no resample).

| rung | um/voxel | 256^3 field | scans snapped here (resample factor) |
|---|---|---|---|
| 0 | 0.6 | 154 um | 0.55 um (x0.917) |
| 1 | 1.2 | 307 um | 1.129 um (x0.941) |
| 2 | 2.4 | 614 um | 2.400 (exact), 2.215 (x0.923), 3.24 (x1.35 up) |
| 3 | 4.8 | 1.23 mm | |
| 4 | 9.6 | 2.46 mm | 7.91 (x0.824), 8.64 (x0.9), 9.36 (x0.975) |
| 5 | 19.2 | 4.9 mm | |
| 6 | 38.4 | 9.8 mm | 45.5 um survey scans (x1.19 up) |
| 7 | 76.8 | 19.7 mm | |
| 8 | 153.6 | 39 mm | |
| 9 | 307.2 | 79 mm | whole cross-section |
| 10 | 614.4 | 157 mm | |
| 11 | 1228.8 | 315 mm | whole Paris 4 (75784 x 32693^2 at 2.4 um -> 148 x 64 x 64) |

12 rungs. The existing "level l" of a 2.4 um mirror is rung l + 2; `data.levels()` and `context()` keep
working, indexed by rung instead of level.

## 2. One sample

A sample is (scan, rung k, 256^3 corner). Model input channels, all 256^3:

- the CT cube at rung k (z-scored),
- 9 context cubes at rungs k+1 .. k+9, same size, same centre (as today; beyond the top of a scan's pyramid
  the cube keeps being pooled, so the scroll just shrinks inside it),
- the radial unit vector (3),
- a scale channel: constant plane (k - 2) / 9 (0 at 2.4 um, log scale in the rung index). It says "this cube
  is 0.6 * 2^k um", and since the context cubes sit at k+1..k+9 by construction it fixes their scales too.

The stack is anchored at the predicted rung: no finer-than-primary cubes exist in the input, so a 2.4 um
sample carries no 0.6/1.2 slots and needs no ignore. The context count is fixed at 9 for every scroll; past
the top of a scan's pyramid the loader keeps pooling, so a 9.6 um scroll's rungs 12 and 13 are the whole
scroll at 1/2 and 1/4 size inside the cube (cheap: cached small arrays, one stem channel each). Nine is
enough for the largest scroll (80k x 40k x 40k at 2.4 um -> 156 x 78 x 78 at rung 11).

14 input channels; the warm start from round 3 (13 channels: CT + 9 ctx + radial) widens the stem and
zero-fills the scale channel (train.py already does "image channels first, radial last"; the scale channel
goes before the radial vector).

Output: one recto channel at rung k, plus the 3 deep-supervision heads at rungs k+1..k+3 (pooled targets,
as now). Verso comes later as a second output channel (warm start copies heads mod n).

Model: 30m6 preset (six levels, 44.9M params), ckpt_act 2, add_skip 1, deep 3, compile, batch 2 at 256^3
on an 80 GB A100 (10.8 Mvox/s, 62 GB, measured). On the desk 5060 Ti's the same model trains at 128^3
patches for pipeline work only (the field of view per cube halves, the rung semantics do not change).

## 3. Targets and weights

Every training source becomes a TARGET PYRAMID: `<name>.zarr/<rung>` uint8 arrays (255 = surface), one per
rung from the source's native rung upward, made by exact 2x mean pooling (so a binary mask at its native
rung becomes the fraction of surface voxels at every coarser rung: the dynamic range the user asked for).
Attrs: `volume` (CT mirror base), `rung` (native), `kind` (binary | prob), `weight` (source weight),
`box` (origin and size at the native rung; whole scroll = the full shape).

Decisions 2026-09-19 (user): published masks are used AS PUBLISHED at their native rung, no softening, no
CT gating, no medial-surface band, no thickness weighting; only our own 2x mean pooling above the native rung
(their own pyramid levels are binary nearest/max and are not used). The loss sees a hard band at a mask's
native rung and fractions above it.

The loader returns (x, target, weight). The per-voxel weight is the product of
- the source weight (published masks 1.0, our own probability stores 1.0, targets upsampled from a coarser
  native rung 0.3),
- 1 inside the target's box at that rung, 0 outside (so a box-limited store may be sampled at coarse
  rungs where the cube is bigger than the box: CT comes from the whole-scroll mirror, the loss only sees
  the box),
- 1 where CT > 0 (masked air never carries surface),
- and, per channel, 0 for a channel the source does not provide (this is the per-channel ignore the
  verso head will need: recto-only sources then train recto only).

Loss stays BCE + soft dice (`losses`) but both take the weight tensor instead of the current `wtgt`
channel convention; `deep_losses` pools targets AND weights.

Both teacher lineages (the 2.4 um recto model and m7) feed the same recto head, as separate stores.
Their bands differ in thickness (recto th0.45 covers ~32% of a mid-scroll cube at 2.4 um, m7 th0.2 ~23%
at 9.6 um); the student learns an average band. No thickness harmonisation in the bootstrap.

## 4. Sampling

Pick a store by voxel weight (as now), then a rung from the store's usable range with probability
proportional to n_k^0.5 where n_k = box voxels / (256 * 2^k)^3 floored at 1, then a corner. For whole
Paris 4 that gives roughly: rung 2 40%, 3 20%, 4 15%, 5 8%, 6..11 about 3% each; a 640x1024^2 teacher box
is sampled at rungs 2 and 3 mostly. Rejection rules (air, low foreground, dense_pow) act on the target at
the chosen rung. The 48 cube symmetries and the GPU augmentations are unchanged; the scale channel and the
context cubes are permuted together with the CT cube.

## 5. Sources for the bootstrap (Paris 4 first)

Measured 2026-09-19 (all published masks are uint8 0/255 at EVERY pyramid level, i.e. their own pyramids
are nearest/max and carry no fraction; we pool level 0 or 1 ourselves):

| source | native rung | published size |
|---|---|---|
| Paris 4 recto-2um-ps256 th0.45, level 0 | 2 | ~540 GiB (extrapolated from 5 z-rows) |
| same, level 1 (4.8 um, binary) | 3 | 115 GiB |
| same, level 2 | 4 | 25 GiB |
| Paris 4 m7 L2 th0.2, level 0 (9.6 um) | 4 | 17 GiB |
| same, level 1 | 5 | 3.9 GiB |
| our teacher stores (probabilities, 73 seeds, recto + m7) | 2 | 58 GB on the desk |

Pull plan: m7 level 0 whole (17 GiB) -> pyramid rungs 4..11. Recto level 1 whole (115 GiB) -> rungs
3..11 (its rung 3 is a binary 4.8 um mask, softened; rung 4 up fractional). Recto level 0 only under the
boxes whose level-0 CT the desk already mirrors (the 299 GB of level-0 chunks) -> rung 2, box-limited.
About 150 GB new on /vesuvius (802 GB free). Level 0 whole (540 GiB) is not pulled: it would double the
mirror for a rung-2 signal our own probability stores already give in the boxed regions.

Training then covers every rung 2..11 of Paris 4 with dense coverage: rung 2 from boxes (published +
ours), rung 3 from the whole scroll, rungs 4+ from two whole-scroll sources.

## 6. Later sources, in order

1. Other 2.4 um scrolls with an m7 L2 mask (PHerc0009B, 0332, 0814, 0841, 0846A, 1203, 1299, 1451, MAN5,
   MANB, MANBp; 0343P and 0500P2's 2.215 um scan after a x0.92 resample): CT mirror levels 2+ whole (rungs
   4..11, tens of GB per scroll), mask 1-17 GiB each. Adds scroll diversity where the coarse rungs live.
   Their rungs 2/3 get targets from our own model later (self-labelling on a few mirrored boxes).
2. Fine scans -> rungs 0 and 1: Paris 4 1.129 um, PHerc1667 1.129 um, PHerc0500P2 0.55 um. Per-box
   resampler (`cloud/resample_boxes.py`): pick boxes, resample CT once onto the canonical grid (x0.941 or
   x0.917), pool the pyramid, store volcomp with the per-level q schedule. Targets: our model run at rung
   2 of the resampled box (it has been trained at rung 2 on the same physics), used at rungs 0 and 1 as
   trilinear upsamples with weight 0.3; then self-training rounds at rungs 0/1 with weight 1. No published
   fine-scan predictions exist, so this cannot be bootstrapped from upstream.
3. ~9 um scans (18 scrolls with m7 at their level 0, 8.64 or 9.36 um -> rung 4 after a x0.9 / x0.975
   resample): whole-scroll CT at rung 4 (1-2e12 voxels each, volcomp q4, ~30-50 GB) plus the mask.
   Legacy 7.91 um scans: see the survey note at the end of this file.
4. Verso head: flipped-student targets (usrm2/verso.py, raw mode) generated by this model at rung 2 and
   above, cout 1 -> 2 by warm start, the overlap metric kept, exclusivity still deferred.

## 7. Code changes (in order)

1. `usrm2/targets.py` (new): `import_mask(src, dst, rung, box=None)` builds a target pyramid from a
   published zarr (level 0 or 1, S3/HTTP) or from one of our probability stores; `pool` (2x mean, exact);
   `soften` (Gaussian at native rung); attrs as in section 3. CLI `usrm2 targets`.
2. `usrm2/data.py`: rung-indexed pyramids (`levels()` returns rungs), `Patches` picks (store, rung, corner),
   reads CT + target + box mask at that rung, returns a weight tensor, adds the scale channel; `val_grid`
   per rung (validation boxes at rungs 2, 3, 4 and 6 at least).
3. `usrm2/train.py`: weight tensor through `losses`/`deep_losses`/`evaluate`; per-rung validation dice;
   warm start widening for the scale channel.
4. `usrm2/predict.py`: `--rung k` sliding window over any level of a mirror (the scale channel and context
   rungs follow), so the model labels its own rungs 0/1/3+ later.
5. `cloud/mirror_scroll.py` (generalised a100_mirror.py): mirror levels >= L whole and level 0/1 chunks under
   given boxes from dl.ash2txt.org, for any scroll; `cloud/resample_boxes.py` for the fine scans.
6. Store list lines become `ct_base,target_store[,target_store...]`; refresh_groups picks up new pyramids.

Tests: pooling exactness (a 2x pool of a 0/255 mask equals the fraction), weight = 0 outside the box and
in masked CT, rung sampling covers the range, the scale channel equals (k-2)/9, warm start from a 13-channel
checkpoint, predict at rung k reproduces training-time inputs.

## 8. Legacy 7.91 um scans (survey 2026-09-19, dl.ash2txt.org)

The open-data bucket has no 7.91 um predictions; the older tree does. Raw CT: `full-scrolls/Scroll{1,2,5}/*.volpkg/
volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr` (14376 x 7888 x 8096, u1, 128^3, blosc-zstd, levels 0..5),
Scroll1B, Scroll2A (14428 x 10112 x 11984), Scroll5 (21000 x 6700 x 9100). Whole-scroll surface predictions on them,
all u1:
- `community-uploads/ryan/3d_predictions_scroll{1,2,3_invariant,4}.zarr` (flat zarr v2, 256^3 chunks, thresholded;
  note the axis order in the metadata is y,x,z-like and must be checked against the CT).
- `community-uploads/bruniss/scrolls/s1/surfaces/full_scroll/s1-surface-{regular,erode}.zarr` and
  `mask-2ext-surface*_ome.zarr` (binary / argmax, 128^3), `s1_059_ome.zarr` (multiscale, possibly probabilities);
  `s3/surfaces/s3-surface-{regular,erode}.zarr` (9778 x 3400 x 3550), `s4/surfaces/s4-surface-{regular,erode}.zarr`
  and `s4_059_medial_ome.zarr` (11174 x 3440 x 3340), `s5/surfaces/090.zarr` (not characterised).
These snap to rung 4 (x0.824 resample, 7.91 -> 9.6 um). Scroll 1A at rung 4 is 1.2e11 voxels resampled (~5 GB at
volcomp q4); the masks resample by mean (fraction) at the same time. Phase 3 material, after the 2.4 um scrolls.
No 3.24 um volumes or predictions exist on either server.

## 9. Export of the published predictions (decided 2026-09-19, later in the day)

Superseding the "as published" rule of section 3: the masks are exported ONCE, by the volcomp fleet (Blue
Lobster VMs, volume-compressor/tools/export, hot path in C), as continuous 0-255 volumes on the exact ladder:
- signed-distance ramp at the native resolution: s = signed distance to the mask boundary in voxels clipped to
  [-3, 3] (positive inside), value = round(127.5 + 42.5 s): inside 170/213/255, outside 85/43/0, the 128
  crossing exactly on the published edge. Continuous data compresses under volcomp; a 0/255 step does not.
- then a trilinear resample of the ramp onto the exact rung grid (9.362 and 8.64 um -> 9.6, 2.215 -> 2.4,
  1.129 -> 1.2, 0.55 -> 0.6, and 2.399/2.401/2.403 -> 2.400: 0.1 % is ~90 voxels of drift across Paris 4).
  Resampling a ramp also blends sub-voxel positions, so the values become a little more continuous than the
  7-level ramp itself. The CT volumes get the same exact-grid resampling later.
- levels 1..3 by 2x mean pooling per unit; levels 4..9 pooled offline from level 3.
- q by PHYSICAL voxel size, one table for predictions and CT: 0.6 um q32, 1.2 q16, 2.4 q8, 4.8 q4, 9.6 q2,
  19.2 q1, 38.4 and coarser q0 lossless (so an m7 prediction native at 9.6 um is q2, q1, then lossless).
Output mirrors the bucket keys under volcomp/<scroll>/representations/predictions/surfaces/<name>.zarr/<level>.
The training loader then reads these directly as target pyramids (values / 255), no importer needed.

## 10. Bootstrap scope (user, 2026-09-19 evening)

The first training of the unified model uses ONLY the exported upstream predictions (the Paris 4 recto and m7 mask
pyramids, then the other scrolls' m7), not our own teacher probability stores and not the verso stores. Those come
back in later rounds. Every rung 2..11 of Paris 4 is covered by the upstream masks alone.

## 11. Implemented (2026-09-19, this commit)

Sections 1-4 and 7.2-7.4 are in the code; the importer of 7.1 is not needed any more (section 9: the
exported pyramids are read directly).

- `data.rungs(base)` -> {rung: array} for both naming schemes (um level names via the group's OME
  multiscales, integer level names via the volume's native um), `data.read_rung` (pooling above the top of a
  pyramid), `data.context(..., rung=k)` (offsets, so `--ctx 1 2 3` means rungs k+1..k+3), `data.scale_plane`,
  `data.inputs(..., rung=k)`.
- `data.Patches(..., rungs=...)`: sources are `ct_base,target_group[,...]` lines, a sample is
  (source, rung, corner), the loader yields `(x, target, weight, rung)`. Rung probability ~ n_k ** 0.5 with an
  optional `--rung-boost`; the measured Paris 4 mix is in the `Patches` docstring and `usrm2 rung-mix` prints
  it for any stores file. `data.chunk_index` / `covered` keep sampling off the partially mirrored CT level 0.
- `train.losses` / `deep_losses` / `evaluate` take the weight tensor (deep heads pool targets and weights);
  the old `wtgt` channel convention still works for old runs. Per-rung validation (`dice_r2`, `dice_r3`, ...,
  `dice` = their mean) on the held-out box read at `--val-rungs`; `train.jsonl` carries the rung histogram.
  `train.warm_start` widens 13 -> 14 input channels (the scale plane sits before the radial vector and is
  zero-filled) and takes head 0 of a 4-head checkpoint.
- `usrm2 predict --rung k` slides over rung k of the CT pyramid (origin and size in rung-k voxels) and
  records `rung` / `voxel_um` in the output attrs; without `--rung` nothing changes for old checkpoints.

Not done here: the target importer / exporter side (volume-compressor does it), `cloud/mirror_scroll.py` and
the fine-scan resampler (section 7.5), and the verso head.

## 12. The loader stays uint8 (2026-09-20)

Measured on the A100, one 256^3 rung-2 sample in a worker while training: CT read 0.07 s, 9 context cubes
3.2 s, target + weight 1.1 s, `inputs()` (z-score 10 cubes to float32 and stack 14 channels, 940 MB) 6.6 s,
`augment()` (the cube symmetry on the 16-channel float32 stack) 6.1 s -- ~17 s per sample, so four workers
fed the card 3.2 Mvox/s instead of the 10.8 it can do. Each worker also held ~5 GB of float32 batches and
`val_grid_rungs` held 14-channel float32 patches (8 x 3 rungs = 22 GB at 256^3).

The worker now only READS. `data.rung_item` is the sample it yields: `ct` uint8 (1 + len(ctx), Z, Y, X) --
the CT cube and the context cubes as read, not z-scored -- `tgt` and `w` uint8 (255 = 1.0), the corner, the
scroll axis (y, x) per z slice (what `radial` interpolates), the rung, the (mean, std) of the z-score and
the cube symmetry index 0..47 the worker draws (`data.draw_sym`, the same rng draws as before, so a seed
still reproduces a run). ~200 MB per 256^3 sample instead of ~1 GB.

`usrm2/prep.py` rebuilds the input on the device: `prepare(batch, dev)` z-scores each cube (global norm or
per-patch, the semantics of `data.zscore`), adds the scale plane, computes the radial unit vector from the
corner and the axis, applies the symmetry to x, target and weight together -- the radial channels are a
vector and are permuted AND negated -- and returns floats. `aug.apply` runs after, unchanged. `evaluate`,
`val_png` and the validation grid use the same path, so a 256^3 val patch costs 12 bytes per voxel (10 cubes
+ target + weight) instead of 56 (14 float32 channels + target + weight): 8 patches x 3 rungs is 4.8 GB
instead of 22 GB, and the 4 default `--val-rungs` at `--val-patches 8` 6.4 GB instead of 30 GB.
`tests/test_prep.py` checks prepare against the CPU path (`data.inputs` + `data.sym_apply`) for all 48
symmetries and both normalizations; `predict.py` still builds its inputs on the host with `data.inputs`,
and `tests/test_rungs.py` asserts the two agree.

Reading was fixed where it was cheap: `read_rung` reads only the part of the source that overlaps the array
(a 256^3 cube five rungs above the top of a pyramid used to allocate 8192^3 float32 first), returns uint8
without a float32 round trip, and `data.full_level` keeps a whole level decoded in the worker when it is
small (<= `CACHE_VOX`, 48 Mvox, within a 192 MB budget), pooling the rungs above the top of a pyramid from
the cached level below instead of re-reading the top once per rung. Measured here on a synthetic pyramid
(rungs 2..11 on disk, 1024^3 at rung 2, patch 256^3, 9 context cubes): 4.2-4.8 s -> 0.14 s per sample
worker-side, 1024 MB -> 192 MB; the context read alone 0.063 s -> 0.021 s from the whole-level cache.

## 13. First results (2026-09-20)

Desk proof of concept `u1_5m_p4` (5m, 128^3, 60k steps, upstream masks only: Paris 4 recto + m7, warm start r3 raw):
val dice r2/r3/r4 = 0.64/0.69/0.62 (from 0.62/0.56/0.33 at step 500). Surface metrics on the val box, head 0,
window 128: recall@4 0.753, continuity 0.600, merge_frac 0.44 (round-3 raw student: 0.729 / 0.584 / 0.42; skin run:
0.722 / 0.548 / 0.47; teacher reference recall@4 0.843). Visually the student's band is broader and softer than the
recto face target at 2.4 um: the one-head compromise between the recto face (rungs 2-3) and the m7 whole-sheet band
(rungs 4+). A100 `u1_30m6_p4` (30m6, 256^3) runs at 12.6 Mvox/s, ~44 h for 60k steps.

## 14. Where the A100 step goes, and what was fixed (2026-09-20)

Profiled at the live configuration (30m6, 256^3, batch 2, add_skip 1, deep 3, compile, 9 context rungs,
6 loader workers) with `cloud/a100_profile.py` (the real loader, the real checkpoint), the micro-probes in
`cloud/a100_micro.py` and the upsample study in `cloud/a100_up.py`.

Per-step wall clock, ms (12-20 steps after warm-up, each phase synchronised):

| phase | ckpt_act 2 (before) | ckpt_act 1 | ckpt_act 0 | ckpt_act 0 + `up2x` |
|---|---|---|---|---|
| loader wait | 0.3 | 0.3 | 0.3 | 0.3 |
| `prep.prepare` (H2D + z-score + radial + symmetry) | 219 | 247 | 247 | 248 |
| `aug.apply` ("geo" = symmetry only) | 2.7 | 2.7 | 2.7 | 0.2 |
| `channels_last_3d` | 3.0 | 3.0 | 3.3 | 3.0 |
| forward | 532 | 420 | 333 | 510 |
| loss (4 deep heads) | 7.5 | 7.1 | 7.1 | 8.2 |
| backward | 2163 | 2023 | 1762 | **651** |
| clip + optimizer | 8.2 | 6.0 | 7.6 | 7.1 |
| EMA | 6.1 | 3.6 | 4.5 | 1.8 |
| **total** | **2942** | **2712** | **2368** | **1429** |
| Mvox/s | 11.4 | 12.4 | 14.2 | **23.5** |
| peak allocated / reserved (GiB) | 40.6 / 47.5 | 44.8 / 50.5 | 41.4 / 51.4 | 43.4 / 48.4 |

Periodic costs (`--eval-every 500`, 21 validation patches): `evaluate` 17-19 s, `val_png` 8.4 s, checkpoint
save 1.6 s -- 27 s per 500 steps, 54 ms/step amortised, i.e. 2% of the old step and 4% of the new one.

The card is NOT starved: during a step `nvidia-smi` reports 100% utilisation, and a bf16 GEMM reaches
290 TFLOPS. The step was memory-traffic bound, and two probes say where:

- `F.interpolate(mode="trilinear")` at the decoder's widest stage (2 x 64 x 256^3 output, 2^31 elements):
  forward 91 ms, **forward+backward 1194 ms**. Its backward is a scatter-add (`_unsafe_index_put`, atomics).
  The 128-channel stage costs another 295 ms. That was ~60% of the whole step.
- eager `GroupNorm+SiLU` on 2 x 32 x 256^3 runs at 50 GiB/s (a plain add on the same tensor runs at
  2300 GiB/s); under `torch.compile` Inductor fuses them into `triton_red_fused_..._native_group_norm_silu_...`
  kernels, which is why compiling matters so much here.
- `conv3d` is fine: 120-137 TFLOPS at 32 channels x 2 x 256^3 (the >1.6e9-element shape), 245-273 TFLOPS at
  every smaller level. Convolutions are only ~320 ms of the step.
- the host->device link is 2.4-2.5 GiB/s (pinned or pageable -- this is a proxied GPU, not a PCIe slot), so
  the ~400 MB of uint8 cubes per step costs ~150 ms. It does NOT overlap with compute: copying the next
  batch on a second CUDA stream made the step 3% SLOWER, so the idea was dropped.

Changes made (all keep the checkpoint format and the model's parameters):

1. `model.up2x` replaces `F.interpolate(..., trilinear)` at an exact factor of 2 with the same arithmetic
   written as gathers (per axis, `0.75 in[m] + 0.25 in[m-+1]` interleaved by `stack`+`flatten`), so the
   backward is a gather. Any other ratio still calls `F.interpolate`. Backward 1762 -> 651 ms.
   Numerics: exact to 2e-7 relative in float32 (`tests/test_smoke.py` checks it in float64); under bf16
   autocast the per-axis rounding differs by up to one bf16 ulp (9e-3 relative on the activation) because
   aten sums all eight corners in float before rounding once. Measured effect on the step-8000 validation:
   dice 0.65403 vs 0.65416, dice_r2/r3/r4 0.68090/0.66978/0.61140 vs 0.68101/0.66995/0.61153.
2. `--ckpt-act 0`: no activation recomputation. It is both faster (no recompute) and, with `torch.compile`,
   no more expensive in memory (Inductor reuses buffers better than the checkpoint boundaries allow).
   NB without `--compile` the same configuration needs 67 GiB and OOMs: compiling is what makes it fit.
3. `aug.apply` no longer concatenates the whole batch back together when no intensity augmentation is
   configured (the "geo" preset): 1.9 GiB of allocation and a full copy per step for nothing.
4. `losses_tw` no longer builds `1 + 0 * (tgt >= 0.9)` when `ridge_w` is 0; the weight tensor is used as is.
5. `ema_update` uses `torch._foreach_mul_`/`_foreach_add_` (same arithmetic, two kernel launches instead of
   two per tensor).

Rejected, with the measurement: `torch.backends.cudnn.benchmark = True` (no speed change, peak allocated
41 -> 71 GiB -- the algorithm search picks huge workspaces); the H2D prefetch stream (above); batch 3
(changes the optimisation, and `--batch` is not in the resume `grow` list); fused AdamW (the optimizer is
0.5% of the step); `max-autotune-no-cudagraphs` (not reached before the win above made it moot).
`TORCH_LOGS=recompiles,graph_breaks` over real steps prints nothing: the rung mode varies tensor VALUES
(the scale plane, the symmetry index), not shapes, so the compiled graph is entered once and kept.

### 14b. The memory format was the other half (2026-09-20, same day)

With `up2x` in place the step was 1429 ms and the backward still 651 ms of it. `cloud/a100_gn.py`
measured GroupNorm + SiLU at each level shape and then the whole net with the norms removed:

| 2 x 32 x 256^3, bf16, GroupNorm(8) + SiLU | eager fwd | eager fwd+bwd | compiled fwd | compiled fwd+bwd | `x + x` |
|---|---|---|---|---|---|
| channels_last_3d | 119.5 | 191.5 | 24.6 | **96.2** | 2.56 ms (2348 GiB/s) |
| contiguous (NCDHW) | 82.7 | 107.1 | 4.0 | **15.7** | 2.51 ms (2386 GiB/s) |

Compiled and contiguous, the norm costs about six passes over the tensor -- near the bandwidth bound.
Compiled and channels_last it costs about thirty-eight. The whole 30m6 step (batch 2, 256^3, synthetic
input, no loader) then measures:

| | ms/step | Mvox/s | peak GiB |
|---|---|---|---|
| channels_last, GroupNorm as is | 1152 | 29.1 | 43.1 |
| channels_last, GroupNorm -> Identity (SiLU kept) | 755 | 44.4 | 37.1 |
| channels_last, a fused `group_norm+silu` written in torch ops | 1479 | 22.7 | 39.1 |
| **contiguous, GroupNorm as is** | **679** | **49.4** | 45.1 |
| contiguous, GroupNorm -> Identity | 601 | 55.8 | 39.1 |

So `channels_last_3d` -- which this net has used since the beginning for cudnn's NDHWC convolution
kernels -- costs more in the normalisations than it saves in the convolutions at these shapes, and
normalisation goes from 397 ms of the step to 78 ms just by dropping it. `usrm2.model.CHANNELS_LAST`
is now False and `model.memfmt()` is what train/predict convert to.

Full step with the real loader, 40 steps, ckpt_act 0, `--eval-every 1000`:

| phase | ms | % |
|---|---|---|
| loader wait | 0.4 | 0.0 |
| `prep.prepare` (H2D 403 MB at 2.4 GiB/s + z-score + radial + symmetry) | 269.7 | 25.8 |
| `aug.apply` | 0.1 | 0.0 |
| memory-format conversion | 3.0 | 0.3 |
| forward | 235.0 | 22.5 |
| loss (4 heads) | 7.5 | 0.7 |
| backward | 512.9 | 49.0 |
| clip + optimizer | 13.7 | 1.3 |
| EMA | 3.8 | 0.4 |
| **total** | **1046** | **32.1 Mvox/s** |

peak allocated 45.4 GiB, reserved 56.8 GiB. Validation at step 8500 is unchanged by the layout:
dice 0.65838 / r2 0.68366 / r3 0.67219 / r4 0.61929 against 0.65862 / 0.68365 / 0.67235 / 0.61987.

Still rejected, with the measurement: a hand-written fused `group_norm+silu` (above); `--batch 3`
(2% better per voxel at 64.7 GiB, and `batch` is not in the resume `grow` list, so it would change
the optimisation mid-run); pinning the loader batch harder (it is already `pin_memory=True` +
`non_blocking=True`, and on this proxied GPU pinned H2D is 2.35 GiB/s against 2.24 pageable, so the
403 MB/step costs ~167 ms either way and does not overlap with compute); a bf16 loss (the loss is
0.7% of the step).

## 15. Other scrolls: lossless masks on the native grid (user, 2026-09-20)

For every published prediction other than Paris 4: a new volcomp mask mode with NO internal 2x2x2 downscale
(spatially lossless binary mask, same context coder, decode = the published 0/255 mask), and NO resampling onto
the 0.6*2^k ladder: levels keep the source grid and are named by their true voxel size (e.g. "9.596", "19.192"
for an m7 mask read at L2 of a 2.399 um scan; "9.362", "18.724" for a 9.362 um scan). The loader snaps a level to
its nearest rung (nearest in log2) and the CT mirrors are on the same native grids, so nothing needs resampling
up front; exact-grid resampling remains available (`--encoding mask`) if a scroll ever needs it. Paris 4 keeps
its existing 2x-mode export (it is exactly 2.400 um and already training).

## 16. Training while streaming (2026-09-20)

The user's requirement: "we only stream in data and buffer it to disk then train over it. the order of
iteration has to be known so we can download some GBs first, wait until we've consumed them, then download
more, so that we don't need to do any pre-downloading but also don't lose time waiting on downloads."

The numbers say it fits: at 32 Mvox/s the card eats ~5 MB/s of compressed CT + context + targets (q8 at the
predicted rung ~1 MB/s, q4/q2 context ~4 MB/s, targets ~0), and the public tree serves 400+ MB/s to an
instance at ~48 concurrent requests. A few GB of buffer is therefore many minutes of training.

`usrm2/stream.py`. **The planner IS the sampler**: `usrm2 stream-plan STORES --queue DIR` runs
`data.Patches._rung_draw` with the same per-worker seeds (`seed + 1000 * w`), the same draw order and the
same air / low-foreground / dense_pow rejection, but a `Hook` is called before every read and fetches, from
the public tree, exactly the chunks that read will touch (`stream.rung_need` mirrors `read_rung` /
`read_block` / `full_level`, including "this level is small enough that the worker keeps it decoded whole").
A 404 is air: a zero-length `<chunk>.absent` marker is written so nothing refetches it and the replaying
worker does not wait for it. Accepted windows are appended to `queue.jsonl` with a monotonic index; entry i
belongs to loader worker i % W, so the queue IS the round robin the DataLoader replays. The CT and the
targets are fetched before the rejection rules run, the nine context cubes only for windows that survive
them. `data.raw` was split into `raw_params` (the rng draws, recorded in the entry) and `raw_apply`, so a
replayed window is bit-identical to the sampled one without carrying an rng.

**The shard is the unit** (user, 2026-09-20). A volcomp level's object is a 1024^3 SHARD holding 512
inner 128^3 chunks (a CT level-0 shard is ~17 MB, a level-2 shard ~35 MB). The planner downloads whole
shards -- no byte ranges, no partial objects -- and writes them into the mirror at their normal paths, so
the buffered copy is the origin's object byte for byte and every later window whose reads land in a resident
shard costs nothing. One shard covers 64 windows' worth of volume at 256^3, and the coarse levels are a
handful of shards each, which is where the hit rate comes from; the window ORDER is still the sampler's own
seeded draw (reordering it would break the guarantee that the queue is what the direct sampler would have
drawn), so reuse is residency, not scheduling. `plan.jsonl` reports `hit_rate` (buffer hits over all shard
lookups) and `B_per_vox` (bytes fetched per training voxel).

`usrm2 train ... --stream DIR` makes `data.Patches` replay instead of sample: worker w takes entries
w, w+W, ..., waits (backoff to 1 s) when the planner has not got there yet, and reads each window from the
buffer exactly as before (uint8 path, `prep.prepare` on the GPU unchanged). Every sample carries `idx` and
`wait`; `train.jsonl` gains `stream_wait_ms` and `stream_idx` per 20-step interval, and the trainer writes
`<DIR>/consumed` ({"i", "margin"}) for the planner's eviction bound.

Buffer management: the planner stays `--ahead N` windows in front of `consumed`, and once the cache passes
`--cache-gb` it deletes the chunks of consumed windows, oldest reference first, down to 90 % of the budget.
Every fetched object is booked (`Planner.charge`), including what a REJECTED candidate pulled before the
air / foreground rules dropped it: those are referenced by no entry (index -1) and are evicted first.
Never evicted: whole levels small enough for `data.full_level` (fetched once), the `.absent` markers, any
`zarr.json`, and the validation grid -- the held-out box at every `--val-rungs` is prefetched and pinned at
startup, since nothing else would ever fetch it out of an empty mirror.

Resume: `queue.jsonl` (torn last line dropped), `state.json` (the index plus each worker stream's
bit-generator state, saved with every emitted entry) and the per-worker `progress/w<k>` files. A restarted
planner continues the same rng streams at the same index; a restarted trainer skips what its workers
already served.

`data.remote()` is the explicit inverse of `data.local()` and knows both subtrees under a scroll: the CT
volumes (`<scroll>/volumes/<rest>` on the origin, `<scroll>/<rest>` in the mirror) and the exported
predictions (`<scroll>/representations/...` either way).

tests/test_stream.py runs a real `python -m http.server` over a synthetic volcomp tree and checks: the
planner's window sequence equals the direct sampler's for the same seed; the chunks land where `read_rung`
expects them; the queue replays across W workers with no gaps or duplicates (including a real 3-worker
DataLoader); a consumer started first waits instead of skipping and reports the wait; eviction respects the
budget and never touches an unconsumed window's chunks; a restarted planner continues the index and the rng;
404s become absent markers; `--require-targets` drops a rung whose export the origin does not serve; and a
20-step `train --stream` run logs `stream_wait_ms` and writes `consumed`.


## 17. Region mode, and the other scrolls (2026-09-20, evening)

**Region mode** (`--region 1024 --windows-per-region N`, on `train` and `stream-plan`). A shard is the unit
of streaming, and a 1024^3 shard covers 64 windows' worth of volume at 256^3; drawing corners uniformly
over a whole scroll therefore touches a new shard almost every window. Region mode keeps the sampler's own
seeded draws but changes their scope: draw a source and a rung as usual, then ONE 1024^3 region inside the
target's box, snapped to the shard grid of the level the CT is read from (so the visit is one shard
footprint per level), take N windows inside it under the usual rejection rules -- with a cap on failed
draws per region -- and then move on. The region visit is per rng stream, not per dataset: the planner
drives one dataset from four threads, one stream each.

Measured on the A6000 against dl.ash2txt.org, same model and flags, whole-shard streaming both times:

| | cache hit rate | bytes fetched per training voxel | buffer for ~50-300 windows |
|---|---|---|---|
| plain (uniform corners) | 0.24 | 39.5 | 20 GiB, 52 windows |
| region 1024, 64 windows | **0.97** | **1.47** | 6.9 GiB, 300 windows |

So region mode is what makes whole-shard streaming affordable: a 27x drop in bytes per training voxel, and
the buffer holds hundreds of windows instead of fifty.

**Other scrolls.** Their predictions are exported on their NATIVE grids (levels named by true voxel size,
`native_voxel_size_um` in the group attrs, `surface-mask-lossless` encoding), and their CT mirrors are the
usual integer levels of a volume whose name carries its voxel size. `data.rungs` snaps both by the
nearest-rung rule: 9.362 / 9.596 / 8.64 um all land on rung 4, their pooled levels on rungs 5..11, and a
CT volume named `...-9.362um-...` has level l on rung 4 + l. Two things had to be added:

- **an axis per scroll.** The radial channel comes from `data.axis`, and a stores file that mixes scrolls
  must not give PHerc0139 Paris 4's umbilicus. `data.source_groups` now takes each source's axis from
  `<UMBILICUS_DIR>/<scroll>/umbilicus-full-resolution.json`, and `usrm2/umbilicus.py` (`usrm2 umbilicus`)
  puts one there: a published file when the origin serves one (the loader's json, or the volpkg
  `umbilicus.txt` with its 1-based `x, y, z` lines), otherwise DERIVED from the scroll's own CT as the
  per-z centroid of the non-air voxels at rung 9 -- a scroll is a roll, so the centroid of its
  cross-section is the umbilicus to the accuracy the radial channel needs. The points are written in
  RUNG-2 voxels, which is what `data.axis_at` expects. The stream planner pulls that one coarse level and
  derives the axis before it opens the pyramids: measured 1.9-22 s per scroll.
- **the lossless mask codec.** The new exports use volcomp mode 5 (`surface-mask-lossless`), which needs a
  volcomp newer than the desk's build; an instance provisioned from the desk's `libvolcomp.so` opens the
  Paris 4 pyramids and fails on every other scroll's. The library also has to be built ON the instance:
  the desk's is compiled for a Ryzen 9950X (AVX-512) and dies with SIGILL on the Xeon E5-2683 v4 that
  serves the A6000s. `gcc -O3 -shared -fPIC -o libvolcomp.so python/volcomp_shim.c -lm` (no arch flags --
  the header dispatches AVX2 at runtime) is the portable build.

**Measured, 5 scrolls, region mode** (A6000, 12m at 256^3, batch 1, ckpt_act 1, deep 3, compile, ctx 1..9,
rungs 2-11 with the usual boosts, `--require-targets`, `--region 1024 --windows-per-region 64`,
`--cache-gb 20 --ahead 300 --workers 4`, stores: Paris 4 recto + Paris 4 m7 + PHerc0139 m7 (9.362 um) +
PHerc0332 m7 (9.596) + PHerc0343 m7 (8.64) + PHercMANB m7 (9.596)):

- planner: cache hit rate 0.979, 1.37 bytes fetched per training voxel, 13.4 GB fetched for 584 windows,
  buffer 12.5 GiB, 0 failed requests; the four new scrolls' axes were derived from their own CT in
  1.9 / 12.1 / 2.3 / 22.0 s.
- training: 6.5-9.0 Mvox/s, `stream_wait_ms` 8-127 per 20 steps (i.e. ~0.5-6 ms per step: the trainer never
  waits for the network), 26.1 GiB of VRAM.
- composition after 584 windows: Paris 4 89 %, PHerc0139 11 %, rungs 2/3/4/6 = 83/3/11/3 %, 12 region
  visits of 48.7 windows each. Region mode draws the source and the rung ONCE PER REGION, so the source
  and rung mix is correct in expectation but coarse-grained: a few hundred windows are a dozen draws, and
  the per-scroll balance only converges over thousands of windows. `--windows-per-region` is the knob.
- a region whose windows carry no weight at all (masked CT, or outside the target box) used to deliver 64
  zero-loss windows in a row; `_rung_draw` now rejects a window with no weighted voxel outright, after the
  existing draws so the rng stream is unchanged.

The A6000s are served by a proxied GPU (Thunder Compute): twice during these runs the trainer's main thread
parked on a futex with the GPU at 0 % and 33 GiB still allocated, and did not recover. That is an
infrastructure hang, not the loader -- the loader workers had items ready and the machine was idle.


## 18. The no-repeat region walk (2026-09-20, night)

The user: *"I don't want to train over the same data multiple times."* Region mode draws a fresh
(source, rung, region) for every visit, so a long run revisits regions by chance -- at 584 windows the
A6000 run had already drawn Paris 4 twelve times. `--walk once` replaces the draw with an enumeration.

**The list.** `data.region_list` walks every source, every usable rung and every shard-aligned `--region`
tile of the target's box at that rung. A tile whose target is all air is dropped before it can cost a
fetch: `data.occupancy` reads ONE coarse level of the export (the finest level under 64 Mvox -- rung 9 for
Paris 4, 38 Mvox, a few MB over the wire) and `tiles_occupied` takes the block maximum of it over each
tile's footprint, so the whole rung's tiles are filtered in one vectorised pass. Each surviving region
carries

    w(region) = (source's physical-volume share) x (its rung's probability from `rung_probs`) / (regions of that source+rung)

so the rung mix and the um^3 source weights of section 17 are honoured IN EXPECTATION while every region is
visited exactly once. `data.walk_order` turns the weights into the visit ORDER by a weighted shuffle
without replacement (Efraimidis-Spirakis, key = Exp(1)/w): P(region r is first) = w_r, and each region
appears exactly once. The order is a pure function of (seed, epoch), so it is never stored -- `regions.jsonl`
holds the list in enumeration order and `walk.json` the cursor, the epoch and a fingerprint of the
configuration the list was built for. A restarted planner recomputes the permutation and continues at the
cursor: nothing before it is ever handed out again.

When the list runs out the planner writes `epoch_done` ({"windows", "index", "regions"}) and stops; the
trainer's replay iterator then ENDS instead of waiting, so `usrm2 train --stream` finishes its epoch, logs
`stream_end`, evaluates and saves. `--epochs N` re-permutes and walks again instead. The epoch's window
count is rounded down to a multiple of the stream count so every replaying (rank, worker) gets the same
number of windows and a DDP run's ranks stop together (the replay holds its last entry back until the
bound is known).

**Interleaving.** `--active-regions K` (default 4) keeps K regions OPEN at once: K `visit` tasks each draw
their own region's windows into their own buffer, and the emitter takes from them round robin, so
consecutive queue entries come from different regions. A slot is released -- free to take the next region
of the walk -- only once every one of its windows has been queued, so all K footprints stay resident for as
long as they are being drawn and the hit rate does not fall. Each region's draws come from its own rng
seeded by its walk ordinal, which is what makes a resume reproducible without carrying bit-generator state.
Every queue entry records its region as `g`.

**Streams, not workers.** `--stream` replay used to shard the queue by loader worker only, so with DDP
every rank replayed the SAME entries and the run saw each window `world` times -- exactly what the walk is
meant to prevent. The share is now taken over `world * num_workers` STREAMS and `stream-plan --workers` is
that total (2 ranks x 8 workers = `--workers 16`).

**The mirror is not the buffer.** The desk already mirrors Paris 4 (levels 1-9 complete, level 0 boxed).
`Planner.mirrored` snapshots `data.chunk_index` for every level BEFORE the first fetch: whatever
`mirror.json` says the mirror owns (a complete level, or the boxes of a partially pulled one, where an
absent file means air) is a hit that is never fetched, never charged against `--cache-gb` and never evicted.
Only planner-fetched shards are the rolling buffer. `plan.jsonl` reports `mirror` (such hits) and folds
them into `hit_rate`, plus `regions` / `regions_left` / `regions_total` / `epoch` / `active` and
`region_MiB` (bytes fetched per region, which is what `--cache-gb` has to hold K of).

### 18.1 The region contract with the teacher service (2026-09-20, coordinator)

`usrm2.stream.region_walk(stores, rungs, seed, patch, region, boost, exclude)` IS the walk, as a plain list:

    [(source line, rung, (z, y, x) origin in rung-k voxels), ...]        # visit order

(`records=True` gives the dicts, with the tile size and the draw weight). The stream planner walks exactly
this list, and `cloud/teacher_regions.py` -- the service that runs the upstream teacher over the regions --
walks it too. Both sides must pass the SAME stores, rungs, rung boosts, patch, region, exclude (the
held-out box) and seed: every one of them is an input to `data.region_list` (which tiles and drops the air)
or to `data.walk_order` (which orders). At rung 2 a region origin is a multiple of 1024: the CT level and
the export are both 1024^3-sharded there, so the walk's tiles ARE the shard grid.

The service writes one probability store per rung-2 region,

    <TEACHER_REGIONS>/<channel>/region_<z>_<y>_<x>.zarr      (z, y, x = the rung-2 origin)
    attrs: origin_zyx (rung-2 voxels), channel, done

and `--teacher-regions DIR` (on `train` and `stream-plan`, recorded in the queue's `meta.json` so a
replaying worker uses the same root) makes the loader PREFER it: a rung-2 or rung-3 window of the first
source that lies inside ONE finished (`done`) store takes its target from that store -- the teacher's soft
probability instead of the thresholded export -- with weight 1 wherever the store covers the voxel and the
CT is not masked. Rung 3 is the 2x mean pool of the same store (`data.read_teacher`). Everything else
falls back to the exported mask pyramid: no store, not `done` yet, a window straddling two regions, another
source, or any other rung. The queue entry records the store path (`t`), so the replaying worker reads what
the planner decided rather than racing the service.

### 18.1b One file per store: every array we write is zarr v3 sharded (2026-09-21)

A 1024^3 region store used to be 512 chunk files (`c/z/y/x`, 128^3 each) plus their directories: 514 files.
Publishing one to the dl.ash2txt.org sftp mirror cost ~590 sftp operations, and with thousands of regions
that hammers the server -- it, not the bandwidth, was the bottleneck (the median store is only ~10 MB).

So `predict.out_array` now creates a zarr v3 SHARDED array: one shard per 1024^3 box (`predict.shard_shape`
= min(1024, shape rounded up to the 128 chunk), so a store smaller than 1024 on an axis is still a single
shard), 128^3 inner chunks, the same `VolcompCodec(q=8)` serializer and the same default zstd compressor.
A finished region is then 2 files, `zarr.json` + `c/0/0/0`, and publishing it is 7 sftp operations, with a
whole batch of stores per sftp session. The rule is general: **every array usrm2 writes is zarr v3 sharded
this way** -- the region/box teacher stores, the m7 and verso stores, `predict`'s output, and the offline
pyramid levels of `cloud/make_levels.py` (whose resume check is now per shard file, and which builds one
whole shard at a time). The old zarr v2 OME "tracer drop-in" export was deleted rather than exempted.

Nothing else changes: the inner chunk encoding is byte-for-byte what it was, so readers are unaffected
(`data.open_zarr` / `data.read_teacher` / `data.chunk_index` already went through zarr and its `shards`
attribute) and `cloud/repack_regions.py` converts the stores already on disk LOSSLESSLY by moving the raw
chunk bytes into a shard file plus its uint64 (offset, nbytes) index and crc32c checksum -- no decode, no
re-encode. Measured on the desk: filling a 1024^3 store in 256^3 blocks takes 41.6 s sharded vs 41.3 s
unsharded (the shard is small enough that even a full rewrite per block is free, and the teacher writes a
region in ONE call anyway); 200 random 128^3 reads cost 1.93 s vs 1.76 s. Converting the 3185 finished
desk stores took 16 s at `--jobs 8`; the decoded arrays are identical to the originals.

The ~505 stores already published in the old layout stay as they are -- deleting them would be another
260k sftp operations, and zarr reads either layout.

### 18.2 `--walk mix`: the coarse rungs must not be used up in the first percent (measured)

A weighted shuffle front-loads the heavy items, and that is a problem for a rung with FEW regions and a
big share. Paris 4 at rung 9 is ONE region (640 x 256 x 256) and `--rung-boost 9=16` asks for a large
slice of the samples; the 10 such regions of the 8-scroll stores file are all drawn in the first few
hundred, and after ~640 windows rung 9 never appears again. Measured on the first 968 windows of the desk
soak: rungs 4/5/6/7/9/11 = 332/192/190/64/64/6 and rung 2 only 120, although rung 2 is 69 % of the regions.
The prefix mix is right (that is what the weights buy), but it is right ONCE.

`--walk mix` (`data.region_visits`) gives a region `round(w * regions)` VISITS instead of one, each of
weight `w / visits` (at most `--visits-max`, 64). Almost every entry then weighs 1 / regions, so the order
is near-uniform, a group's share of the walk is its intended share all the way through, and the fine rungs
still get exactly one visit each. A second visit draws its own windows from its own rng, so it is a denser
sampling of a region, never the same window again -- and it only happens where the weight per region is
above average, i.e. where the ladder has almost no data to sample. `--walk once` remains the strict
no-repeat walk; `mix` is what a long run (the A100's 60k steps) should use.

### 18.3 The desk soak (2026-09-20, 20k steps, 8 scrolls, streamed)

`~/soak_plan.sh` + `~/soak_train.sh` on forlindesk2 (GPU 0 only; GPU 1 was running the region teacher
service): 5m at 128^3, batch 1, deep 3, compile, ctx 1..9, rungs 2-11 with the usual boosts,
`--require-targets --region 1024 --windows-per-region 64 --walk once --active-regions 4 --region-fails 192
--workers 8 --cache-gb 60 --ahead 600`, warm-started from `u1_5m_p4` (60k steps, Paris 4 only). Stores:
10 sources over 8 scrolls (Paris 4 recto + m7; PHerc0139 m7 + recto-090; PHerc0332, PHerc0343, PHercMANB,
PHerc0175A, PHerc0813, PHerc1447 m7) at 2.399 / 2.400 / 8.640 / 9.362 um. Walk: 36329 regions, 2.33 M
windows per epoch, enumerated in 18 s; the 7 new scrolls' axes were derived from their own CT in 1.3-6.4 s
each.

| | |
|---|---|
| windows / regions visited | 20816 / 353 of 36329 (max ordinal 452: 99 regions yielded no window) |
| every window inside its own walk region | 20816 / 20816 |
| regions re-opened after being closed | 0 (max 64 windows each, `--active-regions` never exceeded 4) |
| consecutive entries from a different region | 80.6 % |
| planner cache hit rate | 0.9929 (434009 mirror hits, 3037 shards fetched, 0 failed, 2045 absent) |
| bytes fetched per training voxel | 0.56 (24.4 GB for 20816 windows at 128^3) |
| buffer | median 12.3 GiB, high-water 22.7 GiB of the 60 GB budget, 0 evictions |
| vox_s | 12.14 M (p10 12.09, p90 12.15) -- rock steady over 975 intervals |
| stream_wait_ms per 20 steps | median 1, p99 2, max 3 = 0.02 % of the 56 min of training |
| memory | planner 3.3 GB, trainer 4.6 GB, 8 loader workers 13.0 GB together |
| source mix | 1.5-15.5 % per source, the two Paris 4 sources 28.7 % |
| rung mix | r2 12.3, r3 2.2, r4 49.1, r5 20.7, r6 8.0, r7 1.5, r8 2.5, r9 1.9, r10 1.0, r11 0.8 % |
| val dice (Paris 4's held-out box) | r2 0.613, r3 0.656, r4 0.625, r6 0.338 (warm start: 0.644 / 0.682 / 0.607 / 0.373) |

The trainer never waits for the network: the planner stays parked at `--ahead` and `stream_wait_ms` is a
millisecond per 20 steps. `--cache-gb 60` was never reached, so eviction did not run (the tests cover it).
Val on Paris 4's box moves as the mix moves -- rung 4 up (0.607 -> 0.625, half the windows are rung 4 now),
rung 2 and 3 down a little, rung 6 down (its windows come from other scrolls now) -- which is what a 20k-step
run over eight scrolls at one third the previous rung-2 share should do; it is a pipeline soak, not a
training result.

**`--active-regions` 1 vs 4**, planner only, 1024 windows from a cold buffer, same walk:

| K | hit rate | B per training voxel | windows/s |
|---|---|---|---|
| 1 | 0.955 | 1.91 | 4.6 |
| 4 | 0.960 | 1.82 | 11.6 |

Interleaving costs no residency (all K stay open) and triples the planner's throughput, because the four
regions fetch concurrently.

**Region counts** (patch 256, `--region 1024`, rungs 2-11): the Paris 4 stores file is 30365 regions
(1.94 M windows); the 8-scroll file is 36329 (2.33 M). Per rung, Paris 4 recto is 25093 / 3667 / 599 / 121 /
20 / 3 / 2 / 1 / 1 / 1 at rungs 2..11 out of 76800 / 9728 / 1216 / 160 / 20 / 3 / 2 / 1 / 1 / 1 tiles -- the
all-air check drops two thirds of rung 2 and half of rung 4 before a byte is fetched.

Measured for the 8-scroll file (patch 256): `--walk mix` turns the 36329 regions into 54914 visits
(3.51 M windows per epoch) -- 25093 / 3667 / 15530 / 5428 / 1895 / 665 / 837 / 556 / 606 / 637 at rungs
2..11, i.e. the coarse rungs go from one or two regions each to hundreds of visits spread over the epoch.
`--visits-max` (64) is what stops that: rung 11 of a scroll would need ~470 visits to reach the share that
`--rung-boost 11=80` asks for, and a rung-11 region IS one 256^3 window, so those visits would be the same
16 Mvox over and over. The boosts were written for a sampler that resamples for ever; under a walk they are
capped by what exists, and `mix` gives the coarse rungs as much of their share as the data can carry
(~60x their region count at `--visits-max 64`) without repeating a window.

## 19. Bootstrap run result (A100 `u1_30m6_p4`, finished 2026-09-21 ~04:00 CDT)

30m6 at 256^3, batch 2, 60k steps, upstream masks only (Paris 4 recto 2x-mode + m7), geo augmentation, warm start
r3 raw: val dice r2/r3/r4 = 0.740 / 0.740 / 0.682 (0.59/0.51/0.33 at step 500; 0.73/0.72/0.67 at 30k). Throughput
32 Mvox/s until ~33k, then ~25 (loader-bound on the boxed mirror). Checkpoint backed up to the desk at
/vesuvius/usrm2/runs/u1_30m6_p4_final.pt. Next: `u2_30m6_stream` = streaming region walk (--walk mix), full
augmentation, teacher region stores preferred at rungs 2/3, warm start from this checkpoint, 200k steps.

## 20. Surface metrics: bootstrap final vs the streaming run (2026-09-21)

Val box, head 0, window 128: bootstrap final `u1_30m6_p4` (45M, 256^3, masks only, geo aug): recall@4 0.806,
continuity 0.656, merge_frac 0.44, offset<=3 0.36 -- far above the desk 5m (0.753/0.600/0.44) and round 3
(0.729/0.584/0.42). Streaming run `u2_30m6_stream` at ~7.5k steps (full aug, teacher soft targets at rungs 2/3):
0.748 / 0.623 / 0.48 / 0.29 -- below u1 early on; to be re-measured at 20k and 40k (if it stays below, the
teacher soft targets or the full aug are the suspects: the published mask th0.45 is crisper than the teacher's band).

## 21. Next input channels (direction set 2026-09-21)

Candidate channels are worth adding only when they carry information the network cannot compute from its own
receptive field; local filters (sharpen, Laplacian, Sobel, sheetness, structure-tensor normals) fail that test
since the first 3x3x3 layers learn them. The user picked these to pursue, in order:

1. **Cascade**: the model's own prediction at rung k+1, upsampled, as an input channel at rung k (coarse-to-fine;
   also an iterative-refinement mode when fed its own output). Cross-rung consistency for free, one cheap coarse
   pass per region at inference.
2. **Normalised radius from the umbilicus**: the radial vector gives direction, not how far out the voxel sits
   (core vs mid-wraps vs the outer wrap by the case, which sets sheet spacing, curvature and damage).
3. **Scan-level conditioning planes** from the upstream `metadata.json` next to every open-data volume
   (`<bucket>/<scroll>/volumes/<vol>.zarr/metadata.json`; example kept at
   `docs/example_metadata_paris4_2.4um_78keV.json` without the motor/attenuator dumps). Fields that vary between
   scans and plausibly change what papyrus looks like:
   - `scan.tomo.acquisition.energy` (keV; 74 / 78 / 137 across Paris 4 alone), `detector.samplePixelSize`
     (the true voxel pitch; the scale plane already carries the rung), `detector.scintillator`,
     `sampleDetectorDistance` (phase contrast strength: 220 mm vs 11 m), `expo_time`, `tomo_N`,
     `half_acquisition`;
   - `scan.tomo.processing.preprocessing.phase.delta_beta` and `unsharp_coeff/sigma` (Paganin filtering, i.e.
     how blurred/edge-enhanced the sheets are; 1000/4.0/1.2 on the 2.4 um 78 keV scan, 500/4.0/2.5 on the
     1.1 um mosaic);
   - `scan.tomo.processing.32bitsData.histogram.*` percentiles and `zarr_export.window_u16_min/max` and
     `target_window_f32_*` (the linear map from reconstructed attenuation to the stored uint8, so the absolute
     intensity that the per-window z-score throws away can be restored as a plane);
   - `mosaic.*` (the 1.1 um Paris 4 volume is a 19-tile fused mosaic; per-tile acquisition blocks).
   Encoded as constant planes (log-scaled where the range is wide), zero-filled for the warm start like the scale
   plane was. Our own scan-level papyrus mean/std (data.global_norm) belong in the same set.
4. Distance to the scroll's outer boundary / case: undecided.
5. Existing segmentation meshes as a sparse known-surface channel: later (refinement / interactive mode only).

Angular position about the axis and absolute z stay out: they break the symmetry augmentations and carry little.

## 22. The cascade input channel (implemented 2026-09-21)

Section 21 item 1. At rung k the model also receives, as ONE extra input channel, the prediction at rung
k+1 over the same field of view, upsampled 2x onto the patch grid. Channel order becomes

    [CT, ctx_1..ctx_9, CASCADE, scale plane, radial(3)]   = 15 channels

The cascade channel sits right after the context cubes and before the scale plane. Its value is a
probability in 0..1 and it is NOT z-scored -- it is not an image channel, exactly like the scale plane. The
cube symmetry (`prep.sym_apply_t`) applies to it like any image channel: it is a spatial field with no
vector part, so it is permuted and flipped and never negated. `--cascade off` (the default) changes
nothing: every existing run and checkpoint stays 14-channel.

### Where the channel comes from while training (`--cascade off|mask|self|mix`)

- **mask**: the target pyramid at rung k+1 over the patch FOOTPRINT -- a 128^3 block for a 256^3 patch, read
  by the worker (`data.Patches._cascade_extras` -> `rung_item["cm"]`) with the same `read_rung` machinery the
  rung-k target uses, so beyond the top of a pyramid it is pooled like everything else -- upsampled 2x
  (`model.up2x`, the training-time trilinear) on the GPU. NOTE THE LEAK: the rung-(k+1) level of an exported
  mask pyramid is the 2x pool of the rung-k target, so without noise this channel is a blurred copy of the
  answer and the model will learn to copy it. `--cascade-noise` is therefore ON by default and adds, on top
  of the blur the 2x pool + 2x upsample already applies, a one-voxel erosion or dilation (probability 0.3
  together) and 32^3 block dropout (probability 0.2, four blocks).
- **self**: the model's OWN prediction at rung k+1, computed on the fly with the EMA weights, no grad,
  autocast bf16. The rung-(k+1) input is already mostly in the sample: its CT cube IS ctx_1 (same 256^3 size,
  same centre), its contexts are ctx_2..ctx_9 plus ONE more cube at rung k+10 (the loader loads a tenth
  context cube when cascade is on, pooled past the pyramid top like the others), its scale plane is
  (k+1-2)/9, its radial vector is recomputed at rung k+1 (`prep.radial_t` with the coarser corner and axis,
  carried in the sample as `lo1` / `cyx1`), and its own cascade channel is ZERO -- a one-level truncation, so
  the recursion never runs away. The central 128^3 of that 256^3 output (sigmoid of head 0) is the patch
  footprint; upsampled 2x it is the channel. Cost: one extra forward per sample.
- **mix**: per sample, `self` with probability `--cascade-self-p` (default 0.5), else `mask` (+ noise). This
  is the recommended production mode. `mask` alone leaks the target; `self` alone never shows the model a
  coarse prediction better than its own current one, so early training teaches it that the channel is noise.
  The mixture brackets what inference actually feeds it: at inference the channel is a real prediction from
  the same weights, which is closer to `self`, but a good coarse prediction looks like a soft `mask`.
- All modes: with probability `--cascade-drop` (default 0.1) the channel is zeroed, so inference with a
  MISSING coarse prediction is in distribution -- that is rung 11 (there is no rung 12), `--cascade-depth 0`,
  and the deepest level of any top-down run. Rung 11 samples always get zero.

The exposure mismatch is the whole reason for the mixture and the dropout: the model is trained on one
distribution of coarse channels and run on another. Noise widens the training distribution, dropout puts its
degenerate end (nothing at all) inside it, and `mix` puts a real self-prediction inside it too.

### The tenth context cube

`--ctx 1..9` gives the rung-k input nine context cubes. The rung-(k+1) input needs rungs k+2..k+10, i.e.
ctx_2..ctx_9 plus one more. So with cascade in `self`/`mix` the worker reads a tenth cube at offset
`ctx[-1] + 1` and ships it as `cx`; the stream planner fetches its shards in `fetch_ctx`, and the queue's
`meta.json` records `cascade` so a queue can only be replayed by a run with the same mode.

### Inference: top-down

`predict.probs(..., cascade=..., cascade_depth=3)` runs TOP-DOWN. To predict rung k over a box it first
predicts rung k+1 over the box's footprint (half size on every axis, plus a 16-voxel halo) with cascade
recursively, up to `cascade_depth` rungs above; above that the channel is zero. Each coarse level is
upsampled 2x with the same `model.up2x` and cropped to the box. The cost is geometric, 1/8 per level: three
levels add 1/8 + 1/64 + 1/512 = 14.3 %.

The recursion is OUTSIDE the net: `slide` and `slide_gpu` are unchanged and the compiled / TRT path is just
called on 15-channel input, so nothing about the batched GPU path or an engine build changes. `--cascade
auto|on|off` on `predict`, `evalsurf` and `verso` reads the mode from the checkpoint args by default: a
checkpoint trained with `cascade != off` infers with the channel on, a checkpoint trained with it off is
14-channel as before. `off` still FEEDS the channel (a 15-channel checkpoint always needs 15 channels) --
it feeds zeros, which is exactly the `--cascade-drop` case. `cloud/teacher_regions.py` has no `--cascade`:
it runs the upstream villa teacher, not a usrm2 student, so there is no cascade to pass.

### Checkpoint, resume and the warm start

`cascade`, `cascade_self_p`, `cascade_drop` and `cascade_noise` go into the checkpoint args (only when
cascade is on, so an old checkpoint's args are untouched and an old run resumes unchanged), and `cin`
becomes 15. They are NOT in the resume `grow` tuple: they must match on resume, as `cin` does.

`train.warm_start(src, cin, cout, cascade=True, src_scale=True)` widens 14 -> 15 by keeping the image cubes
first, the radial vector last, the SCALE PLANE lined up with the destination's scale plane, and zeroing only
the cascade slot. The generic "image channels first, radial last" rule is not enough here: it would slide
the source's scale weights into the cascade slot and zero the scale plane instead, which changes the output
at every rung but 2. With the fix the warm start is exact -- with the cascade channel zero the outputs agree
to a float32 ulp (1.3e-6 relative, the stem convolution accumulating 15 products instead of 14), so
`u1_30m6_p4` / `u2_30m6_stream` warm-start into a cascade run with no jump at step 0.

### Measured step cost

`cloud/cascade_bench.py` (synthetic samples, so it isolates the cascade work), A100 while the
`u2_30m6_stream` run was training, so the absolute numbers are contended and only the ratios matter:

| config | off | mask | self | mix (P=0.5) |
|---|---|---|---|---|
| 30m6, 128^3, batch 2, ctx 9, deep 3 | 1.000 (369 ms) | 1.073 | 1.535 | 1.350 |
| 5m, 128^3, batch 1, ctx 9, deep 3 | 1.000 (240 ms) | 1.053 | 1.424 | |

`mask` costs ~6 % (a 1/8-size block, a 2x upsample and one more stem channel). `self` costs ~50 %: one
no-grad forward against a forward+backward+step is about half the work. `mix` at P = 0.5 lands halfway, ~35 %.
Expected A100 throughput for `mix` at the production config (30m6, 256^3, batch 2): 27 / 1.35 = **~20
Mvox/s** (mask would be ~25). The run is often loader-bound rather than GPU-bound, in which case the real
loss is smaller, since the cascade adds GPU work only -- the worker reads one extra 1/8-size target block
and, in self/mix, one more context cube.

The desk could not be measured: both 5060 Ti's were at 100 % with ~0.9 GB free while the teacher region jobs
ran, and taking the last of their VRAM risks failing THEIR allocations.

### One aug detail

`aug.apply` takes the leading `C - 3` channels as "image channels" for the INTENSITY augs, which in a
14-channel run means the scale plane is gamma'd and brightness-shifted along with the cubes. Under
`--cascade` that would also hit the cascade channel, and a brightness shift would move a DROPPED (zero)
channel off the value the model is taught to read as "no coarse prediction". So `apply` now takes an
optional `nimg` and `train` passes the cube count when cascade is on: the intensity augs then act on the
CT and context cubes only, and the cascade channel and the scale plane are left alone. Spatial augs still
act on every channel (the cascade channel is a spatial field and must ride the same grid). A run with
`--cascade off` is bit-for-bit what it was.

## 23. The verso output channel (implemented 2026-09-21)

ONE model, two outputs. The final 1x1x1 head grows from `cout=1` to `cout=2` -- channel 0 recto, channel 1
verso -- and the deep-supervision heads follow (they are `cout`-wide too). There is NO second head, no
second branch and no second decoder: the whole net up to the last convolution is shared, and the verso
output costs `w0` extra parameters per head (32 weights + 1 bias on the 30m6) and nothing measurable per
step. `--verso` on `train` is what switches it on; `--cout 2` is an optional assertion that the output
channel list came out the length you expected.

### Where the verso target comes from

The verso target has no pyramid. Unlike recto -- which has published masks, exported target pyramids and
region teacher stores -- verso exists only as REGION STORES, written by the 5090 pod as it works through the
same region list:

    <root>/verso/region_<z>_<y>_<x>.zarr        z, y, x = the region origin in RUNG-2 voxels

1024^3 uint8 (probability * 255), zarr v3 sharded (one data file, `c/0/0/0`), volcomp q8, written with
`predict.out_array(..., channels=["verso"])`, attrs `origin_zyx` / `voxel_um` / `rung` / `volume` /
`umbilicus` and `done: true` when the pod has finished it -- the same contract as the recto teacher region
stores of section 17, under a sibling directory. `--verso-regions` is that root; it defaults to
`--teacher-regions`, so one directory holds `recto/` and `verso/`.

The loader reads it exactly as it reads a recto teacher store (`data.read_teacher`): rung 2 is the store's
own grid, rung 3 is its 2x mean pool, and a window must lie inside ONE finished store (a window straddling
two regions falls back, as for recto).

### The weights

The target tensor has two channels and so does the weight tensor -- the per-channel ignore of section 3,
which the losses already understand. Per channel:

| | recto (channel 0) | verso (channel 1) |
|---|---|---|
| rung 2 | recto teacher region store if one covers the window, else the published mask pyramid | the verso region store's probability; weight = the store's `inside` mask AND CT > 0 |
| rung 3 | same, the store's 2x pool | the store's 2x pool, same weight |
| rungs >= 4 | the mask pyramid | **weight 0 everywhere** -- there is no verso source above rung 3 yet |
| no store / straddling / blank patch | unchanged | weight 0 |

So a sample with zero verso weight everywhere is a perfectly good recto sample, and most samples are exactly
that. Recto weights are not touched by any of this.

The foreground and density rejection rules (`fg_min`, `dense_pow`, "every voxel masked") look at the RECTO
channels only, so a `--verso` run draws the same windows a `--cout 1` run would from the same seed -- which
is what lets one stream queue feed either.

### The losses with a dead channel

`train.losses_tw` had to change in one place. The BCE is a single weighted mean over the whole tensor, so a
channel that is weight 0 throughout a batch contributes nothing to the numerator AND nothing to the
denominator: it neither adds to the loss nor rescales the recto term, and an all-zero weight tensor (the
blank-patch aug) gives 0, not a NaN. The soft dice is per channel and used to be `.mean()` over them; a
weight-0 channel scores a constant 0 there and would have HALVED the recto dice on every sample without a
verso store. It is now the mean over the channels the batch says anything about:

    live = (wv.sum(dims) > 0)
    dice = (per_channel * live).sum() / live.sum().clamp_min(1)

With one channel, or with every channel live, that is the old value exactly; with nothing live it is 0. The
gradient into the verso row of the head is exactly zero on such a batch (the weight multiplies both `p` and
`tgt` in the dice and the BCE term). `deep_losses` pools the weights alongside the targets as before, so an
ignored channel stays ignored at every supervision level. Measured `overlap` (mean excess
relu(p_recto + p_verso - 1)) is now reported for any even `cout`, so the recto/verso pair gets it too, and
`eval.jsonl` carries `dice_recto` / `dice_verso` -- a channel scored only on the patches that weigh it, so
`dice_verso` is simply absent until a verso store covers the validation box.

### The planner fetches the verso stores

The pod publishes continuously, so the verso stores appear region by region WHILE the run trains. Two paths:

- **local** (`--teacher-regions DIR` on `train`): `<DIR>/verso/` is read the same way `<DIR>/recto/` already
  was. A store that was missing is re-probed after `data.TSTORE_TTL` (30 min) -- a miss must not be cached
  for the life of the run.
- **streamed** (`stream-plan --verso --verso-regions-url URL`): when the planner plans a region it makes that
  region's verso store local first. One GET of `zarr.json` (a few hundred bytes) decides: 404 = not published
  yet, `done` false = still being written (the file is removed again, so no loader ever opens a half store),
  otherwise the single shard `c/0/0/0` is pulled too and the store is complete under
  `<--verso-regions>/verso/`. The result is cached per region -- a hit for good, a miss for
  `stream.VERSO_TTL` (30 min) -- so the cost per window is a dict lookup, and the download goes through the
  existing keep-alive `aiohttp` session and the same `--jobs` semaphore as everything else, with two attempts
  and no `.absent` marker (an absent verso store has to be allowed to become present). The published root is

      https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/representations/predictions/teacher_regions/verso-2.4um/

  (`data.VERSO_REGIONS_URL`). Verso stores are NOT charged to the rolling chunk buffer and are not evicted:
  every window of a region reads the one store, and `<DIR>/verso` therefore grows with the walk (~30-60 MiB
  per region at volcomp q8). `plan.jsonl` reports `verso_stores` and `verso_MiB`.

The queue descriptor gains `"v"`, the store the planner saw, exactly as `"t"` carries the recto one; a
replaying worker that finds no `"v"` looks once itself, so a store published between planning and replay is
still used. `meta.json` carries `verso`, `verso_regions` and `verso_url`, and the queue's channel list now
ends in `verso`, so a `--verso` queue can only be replayed by a `--verso` run and vice versa.

### The warm start

`train.warm_start` already did this: head j takes source head j mod n, for the main head and every
deep-supervision head. A `cout 1 -> 2` warm start therefore copies the recto filter and bias into the verso
channel, so at step 0 the verso output says exactly what the recto output says -- a sane starting point (the
verso face is the same sheet) that the verso loss then pulls off it. The recto channel is untouched: the
head is a 1x1x1 convolution, so channel 0 is the same dot product either way and the outputs agree to one
float32 ulp (measured 1.2e-7 relative; a 2-output convolution accumulates in a different order).

This composes with the cascade 14 -> 15 input warm start of section 22 -- the stem and the head are adapted
independently -- so the A100 restart does both in one `--init-from`: 15 input channels with the cascade slot
zeroed, 2 output channels with verso copied from recto, and the recto prediction at step 0 is the old run's
to a ulp.

### Resume and inference

`verso` and `cout` are NOT in the resume `grow` tuple: they change the head, so they must match a resumed
checkpoint (`cout` was already checked, being in `args`). `verso_regions` and `verso_regions_url` ARE in it,
like `teacher_regions`: where the stores come from may change between restarts without changing the model.
A `cout=1` run resumes exactly as before -- `verso` and friends are only written into `args` when the flag is
on, so an existing checkpoint's args are byte-for-byte what they were.

`predict.probs(head=...)` and `evalsurf --head` take a channel NAME as well as an index: `--head recto` = 0,
`--head verso` = 1, resolved against the checkpoint's own `args["channels"]`. `--head verso` on a cout=1
checkpoint is an error that names the alternative: `--radial-sign -1`, the old flip trick (a recto model
shown a mirrored world, `verso.py`), which is unchanged and still the only way to get a verso band out of a
single-output checkpoint. `predict` writes the channel name into the store's `channels` attr, so the verso
stores the pod publishes and the ones a student writes are self-describing.

`evalsurf --head verso` runs, but there is NO published verso surface to score against. It scores at the
RECTO surface points, so `offset_mean` becomes the sheet thickness rather than a bias and recall/precision
are not comparable with a recto run; the command prints that note. Scoring against the recto surfaces
shifted by a guessed thickness would only measure the guess, so it is deliberately not done. Until verso
surfaces exist, `--head verso` on evalsurf is a smoke test plus a thickness readout.

`val_png` needs no change: it already tiles every target channel and every output channel, so a cout=2 grid
shows CT, recto target, verso target, recto prediction, verso prediction (a verso target with no store is
black).

## 24. The codec chain of the stores we write (2026-09-21)

`zarr.create_array(..., serializer=VolcompCodec(q=8))` leaves zarr-python's DEFAULT compressor in place, so
every store `predict.out_array` and `cloud/make_levels.py` wrote had inner codecs **[volcomp q8, zstd]** --
zstd running over already-compressed volcomp output. Measured on a real 1024^3 region store that zstd layer
saves **0.2 %**, and it costs a decode step on every chunk read. The upstream C-tool exports and the CT
volumes are volcomp-only, so our stores were also not byte-comparable with theirs.

Both creators now pass `compressors=None`: the inner codec chain is exactly `[volcomp]`
(`tests/test_verso_out.py` writes a real store and asserts the `sharding_indexed` configuration's `codecs`
is a single `volcomp` entry, then reads it back).

Stores written before this are converted in place, losslessly, by

    python cloud/repack_regions.py DIR --strip-zstd [--jobs N] [--verify]

which, per shard file, zstd-decodes every chunk payload, rebuilds the payload region, the uint64 LE
(offset, nbytes) index and the index crc32c, and rewrites `zarr.json` without the zstd codec. The volcomp
bytes are never touched, so the decoded array is bit-identical; attributes (including `done`), the
`.published` marker and the atomic `<store>.tmp` swap work as in the sharding repack, and not-done stores
are skipped so the converter never races a writer. Run over the desk
(`/vesuvius/usrm2/teacher_regions/recto --jobs 8`) and the A100 (`~/teacher_regions/recto --jobs 2`, niced).

`--check` is the cheap audit that goes with it: it reads FOUR bytes -- the first stored chunk's first four
-- and compares the zstd frame magic against what `zarr.json` declares. That catches a store whose shard
came from one side of a swap and whose `zarr.json` came from the other.

### An in-place conversion under a live reader corrupts that reader (2026-09-21, learned the hard way)

The directory swap is atomic, but an OPEN zarr array CACHES ITS METADATA. A training worker or a stream
planner that opened a store before the swap keeps decoding with the old codec chain and old chunk grid, and
its first read after the swap raises `numcodecs Zstd decompression error: invalid input data`. That is how
the earlier in-place SHARDING repack killed the A100 `u2_30m6_stream` run at 18:01 UTC on 2026-09-21: a
worker still holding the unsharded `zarr.json` read the new `c/0/0/0` shard file as chunk (0,0,0). An rsync
reading a store across the rename is the same hazard one level out -- it can ship the shard from one side
and the `zarr.json` from the other.

So neither conversion may run while a reader is running:

- **A100** (`~/teacher_regions/recto`): only in a restart window, with no trainer and no planner up.
- **desk** (`/vesuvius/usrm2/teacher_regions/recto`): `bash ~/sync_stop.sh && bash ~/publish_stop.sh`, then
  the strip, then `bash ~/sync_start.sh && bash ~/publish_restart.sh 2`. The two teacher WRITERS may keep
  running: they never read a finished store, and the converter skips stores without `done`.

As a backstop -- not a licence -- every read `usrm2/data.py` makes now retries ONCE against a freshly
opened array when it fails with something that looks like a decode error (`data.read_slice`, used by
`read3`, `read_block`, `full_level` and `read_teacher`; the pyramid and the region-store cache keep the
re-opened handle, so the retry costs one open, not one per read). `tests/test_verso_out.py` reproduces the
A100 failure exactly -- open a store, strip it underneath, watch the raw read raise -- and checks the
recovery, and that a non-decode error is still raised rather than retried.

## 25. Evaluation v2 (Phase A0, implemented 2026-09-21)

Phase A0 / experiment 1 of `docs/research/synthesis_v2_with_literature.md`, from
`docs/research/lit_evaluation_metrics.md`. The complaint it answers: section 20 quotes `recall@4 0.806,
continuity 0.656, merge_frac 0.44, offset<=3 0.36` as one number per box per checkpoint, with **no
confidence interval, no ceiling and no global topology check**, while the labels are a machine teacher
whose own Dice ceiling is ~0.90-0.93. Two adjacent checkpoints differing by 0.02 was an unfalsifiable
claim. Everything here is CPU-only and costs no GPU-hours.

**Nothing old changed.** `metrics()` and `continuity()` in `usrm2/evalsurf.py` compute `recall@{2,4,8}`,
`offset_mean/std/le3/frac`, `precision6`, `pos_frac`, `merge_runs/merge_frac`, `continuity`, `hit_frac`
and `mean_run` exactly as before, and `evalsurf` still prints them on one json line with the same keys, so
every number in sections 19-20 stays comparable. `--window 128 --halo 16` are still the defaults and
`--head` / `--cascade` pass through untouched. The only edit to an old function is
`metrics(..., precision=False)`, which *omits* `precision6`/`pos_frac` (their KD-tree is over the whole
box and is not a per-surface quantity); with the default `precision=True` the output is bit-for-bit what
it was.

### 25.1 The noise ceiling (`--ceiling [STORE]`)

A model scored against a noisy reference cannot, in expectation, beat what that reference's own noise
permits (Metrics Reloaded, Maier-Hein et al. 2024). So: run the **identical** suite with a label source in
the model's place and print every number as `value (ceiling)`.

- Default source: the published recto mask pyramid,
  `.../representations/predictions/surfaces/…-surface-recto-2um-ps256-L0-th0.45.zarr/2.4`
  (`USRM2_CEILING_STORE`). `--ceiling PATH` names another; bare `--ceiling` uses `--teacher` when given.
- Cached per `(box, store, tifxyz, umbilicus)` as
  `<dir of USRM2_VAL>/evalsurf_ceiling_<z>_<y>_<x>_<hash>.json` (`USRM2_CEILING_CACHE`), so every later
  run prints the ceiling for free. `--no-ceiling-cache` recomputes.
- The published mask is a *threshold* of one lineage, so it is not the only ceiling worth having: the
  literature's own caveat is that one teacher's ceiling is itself noisy, and the second one
  (recto lineage vs `m7`, `--ceiling /vesuvius/usrm2/teacher/eval_m7…`) is still to be measured.

How to read it: a metric at or above its ceiling is **saturated on this label source** and can only be
pushed further by better labels or by the mesh-measured metrics; a metric far below its ceiling is where
model work still pays.

### 25.2 Expected run length, in micrometres (`erl_*`)

Januszewski et al. 2018 (flood-filling networks) score a reconstruction by how far a neurite can be traced
before a split or a merge. The papyrus analogue is exact, and the walk graph already exists: the tifxyz
**UV grid** of each published surface.

- A grid vertex is **good** when the probability reaches `thr` within +-`r`(=4) voxels along its normal
  (no break) **and** the ray crosses `thr` exactly once over +-`far`(=40) voxels (no merge -- the same
  `merge_runs` test `metrics()` has always used).
- An edge between two neighbouring in-box vertices is traversable when both endpoints are good; its
  length is the Euclidean distance between the two mesh points **times the voxel size in um** (2.4 um for
  the Paris 4 2.4 um volume; read from the volume name). Both grid axes are walked.
- `erl_um = sum(L_i^2) / sum(L_total)` over maximal traversable runs: the expected length of the run
  containing a point drawn uniformly **by length**, so a 2 mm break costs far more than a 2-voxel one.
  `path_um` is the denominator, the total meshed surface length inside the box.
- `erl_break_um` and `erl_merge_um` repeat the walk with only one of the two stopping conditions, which
  separates "the band is torn" from "the band has fused into a neighbour".
- `lost_break_frac` / `lost_merge_frac` split the surface length between the two failures directly: every
  vertex gets half of each incident edge, and its share is charged to breaks when it has no band, to
  merges when it has a band but the ray crosses twice.

`mean_run` (section 20) is kept and still means grid *cells* along grid *rows* only; `erl_um` is its
physical-units, both-axes, merge-aware replacement. Quote `erl_um`; keep `mean_run` for continuity with
old runs.

### 25.3 Betti-0 / Betti-1 error (`usrm2/topo.py`)

`merge_frac` and `continuity` are local proxies: neither can see a hole spanning more than a couple of
grid cells, nor a handle that is not a normal-ray double crossing. The global check is the Betti number
error (Hu et al. 2019), computed on the **cubical complex** in which every foreground voxel is a closed
unit cube (26-connected foreground, 6-connected background):

```
b0  = connected components of the thresholded band            (scipy.ndimage.label, 3x3x3)
b2  = components of the complement, one zero layer padded on, minus the unbounded one
chi = V - E + F - C over the cells of the cubical complex     (exact, counted, chunked along z)
b1  = b0 + b2 - chi
```

The **approximation** to document: `b1` is *derived* from the Euler characteristic rather than counted.
The count itself is right (a cubical subcomplex of R^3 is torsion-free, by Alexander duality), but unlike
persistence it says nothing about *where* a loop is, and one spurious handle cancels one missing loop.
**TODO**: Betti *matching* error (Stucki et al., ICML 2023; arXiv:2407.04683) -- spatially matched,
differentiable, and the honest version of this metric. It needs a 3D persistent-homology implementation;
the efficient one is C++/CUDA and is not a dependency we carry, so it is deliberately left undone rather
than half-built.

Two masks make the number mean something:

- **Interior margin** (`--betti-margin`, default 8 voxels, >= the halo): a sheet the box merely cuts
  through would otherwise read as a component the model invented and a loop it opened. Every box face is
  cropped by the margin before anything is counted.
- **Band around the mesh** (`--betti-band`, default 6 voxels): the published meshes cover only *some* of
  the sheets crossing the box, so an unrestricted count charges the model for every correctly predicted
  sheet that happens to have no mesh. Both volumes are restricted to voxels within that distance of the
  reference. The band must stay below half the sheet pitch (15-35 voxels here) or two sheets' bands fuse
  and a merge stops being visible; a bridge longer than the band is likewise invisible. This is the
  metric's main blind spot and is why `merge_frac`/`lost_merge_frac` are kept alongside it.

The reference is the mesh itself: `topo.rasterize` fills every quad whose four corners are finite on a
lattice dense enough (0.4 voxel) that the rasterized sheet is 26-connected -- the tifxyz grid is many
voxels coarse, and rasterizing the bare grid points would give a cloud of specks with a meaningless `b0`.

### 25.4 Bootstrap confidence intervals and `--json`

Points inside one surface are far too correlated to resample individually, so the bootstrap resamples
**surfaces** with replacement (200 draws, seeded by `--seed`; `--bootstrap 0` turns it off) and repools.
Pooling is exact, not approximate: point-weighted means for the rate metrics, the pooled-variance formula
for `offset_std`, length-weighted recombination for ERL, and quantiles over the concatenated per-point
offsets for the new `offset_hd95` / `offset_p99` (HD95, section 2 of the metrics survey). `precision6`
and `pos_frac` are box-level, not per-surface, and therefore have no CI.

`evalsurf` now prints, after the legacy json line, a table of `metric  value (ceiling)  [lo, hi]` -- the
form every number should be quoted in from here on. `--json OUT` dumps the whole thing: the box, the
voxel size, per-surface rows, the pooled metrics, the CIs, the Betti block, and the ceiling's own copy of
all of it. The ceiling json line is printed **before** the main source's, so `grep '"recall@4"' | tail -1`
in the existing `~/eval_u2.sh` / `~/eval_u3.sh` still picks the model, not the ceiling.

### 25.5 Declaring a plateau (`usrm2 evalsurf-curve`)

```
usrm2 evalsurf-curve RUN_DIR [--metric dice] [--unbounded] [--json OUT]
```

Reads `RUN_DIR/eval.jsonl` (val dice per step) when it exists, otherwise a directory of
`evalsurf --json` dumps (each carries its `step`), and fits both

```
power     y = c - a * step^-alpha          (Hestness et al. 2017)
logistic  y = c / (1 + exp(-k*(log10 step - m)))
```

keeping the lower-RMSE one -- a power law does not saturate below 1, so for a [0,1] metric the logistic is
usually the honest fit. It prints the fitted **asymptote** `c`, the **step at which 95% of the gain still
outstanding at the last measured step** has been collected, and the **current slope per 10k steps**.

**The decision rule.** Call a metric plateaued when its fitted gain over the next 10k steps is smaller
than the bootstrap CI width from 25.4 -- signal below noise -- and not on a step budget. Do not trust a
saturation call from fewer than ~10-15 checkpoints, and never from an unsmoothed two-point comparison.
Conversely, a metric at its 25.1 ceiling is done regardless of what the fit says: more steps cannot beat
the label.

### 25.6 How a Phase-A0 number is quoted

> `recall@4 0.806 (ceiling 0.9xx) [0.7xx, 0.8xx]`

value, ceiling, 95% CI over surfaces. An experiment in the section-5 table of the synthesis moves a metric
only if it moves **outside that CI**; a change inside it is not evidence. Per-surface rows are in the json
and should be looked at before believing a pooled mean -- one badly broken patch hides inside many good
ones (the survey's first pitfall).

## 27. Physics augmentation v2 (`full2`, implemented 2026-09-21)

Phase D / experiment 10 of `docs/research/synthesis_v2_with_literature.md` (R9), from
`docs/research/lit_ct_physics_augmentation.md` sections 1, 4, 5 and 9. Three changes, all behind the preset
mechanism: **`full` is untouched**, and `full2` = `full` + the two new keys (`paganin`, `shuffle`).

### 27.1 `_paganin_jitter`: re-filtering the cube at a different delta/beta

nabu's phase step is a Fourier-domain low-pass (Paganin 2002) followed by an unsharp mask, both applied to
the projections before reconstruction. Their net effect on the reconstructed volume is close to isotropic,
so we model the pair as one 3D transfer function. With `f` the spatial frequency in cycles/um:

```
H(f; db)    = 1 / (1 + pi * db * lambda * D * f^2)        Paganin: a Lorentzian low-pass
U(f; a, s)  = 1 + a * (1 - exp(-2 pi^2 s^2 f^2))          nabu's unsharp, I' = (1+a) I - a G_s * I
```

with `db = delta/beta`, `lambda = 1.2398e-3 / E[keV]` um, `D = sampleDetectorDistance` (um), `s` the unsharp
sigma in um. (Paganin's own form, `1 / (1 + db*lambda*D*|k|^2/(4 pi))` for the angular wavenumber
`k = 2 pi f`, is the same thing.) A scan is reconstructed ONCE, with its own `(db0, a0, s0)`. To ask what
the cube would have looked like had nabu been run with `(db', a', s')`, we apply the RATIO of the two
transfer functions to the reconstructed cube:

```
T(f) = H(f; db') U(f; a', s') / ( H(f; db0) U(f; a0, s0) )
     = (1 + L0^2 f^2) / (1 + L1^2 f^2) * U(f; a', s') / U(f; a0, s0),    L^2 = pi * db * lambda * D
```

so the scan's own parameters give `T == 1` exactly: **the identity is inside the sampled range**, which is
what makes the op safe to switch on for a scan whose metadata we have (tested to <= 1/255). One
`rfftn`/`irfftn` per batch, cost comparable to `_spectral`.

Everything enters as `L/vox` and `s/vox`, so the frequency grid is in cycles per VOXEL and the op is
**scale-aware for free**: the same physics at a coarser rung is a proportionally smaller filter in voxels.
`vox_um` is set by `for_rung` and may be per sample.

Two guards: `T` is clamped to `[1/gmax, gmax]` (`gmax=4`) because a smaller `db'` is a deconvolution and
would otherwise amplify the noise floor without bound, and the output is kept inside the sample's own tone
range widened by `keep` (0.25 of the range) because a Lorentzian ratio overshoots at sheet edges.

Sampling (`aug.PAGANIN`): `db'` log-uniform, `a'` uniform, `s'` log-uniform. The defaults are the corpus
span widened 2x each way (`scanmeta.WIDEN`):

| parameter | corpus (the two inspected scans) | sampled default |
|---|---|---|
| delta/beta | 1000 (2.4 um Paris 4) / 500 (1.1 um mosaic) | 250 - 2000, log-uniform |
| unsharp coeff | 4.0 / 4.0 | 2.0 - 8.0 |
| unsharp sigma | 1.2 px @ 2.4 um = 2.88 um / 2.5 px @ 1.1 um = 2.75 um | 1.375 - 5.76 um, log-uniform |

The unsharp sigma is very nearly CONSTANT IN MICRONS across the fleet (2.75-2.88 um) while it is 2x apart in
pixels. That is the whole argument for 27.2.

### 27.2 Sigmas in microns, not voxels

The PSF-type ops take a Gaussian sigma whose ranges were calibrated on 2.4 um data, i.e. their numbers are
`sigma_um / 2.4`. `aug.for_rung(cfg, k)` rewrites them for another rung as

```
sigma_vox(k) = sigma_um / rung_um(k) = sigma_vox(rung 2) * 2.4 / rung_um(k) = sigma_vox(2) * 2^(2-k)
```

so one rung coarser is exactly half the voxel sigma, and **rung 2 is bit-identical to today** (tested).
Converted (`aug.SIGMA_KEYS`): `blur.lo/hi`, `sharpen.sigma`, `unsharp.s_lo/s_hi`,
`aniso_blur.z_lo/z_hi/yx_lo/yx_hi`, `haze.s_lo/s_hi`. NOT converted, deliberately: `ring`/`stripe` widths (a
detector line is a fixed number of DETECTOR pixels, not a fixed length in the sample), `elastic.sigma` and
`sheetcomp.smooth` (geometry, not a PSF), `haze.r_*` (a blob-count divisor), and `paganin.s_lo/s_hi`
(already microns).

`aug.apply(x, tg, cfg, rung=k)` is the hook: `rung` is an int, or one per sample (then the scalar-sigma ops
use the batch's median rung -- they draw one sigma for the whole batch anyway -- and `paganin` gets the true
per-sample pitch). `rung=None` keeps rung-2 behaviour, so a caller that does not pass it loses nothing.
**Left to the train.py owner**: passing the sampled rung of the batch through to `apply`.

### 27.3 Shuffled artefact order

SinoSynth (arXiv:2409.18355) randomises the COMPOSITION ORDER of its degradation chain per sample, not just
each step's occurrence: a fixed order lets the network learn order-specific correlations that no real
acquisition chain guarantees. `cfg["shuffle"]` replaces the fixed `INTENS` order with a per-sample
permutation; the geometric ops (`spatial`) and `_cor` (which needs the radial channels) keep their place.

Batched implementation: draw `slot[b, t]` = the op sample `b` applies `t`-th, then walk the slots and run op
`i` only on the samples that both drew it and put it in this slot. An op that no selected sample placed in
slot `t` costs nothing, so the extra work is bounded by the number of distinct (op, slot) pairs actually
used -- about `B x (active ops per sample)` instead of `(active ops per batch)`.

### 27.4 `usrm2/scanmeta.py` and the `--scan-meta` hook

`scanmeta.load(path_or_url)` reads the upstream `metadata.json` next to a volume
(`<bucket>/<scroll>/volumes/<vol>.zarr/metadata.json`, local path or https URL; a directory or a `.zarr`
path gets `/metadata.json` appended) and returns a FLAT dict. It never raises: a missing, unreadable or
partial file falls back to the documented `scanmeta.DEFAULTS` (the 2.4 um 78 keV PHerc-Paris4 B_HA scan, the
one the pipeline is calibrated on) and reports `missing` / `defaulted`.

| key | source in metadata.json | note |
|---|---|---|
| `energy_kev` | `scan.tomo.acquisition.energy` | 74 / 78 / 137 across Paris 4 |
| `pixel_um`, `detector_pixel_um` | `detector.samplePixelSize`, `sensorPixelSize` | upstream is mm; x1000 |
| `distance_mm`, `source_distance_mm` | `sampleDetectorDistance`, `sourceSampleDistance` | propagation distance |
| `delta_beta`, `unsharp_coeff`, `unsharp_sigma_px` | `processing.preprocessing.phase.*` | nabu's phase step |
| `unsharp_sigma_um` | derived: `unsharp_sigma_px * pixel_um` | the pitch-free PSF |
| `hist_min/max`, `hist_p002`, `hist_p998` | `processing.32bitsData.histogram.*` | percentiles |
| `used_min`, `used_max` | `postprocessing.32BitsConversion.dataset_used_*` | |
| `win_f32_lo/hi`, `win_u16_lo/hi` | `zarr_export.target_window_f32_*`, `window_u16_*` | the f32 -> uint8 window |
| `mosaic`, `mosaic_tiles` | `mosaic.*` | the 1.1 um Paris 4 volume is a 19-tile fusion |
| `helical`, `half_acquisition`, `expo_time`, `tomo_n`, `scintillator`, `phase_method` | `acquisition.*` | |
| `rung` | derived: `round(log2(pixel_um / 0.6))` | 2.4 um -> 2 |

`scanmeta.ranges_for(meta)` turns that into augmentation overrides centred on the scan: the `paganin` block
(the scan's own `db/a/s_um` as the filter reference, plus sampled ranges that are the corpus span union the
scan's own value, so the identity is always reachable) and `bias.max` scaled by `78 / energy_kev` clamped to
0.5-2x (cupping is a low-energy effect; lit section 2). `aug.get(name, meta=..., rung=...)` merges it PER
OP, so a preset that does not configure an op does not grow one -- `get("full", meta=...)` still has no
`paganin`.

Scan-metadata planes as INPUT channels (section 21.3) are a separate change; this section only exposes the
dict and the ranges. A `--scan-meta PATH` CLI flag threading `scanmeta.load(PATH)` into `A.get` is the
remaining wiring, and belongs to the `cli.py`/`train.py` owner.

### 27.5 Presets and cost

`full2` = `full` + `paganin` + `shuffle`; ablations: `geo+paganin`, `geo+shuffle`. Measured on CPU (8
threads, B=2, 14 channels, 128^3, median of 10): see the commit message / README -- the jitter is one FFT
pair and the shuffle adds only the (op, slot) pairs actually used, so `full2` is a small constant factor
over `full` and negligible next to the GPU step it overlaps with.

## 28. Masked-cube pretraining (`usrm2 pretrain`, implemented 2026-09-21)

`usrm2/pretrain.py`, `tests/test_pretrain.py`, `cloud/pretrain_a6000.sh`. The R10 item of
`docs/research/synthesis_v2_with_literature.md` and its experiment 11; the evidence is in
`docs/research/lit_pretraining_foundation.md` (Wald/Isensee CVPR 2025: a CNN-native MAE stage on a
ResEnc U-Net, +3 Dice average over 8 downstream sets, the gain concentrated at low label counts; VAMAE:
structure-aware masking is what moves the topology metrics specifically). This is the replacement for
tsm's dead DINO/feature-distillation line (L11): the same instinct, done the way the literature says works
-- pretrain OUR OWN CNN on OUR OWN CT, do not import a clinical-CT foundation checkpoint and do not swap to
a ViT to make somebody's recipe drop in.

### The objective

A pretraining sample is the ORDINARY rung sample of section 2, built by the ordinary loader and
`prep.prepare`: CT cube at rung k, 9 context cubes at rungs k+1..k+9, the cascade slot, the scale plane,
the radial vector. Then

- **mask** the CT channel: `mask_block`^3 blocks (32 by default), a ratio drawn per sample from
  [`--mask-lo`, `--mask-hi`] = [0.5, 0.75], masked voxels set to 0 (the mean of the z-scored cube, the same
  "no information" value a dropped cascade channel carries);
- with probability `--sheet-p` (0.5) the blocks are drawn **structure-aware** instead of uniformly: a block
  is sampled with probability proportional to its foreground fraction, foreground being CT above the
  `--sheet-pct` (0.7) quantile of that cube. That is the cheap sheet proxy. Sheet-heavy blocks are then what
  vanishes, so the model has to reconstruct sheet TEXTURE and cannot score well by interpolating air;
- **target** = the z-scored CT cube before masking (after the augmentations, so the target is what the model
  would have seen);
- **loss** = L1 (`--loss l2` for squared) over the MASKED voxels only. An unmasked voxel is a copy, not a
  prediction, and scoring it would let the model win by learning the identity.

No target pyramid is read. `Patches` runs with `require_targets=False` and `--fg-min 0` (take any non-air
window: pretraining wants texture, not labels).

**The context cubes are masked too, and this matters.** Context cube j sits at rung k+j over the same
centre with the same voxel count, so its central 2^-j box is a 2^j-times coarser copy of the CT cube --
left alone it is a free low-frequency answer key and the "reconstruction" is an upsample. The voxel mask is
therefore max-pooled by 2^j and pasted into channel j's central footprint (`mask_ctx_`), for every j whose
footprint is still at least one voxel. `--no-mask-ctx` turns it off for an ablation.

### Which rungs

`--rungs 0-4` by default: the fine half of the ladder, where the texture the fine-tuning run has to model
lives and where R10 predicts the gain. **Rungs 0 and 1 (0.6 and 1.2 um) exist only where a fine scan does**:
a source's usable rungs start at its NATIVE rung (`data.usable_rungs`), and every 2.4 um mirror is native at
rung 2. `pretrain.available_rungs` intersects the request with what is actually on disk, drops a source that
has none of the requested rungs (instead of letting `data.rung_probs` assert), and prints what it dropped --
so `--rungs 0-4` on the Paris 4 corpus trains at 2-4 and says so.

### The warm start: same trunk, renamed head

The point of the stage is that `usrm2 train --init <run>/ckpt.pt` is EXACT. `pretrain` calls
`model.build(size, cin=..., ckpt_act=..., add_skip=..., deep=...)` -- the very same builder, so `enc.*`,
`down.*`, `dec.*` and `proj.*` have the state-dict keys `train` expects, byte for byte. Only the 1x1 output
head is repurposed as the reconstruction head, and in the checkpoint it is stored as `recon_head.*`
(`recon_deep_heads.*` under `--deep`). `train`'s `load_state_dict(..., strict=False)` then reports it as an
UNEXPECTED key and drops it: the trunk is pretrained, the segmentation head starts random.
`tests/test_pretrain.py` asserts exactly this -- missing keys are `{head.weight, head.bias}` and nothing
else, unexpected keys are the `recon_head` pair, every trunk tensor equals the checkpoint's, and the warm
started net's output is finite.

`cin` must line up. By default the stem is the 15-channel one of a `--cascade` run, with the cascade slot
held at ZERO -- which is exactly the in-distribution "no coarse prediction" value that `--cascade-drop`
teaches. For a 14-channel fine-tuning run pretrain with `--no-cascade-slot`: `train.warm_start` can widen a
stem, never narrow one. The checkpoint sets `args["scale_plane"] = True`, which is what makes `warm_start`'s
`src_scale` path line the scale plane up rather than sliding it into the cascade slot.

The checkpoint is train's shape -- `{"model", "ema", "opt", "step", "args"}` -- with the same EMA, bf16
autocast, `--compile`, activation checkpointing, atomic save and `--resume` argument check. The few small
pieces (`ema_update`, `autocast`) are COPIED from train.py rather than imported, so this stage does not
break when train's internals move.

### The optional rung head (VoCo flavour)

`--rung-aux W` (off by default) adds a linear head on the mean-pooled bottleneck predicting the rung index,
cross entropy, weight W. It is the one part of VoCo (CVPR 2024) that transfers: VoCo's own pretext task is
"where is this crop in the body", which needs a fixed macro-anatomy a scroll does not have, but "how coarse
is this cube" is the same idea on the axis we actually have. The catch, and the reason it is off: **the
scale plane hands the model the answer**. So when it is on, a fraction `--rung-aux-p` (0.5) of the steps
zero the scale plane and score the aux loss; the rest are plain reconstruction with the plane intact. The
head lives outside the model state dict (`ckpt["rung_head"]`), so it can never reach a warm start.

### The ablation protocol (experiment 11)

Budget ~16 GPU-h: pretrain ~20k steps (~7 GPU-h) + two fine-tuning arms x 10k steps (~9 GPU-h).

1. **Audit the corpus first.** Count DISTINCT SCANS in the stores file, not voxels. The R10 evidence comes
   from ~39k volumes; two scrolls is a different regime and the number belongs in the write-up.
2. `bash cloud/pretrain_a6000.sh pre1` -- planner-free, it samples the local CT mirror directly (there is no
   target to wait for, so there is nothing for `stream-plan` to do).
3. Two fine-tuning arms, IDENTICAL but for one flag and at MATCHED steps:
   - from scratch: `usrm2 train ~/runs/ft_scratch ... --steps 10000`
   - pretrained: the same command `+ --init-from ~/runs/pre1/ckpt.pt`
   Same `--size`, `--patch`, `--ctx`, `--aug`, `--deep`, `--add-skip`, `--ckpt-act`, seed and val boxes.
4. **Metric: `usrm2 evalsurf` continuity / ERL on the held-out box**, plus val dice per rung. Dice alone is
   the wrong headline here -- the claimed mechanism (structure-aware masking, thin structures) is topological,
   so ERL is what has to move.
5. **Decision rule** (from synthesis_v2): adopt only if the gain survives at the label counts we actually
   have. A gain that only appears at an artificially reduced label count is a note, not an adoption.
6. Worth a third arm if the first two are close: `--no-mask-ctx`, which tells you how much of any gain was
   real reconstruction and how much was the context channels leaking a coarse answer.
