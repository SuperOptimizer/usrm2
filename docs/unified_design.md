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
