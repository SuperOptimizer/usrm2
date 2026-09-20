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
