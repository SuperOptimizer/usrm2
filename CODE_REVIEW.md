# usrm2 code review

Review date: 2026-09-17  
Revision: `795370c9e4fcc78162eb236876b66f4722794c1a`

## Scope and method

This review covers all 11 Python modules in `usrm2`, all three test modules, the README, package configuration, and launcher script. Three independent review agents examined training/data/teacher generation, inference/surface evaluation, and augmentation. The primary reviewer examined CLI/ablation integration, ran the test suite, checked evidence, and consolidated findings.

No implementation, test, configuration, or dataset files were changed. This document is the sole repository addition. Reproductions used in-memory tensors, mocks, and temporary files; they did not run against production volumes or checkpoints.

Severity: **P1** means a high-priority risk of silently incorrect data/results; **P2** means a correctness or reliability defect in a supported path; **P3** means a narrower usability issue. Findings distinguish runtime evidence from code inspection. Suggested fixes are recommendations, not implemented changes.

## Assessment

The largest risks are at integration boundaries: selecting the correct scroll axis, preserving volume identity, recognizing completed teacher outputs, and reusing evaluation state. Augmentation has two confirmed semantic errors that shape/finite-value checks miss. Multi-head support is implemented in training and prediction but incomplete in evaluation and previews.

The available test suite does not currently pass in the review environment. Several confirmed bugs affect workflows absent from the tests.

## Findings

### 1. P1 — Custom scroll axes are ignored during inference

**Locations:** `usrm2/data.py:40`, `usrm2/cli.py:87`, `usrm2/predict.py:131`, `usrm2/evalsurf.py:39`.

`axis(path=UMBILICUS)` captures the Paris4 path when the module is imported. The CLI changes `data.UMBILICUS` for `--umbilicus`, but prediction and surface evaluation call `data.axis()` without an argument. They consequently read the original axis. Training explicitly supplies the current global or store metadata, so the same custom-scroll workflow can train and infer with different radial fields. Surface-normal orientation can also use the wrong axis.

**Evidence:** After changing the global, `axis.__defaults__` still contains the Paris4 path. The existing smoke test, with CPU autocast bypassed in memory, finishes training and then fails in prediction with `FileNotFoundError` for the Paris4 axis despite its synthetic axis override. On a machine where that path exists, this can silently use incorrect geometry.

**Recommendation:** Resolve the current default at call time and pass the selected axis explicitly through inference/evaluation. Test a custom axis whose vectors differ from the default.

### 2. P1 — Prediction output loses its source-volume identity

**Locations:** `usrm2/predict.py:66-69`, `usrm2/predict.py:84`, `usrm2/predict.py:142-148`.

`predict()` receives `volume`, but does not forward it through `write()` to `out_array()`. Ordinary outputs therefore record `data.CT` as their source volume even when prediction used another scroll. The data loader trusts this attribute when pairing probabilities with CT, so reusing these outputs can pair labels with the wrong scan.

The OME branch similarly omits `full_shape`, causing `write_ome()` to open the default Paris4 CT for its full-volume shape. A custom-volume prediction may fail when Paris4 is unavailable or produce the wrong spatial extent when it is available.

**Evidence:** With inference mocked and a real temporary plain-Zarr output, a custom-volume prediction recorded the Paris4 volume attribute. A mocked OME call attempted to open Paris4 for shape discovery.

**Recommendation:** Carry the actual source volume and axis through output creation; derive OME shape from that volume. Test both output modes against a custom volume with a distinct shape.

### 3. P1 — Incomplete teacher stores are accepted as completed work

**Locations:** `usrm2/teacher.py:56-63`, `usrm2/teacher.py:84-88`.

Teacher generation creates the output store before processing tiles. A crash or interruption leaves a directory with unwritten, fill-zero chunks. On rerun, `boxes()` treats path existence as success, skips generation, and increments its completion count. There is no completion marker or completeness validation. Partial stores can then supply false negative labels for distillation.

**Evidence:** A controlled reproduction with an existing output path reported one completed box and made zero runner calls. The progressive creation/write order is directly visible in `run()`.

**Recommendation:** Publish an output only after successful completion, or require a final completion marker. Reject or regenerate partial outputs, and test interruption followed by retry.

### 4. P2 — Both teacher writers fail with plain-Zarr fallback

**Locations:** `usrm2/predict.py:54-57`, `usrm2/teacher.py:63`, `usrm2/m7.py:53`.

When the volcomp codec cannot import, `out_array()` returns a `(1,Z,Y,X)` array. The recto teacher still indexes it as a three-dimensional array, putting tile offsets on the wrong axes. The M7 teacher assigns a three-dimensional value to the whole four-dimensional Zarr array. Neither handles the channel dimension as `predict.write()` does.

**Evidence:** Recto reproduction with output `(1,4,8,8)` and tile size 4 failed with `ValueError: could not broadcast input array from shape (4,4,4) into shape (1,4,4,8)`. M7 reproduction with mocked inference and a real temporary fallback Zarr failed with an indexing error for three-dimensional input indexed in four dimensions.

**Recommendation:** Centralize writes for both supported layouts, including tiled slices. Cover codec-unavailable recto and M7 output paths.

### 5. P2 — Surface-site caching can silently change evaluation results

**Locations:** `usrm2/evalsurf.py:36-38`, `usrm2/evalsurf.py:130`.

The automatic cache key includes the store and origin, but omits box size, surface source, and scroll axis. `sites()` returns cached points and normals without checking any request parameters. Repeating an evaluation with a different size or `--tifxyz` can score stale points, including points outside the requested box; changing axes can retain stale normal orientations.

**Evidence:** A cached 128-cubed synthetic plane returned 1,681 points after the z extent was reduced to 64. A fresh computation returned 902; 779 cached points were outside the new box.

**Recommendation:** Store and validate cache inputs, including geometry and source identity. Include invalidation tests for size, surface source, and axis changes.

### 6. P2 — CLI evaluation and ablation previews cannot load multi-head checkpoints

**Locations:** `usrm2/cli.py:104-105`, `usrm2/ablate.py:18-19`.

Training records `args.cout` and prediction uses it, but these two consumers build the default one-head model. Checkpoints from the documented comma-separated teacher-store workflow fail to load. In ablation, preview failure is caught and recorded, so a sweep can finish without its promised preview.

**Evidence:** A generated two-head checkpoint passed to `cli.main(['eval', ...])` failed with a head-weight/bias size mismatch, including checkpoint bias shape `[2]` versus model shape `[1]`. The preview uses the same construction pattern.

**Recommendation:** Restore `cout` consistently in every checkpoint consumer and define which heads a preview displays.

### 7. P2 — CLI evaluation ignores the checkpoint's validation dataset

**Location:** `usrm2/cli.py:106`.

Training accepts and records custom `val`/`ct` settings, but `eval` always calls `val_grid()` with the default stores. There is no evaluation CLI override for those paths. A custom-store run can thus be evaluated on a different dataset, or fail on a machine without the default paths. Restoring multi-head model construction alone would still leave the wrong target-head configuration.

**Evidence:** A checkpoint with custom `args.val` and `args.ct` produced the observed call `val_grid(patch=128, limit=32)`, without either saved setting.

**Recommendation:** Default to the saved validation configuration, permit explicit overrides, and print the dataset identity with metrics.

### 8. P2 — Sliding-window inference breaks on dimensions smaller than the window

**Locations:** `usrm2/predict.py:15-19`, `usrm2/predict.py:27-40`.

`starts()` appends `n-window` when `n < window`, producing a negative start. Even the initial crop is smaller than the fixed Gaussian window. Multiplication/blending fails for an ordinary non-air ROI. The CLI accepts such boxes without validation; this also affects teacher ROIs near small dimensions. M7 explicitly pads its ROI, but the shared student/recto path does not.

**Evidence:** `starts(8,16,8)` returned `[0,-8]`. Sliding over a nonzero `(8,16,16)` array with window 16 and halo 4 failed with a broadcast error between `(8,16,16)` and `(16,16,16)`.

**Recommendation:** Pad undersized ROIs and crop output back, or reject them clearly before inference. Validate a positive stride and supported window size as well.

### 9. P2 — Sheet compression expands sheet separation

**Location:** `usrm2/aug.py:319-325`.

`grid_sample` consumes output-to-input coordinates. Subtracting a positive cumulative gap displacement gives a sampling derivative of approximately `1-a` in air, expanding output gaps by approximately `1/(1-a)`. This reverses the documented purpose of pushing sheets together and can push later content beyond the crop.

**Evidence:** In a 64-cubed synthetic volume, sheets and matching targets at x=10 and x=30 moved to x=13 and x=40 with strength 0.25, smoothing 0, and seed 0. Separation increased from 20 to 27.

**Recommendation:** Construct the correct inverse sampling map for the intended forward compression. Test decreased separation, preserved ordering, and CT/target alignment.

### 10. P2 — Pool augmentation crops and stretches CT without transforming targets

**Location:** `usrm2/aug.py:289-303`.

Pooling truncates every dimension to a multiple of the kernel, then resizes the cropped result to the original shape. With non-divisible dimensions, it drops border content and changes spatial scale only in CT; targets and radial channels remain unchanged. Kernel 3 makes this relevant to ordinary power-of-two patches.

**Evidence:** A 32-cubed CT containing ones only in the last two x planes had input sum 2,048 and output sum zero with average pooling, kernel 3, and nearest upsampling. Those planes are cut off before pooling. The existing pooling test uses size 24 with kernel 3, which avoids this case.

**Recommendation:** Preserve the original spatial extent using padding or partial edge windows, then restore and crop in the same coordinates. Test non-divisible dimensions and boundary features.

### 11. P2 — CPU fallback unconditionally selects unsupported bfloat16 training

**Locations:** `usrm2/train.py:28-29`, `usrm2/train.py:96-99`.

The program automatically falls back to CPU, but `autocast()` always chooses CPU bfloat16. The installed Torch/oneDNN combination on this review machine cannot execute the resulting backward operation. Training therefore fails on its first step rather than using working float32 execution.

**Evidence:** The unmodified smoke test failed at `loss.backward()` with `DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2`. Replacing autocast with `nullcontext()` only in the review process allowed all three training steps and validation to complete, exposing finding 1 afterward.

**Recommendation:** Provide a supported float32 CPU path and make reduced precision optional or capability-aware. This failure is environment-specific; it is not evidence of failure on every CPU or on CUDA.

### 12. P2 — Teacher-box completion counts duplicate origins

**Location:** `usrm2/teacher.py:78-88`.

Random origins are rounded down to a 128-voxel grid. Duplicate accepted coordinates increment `done` even though generation is skipped for the existing path. The command can report N completed boxes while yielding fewer than N unique training regions.

**Evidence:** With CT shape `(257,257,257)`, box size `(256,256,256)`, and `n=3`, the same origin was logged three times and the runner executed once.

**Recommendation:** Count unique completed boxes and report when the requested number of distinct acceptable placements cannot be satisfied.

### 13. P2 — Teacher rejection sampling has no termination bound

**Locations:** `usrm2/teacher.py:35-38`, `usrm2/teacher.py:77-83`.

Histogram matching loops until it finds enough bright, mostly non-air patches. Teacher-box sampling loops until enough boxes survive masking and exclusion. Neither bounds rejected attempts. All-air or sufficiently dark volumes, or exclusion of every possible placement, cause an indefinite loop without an actionable error.

**Evidence:** Control-flow inspection; no infinite loop was deliberately run. The patch-training sampler already uses a rejection limit, but these loops do not.

**Recommendation:** Bound sampling attempts and report why candidates were rejected, including when too few distinct placements exist.

### 14. P3 — Short ablation runs crash after successful training

**Locations:** `usrm2/train.py:104-109`, `usrm2/ablate.py:39-43`.

Throughput records are emitted only every 20 steps. A positive `--steps` value below 20 can finish training and evaluation, but `sweep()` computes the median of an empty throughput list before writing its summary or continuing to the next preset.

**Evidence:** A controlled completed-run fixture containing evaluation metrics and only the initial training record raised `StatisticsError: no median for empty data` for a three-step sweep.

**Recommendation:** Record final throughput for short runs or permit a missing throughput statistic.

### 15. P3 — Surface preview can select a slice outside the volume

**Location:** `usrm2/evalsurf.py:116-117`.

Site selection accepts continuous z coordinates strictly below the ROI upper boundary, but preview rounds them before choosing the slice. A valid point in the last half voxel can round to `ct.shape[0]`, making `ct[zi]` invalid. This can abort PNG creation after scoring has completed.

**Evidence:** A valid local point at z=7.9 in an eight-slice ROI reproduced an out-of-bounds access at index 8.

**Recommendation:** Clamp rounded slice coordinates before histogramming and index selection. Test fractional points near both boundaries.

## Additional risks and limitations

These deserve follow-up, but are separated from the demonstrated defects above because they involve implicit contracts, interruption behavior, or metric interpretation.

- **Resume configuration:** `train.py:53-78` reloads model/optimizer/EMA/step but does not restore or compare saved arguments. Resuming a custom augmentation, no-radial, or custom-data run with invocation defaults can silently change the experiment and overwrite its recorded configuration. Explicit overrides may be intentional; the interface should distinguish them from omitted options.
- **Checkpoint durability:** `train.py:80-82` writes directly over the sole checkpoint. An interrupted write can lose the last resumable state. Atomic replacement and a retained prior checkpoint would reduce this risk; interruption was not injected.
- **Multi-teacher integrity:** `data.py:126-132,161,173-186` assumes every teacher in a comma-separated group shares the first teacher's origin, dimensions, and volume. The documented contract requires alignment, but there is no validation: same-shaped misregistered stores silently supply different locations as corresponding labels.
- **Empty validation:** A validation box smaller than the requested patch yields no grid entries. Training indexes `grid[0]`; standalone evaluation instead returns all-zero metrics for an empty grid. Validate nonempty sampling and report the box/patch mismatch.
- **Surface precision proxy:** `evalsurf.py:93-99` measures proximity to sampled points, not continuous surfaces. A reproduced perfect plane sampled every 20 voxels scored about 0.279 for `precision6`, whereas the dense two-voxel test scores 1.0. The metric is already named a proxy; interpret it with sampling density and annotation coverage, and avoid treating it as conventional precision across differently sampled surfaces.
- **OME spatial units:** `predict.py:91-96` declares micrometer axes with scale `[1,1,1]` at level zero while separately recording `voxel_um=2.4`. These fields express inconsistent physical spacing to consumers that use the transform. Verify intended tracer behavior before changing metadata; external-consumer compatibility was not exercised.
- **Augmentation runtime cost:** Intensity operations are computed before the per-sample activation mask chooses their results. FFTs, sorts, and convolutions may run even when no sample activates the operation. This is a code-level performance observation, not a measured throughput regression.
- **Input and environment validation:** Window/halo relationships, model-compatible patch dimensions, head indices, teacher-store size margins, and required external checkpoint paths generally fail deep in execution. The M7 path imports `dynamic_network_architectures`, which is not explicitly listed in the package dependencies; availability through optional/transitive environments was not established.

## Validation performed

The repository had a clean working tree before review. The system Python lacked dependencies, so tests used the existing `/home/forrest/usrm/.venv/bin/python` environment: Python 3.12, Torch `2.14.0+cu130`, NumPy `2.5.3`, Zarr `3.3.0`, and pytest `9.1.1`. No dependencies were installed or updated.

Unmodified suite command:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/forrest/usrm/.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Result: **49 passed, 1 skipped, 1 failed in 8.43 seconds**. The skipped case was the native volcomp roundtrip. The failure was CPU bfloat16 backward in `test_train_and_predict` (finding 11).

A second diagnostic run invoked the smoke test with `train.autocast` and `predict.autocast` temporarily replaced by `nullcontext()` in memory. It completed three training steps and validation, then failed on the ignored custom axis (finding 1). This was diagnostic isolation, not a passing test or a repository modification.

Focused reproductions covered custom-axis defaults; plain/OME volume propagation; both teacher fallback writers; partial-output skipping; duplicate box counts; multi-head CLI loading; saved validation arguments; short ablation summaries; undersized sliding windows; stale surface caches; PNG boundaries; sparse-surface precision; sheet compression; and non-divisible pooling. Mocks isolated external model loading and production storage where needed, so those cases establish control-flow/array behavior rather than full upstream-model integration.

## Coverage gaps and suggested regression order

1. Custom-scroll training → prediction → output readback, asserting axis, volume identity, and OME shape.
2. Teacher generation with unavailable codec, multiple tiles, interruption, and retry; both teacher models.
3. Multi-head train/eval/predict/preview with saved custom validation settings.
4. Evaluation cache invalidation and fractional boundary points.
5. Semantic augmentation tests: compression must reduce sheet separation; pooling must preserve coordinate extent for non-divisible sizes.
6. Float32 CPU smoke execution, supported CUDA execution, and resume/checkpoint recovery.
7. Sampling exhaustion, unique teacher-box counts, short ablations, empty validation, and invalid window/halo configurations.

Existing augmentation tests mostly check shape, finiteness, range, and vector norm. The sheet-compression preset's seeded activation can leave both tested samples unaugmented, and there is no compression-separation assertion. Pool tests use a divisible shape. No teacher-generation, retry/completion, LUT-exhaustion, multi-head lifecycle, resume, or OME integration tests are present.

The review did not execute production-scale GPU training, real upstream teacher checkpoints, native volcomp roundtrips, production tifxyz/CT datasets, or the external vc3d tracer. Model architecture and BCE/Dice implementation were inspected without identifying an additional concrete defect; this is not a claim of numerical or scientific validation. No fixes were applied.
