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
    t.add_argument("--ema", default="0.999", help="EMA decay of the evaluated weights (0.9995 for 100k+ "
                   "steps), or 'auto' = 1 - K/--steps with K from --ema-k (a window that is a fixed "
                   "fraction of the run instead of a fixed 1000 steps)")
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
    t.add_argument("--scan-meta", default=None, metavar="PATH", help="recentre the scan-domain "
                   "augmentation ranges on this scan's recorded metadata (usrm2/scanmeta.py); a path or "
                   "URL to the metadata json. Without it the preset's own ranges are used, as before")
    t.add_argument("--no-radial", action="store_true", help="zero the radial channels 1..3")
    t.add_argument("--region", type=int, default=0, help="region mode: visit one REGION^3 region of one "
                   "source at one rung, take --windows-per-region windows inside it, then move on")
    t.add_argument("--windows-per-region", type=int, default=64)
    t.add_argument("--teacher-regions", default=None, help="prefer a region's teacher probability store "
                   "(usrm2.data.teacher_region_path under this root) over the exported mask as the rung-2/3 "
                   "target; rung 3 is its 2x mean pool")
    t.add_argument("--verso", action="store_true", help="add the VERSO OUTPUT CHANNEL (docs/unified_design.md "
                   "section 23): ONE head with cout 2, channel 0 recto and channel 1 verso (the deep heads "
                   "follow). Its target is the verso region stores under --verso-regions (default: "
                   "--teacher-regions); every voxel no finished verso store covers has weight 0 in that "
                   "channel, so a sample with no verso is still a valid recto sample")
    t.add_argument("--verso-regions", default=None, help="root of the verso region stores "
                   "(<root>/verso/region_<z>_<y>_<x>.zarr); default: --teacher-regions")
    t.add_argument("--verso-regions-url", default=None, help="where `stream-plan` fetches verso region "
                   "stores from (recorded in the args; the planner is what downloads them)")
    t.add_argument("--cout", type=int, default=None, help="assert the number of output channels (1 recto, "
                   "2 recto+verso): the head width is derived from the target channels, this just checks it")
    t.add_argument("--cascade", default="off", choices=["off", "mask", "self", "mix"], help="cascade input channel (docs/unified_design.md section 22): the rung-(k+1) prediction over the same field of view, upsampled 2x. off = 14 channels as before; mask = the rung-(k+1) target block (+ noise); self = the model's own coarse prediction (one extra forward per sample); mix = self with probability --cascade-self-p, else mask")
    t.add_argument("--cascade-self-p", type=float, default=0.5, help="--cascade mix: probability a sample uses the self source")
    t.add_argument("--cascade-drop", type=float, default=0.1, help="probability a sample's cascade channel is zeroed (a missing coarse prediction stays in distribution)")
    t.add_argument("--no-cascade-noise", action="store_true", help="do NOT roughen the mask-derived cascade channel (it is then a blurred copy of the target: a leak)")
    t.add_argument("--stream", default=None, help="replay a `usrm2 stream-plan` queue directory instead of "
                   "sampling: the windows come from the rolling local buffer the planner fills")
    # ---- Phase A losses and training recipe (docs/unified_design.md section 26). Every one is OFF by
    # default and is recorded in the checkpoint args only when it is on, so a run without them is
    # byte-identical to one started before they existed.
    t.add_argument("--loss-excl", type=float, default=0.0, metavar="W", help="L3 soft exclusivity: "
                   "relu(p_recto + p_verso - 1) averaged where BOTH channels carry weight (needs --verso)")
    t.add_argument("--loss-selfcons", type=float, default=0.0, metavar="W", help="L4 cascade "
                   "self-consistency: |pool2(p) - pool2(CASCADE)| on the samples whose cascade channel came "
                   "from the model's own coarse forward (--cascade self|mix). No extra forward")
    t.add_argument("--loss-skel", type=float, default=0.0, metavar="W", help="L8 skeleton recall: "
                   "1 - mean predicted probability along the TARGET's medial surface (a GAPS term; watch "
                   "merge_frac while it is on, a bridge scores well under it)")
    t.add_argument("--skel-iters", type=int, default=4, help="--loss-skel: erosions used to build the "
                   "medial surface (the cap of the distance transform, in voxels)")
    t.add_argument("--affinity", default=None, metavar="OFFSETS", help="O12 long-range affinity: EVEN voxel "
                   "offsets, e.g. '16,32'. The head grows by 3 channels per offset (one per axis) which "
                   "predict whether the voxels d/2 back and d/2 forward along that axis are the SAME sheet. "
                   "Inference and evaluation never read them; a warm start zero-inits their rows")
    t.add_argument("--loss-affinity", type=float, default=0.0, metavar="W", help="weight of the affinity BCE")
    t.add_argument("--affinity-all", action="store_true", help="score the affinity channels everywhere "
                   "instead of only where BOTH voxels of the pair are foreground (the default)")
    t.add_argument("--cascade-self-p-anneal", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="scheduled sampling: anneal --cascade-self-p linearly from START to END over the run")
    t.add_argument("--sched", default="cosine", choices=["cosine", "wsd"], help="LR schedule: cosine over "
                   "--steps (unchanged), or warmup-stable-decay (flat after warmup until --stable-until, "
                   "then cosine over --cooldown). WSD's plateau and cooldown MAY be changed on a resume")
    t.add_argument("--stable-until", type=int, default=None, help="--sched wsd: step the plateau ends "
                   "(default: --steps minus --cooldown)")
    t.add_argument("--cooldown", type=int, default=0, help="--sched wsd: cooldown length (default 10%% of --steps)")
    t.add_argument("--ema-k", type=float, default=None, metavar="K", help="ema_decay = 1 - K / --steps "
                   "(an averaging window of steps/K; K=50 is 2%%, the middle of the literature's 1-3%%). "
                   "Overrides --ema")
    t.add_argument("--rewarm", type=int, default=0, metavar="N", help="on a warm start (--init-from) warm "
                   "the LR up over N steps instead of --warmup: resuming on a decayed tail generalises worse")
    t.add_argument("--new-param-lr-mult", type=float, default=1.0, metavar="M", help="a second AdamW param "
                   "group, at M x LR through the stable phase, for the tensors a warm start GREW (the stem "
                   "convolution and the heads): new rows carry no memory to protect")
    t.add_argument("--fuse", default="off", choices=["off", "agreement"], help="how a region teacher store "
                   "and the exported mask are combined where BOTH cover a voxel: 'off' = the store replaces "
                   "the mask (what every run so far did), 'agreement' = a confidence-weighted mean whose "
                   "loss weight is the sources' agreement (usrm2/data.py fuse_agreement)")
    t.add_argument("--source-w", nargs="*", default=(), metavar="SRC=W", help="per-source loss weights, e.g. "
                   "'mask=1 store=1'; `usrm2 glc-weights` suggests them from the published meshes")
    # ---- Phase B / C: distance, normals, pairing, planes (docs/unified_design.md section 29).
    # Every one is OFF by default and is recorded in the checkpoint args only when it is on.
    t.add_argument("--sdist", default=None, choices=["face", "midline"], help="add ONE regression "
                   "output channel holding the signed distance (voxels at the sample's rung, + on the "
                   "recto side) to the recto FACE or to the sheet MIDLINE. Its target is a distance "
                   "pyramid from `usrm2 dist-pyramid`, added to the source line like any other target "
                   "group (channel 'sdist' / 'midline'); it is never pooled, so a rung the store has no "
                   "level for has weight 0")
    t.add_argument("--thickness", action="store_true", help="--sdist: one more regression channel, the "
                   "recto-to-verso separation along the normal, read as TMIN + softplus(raw) so it can "
                   "never fall below the minimum physical sheet thickness")
    t.add_argument("--normals", default="off", choices=["off", "derive", "head"], help="--sdist: "
                   "'derive' = normals are the normalised gradient of the PREDICTED distance field (no "
                   "extra channels), 'head' = three explicit channels distilled from that gradient")
    t.add_argument("--sdist-hetero", action="store_true", help="--sdist: one more head channel holding "
                   "the LOG-VARIANCE of the distance regression (heteroscedastic Huber), which doubles "
                   "as the tracer contract's `conf` channel")
    t.add_argument("--loss-sdist", type=float, default=1.0, metavar="W", help="weight of the distance Huber")
    t.add_argument("--loss-eikonal", type=float, default=0.0, metavar="W", help="L5: weight of "
                   "(|grad d| - 1)^2 in the band -- the regulariser that makes a regressed field an "
                   "actual distance function between the voxels that pin it down")
    t.add_argument("--loss-normals", type=float, default=0.0, metavar="W", help="--normals head: weight "
                   "of the normal-head distillation")
    t.add_argument("--sdist-delta", type=float, default=2.0, help="Huber delta, in voxels")
    t.add_argument("--sdist-band", type=float, default=8.0, help="voxels around the surface the Eikonal "
                   "and normal terms are evaluated in")
    t.add_argument("--pair", default="off", choices=["off", "construct", "construct-only"],
                   help="PHASE C: with --sdist midline --thickness, derive p_recto / p_verso from the "
                   "midline distance m and the thickness t as soft bands at m = +- t/2, so they cannot "
                   "cross BY CONSTRUCTION, and score those. 'construct' keeps the learned recto/verso "
                   "channels as an auxiliary; 'construct-only' drops their loss term (the rows stay in "
                   "the head and keep their weights). --loss-excl remains as a backstop")
    t.add_argument("--pair-band", type=float, default=1.5, help="--pair: half-width of the soft band, in "
                   "voxels (the thickness floor is 2x this, which is what makes the two bands disjoint)")
    t.add_argument("--pair-tau", type=float, default=0.5, help="--pair: softness of the band edge, in voxels")
    t.add_argument("--loss-ect", type=float, default=0.0, metavar="W", help="L7: the fast "
                   "Euler-characteristic-transform topology loss on INTERIOR sub-blocks of the samples "
                   "at --ect-rung. May be changed on a resume (it touches no weight)")
    t.add_argument("--ect-dirs", type=int, default=8, help="--loss-ect: directions of the ECT sweep")
    t.add_argument("--ect-res", type=int, default=16, help="--loss-ect: filtration heights per direction")
    t.add_argument("--ect-margin", type=int, default=8, help="--loss-ect: voxels cropped off every patch "
                   "face before the sub-blocks are taken (a topology loss on a cropped patch sees every "
                   "sheet truncated at the face)")
    t.add_argument("--ect-block", type=int, default=32, help="--loss-ect: sub-block edge")
    t.add_argument("--ect-n", type=int, default=4, help="--loss-ect: sub-blocks per sample")
    t.add_argument("--ect-rung", type=int, default=2, help="--loss-ect: the ONE rung it is computed at")
    t.add_argument("--planes", default=None, metavar="LIST", help="extra constant input planes between "
                   "the cascade channel and the scale plane (section 21 items 2 and 3): 'radius' = "
                   "r / r_max from the umbilicus, 'meta' = five scan planes from the volume's own "
                   "metadata.json (energy, log delta/beta, unsharp sigma in um, sample-detector "
                   "distance, pixel pitch), each min-max normalised over a documented corpus range and "
                   "ZERO where the file does not say. e.g. --planes meta,radius. They grow `cin`, so "
                   "they must match on a resume; a warm start zero-fills them")
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
    sp.add_argument("--verso", action="store_true", help="plan for the verso output channel (must match the "
                    "training run's --verso: it changes the queue's channel list)")
    sp.add_argument("--verso-regions", default=None, help="see `train --verso-regions`")
    sp.add_argument("--verso-regions-url", default=None, metavar="URL",
                    help="fetch verso region stores from this published root as the pod writes them, e.g. "
                    f"{__import__('usrm2.data', fromlist=['data']).VERSO_REGIONS_URL} . The planner checks a "
                    "region once when it plans it (a GET of zarr.json; a 404 = not published yet, retried "
                    "after 30 min) and downloads zarr.json + c/0/0/0 into <--verso-regions>/verso/")
    sp.add_argument("--region-fails", type=int, default=0, help="consecutive rejected draws that abandon a "
                    "region (0 = 8 x --windows-per-region)")
    sp.add_argument("--cascade", default="off", choices=["off", "mask", "self", "mix"], help="must match the "
                    "training run's --cascade: the planner fetches the coarse target block and the tenth context cube")
    sp.add_argument("--planes", default=None, metavar="LIST", help="must match the training run's "
                    "--planes: the plane set is recorded in the queue's meta.json and a queue can only "
                    "be replayed by a run that builds the same stem")
    sp.add_argument("--scan-meta", default=None, help="see `train --scan-meta`; with --planes meta it "
                    "also fixes the five scan planes for every source")
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
    p.add_argument("--head", default="0", help="which output channel to write: an index, a channel NAME of "
                   "the checkpoint (recto / verso -- see `train --verso`), all, or mean / prod / max")
    p.add_argument("--radial-sign", type=float, default=1.0, help="-1 negates the radial vector (the student then predicts the verso face)")
    p.add_argument("--lut-to", nargs="*", default=(), metavar="REF", help="also average with the input histogram-matched to REF volumes")
    p.add_argument("--cascade", default="auto", choices=["auto", "on", "off"], help="top-down cascade inference: "
                   "'auto' follows the checkpoint's own --cascade, 'off' feeds a zero cascade channel")
    p.add_argument("--cascade-depth", type=int, default=3, help="rungs above k predicted top-down to fill the cascade channel")
    p.add_argument("--no-calib", action="store_true", help="do NOT apply the checkpoint's per-rung "
                   "temperature (`usrm2 calibrate`); a checkpoint with none is unaffected either way")
    dp = sub.add_parser("dist-pyramid", help="PHASE B (docs/unified_design.md section 29): turn an "
                        "exported MASK pyramid into a signed-distance / midline / thickness pyramid, one "
                        "level per rung, each computed from THAT rung's own mask (a distance is never "
                        "pooled). CPU only -- do not run it on a card a training job is using")
    dp.add_argument("mask", help="the mask pyramid group (levels named by voxel size in um)")
    dp.add_argument("--out", default=None, help="output group (default: <mask>_sdist.zarr / "
                    "_midline.zarr / _thick.zarr beside it); only honoured with a single --kind")
    dp.add_argument("--kind", nargs="+", default=["face"], choices=["face", "midline", "thickness"],
                    help="face = signed distance to the RECTO FACE; midline = to the sheet MIDLINE "
                         "(needs --verso); thickness = the recto-to-verso separation (needs --verso)")
    dp.add_argument("--verso", default=None, help="the paired verso source: a verso pyramid group, or the "
                    "root of the published verso REGION stores (<root>/verso/region_<z>_<y>_<x>.zarr, "
                    "rung 2 and its 2x pool). Without it the midline falls back to the recto band's own "
                    "medial surface and the thickness is not measurable (weight 0)")
    dp.add_argument("--rungs", default="0-4", help="rungs to write; a rung with no level in the mask "
                    "pyramid, above --max-rung, or above the mask's native rung + 1 is skipped")
    dp.add_argument("--max-rung", type=int, default=None, help="default 4: above it an exported 'mask' is "
                    "a pooled area FRACTION, and its 0.5 level set is not a surface")
    dp.add_argument("--axis-r-um", type=float, default=None, metavar="R", help="microns around the "
                    "umbilicus axis that are marked no-data (default 400 um: the core is crushed and the "
                    "published masks put recto_is_in near 0.5 there)")
    dp.add_argument("--tmin", type=float, default=None, help="floor on the stored thickness, in voxels "
                    "(default 3.0 = 2 x the default --pair-band)")
    dp.add_argument("--block", type=int, default=128, help="core block edge")
    dp.add_argument("--halo", type=int, default=48, help="context around each block, >= the +-32 clamp")
    dp.add_argument("--volume", default=None)
    dp.add_argument("--box", type=int, nargs=6, default=None, metavar=("Z0", "Y0", "X0", "Z", "Y", "X"),
                    help="only visit the blocks inside this box, given at RUNG 2 (a whole Paris 4 level "
                         "is 10^12 rung-2 voxels: the full pass is a region-by-region job)")
    dp.add_argument("--dry-run", action="store_true", help="print what would be written and stop")
    xt = sub.add_parser("export-tracer", help="write the tracer contract (docs/research/"
                        "synthesis_v2_with_literature.md section 2) over a box from a --sdist checkpoint: "
                        "recto/verso, surf_sdist, nz/ny/nx, gmag and conf as sharded volcomp stores")
    xt.add_argument("ckpt")
    xt.add_argument("out", help="output DIRECTORY; one store per field goes inside it")
    xt.add_argument("--volume", default=None)
    xt.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    xt.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    xt.add_argument("--rung", type=int, default=None, help="predict at rung k (--origin/--size are then rung-k voxels)")
    xt.add_argument("--window", type=int, default=128)
    xt.add_argument("--halo", type=int, default=16)
    xt.add_argument("--tta", type=int, default=0)
    xt.add_argument("--cascade", default="auto", choices=["auto", "on", "off"])
    xt.add_argument("--cascade-depth", type=int, default=3)
    xt.add_argument("--device", default=None)
    xt.add_argument("--plain", action="store_true", help="plain zarr instead of volcomp")
    xt.add_argument("--marching-cubes", action="store_true", help="also run marching cubes (skimage) per "
                    "SHARD on the zero level of the distance field and write one .obj per shard under "
                    "<out>/mesh/ (vertices in GLOBAL ZYX voxels of this rung)")
    xt.add_argument("--mc-level", type=float, default=0.0, help="the level set to extract, in voxels")
    cb = sub.add_parser("calibrate", help="fit one TEMPERATURE per rung on the checkpoint's own held-out "
                        "grid and store them in its args (docs/unified_design.md section 26); "
                        "`predict`/`evalsurf` then divide the logits by it unless --no-calib")
    cb.add_argument("ckpt")
    cb.add_argument("--val", nargs="+", default=None, help="validation box(es) (default: the checkpoint's own)")
    cb.add_argument("--val-rungs", default=None, help="rungs to fit at (default: the checkpoint's own)")
    cb.add_argument("--val-patches", type=int, default=None)
    cb.add_argument("--device", default=None)
    cb.add_argument("--all-rungs", action="store_true", help="also fit the rungs whose target is a pooled "
                    "FRACTION rather than a binary band (a temperature there is not a calibration)")
    cb.add_argument("--dry-run", action="store_true", help="print the fit but do not write the checkpoint")
    gw = sub.add_parser("glc-weights", help="GLC-style per-source loss weights: score each teacher source "
                        "against the published meshes on the val box and print a --source-w line")
    gw.add_argument("sources", nargs="+", metavar="NAME=STORE", help="e.g. mask=/vesuvius/usrm2/teacher/eval.zarr "
                    "store=/vesuvius/usrm2/teacher_regions/recto/region_34816_14336_17408.zarr")
    gw.add_argument("--box", type=int, nargs=6, default=None, metavar=("Z0", "Y0", "X0", "Z", "Y", "X"))
    gw.add_argument("--tifxyz", default=None)
    gw.add_argument("--volume", default=None)
    gw.add_argument("--r", type=float, default=4.0, help="voxels along the normal that count as a hit")
    gw.add_argument("--thr", type=float, default=0.5)
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
    s.add_argument("--head", default="0", help="which output channel to score: an index, a channel NAME "
                   "(recto / verso), or mean / prod / max. NOTE: there is no published verso surface, so "
                   "`--head verso` measures the verso band at the RECTO points (see usrm2/evalsurf.py)")
    s.add_argument("--lut-to", nargs="*", default=(), metavar="REF")
    s.add_argument("--halo", type=int, default=16)
    s.add_argument("--cascade", default="auto", choices=["auto", "on", "off"], help="see `predict --cascade`")
    s.add_argument("--cascade-depth", type=int, default=3)
    s.add_argument("--device", default=None)
    s.add_argument("--ceiling", nargs="?", const="", default=None, metavar="STORE",
                   help="also score the noise ceiling and print every number as \"value (ceiling)\": the "
                        "store given here, else --teacher, else the published recto mask pyramid. Cached "
                        "per (box, store) next to the eval box (docs/unified_design.md section 25)")
    s.add_argument("--no-ceiling-cache", action="store_true", help="recompute the ceiling, ignoring the cache")
    s.add_argument("--json", dest="json_out", default=None, metavar="OUT",
                   help="dump everything (per-surface rows, pooled metrics, bootstrap CIs, Betti, ceiling) as json")
    s.add_argument("--bootstrap", type=int, default=200, help="bootstrap draws over surfaces for the 95%% CIs (0 = off)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-betti", action="store_true", help="skip the Betti-0/1 pass (the one costly new metric)")
    s.add_argument("--betti-margin", type=int, default=8, help="voxels cropped off every box face before counting")
    s.add_argument("--betti-band", type=float, default=6.0, help="voxels around the mesh the topology is counted in")
    s.add_argument("--betti-dilate", type=float, default=2.0, help="thicken the rasterized mesh by this many "
                   "voxels before counting, so a one-voxel staircase is not compared with a 3-5 voxel band")
    sc = sub.add_parser("evalsurf-curve", help="fit the plateau of a metric over a run (eval.jsonl or evalsurf --json dumps)")
    sc.add_argument("run_dir")
    sc.add_argument("--metric", default="dice", help="a key of eval.jsonl (dice, dice_r2, bce, ...) or of an evalsurf json")
    sc.add_argument("--unbounded", action="store_true", help="the metric is not confined to [0,1]")
    sc.add_argument("--smooth", type=int, default=5, help="width of the running median applied before fitting (1 = off)")
    sc.add_argument("--tail", type=float, default=1.0, help="fit only the last fraction of the checkpoints")
    sc.add_argument("--json", dest="json_out", default=None)
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
    v.add_argument("--cascade", default="auto", choices=["auto", "on", "off"], help="see `predict --cascade`")
    v.add_argument("--cascade-depth", type=int, default=3)
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
    # ----------------------------------------------------- masked-cube pretraining (docs/unified_design.md 28)
    pt = sub.add_parser("pretrain", help="in-domain masked-cube (MAE-style) pretraining of the SAME encoder/"
                        "decoder `train` uses: no labels, reconstruct the masked CT. The checkpoint warm-"
                        "starts a fine-tuning run with `usrm2 train --init <out>/ckpt.pt`")
    pt.add_argument("out_dir")
    pt.add_argument("--size", default="1m", choices=list(model_presets()))
    pt.add_argument("--steps", type=int, default=20000)
    pt.add_argument("--patch", type=int, nargs="+", default=[256])
    pt.add_argument("--batch", type=int, default=1)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--accum", type=int, default=1)
    pt.add_argument("--warmup", type=int, default=200)
    pt.add_argument("--ema", type=float, default=0.999)
    pt.add_argument("--lr-floor", type=float, default=0.0)
    pt.add_argument("--eval-every", type=int, default=500)
    pt.add_argument("--val-patches", type=int, default=4)
    pt.add_argument("--val", nargs="+", default=None, help="held-out box, excluded from sampling and used for the reconstruction metric")
    pt.add_argument("--stores", nargs="+", default=None, help="'ct_base,target_group' lines, as `train --stores`: no label is read, the target group only bounds the sampled box")
    pt.add_argument("--stores-file", default=None, help="text file of those lines (one per line)")
    pt.add_argument("--rungs", default="0-4", help="rungs to pretrain at (default the fine half, 0-4); rungs no scan is native at are dropped with a message")
    pt.add_argument("--ctx", nargs="*", default=(), help="context offsets, e.g. 1..9 (must match the fine-tuning run's)")
    pt.add_argument("--aug", default="geo", help="augmentation preset (see aug.PRESETS)")
    pt.add_argument("--no-radial", action="store_true")
    pt.add_argument("--norm", default="patch", choices=["patch", "global"])
    pt.add_argument("--ckpt-act", type=int, default=0)
    pt.add_argument("--add-skip", type=int, default=0)
    pt.add_argument("--deep", type=int, default=0, help="deep supervision levels; the coarse reconstruction heads are dropped by the warm start like the main one")
    pt.add_argument("--compile", action="store_true")
    pt.add_argument("--resume", action="store_true")
    pt.add_argument("--mask-block", type=int, default=32, help="edge of the masked blocks, in voxels")
    pt.add_argument("--mask-lo", type=float, default=0.5, help="lowest per-sample masking ratio")
    pt.add_argument("--mask-hi", type=float, default=0.75, help="highest per-sample masking ratio")
    pt.add_argument("--sheet-p", type=float, default=0.5, help="probability a sample is masked ALONG SHEETS (blocks drawn proportional to their foreground fraction) instead of uniformly")
    pt.add_argument("--sheet-pct", type=float, default=0.7, help="the CT quantile the foreground proxy thresholds at")
    pt.add_argument("--no-mask-ctx", action="store_true", help="do NOT blank the context channels' footprint (they then hand the model a coarse copy of the answer)")
    pt.add_argument("--loss", default="l1", choices=["l1", "l2"], help="reconstruction loss on the masked voxels")
    pt.add_argument("--no-cascade-slot", action="store_true", help="build the 14-channel stem instead of the 15-channel one (pretrain this way only for a fine-tuning run with --cascade off: warm_start can widen a stem, never narrow one)")
    pt.add_argument("--rung-aux", type=float, default=0.0, help="weight of the VoCo-flavoured 'which rung' head on the bottleneck (off by default; it hides the scale plane on the steps it scores)")
    pt.add_argument("--rung-aux-p", type=float, default=0.5, help="fraction of steps that hide the scale plane and score the aux head")
    pt.add_argument("--fg-min", type=float, default=0.0, help="foreground rejection threshold (0 = take every non-air window: pretraining wants texture, not labels)")
    pt.add_argument("--air-keep", type=float, default=0.1, help="probability an all-air window is kept")
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

    def parse_kv(vs):
        """'mask=1 store=0.7' -> {"mask": 1.0, "store": 0.7}."""
        return {q.split("=")[0]: float(q.split("=")[1]) for q in vs}

    def parse_kv_str(vs):
        """'mask=/a.zarr store=/b.zarr' -> {"mask": "/a.zarr", "store": "/b.zarr"}."""
        return {q.split("=", 1)[0]: q.split("=", 1)[1] for q in vs}

    def parse_cascade(v):
        """`--cascade auto|on|off` at INFERENCE -> what predict.probs wants: None = follow the checkpoint."""
        return None if str(v) == "auto" else (str(v) != "off")
    if a.umbilicus:
        data.UMBILICUS = a.umbilicus
    if a.cmd == "rung-mix":
        lines = [l.strip() for l in open(a.stores_file) if l.strip() and not l.startswith("#")]
        rs = parse_rungs(a.rungs)
        rows = data.rung_mix(lines, patch=a.patch if len(a.patch) > 1 else a.patch[0],
                             allowed=None if rs is True else rs, boost=parse_boost(a.rung_boost))
        print(data.format_rung_mix(rows))
    elif a.cmd == "train":
        ema_k = a.ema_k if a.ema_k else (T.EMA_K if str(a.ema).lower() == "auto" else None)
        T.train(a.out_dir, accum=a.accum, ema_decay=(0.999 if str(a.ema).lower() == "auto" else float(a.ema)), lr_floor=a.lr_floor, ridge_w=a.ridge_w, dense_pow=a.dense_pow,
                norm=a.norm, ctx=parse_ctx(a.ctx), stream=a.stream, init_from=a.init_from, wtgt=tuple(a.wtgt), compile=a.compile, ckpt_act=a.ckpt_act, add_skip=a.add_skip, deep=a.deep, size=a.size, steps=a.steps, patch=a.patch if len(a.patch) > 1 else a.patch[0], batch=a.batch, lr=a.lr,
                workers=a.workers, eval_every=a.eval_every, val_patches=a.val_patches, resume=a.resume,
                aug=a.aug, no_radial=a.no_radial,
                cascade=a.cascade, cascade_self_p=a.cascade_self_p, cascade_drop=a.cascade_drop,
                cascade_noise=not a.no_cascade_noise,
                loss_excl=a.loss_excl, loss_selfcons=a.loss_selfcons, loss_skel=a.loss_skel,
                loss_affinity=a.loss_affinity, affinity=a.affinity, skel_iters=a.skel_iters,
                affinity_all=a.affinity_all, cascade_self_p_anneal=a.cascade_self_p_anneal,
                sched=a.sched, stable_until=a.stable_until, cooldown=a.cooldown,
                rewarm=a.rewarm, new_param_lr_mult=a.new_param_lr_mult, ema_k=ema_k,
                fuse=a.fuse, source_w=parse_kv(a.source_w), scan_meta=a.scan_meta,
                sdist=a.sdist, thickness=a.thickness, normals=a.normals, sdist_hetero=a.sdist_hetero,
                loss_sdist=a.loss_sdist, loss_eikonal=a.loss_eikonal, loss_normals=a.loss_normals,
                sdist_delta=a.sdist_delta, sdist_band=a.sdist_band,
                pair=a.pair, pair_band=a.pair_band, pair_tau=a.pair_tau,
                loss_ect=a.loss_ect, ect_dirs=a.ect_dirs, ect_res=a.ect_res, ect_margin=a.ect_margin,
                ect_block=a.ect_block, ect_n=a.ect_n, ect_rung=a.ect_rung, planes=a.planes,
                **({"rungs": parse_rungs(a.rungs), "rung_boost": parse_boost(a.rung_boost),
                    "val_rungs": [int(q) for q in a.val_rungs.split(",")], "require_targets": a.require_targets,
                    "region": a.region, "windows_per_region": a.windows_per_region,
                    "teacher_regions": a.teacher_regions, "verso": a.verso,
                    "verso_regions": a.verso_regions, "verso_regions_url": a.verso_regions_url,
                    "cout": a.cout} if a.rungs else {}),
                **{k: v for k, v in dict(stores=a.stores, stores_file=a.stores_file, val=a.val).items() if v})
    elif a.cmd == "pretrain":
        from usrm2 import pretrain as PT
        PT.pretrain(a.out_dir, size=a.size, steps=a.steps, patch=a.patch if len(a.patch) > 1 else a.patch[0],
                    batch=a.batch, lr=a.lr, workers=a.workers, warmup=a.warmup, eval_every=a.eval_every,
                    val_patches=a.val_patches, resume=a.resume, aug=a.aug, no_radial=a.no_radial,
                    accum=a.accum, ema_decay=a.ema, lr_floor=a.lr_floor, norm=a.norm, ctx=parse_ctx(a.ctx),
                    compile=a.compile, ckpt_act=a.ckpt_act, add_skip=a.add_skip, deep=a.deep,
                    rungs=parse_rungs(a.rungs), mask_block=a.mask_block, mask_lo=a.mask_lo,
                    mask_hi=a.mask_hi, sheet_p=a.sheet_p, sheet_pct=a.sheet_pct,
                    mask_ctx=not a.no_mask_ctx, loss=a.loss, cascade_slot=not a.no_cascade_slot,
                    rung_aux=a.rung_aux, rung_aux_p=a.rung_aux_p, fg_min=a.fg_min, air_keep=a.air_keep,
                    **{k: v for k, v in dict(stores=a.stores, stores_file=a.stores_file, val=a.val).items() if v})
    elif a.cmd == "dist-pyramid":
        from usrm2 import targets as TG
        rs = parse_rungs(a.rungs)
        TG.dist_pyramid(a.mask, out=a.out, verso=a.verso, kinds=tuple(a.kind),
                        rungs=(range(0, TG.MAX_RUNG + 1) if rs is True else sorted(rs)),
                        volume=a.volume, umbilicus=a.umbilicus, block=a.block, halo=a.halo,
                        box=((a.box[:3], a.box[3:]) if a.box else None), dry_run=a.dry_run,
                        **{k: v for k, v in dict(axis_r_um=a.axis_r_um, tmin=a.tmin,
                                                 max_rung=a.max_rung).items() if v is not None})
    elif a.cmd == "export-tracer":
        P.export_tracer(a.ckpt, a.volume or data.CT, *a.origin, *a.size, a.out, window=a.window,
                        halo=a.halo, device=a.device, rung=a.rung, tta=a.tta,
                        cascade=parse_cascade(a.cascade), cascade_depth=a.cascade_depth,
                        marching_cubes=a.marching_cubes, mc_level=a.mc_level, volcomp=not a.plain)
    elif a.cmd == "calibrate":
        from usrm2 import calib
        calib.main(a.ckpt, val=a.val, val_rungs=(None if a.val_rungs is None else
                                                 [int(q) for q in str(a.val_rungs).split(",")]),
                   val_patches=a.val_patches, device=a.device, all_rungs=a.all_rungs, write=not a.dry_run)
    elif a.cmd == "glc-weights":
        from usrm2 import glc
        b = a.box
        glc.main(parse_kv_str(a.sources), origin=(b[:3] if b else None), size=(b[3:] if b else None),
                 tifxyz=a.tifxyz, volume=a.volume, r=a.r, thr=a.thr)
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
               teacher_regions=a.teacher_regions, visits_max=a.visits_max, cascade=a.cascade,
               verso=a.verso, verso_regions=a.verso_regions, verso_url=a.verso_regions_url,
               planes=a.planes, scan_meta=a.scan_meta)
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
              head=a.head,  # resolved against the checkpoint's own channel list in predict.probs
              cascade=parse_cascade(a.cascade), cascade_depth=a.cascade_depth,
              ceil=a.ceiling, json_out=a.json_out, boot=a.bootstrap, seed=a.seed, betti=not a.no_betti,
              betti_margin=a.betti_margin, betti_band=a.betti_band, betti_dilate=a.betti_dilate,
              no_ceiling_cache=a.no_ceiling_cache)
    elif a.cmd == "evalsurf-curve":
        from usrm2 import evalsurf as E
        E.curve(a.run_dir, metric=a.metric, bounded=not a.unbounded, out=a.json_out, smooth=a.smooth, tail=a.tail)
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
                outs = verso.run(g, a.ckpt, window=a.window, halo=a.halo, tile=a.tile, margin=a.margin, force=a.force, batch=a.batch, modes=tuple(a.modes),
                                 cascade=parse_cascade(a.cascade), cascade_depth=a.cascade_depth)
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
                  tta=a.tta, luts=[teacher.lut_to(vol, r) for r in a.lut_to], head=a.head,
                  radial_sign=a.radial_sign, cascade=parse_cascade(a.cascade), cascade_depth=a.cascade_depth,
                  calib=not a.no_calib)


if __name__ == "__main__":
    main()
