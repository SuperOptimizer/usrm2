"""Distillation training: BCE + soft dice against the teacher's soft probabilities."""
import json
import math
import os

import numpy as np
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from usrm2 import aug as A, data, losses as L, model as M, prep


def weighted(tgt, wtgt=()):
    """Targets and per-voxel weights. Channels in `wtgt` (verso targets) store a WEIGHT where they are positive:
    target 1 with weight = value (255 -> 1, verso.WEAK -> 0.3); zeros keep weight 1. Other channels: weight 1."""
    if not wtgt:
        return tgt, None
    w = torch.ones_like(tgt)
    t = tgt.clone()
    for c in wtgt:
        pos = tgt[:, c] > 0
        w[:, c] = torch.where(pos, tgt[:, c], torch.ones_like(tgt[:, c]))
        t[:, c] = pos.float()
    return t, w


DEEP_W = (1.0, 0.5, 0.25, 0.125)  # loss weight of the level-0 head and the coarser (deep supervision) heads


def deep_losses(logits, tgt, ridge_w=0.0, wtgt=(), w=None):
    """`losses` summed over the multi-resolution outputs: level k is scored against the target average-pooled
    2^k times, with the per-voxel weights pooled alongside (so the pooled target stays a probability and a
    voxel that is ignored at level 0 stays ignored above it)."""
    if not isinstance(logits, (list, tuple)):
        return losses(logits, tgt, ridge_w, wtgt, w)
    t, wv = weighted(tgt, wtgt)
    wv = w if wv is None else (wv if w is None else wv * w)
    bce = dice = 0.0
    for k, lg in enumerate(logits):
        tk = F.avg_pool3d(t, 2 ** k) if k else t
        wk = F.avg_pool3d(wv, 2 ** k) if (k and wv is not None) else wv
        b, d = losses_tw(lg, tk, wk, ridge_w)
        bce, dice = bce + DEEP_W[k] * b, dice + DEEP_W[k] * d
    return bce, dice


def losses_tw(logit, tgt, wv, ridge_w=0.0):
    """`losses` on an already-converted (target, weight) pair. The weight scales the BCE and masks the soft
    dice (a voxel with weight 0 contributes to neither, so it carries no gradient).

    The BCE is one weighted mean over the whole tensor, so a channel that is weight 0 throughout the batch
    adds nothing to the numerator AND nothing to the denominator: it neither contributes nor rescales the
    others, and an all-zero weight tensor (the blank-patch aug) gives 0, not a NaN. The soft dice is per
    channel and is averaged over the channels that HAVE weight somewhere in the batch -- an ignored channel
    would otherwise score a constant 0 and halve the loss of the channel that is being trained. That is what
    lets the verso output (docs/unified_design.md section 23) sit in the same head as the recto one: on a
    sample with no verso store the verso channel simply is not there."""
    if ridge_w > 0 or wv is not None:
        w = wv if ridge_w == 0 else (1 + ridge_w * (tgt >= 0.9).float())  # ridge_w 0: 1 * wv is wv
        if ridge_w > 0 and wv is not None:
            w = w * wv
        bce = (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum() / w.sum().clamp_min(1e-6)
    else:
        bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p, d = torch.sigmoid(logit), (0, 2, 3, 4)
    if wv is not None:
        p, tgt = p * wv, tgt * wv
    per = 1 - (2 * (p * tgt).sum(d) + 1) / (p.sum(d) + tgt.sum(d) + 1)  # per head
    if wv is None:
        return bce, per.mean()
    live = (wv.sum(d) > 0).to(per.dtype)          # the channels this batch says anything about
    return bce, (per * live).sum() / live.sum().clamp_min(1.0)


def losses(logit, tgt, ridge_w=0.0, wtgt=(), w=None):
    """BCE (+ ridge_w extra weight on the band's core, target >= 0.9: the student must commit there) + soft dice.
    w: the per-voxel, per-channel weight tensor the rung loader returns (None = every voxel counts).
    wtgt: the older convention, channels whose stored value is a loss weight (verso targets, see `weighted`)."""
    tgt, wv = weighted(tgt, wtgt)
    wv = w if wv is None else (wv if w is None else wv * w)
    return losses_tw(logit, tgt, wv, ridge_w)


# ----------------------------------------------------------------- the training recipe (section 26)

def lr_lambda(steps, warmup, lr_floor=0.0, sched="cosine", stable_until=None, cooldown=0):
    """The LambdaLR factor. `cosine` is the original formula, unchanged to the last float op.

    `wsd` (warmup-stable-decay, MiniCPM / `lit_optimisation_schedules.md`): linear warmup, then a FLAT
    plateau until step `stable_until`, then the same cosine cooling over `cooldown` steps. Its point is
    that the step budget need not be committed at run start: `stable_until` and `cooldown` are not part
    of the weights, so a resume may move them (they are in the resume `grow` tuple) and the plateau
    simply runs longer. The defaults follow the literature's 10 % cooldown: `cooldown = 0.1 * steps`
    and `stable_until = steps - cooldown`.
    """
    if str(sched) != "wsd":
        return lambda s: min((s + 1) / warmup, 1.0) * \
            (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    C = int(cooldown) if cooldown else max(int(round(0.1 * steps)), 1)
    S = int(stable_until) if stable_until is not None else max(int(steps) - C, 1)

    def f(s):
        wu = min((s + 1) / warmup, 1.0)
        if s < S:
            return wu
        t = min((s - S) / max(C, 1), 1.0)
        return wu * (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * t)))
    return f


EMA_K = 50  # `--ema auto` = 1 - k / steps: the averaging window is steps / k, i.e. 1 / 50 = 2 % of the
            # run, the middle of `lit_optimisation_schedules.md`'s 1-3 % recommendation. k = 10 would be
            # a 10 % window. A fixed 0.999 is a 1000-step window: 1.7 % of a 60k run and 0.5 % of a 200k one.


def ema_auto(steps, k=EMA_K):
    """`1 - k / steps`, clamped to [0.9, 0.9999] so a very short or very long run stays sane."""
    return float(min(max(1.0 - float(k) / max(int(steps), 1), 0.9), 0.9999))


def new_param_names(cin_grew, cout_grew, net):
    """State-dict names of the tensors that hold NEWLY INITIALISED rows after a warm start: the stem
    convolution (new input planes) and the heads (new output rows).

    Param groups are per TENSOR, not per row, so putting the whole stem/head in the boosted group also
    boosts the warm-started rows inside them. That is the standard practical form of the
    "new parameters take full LR" recipe and it is cheap here: the head is a 1x1x1 convolution and the
    stem is one 3x3x3 convolution out of ~200 tensors."""
    out = []
    if cin_grew:
        out.append("enc.0.0.weight")
    if cout_grew:
        out += [k for k in net.state_dict() if k.startswith("head.") or k.startswith("deep_heads.")]
    return set(out)


def param_groups(net, new_names, mult):
    """[(params), (new params)] for AdamW when `mult != 1`, else one group exactly as before."""
    if mult == 1.0 or not new_names:
        return [{"params": list(net.parameters())}], False
    new = [p for n, p in net.named_parameters() if n in new_names]
    old = [p for n, p in net.named_parameters() if n not in new_names]
    return [{"params": old}, {"params": new}], bool(new)


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    """Same update as `e.mul_(decay).add_(v, alpha=1-decay)` per tensor, batched with the foreach kernels:
    one pair of kernel launches for the whole 45M-parameter state instead of two per tensor."""
    es, vs = [], []
    for k, v in model.state_dict().items():
        e = ema[k]
        if e.is_floating_point():
            es.append(e), vs.append(v.detach())
        else:
            e.copy_(v)
    if es:
        torch._foreach_mul_(es, decay)
        torch._foreach_add_(es, vs, alpha=1 - decay)


def autocast(dev):
    import contextlib
    return torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(net, grid, dev, wtgt=(), norad=False, cascade=None, channels=None, cout_t=None):
    """bce / dice / mae over the val patches. A grid entry is (x, tgt) -- the old convention, optionally with
    `wtgt` weight channels -- or a compact rung sample (data.rung_item), whose input is built on the device
    by `prep.prepare`; its weights then scale every metric and each rung is also scored on its own
    (`dice_r2`, `dice_r3`, ...), `dice` being the mean over the rungs present. With an even number of output
    channels (the unified model's recto/verso pair, or the older recto.../verso... heads) also `overlap`,
    the mean excess relu(p_recto + p_verso - 1) per lineage pair, measured only (no loss term).

    `channels`: the output channel names, which give the per-channel dice a name (`dice_recto`,
    `dice_verso`); without them the channels are numbered. A channel is only scored on the patches whose
    WEIGHT says anything about it, so the verso channel is silently absent until a verso store covers the
    val box."""
    net.eval()
    m = torch.zeros(4)
    per, pch = {}, {}
    for item in grid:
        if isinstance(item, dict):
            ct, tg, ww = prep.prepare(prep.batch1(item), dev, norad=norad, cascade=cascade)
            ct, rung = ct.to(memory_format=M.memfmt()), int(item["rung"])
        else:
            ct, tg = item[0], item[1]
            w = item[2] if len(item) > 2 else None
            rung = item[3] if len(item) > 3 else None
            ct, tg = ct[None].to(dev).to(memory_format=M.memfmt()), tg[None].to(dev)
            ww = None if w is None else w[None].to(dev)
        if ww is None:
            tg = weighted(tg, wtgt)[0]
            ww = torch.ones_like(tg)
        with autocast(dev):
            logit = net(ct).float()
        if cout_t is not None:  # the AFFINITY channels are a training-only head: never scored, never shown
            logit = logit[:, :int(cout_t)]
        p = torch.sigmoid(logit)
        h, t = (p >= 0.5).float(), (tg >= 0.5).float()
        C = p.shape[1]
        ov = (p[:, :C // 2] + p[:, C // 2:] - 1).clamp_min(0).mean().item() if C >= 2 and C % 2 == 0 else 0.0
        n = ww.sum().clamp_min(1e-6)
        dice = (2 * (h * t * ww).sum() / ((h * ww).sum() + (t * ww).sum() + 1)).item()
        for c in range(C if C > 1 else 0):  # per output channel, over the voxels it weighs
            wc = ww[:, c]
            if float(wc.sum()) > 0:
                pch.setdefault(c, []).append(float(2 * (h[:, c] * t[:, c] * wc).sum() /
                                                  ((h[:, c] * wc).sum() + (t[:, c] * wc).sum() + 1)))
        m += torch.tensor([((F.binary_cross_entropy_with_logits(logit, tg, reduction="none") * ww).sum() / n).item(),
                           dice, (((p - tg).abs() * ww).sum() / n).item(), ov])
        if rung is not None:
            per.setdefault(int(rung), []).append(dice)
    net.train()
    m /= max(len(grid), 1)
    out = {"bce": m[0].item(), "dice": m[1].item(), "mae": m[2].item()}
    if per:
        for k in sorted(per):
            out[f"dice_r{k}"] = float(np.mean(per[k]))
        out["dice"] = float(np.mean([out[f"dice_r{k}"] for k in sorted(per)]))  # every rung counts the same
    for c in sorted(pch):
        out[f"dice_{(list(channels)[c] if channels and c < len(channels) else f'c{c}')}"] = float(np.mean(pch[c]))
    if grid and grid_cout(grid[0]) >= 2 and grid_cout(grid[0]) % 2 == 0:
        out["overlap"] = m[3].item()
    return out


def grid_cin_cout(item):
    """(input channels, output channels) of a validation grid entry, compact (dict) or legacy (tuple)."""
    return prep.shapes(item) if isinstance(item, dict) else (item[0].shape[0], item[1].shape[0])


def grid_cout(item):
    return grid_cin_cout(item)[1]


def val_png(path, net, grid, dev, norad=False, cascade=None, cout_t=None):
    """Middle z-slice of the first 4 val patches: CT (gray), each teacher target and each student OUTPUT
    CHANNEL as a red opacity overlay (no threshold), tiled patches x [CT, targets..., channels...]. With
    `--verso` (cout 2) it is three tiles: CT | recto target red + verso target blue | recto prediction red +
    verso prediction blue, both on the same tile; a verso target the val box has no store for adds nothing."""
    from PIL import Image
    rows = []
    with torch.no_grad():
        for item in grid[:4]:
            if isinstance(item, dict):
                xd, td = prep.prepare(prep.batch1(item), dev, norad=norad, cascade=cascade)[:2]
                x, t = xd[0].cpu(), td[0].cpu()
            else:
                x, t = item[0], item[1]
            with autocast(dev):
                y = net(x[None].to(dev).to(memory_format=M.memfmt()))
                y = y[0] if isinstance(y, (list, tuple)) else y  # deep supervision returns [main, coarse...]
                y = y if cout_t is None else y[:, :int(cout_t)]  # never draw the affinity channels
                p = torch.sigmoid(y.float())[0].cpu()
            z = x.shape[1] // 2
            c = x[0, z].numpy()
            c = (c - c.min()) / (c.max() - c.min() + 1e-6) * 255 * 0.9
            gray = np.repeat(c[..., None], 3, -1)

            def overlay(chans):  # channel 0 red, channel 1 blue, mixed on one tile (overlap = purple)
                a = [np.clip(v, 0, 1)[..., None] for v in chans]
                cols = [np.array([255, 40, 40]), np.array([40, 90, 255])]
                wsum = sum(a); al = np.maximum.reduce(a) * 0.85
                col = sum(ai * ci for ai, ci in zip(a, cols)) / np.maximum(wsum, 1e-6)
                return gray * (1 - al) + col * al
            tt, pp = t[:, z].numpy(), p[:, z].numpy()
            if pp.shape[0] == 2:  # --verso: CT | targets (recto red + verso blue) | prediction (recto red + verso blue)
                tiles = [gray, overlay(tt[:2]), overlay(pp[:2])]
            else:
                tiles = [gray] + [overlay([a]) for a in list(tt) + list(pp)]
            rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)


def warm_start(src, cin, cout, cascade=False, src_scale=False, ncopy=None):
    """Adapt another run's weights to (cin, cout). Extra input channels get zero weights, so the net starts
    with the same output: the image channels stay first and the radial vector stays last, which is what
    zero-fills the scale plane of the unified model (13 -> 14 channels). Heads: more heads are copies of the
    source heads (head j <- source head j mod n), fewer keep the first cout (a 4-head recto/m7 student warm-
    starting the single-headed unified model keeps head 0); the deep-supervision heads follow.

    `cascade`: the DESTINATION has a cascade channel, so its stem is [image..., CASCADE, scale, radial(3)].
    A source that already carries a scale plane (`src_scale`, 14 channels) must have it lined up with the
    destination's scale plane and only the cascade slot zeroed -- "image channels first, radial last" alone
    would slide the scale weights into the cascade slot and zero the scale plane instead, which changes the
    output at every rung but 2. With both flags the 14 -> 15 warm start is exact: the new channel's weights
    are zero, so it contributes nothing whatever the channel holds, and the outputs agree with the source's
    to a float32 ulp (the stem convolution accumulates 15 products instead of 14).

    `ncopy`: how many of the `cout` head rows follow the copy rule. The rows at and above it are NEW and
    are zero-initialised (weight and bias both 0, i.e. p = 0.5) instead of copying a probability filter
    -- that is the per-channel policy the AFFINITY channels need (`--affinity`, section 26): they are not
    another recto/verso probability, so `copy mod n` would be wrong, and because the head is 1x1x1 the
    recto/verso rows are bit-identical whatever the new rows hold."""
    src = dict(src)
    w = src["enc.0.0.weight"]
    if w.shape[1] != cin:
        assert w.shape[1] < cin, "cannot drop input channels on a warm start"
        n = w.shape[1]
        w2 = torch.zeros(w.shape[0], cin, *w.shape[2:], device=w.device, dtype=w.dtype)
        if cascade and src_scale:  # [img..., scale, radial] -> [img..., CASCADE(0), scale, radial]
            w2[:, :n - 4], w2[:, cin - 4], w2[:, cin - 3:] = w[:, :n - 4], w[:, n - 4], w[:, n - 3:]
        else:
            w2[:, :n - 3], w2[:, cin - 3:] = w[:, :n - 3], w[:, n - 3:]  # image chans first, radial last
        src["enc.0.0.weight"] = w2
    nc = cout if ncopy is None else int(ncopy)
    for hk in [k for k in src if k == "head.weight" or (k.startswith("deep_heads.") and k.endswith(".weight"))]:
        hw, hb = src[hk], src[hk[:-6] + "bias"]
        if hw.shape[0] == cout:
            continue   # the source head already has exactly these rows: nothing is new, nothing is zeroed
        grew = hw.shape[0] < cout
        j = torch.arange(cout, device=hw.device) % hw.shape[0] if grew else torch.arange(cout, device=hw.device)
        w2, b2 = hw[j].clone(), hb[j].clone()
        if grew and nc < cout:  # rows >= ncopy are NEW (the affinity channels): zero weight and bias, so
            w2[nc:], b2[nc:] = 0, 0   # they start at p = 0.5 and the recto/verso rows are untouched
        src[hk], src[hk[:-6] + "bias"] = w2, b2
    return src


def train(out_dir, size="1m", steps=20000, patch=128, batch=1, lr=3e-4, workers=4, warmup=200,
          eval_every=500, val_patches=32, resume=False, device=None, aug="geo", no_radial=False, accum=1,
          ema_decay=0.999, lr_floor=0.0, ridge_w=0.0, dense_pow=0.0, norm="patch", ctx=(), init_from=None, wtgt=(),
          compile=False, ckpt_act=0, add_skip=0, deep=0, rungs=None, rung_boost=None, val_rungs=data.VAL_RUNGS,
          require_targets=False, stream=None, cascade="off", cascade_self_p=0.5, cascade_drop=0.1,
          cascade_noise=True, verso=False, verso_regions=None, verso_regions_url=None, cout=None,
          loss_excl=0.0, loss_selfcons=0.0, loss_skel=0.0, loss_affinity=0.0, affinity=None,
          skel_iters=4, affinity_all=False, cascade_self_p_anneal=None,
          sched="cosine", stable_until=None, cooldown=0, ema_k=None, rewarm=0, new_param_lr_mult=1.0,
          fuse="off", source_w=None, **kw):
    """accum: gradient accumulation (micro-batches per optimizer step), for big models on small cards.
    lr_floor: the cosine decays to lr_floor * lr instead of 0. norm: "patch" (per-patch z-score) or "global"
    (fixed scan mean/std, stored in the checkpoint). dense_pow / ridge_w: see data.Patches / losses.
    wtgt: target channels that carry loss weights (verso targets from verso.py), e.g. (2, 3) for a 4-head student.
    Multi-GPU: launch with `torchrun --nproc_per_node N -m usrm2.cli train ...`; every rank draws its own patches
    (seed offset), gradients are all-reduced (DDP), rank 0 evaluates, logs and saves. One optimizer step then sees
    N * batch * accum patches. compile: torch.compile the training forward/backward (1.4x on the 5m at 128^3;
    the raw module keeps serving EMA, checkpoints and evaluation).
    rungs: train the unified multi-resolution model (docs/unified_design.md). The stores are then
    `ct_base,target_group[,...]` lines of whole-scroll pyramids, a sample is (source, rung, corner), the input
    carries the constant scale plane (k - 2) / 9 right before the radial vector, and the loader's per-voxel
    weights go into the loss. `rungs` is True (every usable rung) or the allowed rungs; rung_boost {k: m}
    skews the mix; val_rungs are the rungs the held-out box (kw["val"], given at rung 2) is scored at.
    cascade: the CASCADE input channel (docs/unified_design.md section 22), "off" | "mask" | "self" | "mix".
    The model then takes 15 channels, [CT, ctx_1..9, CASCADE, scale, radial(3)]: the rung-(k+1) prediction
    over the same field of view, upsampled 2x. `cascade_self_p` is P in "mix", `cascade_drop` the per-sample
    probability the channel is zeroed (so a missing coarse prediction is in distribution), `cascade_noise`
    the roughening of the mask-derived channel. "off" (the default) leaves every existing run untouched.
    verso: add the VERSO OUTPUT CHANNEL (docs/unified_design.md section 23). The final 1x1x1 head grows
    from cout 1 to cout 2 -- channel 0 recto, channel 1 verso, ONE head, and the deep-supervision heads
    follow -- and the loader gains a second target channel whose only source is the verso region stores
    under `verso_regions` (default: `--teacher-regions`). Where no finished verso store covers a voxel its
    weight is 0, so the sample trains recto alone; at rungs >= 4 the whole channel is weight 0 for now.
    `verso_regions_url` is only for `stream-plan` (the planner fetches the stores as they are published);
    it is recorded in the args so a resume can tell how the run was fed. `cout`, when given, is an
    assertion: the number of output channels is derived from the target channels (+ verso), and `--cout 2`
    just says out loud that you expect two.

    PHASE A (docs/unified_design.md section 26). Every one of these defaults to OFF and is recorded in the
    checkpoint args only when it is on, so a run started without them is byte-identical to one started
    before they existed, and an existing run resumes unchanged.
    loss_excl: weight of L3 soft exclusivity, relu(p_recto + p_verso - 1) where both channels have weight.
    loss_selfcons: weight of L4 cascade self-consistency, |pool2(p) - pool2(CASCADE)| on the samples whose
    cascade channel came from the model's own coarse forward (`--cascade self|mix`); no extra forward.
    loss_skel: weight of L8 skeleton recall, 1 - mean p along the TARGET's medial surface (usrm2/losses.py).
    affinity / loss_affinity: O12. `affinity` is a list of EVEN voxel offsets (e.g. "16,32"); the head grows
    by 3 channels per offset (one per axis) which predict "are the voxels d/2 back and d/2 forward along
    this axis the same sheet", and `loss_affinity` weighs their BCE. Inference never reads them.
    cascade_self_p_anneal (START, END): linear scheduled-sampling anneal of `--cascade-self-p` over the run.
    sched / stable_until / cooldown: "cosine" (unchanged) or "wsd" (see `lr_lambda`).
    ema_k: `--ema auto`, ema_decay = 1 - ema_k / steps (see `ema_auto`).
    rewarm: on a warm start (`init_from`) warm the LR up over this many steps instead of `warmup`.
    new_param_lr_mult: LR multiplier of a second AdamW param group holding the tensors a warm start grew.
    fuse / source_w: teacher fusion and per-source loss weights (usrm2/data.py, `usrm2 glc-weights`).

    stream: a queue directory filled by `usrm2 stream-plan` (usrm2/stream.py). The loader then replays that
    queue out of a rolling local buffer instead of sampling, and every 20 steps `train.jsonl` carries
    `stream_wait_ms` (how long the workers waited for the planner, 0 once the buffer is ahead) and
    `stream_idx` (the highest queue index consumed, also written to <stream>/consumed for the planner's
    eviction bound)."""
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    main = rank == 0
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(A.get(aug), **({"norad": True} if no_radial else {}))
    no_radial = bool(cfg.get("norad"))
    patch = int(patch[0]) if not np.isscalar(patch) and len(patch) == 1 else (patch if np.isscalar(patch) else [int(v) for v in patch])
    args = dict(size=size, steps=steps, patch=patch, batch=batch, lr=lr, aug=aug, aug_cfg=cfg,
                no_radial=no_radial, accum=accum, ema_decay=ema_decay, lr_floor=lr_floor, ridge_w=ridge_w, wtgt=list(wtgt),
                dense_pow=dense_pow, norm=norm, ctx=list(ctx), world=world,
                stream=str(stream) if stream else None, **kw)
    cascade = str(cascade or "off")
    assert cascade in data.CASCADE_MODES, f"--cascade {cascade}: one of {data.CASCADE_MODES}"
    assert cascade == "off" or rungs is not None, "--cascade needs the rung ladder (--rungs)"
    if cascade != "off":
        args.update(cascade=cascade, cascade_self_p=float(cascade_self_p), cascade_drop=float(cascade_drop),
                    cascade_noise=bool(cascade_noise))
    verso = bool(verso)
    assert not verso or rungs is not None, "--verso needs the rung ladder (--rungs)"
    verso_regions = str(verso_regions) if verso_regions else (kw.get("teacher_regions") or None)
    if verso:  # only recorded when on, so an existing run's args -- and its resume check -- are untouched
        args.update(verso=True, verso_regions=str(verso_regions) if verso_regions else None,
                    verso_regions_url=str(verso_regions_url) if verso_regions_url else None)
        assert verso_regions, "--verso needs --teacher-regions (or --verso-regions): the verso targets are " \
                              "region stores, there is no verso pyramid"
        if verso_regions_url and not stream:  # only the planner downloads; train reads the directory
            main and print(f"--verso-regions-url {verso_regions_url} is recorded but NOT fetched by train: "
                           f"`usrm2 stream-plan --verso --verso-regions-url ...` downloads into "
                           f"{verso_regions}; this run only reads what is there", flush=True)
    # ---- Phase A (section 26). Recorded only when ON, so every older run's args -- and its resume
    # check -- are untouched and a run without these flags is byte-identical to one built before them.
    offsets = L.parse_offsets(affinity)
    naff = L.n_affinity(offsets)
    assert not naff or rungs is not None, "--affinity needs the rung ladder (--rungs)"
    assert not naff or loss_affinity > 0, "--affinity adds output channels: give it --loss-affinity W"
    phase_a = dict(loss_excl=float(loss_excl), loss_selfcons=float(loss_selfcons), loss_skel=float(loss_skel),
                   loss_affinity=float(loss_affinity))
    for k, v in list(phase_a.items()):
        if not v:
            phase_a.pop(k)
    if naff:
        phase_a.update(affinity=list(offsets), affinity_all=bool(affinity_all))
    if loss_skel:
        phase_a["skel_iters"] = int(skel_iters)
    if cascade_self_p_anneal:
        a0, a1 = (float(v) for v in cascade_self_p_anneal)
        phase_a["cascade_self_p_anneal"] = [a0, a1]
    if str(sched) != "cosine":
        phase_a.update(sched=str(sched), stable_until=(None if stable_until is None else int(stable_until)),
                       cooldown=int(cooldown or 0))
    if ema_k:
        phase_a["ema_k"] = float(ema_k)
        ema_decay = ema_auto(steps, ema_k)
        args["ema_decay"] = ema_decay
    if rewarm:
        phase_a["rewarm"] = int(rewarm)
    if new_param_lr_mult and float(new_param_lr_mult) != 1.0:
        phase_a["new_param_lr_mult"] = float(new_param_lr_mult)
    if str(fuse or "off") != "off":
        phase_a["fuse"] = str(fuse)
    if source_w:
        phase_a["source_w"] = dict(source_w)
    args.update(phase_a)
    if loss_excl and not verso:
        main and print("--loss-excl needs two probability channels (--verso); it will be inactive", flush=True)
    if norm == "global":
        args["norm_stats"] = data.global_norm(kw.get("ct", data.CT))
        main and print("global normalization", args["norm_stats"], flush=True)
    cout_arg, dev = cout, torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if rungs is not None:
        lines = data.Patches(patch=patch, stores=kw.get("stores", data.TRAIN), stores_file=kw.get("stores_file"),
                             rungs=rungs).paths
        lines = [",".join(q) for q in lines]
        args["rungs"], args["rung_boost"], args["val_rungs"] = rungs if rungs is True else list(rungs), dict(rung_boost or {}), list(val_rungs)
        args["scale_plane"] = True
        chans = list(dict.fromkeys(c for s in data.source_groups(lines) for c in s["targets"]))
        if verso and data.VERSO not in chans:  # channel 1 of the same head, not a second branch
            chans.append(data.VERSO)
        args["channels"] = chans
        grid = data.val_grid_rungs(patch, lines, kw.get("val", data.VAL), rungs=val_rungs, limit=val_patches,
                                   ctx=ctx, cascade=cascade, channels=chans, verso=verso,
                                   verso_regions=verso_regions)
    else:
        grid = data.val_grid(patch=patch, ct=kw.get("ct", data.CT), store=kw.get("val", data.VAL), limit=val_patches, ctx=ctx)
    assert grid, f"validation store {kw.get('val', data.VAL)} is smaller than the patch ({patch})"
    # CT + context cubes (+ scale plane) + radial vector; one output channel per target channel (+ verso)
    cin, cout_t = grid_cin_cout(grid[0])
    assert cout_arg is None or int(cout_arg) == cout_t, \
        f"--cout {cout_arg} but the output channels are {args.get('channels')} (cout {cout_t}): " \
        "--verso is what adds the verso output channel"
    # the AFFINITY channels sit AFTER the target channels in the same 1x1x1 head; `cout_t` is what the
    # loader, the loss, the evaluation and inference see, `cout` is the head width
    cout = cout_t + naff
    if naff:
        args["cout_t"], args["channels"] = cout_t, list(args.get("channels") or []) + L.affinity_names(offsets)
    args["cin"], args["cout"] = cin, cout
    net = M.build(size, cout=cout, cin=cin, ckpt_act=ckpt_act, add_skip=add_skip, deep=deep).to(dev)
    args["ckpt_act"], args["add_skip"], args["deep"] = ckpt_act, add_skip, deep
    newp = set()
    if init_from:  # warm start from another run's EMA weights; extra input channels get zero weights (same output at step 0)
        sst = torch.load(init_from, map_location=dev)
        src = warm_start(sst["ema"], cin, cout, cascade=cascade != "off",
                         src_scale=bool(sst.get("args", {}).get("scale_plane")),
                         ncopy=cout_t if naff else None)
        own = net.state_dict()
        skipped = [k for k, v in src.items() if k in own and tuple(own[k].shape) != tuple(v.shape)]
        src = {k: v for k, v in src.items() if k not in skipped}  # e.g. dec.0 under --add-skip, new deeper levels
        missing = net.load_state_dict(src, strict=False)
        main and print(f"warm start from {init_from}: {len(missing.missing_keys)} missing, {len(missing.unexpected_keys)} unexpected, "
                       f"{len(skipped)} shape-mismatched skipped", flush=True)
        args["init_from"] = str(init_from)
        sa = sst.get("args", {})
        newp = new_param_names(int(sa.get("cin", 0) or 0) < cin, int(sa.get("cout", 0) or 0) < cout, net)
    warm = int(rewarm) if (rewarm and init_from) else warmup  # a decayed tail is not where a warm start resumes
    groups, split = param_groups(net, newp, float(new_param_lr_mult or 1.0))
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.01)
    sched_name = str(sched or "cosine")
    base = lr_lambda(steps, warm, lr_floor, sched=sched_name, stable_until=stable_until, cooldown=cooldown)
    if split:  # the new rows carry no memory to protect: M x LR through the stable phase, then M = 1
        S = (int(stable_until) if stable_until is not None else
             max(int(steps) - (int(cooldown) if cooldown else max(int(round(0.1 * steps)), 1)), 1)) \
            if sched_name == "wsd" else int(steps)
        m = float(new_param_lr_mult)
        base = [base, (lambda s, f=base, m=m, S=S: f(s) * (m if s < S else 1.0))]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, base)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step = 0
    ck = out / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        # `verso` and `cout` are NOT here: they change the head, so they must match. `verso_regions` /
        # `verso_regions_url` are, like `teacher_regions`: WHERE the verso stores come from may change
        # between restarts (a different mirror, a planner that fetches them) without changing the model.
        grow = ("steps", "stores", "stores_file", "val", "val_rungs", "val_patches", "compile", "workers",
                "require_targets", "rung_boost", "eval_every", "continued_from", "ckpt_act",
                "stream", "teacher_regions", "region", "windows_per_region",
                "verso_regions", "verso_regions_url",
                # WSD's whole point is that the budget is NOT committed at run start: the plateau may be
                # extended and the cooldown moved on a resume, because neither is part of the weights.
                "sched", "stable_until", "cooldown") + \
               (("ema_decay",) if (ema_k or st["args"].get("ema_k")) else ())
        # a continued run may train longer, on more data, with other bookkeeping -- and from another queue
        diff = {k: (st["args"][k], args.get(k)) for k in st["args"] if k not in grow and k != "aug_cfg" and st["args"][k] != args.get(k)}
        assert not diff, f"resume with different arguments (saved, now): {diff}"
        if st["args"].get("stores") != args.get("stores"):
            print(f"resume: {len(st['args'].get('stores') or [])} -> {len(args.get('stores') or [])} stores", flush=True)
        args["continued_from"] = st["step"]
        net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        ema, step = {k: v.to(dev) for k, v in st["ema"].items()}, st["step"]
        for _ in range(step):
            sched.step()
    if no_radial:  # compact grid entries get their radial channels zeroed by prep.prepare(norad=True)
        for item in grid:
            if not isinstance(item, dict):
                item[0][-3:] = 0
    evnet = M.build(size, verbose=False, cout=cout, cin=cin, add_skip=add_skip, deep=deep).to(dev)
    # CASCADE: one Cascade builds the training channel (stochastic: mix / dropout / noise), the other the
    # validation one (deterministic: self when the run trains with self, mask otherwise, no noise, no drop).
    # The self mode runs its own copy of the net, kept on the EMA weights (`cas.sync`) -- never the DDP or
    # compiled module, and never with a grad path.
    cas = casval = None
    if cascade != "off":
        casnet = None
        if cascade in ("self", "mix"):
            casnet = M.build(size, verbose=False, cout=cout, cin=cin, add_skip=add_skip, deep=deep).to(dev)
            casnet.eval()
        cas = prep.Cascade(cascade, self_p=cascade_self_p, drop=cascade_drop, noise=cascade_noise, net=casnet)
        casval = prep.Cascade("self" if cascade in ("self", "mix") else "mask", self_p=1.0, drop=0.0,
                              noise=False, net=evnet)
    dl = data.loader(patch, batch, workers, ct=kw.get("ct", data.CT), stores=kw.get("stores", data.TRAIN),
                     exclude=kw.get("val", data.VAL), seed=step + 7919 * rank, sym=cfg.get("sym", True), aug=cfg, dense_pow=dense_pow, ctx=ctx,
                     stores_file=kw.get("stores_file"),  # a stores file is re-read as it grows (data.Patches)
                     # region mode / the region teacher stores reach the DATASET, not just the args record
                     **(dict(rungs=rungs, rung_boost=rung_boost,
                             channels=(args.get("channels") or [])[:cout_t] or None,
                             require_targets=require_targets, cascade=cascade,
                             verso=verso, verso_regions=verso_regions,
                             fuse=fuse, source_w=source_w,
                             **{q: kw[q] for q in ("region", "windows_per_region", "region_fails",
                                                   "teacher_regions") if kw.get(q)}) if rungs is not None else {}),
                     **(dict(stream=stream) if stream else {}))
    model = torch.nn.parallel.DistributedDataParallel(net, device_ids=[dev.index]) if world > 1 else net
    if compile:
        model = torch.compile(model)
        args["compile"] = True

    def save():  # atomic: an interrupted write never loses the last resumable state
        if not main:
            return
        torch.save({"model": net.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "step": step, "args": args}, ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").replace(ck)

    def log(name, rec):
        if not main:
            return
        with open(out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(name, rec, flush=True)

    log("train.jsonl", {"step": step, "aug": aug, "cfg": cfg, "size": size, "patch": patch, "batch": batch})
    t0, micro, rung_n = time.time(), 0, {}
    wait_ms, stream_idx = 0.0, -1
    aux_on = bool(loss_excl or loss_selfcons or loss_skel or loss_affinity)
    aux_dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    aux_log = {}
    # the planner may evict a chunk once every worker is past it; the DataLoader is up to this many entries
    # ahead of what the training loop has actually seen
    # (x world: every rank replays its own share of the one queue, and the bound must clear the slowest)
    margin = world * max(workers, 1) * batch * (2 + 2)
    for item in dl:
        if step >= steps:
            break
        wt = None
        if isinstance(item, dict):  # rung mode: uint8 cubes + metadata; the input is built on the device
            for r in item["rung"].tolist():
                rung_n[r] = rung_n.get(r, 0) + 1
            if "wait" in item:
                wait_ms += float(item["wait"].sum())
                stream_idx = max(stream_idx, int(item["idx"].max()))
            if cas is not None:
                cas.sync(ema)  # the self-mode coarse pass always runs on the current EMA weights
                if cascade_self_p_anneal:  # scheduled sampling: mostly `mask` early, mostly `self` late
                    a0, a1 = (float(v) for v in cascade_self_p_anneal)
                    cas.self_p = a0 + (a1 - a0) * min(step / max(steps, 1), 1.0)
            ct, tg, wt = prep.prepare(item, dev, norad=no_radial, cascade=cas)
            tg = torch.cat([tg, wt], 1)  # the weights ride along as extra target channels so every
        else:                            # geometric aug transforms them identically
            ct, tg = item[0].to(dev, non_blocking=True), item[1].to(dev, non_blocking=True)
        ct, tg = A.apply(ct, tg, cfg, nimg=(cin - 5) if cascade != "off" else None)
        if wt is not None:
            tg, wt = tg[:, :cout_t], tg[:, cout_t:]
        ct = ct.to(memory_format=M.memfmt())
        with autocast(dev):
            pred = model(ct)
            outs = [o.float() for o in pred] if isinstance(pred, (list, tuple)) else [pred.float()]
            bce, dice = deep_losses([o[:, :cout_t] for o in outs] if len(outs) > 1 else outs[0][:, :cout_t],
                                    tg, ridge_w, wtgt, wt)
        loss = bce + dice
        if aux_on:  # Phase A: every term is computed from tensors this step already holds
            t_a, w_a = weighted(tg, wtgt)
            w_a = wt if w_a is None else (w_a if wt is None else w_a * wt)
            w_a = torch.ones_like(t_a) if w_a is None else w_a
            # in the AUX dtype (bf16 on the card): the skeleton and the affinity targets are 0/1 fields
            # the size of the target, and at 256^3 fp32 they alone would be ~1.6 GB of temporaries
            cv = (lambda q: None if q is None else q.to(aux_dt))
            ax = L.aux_losses(cv(outs[0]), cv(t_a), cv(w_a),
                              w_excl=loss_excl, w_selfcons=loss_selfcons, w_skel=loss_skel,
                              w_affinity=loss_affinity, offsets=offsets, skel_iters=skel_iters,
                              aff_fg_only=not affinity_all,
                              cascade=(cv(ct[:, cin - 5:cin - 4]) if cascade != "off" else None),
                              cascade_self=(cas.last_self if cas is not None else None))
            if "aux" in ax:
                loss = loss + ax["aux"].float()
            aux_log = {k: float(v) for k, v in ax.items()}
        (loss / accum).backward()
        micro += 1
        if micro < accum:
            continue
        micro = 0
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(), opt.zero_grad(set_to_none=True), sched.step()
        ema_update(ema, net, ema_decay)
        step += 1
        if step % 20 == 0:
            dt = time.time() - t0
            log("train.jsonl", {"step": step, "loss": loss.item(), "bce": bce.item(), "dice": dice.item(),
                                **aux_log,
                                "lr": sched.get_last_lr()[0], "vox_s": round(20 * accum * batch * world * int(np.prod(data.shape3(patch))) / dt),
                                "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0,
                                **({"rung": {str(k): rung_n[k] for k in sorted(rung_n)}} if rung_n else {}),
                                **({"stream_wait_ms": round(wait_ms), "stream_idx": stream_idx} if stream else {})})
            rung_n, wait_ms = {}, 0.0
            if stream and main and stream_idx >= 0:  # the planner's eviction bound
                tmp = os.path.join(str(stream), "consumed.tmp")
                with open(tmp, "w") as f:
                    json.dump({"i": stream_idx, "margin": margin}, f)
                os.replace(tmp, os.path.join(str(stream), "consumed"))
            t0 = time.time()
        if (step % eval_every == 0 or step == steps) and main:
            evnet.load_state_dict(ema)
            log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev, wtgt, norad=no_radial, cascade=casval,
                                            channels=args.get("channels"), cout_t=cout_t)})
            try:
                val_png(out / f"val_{step:06d}.png", evnet, grid, dev, norad=no_radial, cascade=casval, cout_t=cout_t)
            except Exception as e:  # a missing PIL must not stop training
                print("val_png:", repr(e))
            save()
            t0 = time.time()
    else:  # the loader ran out: a streamed walk finished its epoch (nothing is ever trained on twice)
        if stream:
            log("train.jsonl", {"step": step, "stream_end": stream_idx,
                                **({"epoch_done": json.load(open(os.path.join(str(stream), "epoch_done")))}
                                   if os.path.exists(os.path.join(str(stream), "epoch_done")) else {})})
            if main:
                evnet.load_state_dict(ema)
                log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev, wtgt, norad=no_radial, cascade=casval,
                                            channels=args.get("channels"), cout_t=cout_t)})
    save()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return ck
