"""In-domain masked-cube pretraining (MAE-style) for the unified model's encoder/decoder.

docs/unified_design.md section 28; docs/research/synthesis_v2_with_literature.md item R10 / experiment 11;
docs/research/lit_pretraining_foundation.md.

The objective: take the SAME sample the rung loader builds for `train` (docs/unified_design.md sections 2
and 12 -- CT cube + 9 context cubes + the cascade slot + the scale plane + the radial vector), blank out a
large fraction of the CT cube, and ask the SAME trunk to put the z-scored CT back. No labels are read, so
this runs anywhere a CT pyramid is mirrored.

What makes the weights reusable is that nothing about the trunk changes: `usrm2.model.build` is called with
the same `size`, `cin`, `ckpt_act`, `add_skip` and `deep` a fine-tuning run would use, and only the 1x1
OUTPUT head is repurposed. In the checkpoint that head is stored under `recon_head.*` (and, with deep
supervision, `recon_deep_heads.*`) instead of `head.*` / `deep_heads.*`, so `train --init`'s
`load_state_dict(..., strict=False)` sees it as an unexpected key and drops it: the segmentation head starts
random, the trunk starts pretrained. `tests/test_pretrain.py` asserts exactly that.

Why the context cubes are masked too: context cube j sits at rung k + j over the same centre, so its central
2^-j box is a 2^j-times coarser copy of the CT cube. Left alone it is a free low-frequency answer key. The
same voxel mask is therefore pooled down and pasted into each context channel's central footprint, so the
model has to invent the texture rather than upsample it (`--no-mask-ctx` turns that off).
"""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from usrm2 import aug as A, data, model as M, prep

# The checkpoint stores the reconstruction head under these names, so a fine-tuning warm start drops it
# instead of loading a reconstruction head into the segmentation head. Everything else -- enc.*, down.*,
# dec.*, proj.* -- keeps the state-dict keys `train` expects, byte for byte.
HEAD_RENAME = (("head.", "recon_head."), ("deep_heads.", "recon_deep_heads."))


def rename_out(sd):
    """UNet state dict -> checkpoint keys (the output head becomes `recon_head.*`)."""
    out = {}
    for k, v in sd.items():
        for a, b in HEAD_RENAME:
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    return out


def rename_in(sd):
    """Checkpoint keys -> UNet state dict (the inverse of `rename_out`, for a resume)."""
    out = {}
    for k, v in sd.items():
        for a, b in HEAD_RENAME:
            if k.startswith(b):
                k = a + k[len(b):]
                break
        out[k] = v
    return out


# ------------------------------------------------------------------------------- the masking

def quantile_(x, q, cap=1 << 20):
    """Per-sample quantile of (B,1,Z,Y,X), computed on a strided subsample so a 256^3 cube does not hit
    `torch.quantile`'s 16M-element limit (and costs a sort of 1M values, not 16M)."""
    f = x.reshape(x.shape[0], -1)
    if f.shape[1] > cap:
        f = f[:, :: max(f.shape[1] // cap, 1)]
    return torch.quantile(f.float(), float(q), dim=1)


def block_mask(ct, block=32, lo=0.5, hi=0.75, sheet_p=0.5, pct=0.7, gen=None):
    """A (B,1,Z,Y,X) float mask, 1 where the CT voxel is MASKED, built out of `block`^3 blocks.

    Per sample a masking ratio r is drawn uniformly from [lo, hi] and round(r * nblocks) blocks are hidden.
    With probability `sheet_p` the blocks are drawn STRUCTURE-AWARE: the block is sampled with probability
    proportional to its foreground fraction, foreground being "z-scored CT above the `pct` quantile of this
    cube" -- a one-line proxy for "on a sheet". Sheet-heavy blocks are then what disappears, so the model
    has to reconstruct sheet texture and cannot get a good score by interpolating through air. Otherwise the
    blocks are drawn uniformly (plain MAE). Returns (mask, ratios, sheet) -- `ratios` is the ACHIEVED masked
    fraction of blocks per sample and `sheet` the per-sample bool, which is what the tests check."""
    B, S = ct.shape[0], ct.shape[2:]
    g = [max(int(math.ceil(int(s) / block)), 1) for s in S]
    n = int(np.prod(g))
    thr = quantile_(ct, pct, ).view(B, 1, 1, 1, 1)
    fg = F.avg_pool3d((ct > thr).to(ct.dtype), block, stride=block, ceil_mode=True).reshape(B, n)
    r = torch.rand(B, generator=gen).to(ct.device) * (hi - lo) + lo
    sheet = torch.rand(B, generator=gen).to(ct.device) < sheet_p
    keep = torch.zeros(B, n, device=ct.device, dtype=ct.dtype)
    ratios = []
    for i in range(B):
        k = max(int(round(float(r[i]) * n)), 1)
        w = (fg[i] + 1e-3) if bool(sheet[i]) else torch.ones(n, device=ct.device, dtype=ct.dtype)
        idx = torch.multinomial(w.float(), k, replacement=False,
                                generator=(gen if gen is not None and gen.device == w.device else None))
        keep[i, idx] = 1.0
        ratios.append(k / n)
    m = keep.view(B, 1, *g)
    for d in range(3):  # blocks -> voxels, then crop the ceil_mode overhang
        m = m.repeat_interleave(block, dim=2 + d)
    m = m[:, :, : int(S[0]), : int(S[1]), : int(S[2])].contiguous()
    return m, torch.tensor(ratios), sheet


def mask_ctx_(x, m, ctx):
    """Paste the CT mask into the context channels, at each one's own scale, in place.

    Context channel j (x[:, 1 + j]) is the cube at rung k + ctx[j]: same voxel count, 2^ctx[j] times coarser,
    same centre. The CT cube's footprint inside it is therefore the CENTRAL box of side S / 2^ctx[j], and the
    mask pooled down by that factor (max pool: a coarse voxel that sees any masked fine voxel is masked) is
    what has to be blanked there. Past the point where the footprint is under one voxel there is nothing
    left to leak and the channel is left alone."""
    S = [int(v) for v in m.shape[2:]]
    for j, off in enumerate(ctx):
        f = 1 << int(off)
        sz = [s // f for s in S]
        if min(sz) < 1:
            break
        mj = F.max_pool3d(m, f)
        o = [(s - q) // 2 for s, q in zip(S, sz)]
        sl = (slice(None), slice(1 + j, 2 + j)) + tuple(slice(a, a + q) for a, q in zip(o, sz))
        x[sl] = x[sl] * (1 - mj)
    return x


def mask_input(x, ctx=(), nimg=None, block=32, lo=0.5, hi=0.75, sheet_p=0.5, pct=0.7, mask_ctx=True, gen=None):
    """(x, target, mask): blank the CT channel (and, with `mask_ctx`, the context channels' footprint).

    `x` is the finished model input from `prep.prepare` (+ `aug.apply`): channel 0 the z-scored CT, channels
    1..nimg-1 the context cubes, then the cascade slot / scale plane / radial vector. The target is the CT
    channel BEFORE masking; masked voxels are set to 0, which is the mean of the z-scored cube (the "no
    information" value, exactly what a dropped cascade channel is)."""
    tgt = x[:, :1].clone()
    m, ratios, sheet = block_mask(tgt, block, lo, hi, sheet_p, pct, gen)
    x[:, :1] = x[:, :1] * (1 - m)
    if mask_ctx and ctx:
        nc = (int(nimg) - 1) if nimg else len(ctx)
        mask_ctx_(x, m, tuple(ctx)[:nc])
    return x, tgt, m, ratios, sheet


def recon_loss(pred, tgt, m, kind="l1"):
    """Reconstruction loss on the MASKED voxels only (an unmasked voxel is a copy, not a prediction)."""
    d = (pred - tgt).abs() if kind == "l1" else (pred - tgt) ** 2
    return (d * m).sum() / m.sum().clamp_min(1.0)


# ------------------------------------------------------- the small pieces copied from train.py
# Deliberately copied, not imported: usrm2/train.py is being edited in parallel and this stage must not
# break when its internals move. These are ~15 lines and have been stable since the first commit.

@torch.no_grad()
def ema_update(ema, model, decay=0.999):
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


# ------------------------------------------------------------------------------- rungs and sources

def available_rungs(lines, want, quiet=False, label_free=False):
    """(usable lines, usable rungs): the requested rungs intersected with what the sources actually have.

    A source's usable rungs start at its NATIVE rung (`data.usable_rungs`), so rungs 0 and 1 (0.6 and 1.2 um)
    exist only for a source built from a fine scan -- the 2.4 um mirrors start at rung 2 and can never serve
    them. `--rungs 0-4` on a corpus of 2.4 um scrolls is therefore silently rungs 2-4, and a source with no
    requested rung at all is dropped rather than left to trip `data.rung_probs`' assertion."""
    allowed = None if want is True else {int(k) for k in want}
    keep, have = [], set()
    for line in lines:
        s = data.source_groups([line], label_free=label_free)[0]
        ks = data.usable_rungs(s, allowed)
        if ks:
            keep.append(line)
            have.update(ks)
        elif not quiet:
            print(f"pretrain: {s['line']} has no requested rung (native {s['native']}), dropped", flush=True)
    assert keep, f"no source provides any of the requested rungs {sorted(allowed) if allowed else 'all'}"
    if allowed and not quiet:
        miss = sorted(allowed - have)
        if miss:
            print(f"pretrain: rungs {miss} are not on disk anywhere (no scan is native there); "
                  f"training at {sorted(have)}", flush=True)
    return keep, sorted(have)


# ------------------------------------------------------------------------------------ the run

def val_png(path, x, tgt, m, pred):
    """CT | masked input | reconstruction, middle z slice, one row per sample."""
    from PIL import Image
    rows = []
    for i in range(min(x.shape[0], 4)):
        z = x.shape[2] // 2
        tiles = [tgt[i, 0, z], (tgt[i, 0, z] * (1 - m[i, 0, z])), pred[i, 0, z]]
        tiles = [t.float().cpu().numpy() for t in tiles]
        lo, hi = float(np.min(tiles[0])), float(np.max(tiles[0]))
        rows.append(np.concatenate([np.clip((t - lo) / (hi - lo + 1e-6), 0, 1) * 255 for t in tiles], 1))
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)


def pretrain(out_dir, size="1m", steps=20000, patch=128, batch=1, lr=3e-4, workers=4, warmup=200,
             eval_every=500, val_patches=8, resume=False, device=None, aug="geo", no_radial=False, accum=1,
             ema_decay=0.999, lr_floor=0.0, norm="patch", ctx=(), compile=False, ckpt_act=0, add_skip=0,
             deep=0, rungs=(0, 1, 2, 3, 4), mask_block=32, mask_lo=0.5, mask_hi=0.75, sheet_p=0.5,
             sheet_pct=0.7, mask_ctx=True, loss="l1", cascade_slot=True, rung_aux=0.0, rung_aux_p=0.5,
             fg_min=0.0, air_keep=0.1, label_free=False, stream=None, stream_tag=None, **kw):
    """Masked-cube pretraining of the trunk `train` uses. Writes `<out_dir>/ckpt.pt` with `args`, `ema` and
    `model`, loadable by `usrm2 train --init <out_dir>/ckpt.pt`.

    rungs: which rungs to sample (default the fine half of the ladder, 0-4: that is where the texture the
    fine-tuning run must model lives, and it is where R10 predicts the gain). Rungs with no native source
    are dropped with a message (`available_rungs`).
    mask_block / mask_lo / mask_hi: the masking grid and the ratio drawn per sample (0.5-0.75 by default).
    sheet_p: probability a sample is masked STRUCTURE-AWARE (blocks drawn proportional to their foreground
    fraction, foreground = CT above the `sheet_pct` quantile) instead of uniformly.
    cascade_slot: build the 15-channel input of a `--cascade` run, with the cascade channel held at zero --
    which is exactly the in-distribution "no coarse prediction" value. A 14-channel fine-tuning run should
    pretrain with `--no-cascade-slot`: `train.warm_start` can widen a stem, never narrow one.
    rung_aux: weight of a VoCo-flavoured auxiliary head that predicts the rung index from the bottleneck
    (mean-pooled) with a cross entropy. Off by default, and for a reason: the scale plane HANDS the model
    the rung, so the task is trivial unless that plane is hidden. When it is on, each step zeroes the scale
    plane and scores the aux loss with probability `rung_aux_p` (the other steps are plain reconstruction
    with the plane intact). The head lives OUTSIDE the model state dict (`ckpt["rung_head"]`), so it can
    never reach a fine-tuning warm start.
    Everything else -- EMA, bf16 autocast, `--compile`, activation checkpointing, resume, the atomic save --
    follows train.py."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(A.get(aug), **({"norad": True} if no_radial else {}))
    no_radial = bool(cfg.get("norad"))
    patch = int(patch[0]) if not np.isscalar(patch) and len(patch) == 1 else (patch if np.isscalar(patch) else [int(v) for v in patch])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ctx = tuple(int(v) for v in ctx)

    lines = kw.get("stores") or []
    if kw.get("stores_file"):
        lines = [l.strip() for l in open(kw["stores_file"]) if l.strip() and not l.startswith("#")]
    lines = [",".join(q) if isinstance(q, (list, tuple)) else str(q) for q in lines]
    assert lines, "pretrain needs --stores or --stores-file (CT pyramids; the target group is only used to " \
                  "bound the sampling box, no labels are read)"
    # LABEL-FREE sampling (section 30): the window is drawn anywhere the CT is not air rather than inside
    # the first target group's box, and a line may be a bare `ct_base` with no target group at all. That
    # is the sampling R10 actually wants -- the pretraining corpus is the mirrored CT, not the labelled
    # part of it -- and it is what `--label-free` switches on. Off, everything is as it was.
    lines, ks = available_rungs(lines, True if rungs is True else set(rungs), label_free=label_free)

    args = dict(size=size, steps=steps, patch=patch, batch=batch, lr=lr, aug=aug, aug_cfg=cfg,
                no_radial=no_radial, accum=accum, ema_decay=ema_decay, lr_floor=lr_floor, norm=norm,
                ctx=list(ctx), rungs=list(ks), scale_plane=True, pretrain=True,
                mask_block=mask_block, mask_lo=mask_lo, mask_hi=mask_hi, sheet_p=sheet_p,
                sheet_pct=sheet_pct, mask_ctx=bool(mask_ctx), loss=loss,
                cascade_slot=bool(cascade_slot), rung_aux=float(rung_aux),
                **({"label_free": True} if label_free else {}),
                **({"stream": str(stream)} if stream else {}), **kw)
    if norm == "global":
        args["norm_stats"] = data.global_norm(lines[0].split(",")[0])

    # CT + context cubes (+ the zero cascade slot) + scale plane + radial vector -- the SAME `cin` a
    # fine-tuning run will build, which is what makes the stem weights transfer without a reshape.
    cin = 1 + len(ctx) + (1 if cascade_slot else 0) + 4
    args["cin"], args["cout"] = cin, 1
    args["ckpt_act"], args["add_skip"], args["deep"] = ckpt_act, add_skip, deep
    if cascade_slot:
        args["cascade"] = "off"  # the slot exists and is held at zero; no coarse prediction is ever read
    net = M.build(size, cout=1, cin=cin, ckpt_act=ckpt_act, add_skip=add_skip, deep=deep).to(dev)
    nimg = 1 + len(ctx)

    aux = None
    if rung_aux > 0:
        w = M.PRESETS[size][-1]
        aux = torch.nn.Linear(w, data.NRUNGS).to(dev)
        feats = {}
        net.enc[-1].register_forward_hook(lambda mod, i, o: feats.__setitem__("b", o))

    params = list(net.parameters()) + (list(aux.parameters()) if aux is not None else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0) *
                                              (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step = 0
    ck = out / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        grow = ("steps", "stores", "stores_file", "val", "val_patches", "compile", "workers", "eval_every",
                "continued_from")
        diff = {k: (st["args"][k], args.get(k)) for k in st["args"]
                if k not in grow and k != "aug_cfg" and st["args"][k] != args.get(k)}
        assert not diff, f"resume with different arguments (saved, now): {diff}"
        args["continued_from"] = st["step"]
        net.load_state_dict(rename_in(st["model"])), opt.load_state_dict(st["opt"])
        ema, step = {k: v.to(dev) for k, v in rename_in(st["ema"]).items()}, st["step"]
        if aux is not None and st.get("rung_head"):
            aux.load_state_dict({k: v.to(dev) for k, v in st["rung_head"].items()})
        for _ in range(step):
            sched.step()

    grid = data.val_grid_rungs(patch, lines, kw.get("val", data.VAL), rungs=ks, limit=val_patches, ctx=ctx,
                               label_free=bool(label_free))
    evnet = M.build(size, verbose=False, cout=1, cin=cin, add_skip=add_skip, deep=deep).to(dev)
    dl = data.loader(patch, batch, workers, stores=lines, exclude=kw.get("val", data.VAL),
                     seed=step + 7919, sym=cfg.get("sym", True), aug=cfg, ctx=ctx, rungs=set(ks),
                     air_keep=air_keep, fg_min=fg_min, fg_keep=1.0, require_targets=False,
                     label_free=bool(label_free),
                     **(dict(stream=stream, stream_tag=stream_tag) if stream else {}))
    model = torch.compile(net) if compile else net
    if compile:
        args["compile"] = True

    def save():
        torch.save({"model": rename_out(net.state_dict()), "ema": rename_out(ema), "opt": opt.state_dict(),
                    "step": step, "args": args,
                    **({"rung_head": aux.state_dict()} if aux is not None else {})}, ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").replace(ck)

    def log(name, rec):
        with open(out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(name, rec, flush=True)

    def build_x(item):
        """A collated loader batch -> the masked model input, the target and the mask."""
        x = prep.prepare(item, dev, norad=no_radial)[0]
        t0 = torch.zeros((x.shape[0], 1) + tuple(x.shape[2:]), device=dev, dtype=x.dtype)
        x, _ = A.apply(x, t0, cfg, nimg=nimg)
        if cascade_slot:  # [CT, ctx..., CASCADE(0), scale, radial] -- the 15-channel stem, slot held at zero
            x = torch.cat([x[:, :nimg], torch.zeros_like(x[:, :1]), x[:, nimg:]], 1)
        return x

    log("pretrain.jsonl", {"step": step, "size": size, "patch": patch, "batch": batch, "rungs": ks,
                           "cin": cin, "mask": [mask_lo, mask_hi], "block": mask_block, "sheet_p": sheet_p,
                           "sources": len(lines)})
    t0, micro, rung_n = time.time(), 0, {}
    for item in dl:
        if step >= steps:
            break
        for r in item["rung"].tolist():
            rung_n[r] = rung_n.get(r, 0) + 1
        x = build_x(item)
        aux_on = aux is not None and float(torch.rand(())) < rung_aux_p
        if aux_on:
            x[:, nimg + (1 if cascade_slot else 0)] = 0  # hide the scale plane: the aux task must be earned
        x, tgt, m, ratios, _ = mask_input(x, ctx, nimg, mask_block, mask_lo, mask_hi, sheet_p, sheet_pct,
                                          mask_ctx)
        x = x.to(memory_format=M.memfmt())
        with autocast(dev):
            pred = model(x)
            pred = pred[0] if isinstance(pred, (list, tuple)) else pred
            rl = recon_loss(pred.float(), tgt, m, loss)
            al = torch.zeros((), device=dev)
            if aux_on:
                b = feats["b"].float().mean((2, 3, 4))
                al = F.cross_entropy(aux(b), item["rung"].reshape(-1).to(dev).long())
        total = rl + rung_aux * al
        (total / accum).backward()
        micro += 1
        if micro < accum:
            continue
        micro = 0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(), opt.zero_grad(set_to_none=True), sched.step()
        ema_update(ema, net, ema_decay)
        step += 1
        if step % 20 == 0:
            dt = time.time() - t0
            log("pretrain.jsonl", {"step": step, "loss": float(total), "recon": float(rl),
                                   "ratio": round(float(ratios.mean()), 3), "lr": sched.get_last_lr()[0],
                                   "vox_s": round(20 * accum * batch * int(np.prod(data.shape3(patch))) / dt),
                                   "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0,
                                   **({"aux": float(al)} if aux is not None else {}),
                                   **({"rung": {str(k): rung_n[k] for k in sorted(rung_n)}} if rung_n else {})})
            rung_n = {}
            t0 = time.time()
        if step % eval_every == 0 or step == steps:
            evnet.load_state_dict(ema)
            log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev, cfg, nimg, ctx, cascade_slot,
                                                        no_radial, args, png=out / f"val_{step:06d}.png")})
            save()
            t0 = time.time()
    save()
    return ck


@torch.no_grad()
def evaluate(net, grid, dev, cfg, nimg, ctx, cascade_slot, no_radial, args, png=None):
    """Masked L1/L2 on the held-out box, with a FIXED mask (seed 0): the number moves because the model
    improved, not because a different set of blocks was hidden. No augmentation here either."""
    net.eval()
    tot, n, per = 0.0, 0, {}
    shown = None
    for item in grid:
        x = prep.prepare(prep.batch1(item), dev, norad=no_radial)[0]
        if cascade_slot:
            x = torch.cat([x[:, :nimg], torch.zeros_like(x[:, :1]), x[:, nimg:]], 1)
        g = torch.Generator().manual_seed(1234 + int(item["rung"]))
        x, tgt, m, _, _ = mask_input(x, ctx, nimg, args["mask_block"], args["mask_lo"], args["mask_hi"],
                                     args["sheet_p"], args["sheet_pct"], args["mask_ctx"], gen=g)
        with autocast(dev):
            p = net(x.to(memory_format=M.memfmt()))
        p = (p[0] if isinstance(p, (list, tuple)) else p).float()
        v = float(recon_loss(p, tgt, m, args["loss"]))
        tot, n = tot + v, n + 1
        per.setdefault(int(item["rung"]), []).append(v)
        shown = shown or (x, tgt, m, p)
    net.train()
    out = {"recon": tot / max(n, 1)}
    for k in sorted(per):
        out[f"recon_r{k}"] = float(np.mean(per[k]))
    if png is not None and shown is not None:
        try:
            val_png(png, *shown)
        except Exception as e:  # noqa: BLE001  a missing PIL must not stop pretraining
            print("val_png:", repr(e))
    return out
