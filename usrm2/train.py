"""Distillation training: BCE + soft dice against the teacher's soft probabilities."""
import json
import math

import numpy as np
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from usrm2 import aug as A, data, model as M


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


def losses(logit, tgt, ridge_w=0.0, wtgt=()):
    """BCE (+ ridge_w extra weight on the band's core, target >= 0.9: the student must commit there) + soft dice.
    wtgt: channels whose stored value is a loss weight (verso targets, see `weighted`)."""
    tgt, wv = weighted(tgt, wtgt)
    if ridge_w > 0 or wv is not None:
        w = 1 + ridge_w * (tgt >= 0.9).float()
        if wv is not None:
            w = w * wv
        bce = (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum() / w.sum()
    else:
        bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p, d = torch.sigmoid(logit), (0, 2, 3, 4)
    dice = (1 - (2 * (p * tgt).sum(d) + 1) / (p.sum(d) + tgt.sum(d) + 1)).mean()  # per head
    return bce, dice


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for k, v in model.state_dict().items():
        e = ema[k]
        e.mul_(decay).add_(v.detach(), alpha=1 - decay) if e.is_floating_point() else e.copy_(v)


def autocast(dev):
    import contextlib
    return torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(net, grid, dev, wtgt=()):
    net.eval()
    m = torch.zeros(3)
    for ct, tg in grid:
        ct, tg = ct[None].to(dev).to(memory_format=torch.channels_last_3d), tg[None].to(dev)
        tg = weighted(tg, wtgt)[0]
        with autocast(dev):
            logit = net(ct).float()
        p = torch.sigmoid(logit)
        h, t = (p >= 0.5).float(), (tg >= 0.5).float()
        m += torch.tensor([F.binary_cross_entropy_with_logits(logit, tg).item(),
                           (2 * (h * t).sum() / (h.sum() + t.sum() + 1)).item(),
                           (p - tg).abs().mean().item()])
    net.train()
    m /= max(len(grid), 1)
    return {"bce": m[0].item(), "dice": m[1].item(), "mae": m[2].item()}


def val_png(path, net, grid, dev):
    """Middle z-slice of the first 4 val patches: CT (gray), each teacher target and each student head as a red
    opacity overlay (no threshold), tiled patches x [CT, targets..., heads...]."""
    from PIL import Image
    rows = []
    with torch.no_grad():
        for x, t in grid[:4]:
            with autocast(dev):
                p = torch.sigmoid(net(x[None].to(dev).to(memory_format=torch.channels_last_3d)).float())[0].cpu()
            z = x.shape[1] // 2
            c = x[0, z].numpy()
            c = (c - c.min()) / (c.max() - c.min() + 1e-6) * 255 * 0.9
            tiles = [np.repeat(c[..., None], 3, -1)]
            for a in list(t[:, z].numpy()) + list(p[:, z].numpy()):
                al = np.clip(a, 0, 1)[..., None] * 0.85
                tiles.append(np.repeat(c[..., None], 3, -1) * (1 - al) + np.array([255, 40, 40]) * al)
            rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0).astype(np.uint8)).save(path)


def train(out_dir, size="1m", steps=20000, patch=128, batch=1, lr=3e-4, workers=4, warmup=200,
          eval_every=500, val_patches=32, resume=False, device=None, aug="geo", no_radial=False, accum=1,
          ema_decay=0.999, lr_floor=0.0, ridge_w=0.0, dense_pow=0.0, norm="patch", ctx=(), init_from=None, wtgt=(), **kw):
    """accum: gradient accumulation (micro-batches per optimizer step), for big models on small cards.
    lr_floor: the cosine decays to lr_floor * lr instead of 0. norm: "patch" (per-patch z-score) or "global"
    (fixed scan mean/std, stored in the checkpoint). dense_pow / ridge_w: see data.Patches / losses.
    wtgt: target channels that carry loss weights (verso targets from verso.py), e.g. (2, 3) for a 4-head student."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(A.get(aug), **({"norad": True} if no_radial else {}))
    no_radial = bool(cfg.get("norad"))
    args = dict(size=size, steps=steps, patch=patch, batch=batch, lr=lr, aug=aug, aug_cfg=cfg,
                no_radial=no_radial, accum=accum, ema_decay=ema_decay, lr_floor=lr_floor, ridge_w=ridge_w, wtgt=list(wtgt),
                dense_pow=dense_pow, norm=norm, ctx=list(ctx), **kw)
    if norm == "global":
        args["norm_stats"] = data.global_norm(kw.get("ct", data.CT))
        print("global normalization", args["norm_stats"], flush=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    grid = data.val_grid(patch=patch, ct=kw.get("ct", data.CT), store=kw.get("val", data.VAL), limit=val_patches, ctx=ctx)
    assert grid, f"validation store {kw.get('val', data.VAL)} is smaller than the patch ({patch})"
    args["cout"] = cout = grid[0][1].shape[0]  # one head per teacher store
    args["cin"] = cin = grid[0][0].shape[0]  # CT + context cubes + radial vector
    net = M.build(size, cout=cout, cin=cin).to(dev)
    if init_from:  # warm start from another run's EMA weights; extra input channels get zero weights (same output at step 0)
        src = torch.load(init_from, map_location=dev)["ema"]
        w = src["enc.0.0.weight"]
        if w.shape[1] != cin:
            assert w.shape[1] < cin, "cannot drop input channels on a warm start"
            w2 = torch.zeros(w.shape[0], cin, *w.shape[2:], device=w.device, dtype=w.dtype)
            w2[:, :w.shape[1] - 3], w2[:, cin - 3:] = w[:, :w.shape[1] - 3], w[:, w.shape[1] - 3:]  # image chans first, radial last
            src["enc.0.0.weight"] = w2
        hw, hb = src["head.weight"], src["head.bias"]
        if hw.shape[0] != cout:  # new heads start as copies of the source heads (head j <- source head j mod n)
            assert hw.shape[0] < cout, "cannot drop heads on a warm start"
            j = torch.arange(cout, device=hw.device) % hw.shape[0]
            src["head.weight"], src["head.bias"] = hw[j].clone(), hb[j].clone()
        missing = net.load_state_dict(src, strict=False)
        print(f"warm start from {init_from}: {len(missing.missing_keys)} missing, {len(missing.unexpected_keys)} unexpected", flush=True)
        args["init_from"] = str(init_from)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0) *
                                              (lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0)))))
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step = 0
    ck = out / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        grow = ("steps", "stores", "val")  # a continued run may train longer and on more data
        diff = {k: (st["args"][k], args.get(k)) for k in st["args"] if k not in grow and k != "aug_cfg" and st["args"][k] != args.get(k)}
        assert not diff, f"resume with different arguments (saved, now): {diff}"
        if st["args"].get("stores") != args.get("stores"):
            print(f"resume: {len(st['args'].get('stores') or [])} -> {len(args.get('stores') or [])} stores", flush=True)
        args["continued_from"] = st["step"]
        net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        ema, step = {k: v.to(dev) for k, v in st["ema"].items()}, st["step"]
        for _ in range(step):
            sched.step()
    if no_radial:
        for x, _ in grid:
            x[-3:] = 0
    evnet = M.build(size, verbose=False, cout=cout, cin=cin).to(dev)
    dl = data.loader(patch, batch, workers, ct=kw.get("ct", data.CT), stores=kw.get("stores", data.TRAIN),
                     exclude=kw.get("val", data.VAL), seed=step, sym=cfg.get("sym", True), aug=cfg, dense_pow=dense_pow, ctx=ctx)

    def save():  # atomic: an interrupted write never loses the last resumable state
        torch.save({"model": net.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "step": step, "args": args}, ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").replace(ck)

    def log(name, rec):
        with open(out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(name, rec, flush=True)

    log("train.jsonl", {"step": step, "aug": aug, "cfg": cfg, "size": size, "patch": patch, "batch": batch})
    t0, micro = time.time(), 0
    for ct, tg in dl:
        if step >= steps:
            break
        ct, tg = A.apply(ct.to(dev, non_blocking=True), tg.to(dev, non_blocking=True), cfg)
        ct = ct.to(memory_format=torch.channels_last_3d)
        with autocast(dev):
            bce, dice = losses(net(ct).float(), tg, ridge_w, wtgt)
        loss = bce + dice
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
                                "lr": sched.get_last_lr()[0], "vox_s": round(20 * accum * batch * patch ** 3 / dt),
                                "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0})
            t0 = time.time()
        if step % eval_every == 0 or step == steps:
            evnet.load_state_dict(ema)
            log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev, wtgt)})
            try:
                val_png(out / f"val_{step:06d}.png", evnet, grid, dev)
            except Exception as e:  # a missing PIL must not stop training
                print("val_png:", repr(e))
            save()
            t0 = time.time()
    save()
    return ck
