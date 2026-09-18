"""Distillation training: BCE + soft dice against the teacher's soft probabilities."""
import json
import math

import numpy as np
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from usrm2 import aug as A, data, model as M


def losses(logit, tgt, ridge_w=0.0):
    """BCE (+ ridge_w extra weight on the band's core, target >= 0.9: the student must commit there) + soft dice."""
    if ridge_w > 0:
        w = 1 + ridge_w * (tgt >= 0.9).float()
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
def evaluate(net, grid, dev):
    net.eval()
    m = torch.zeros(3)
    for ct, tg in grid:
        ct, tg = ct[None].to(dev).to(memory_format=torch.channels_last_3d), tg[None].to(dev)
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
          ema_decay=0.999, lr_floor=0.0, ridge_w=0.0, dense_pow=0.0, norm="patch", **kw):
    """accum: gradient accumulation (micro-batches per optimizer step), for big models on small cards.
    lr_floor: the cosine decays to lr_floor * lr instead of 0. norm: "patch" (per-patch z-score) or "global"
    (fixed scan mean/std, stored in the checkpoint). dense_pow / ridge_w: see data.Patches / losses."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(A.get(aug), **({"norad": True} if no_radial else {}))
    no_radial = bool(cfg.get("norad"))
    args = dict(size=size, steps=steps, patch=patch, batch=batch, lr=lr, aug=aug, aug_cfg=cfg,
                no_radial=no_radial, accum=accum, ema_decay=ema_decay, lr_floor=lr_floor, ridge_w=ridge_w,
                dense_pow=dense_pow, norm=norm, **kw)
    if norm == "global":
        args["norm_stats"] = data.global_norm(kw.get("ct", data.CT))
        print("global normalization", args["norm_stats"], flush=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    grid = data.val_grid(patch=patch, ct=kw.get("ct", data.CT), store=kw.get("val", data.VAL), limit=val_patches)
    assert grid, f"validation store {kw.get('val', data.VAL)} is smaller than the patch ({patch})"
    args["cout"] = cout = grid[0][1].shape[0]  # one head per teacher store
    net = M.build(size, cout=cout).to(dev)
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
            x[1:] = 0
    evnet = M.build(size, verbose=False, cout=cout).to(dev)
    dl = data.loader(patch, batch, workers, ct=kw.get("ct", data.CT), stores=kw.get("stores", data.TRAIN),
                     exclude=kw.get("val", data.VAL), seed=step, sym=cfg.get("sym", True), aug=cfg, dense_pow=dense_pow)

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
            bce, dice = losses(net(ct).float(), tg, ridge_w)
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
            log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev)})
            try:
                val_png(out / f"val_{step:06d}.png", evnet, grid, dev)
            except Exception as e:  # a missing PIL must not stop training
                print("val_png:", repr(e))
            save()
            t0 = time.time()
    save()
    return ck
