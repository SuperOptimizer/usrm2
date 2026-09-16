"""Distillation training: BCE + soft dice against the teacher's soft probabilities."""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from usrm2 import data, model as M


def losses(logit, tgt):
    bce = F.binary_cross_entropy_with_logits(logit, tgt)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * tgt).sum() + 1) / (p.sum() + tgt.sum() + 1)
    return bce, dice


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for k, v in model.state_dict().items():
        e = ema[k]
        e.mul_(decay).add_(v.detach(), alpha=1 - decay) if e.is_floating_point() else e.copy_(v)


def autocast(dev):
    return torch.autocast("cuda", torch.bfloat16) if dev.type == "cuda" else torch.autocast("cpu", torch.bfloat16)


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


def train(out_dir, size="1m", steps=20000, patch=128, batch=1, lr=3e-4, workers=4, warmup=200,
          eval_every=500, val_patches=32, resume=False, device=None, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = dict(size=size, steps=steps, patch=patch, batch=batch, lr=lr, **kw)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = M.build(size).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 1.0) *
                                              0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    step = 0
    ck = out / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location=dev)
        net.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        ema, step = {k: v.to(dev) for k, v in st["ema"].items()}, st["step"]
        for _ in range(step):
            sched.step()
    grid = data.val_grid(patch=patch, ct=kw.get("ct", data.CT), store=kw.get("val", data.VAL), limit=val_patches)
    evnet = M.build(size, verbose=False).to(dev)
    dl = data.loader(patch, batch, workers, ct=kw.get("ct", data.CT), stores=kw.get("stores", data.TRAIN),
                     exclude=kw.get("val", data.VAL), seed=step)

    def save():
        torch.save({"model": net.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "step": step, "args": args}, ck)

    def log(name, rec):
        with open(out / name, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(name, rec, flush=True)

    t0 = time.time()
    for ct, tg in dl:
        if step >= steps:
            break
        ct = ct.to(dev, non_blocking=True).to(memory_format=torch.channels_last_3d)
        tg = tg.to(dev, non_blocking=True)
        with autocast(dev):
            bce, dice = losses(net(ct).float(), tg)
        loss = bce + dice
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(), opt.zero_grad(set_to_none=True), sched.step()
        ema_update(ema, net)
        step += 1
        if step % 20 == 0:
            dt = time.time() - t0
            log("train.jsonl", {"step": step, "loss": loss.item(), "bce": bce.item(), "dice": dice.item(),
                                "lr": sched.get_last_lr()[0], "vox_s": round(20 * batch * patch ** 3 / dt),
                                "vram_MiB": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else 0})
            t0 = time.time()
        if step % eval_every == 0 or step == steps:
            evnet.load_state_dict(ema)
            log("eval.jsonl", {"step": step, **evaluate(evnet, grid, dev)})
            save()
            t0 = time.time()
    save()
    return ck
