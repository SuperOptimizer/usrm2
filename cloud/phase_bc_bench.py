"""Step cost of the Phase B/C terms (docs/unified_design.md section 29), measured the way section 26.5
measured Phase A: one `5m` net at 128^3, batch 1, 15 input channels, bf16 autocast, `--deep 2`, the
WHOLE training step (forward + every loss + backward + optimizer), two passes, on the laptop RTX 5080.

    uv run python cloud/phase_bc_bench.py [--size 5m] [--patch 128] [--batch 1] [--iters 12]

Synthetic tensors: the point is the marginal cost of each term, and every term is computed from tensors
the step already holds, so a real loader changes the absolute numbers and not the ratios.
"""
import argparse
import time

import torch

from usrm2 import losses as L, model as M, train as T


def arms(nprob):
    """(name, kwargs) per arm. `sd` = the distance channels exist; the rest are loss switches."""
    return [
        ("baseline",              dict()),
        ("+planes (6)",           dict(planes=6)),
        ("+sdist (Huber)",        dict(sd=1)),
        ("+sdist +eikonal",       dict(sd=1, eik=0.1)),
        ("+sdist +thickness",     dict(sd=2, eik=0.1)),
        ("+normals head",         dict(sd=2, eik=0.1, nrm=0.1)),
        ("+pair construct",       dict(sd=2, eik=0.1, nrm=0.1, pair=True)),
        ("+loss-ect 4x32^3",      dict(ect=0.01, ect_n=4)),
        ("+loss-ect 1x32^3",      dict(ect=0.01, ect_n=1)),
        ("all of the above",      dict(planes=6, sd=2, eik=0.1, nrm=0.1, pair=True, ect=0.01, ect_n=4)),
    ]


def build(size, cin, cout, deep, dev):
    net = M.build(size, verbose=False, cin=cin, cout=cout, deep=deep).to(dev).to(memory_format=M.memfmt())
    return net, torch.optim.AdamW(net.parameters(), lr=1e-4)


def step(net, opt, x, tg, w, cfg, nprob, dev):
    from usrm2.train import autocast, deep_losses, losses_tw
    sd, eik, nrm = cfg.get("sd", 0), cfg.get("eik", 0.0), cfg.get("nrm", 0.0)
    with autocast(dev):
        pred = net(x)
        outs = [o.float() for o in pred] if isinstance(pred, (list, tuple)) else [pred.float()]
        bce, dice = deep_losses([o[:, :nprob] for o in outs] if len(outs) > 1 else outs[0][:, :nprob],
                                tg[:, :nprob], 0.0, (), w[:, :nprob])
    loss = bce + dice
    y0 = outs[0]
    if sd:
        d = y0[:, nprob:nprob + 1]
        td, wd = tg[:, nprob:nprob + 1], L.dist_weight(w[:, nprob:nprob + 1])
        lv = y0[:, nprob + sd:nprob + sd + 1]
        loss = loss + L.sdist_loss(d, td, wd, logvar=lv)
        if eik:
            loss = loss + eik * L.eikonal(d, wd, tgt=td)
        if sd > 1:
            th = L.soft_thickness(y0[:, nprob + 1:nprob + 2])
            loss = loss + L.thickness_loss(th, tg[:, nprob + 1:nprob + 2],
                                           L.dist_weight(w[:, nprob + 1:nprob + 2]))
            if nrm:
                j = nprob + sd + 1
                loss = loss + nrm * L.normal_head_loss(y0[:, j:j + 3], d, wd, tgt=td)
            if cfg.get("pair"):
                a, b = L.pair_logits(d, th)
                pb, pd = losses_tw(torch.cat([a, b], 1), tg[:, :2], w[:, :2], 0.0)
                loss = loss + pb + pd
    if cfg.get("ect"):
        loss = loss + cfg["ect"] * L.ect_loss(torch.sigmoid(y0[:, :1]), tg[:, :1], dirs=8, res=16,
                                              margin=8, block=32, nblocks=cfg.get("ect_n", 4))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step(), opt.zero_grad(set_to_none=True)
    return float(loss.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="5m")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--deep", type=int, default=2)
    ap.add_argument("--nprob", type=int, default=2)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, B, nprob = a.patch, a.batch, a.nprob
    torch.manual_seed(0)
    rows = []
    for name, cfg in arms(nprob):
        sd, extra = cfg.get("sd", 0), 0
        if sd:
            extra = 1 + (3 if cfg.get("nrm") else 0)     # log-variance [+ 3 normal channels]
        cout = nprob + sd + extra
        cin = 15 + cfg.get("planes", 0)
        net, opt = build(a.size, cin, cout, a.deep, dev)
        x = torch.randn(B, cin, P, P, P, device=dev).to(memory_format=M.memfmt())
        tg = torch.rand(B, nprob + sd, P, P, P, device=dev)
        w = torch.ones_like(tg)
        ms = []
        for _ in range(a.passes):
            for i in range(a.iters):
                if i == 2:
                    torch.cuda.synchronize() if dev.type == "cuda" else None
                    t0 = time.time()
                step(net, opt, x, tg, w, cfg, nprob, dev)
            torch.cuda.synchronize() if dev.type == "cuda" else None
            ms.append(1000 * (time.time() - t0) / (a.iters - 2))
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else 0
        torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
        rows.append((name, ms, peak, cin, cout))
        del net, opt, x, tg, w
        torch.cuda.empty_cache() if dev.type == "cuda" else None
        print(f"{name:24s} {'  '.join(f'{q:7.1f}' for q in ms)} ms/step  cin={cin} cout={cout} "
              f"peak={peak:.0f} MiB", flush=True)
    base = rows[0][1]
    print(f"\n{a.size} at {P}^3 batch {B}, deep {a.deep}, bf16, {dev}")
    print("| arm | ms / step | vs baseline | peak MiB |")
    print("|---|---|---|---|")
    for name, ms, peak, cin, cout in rows:
        rel = ", ".join(f"{100 * (m / b - 1):+.1f} %" for m, b in zip(ms, base))
        print(f"| {name} | {', '.join(f'{m:.1f}' for m in ms)} | {rel if name != rows[0][0] else '--'} "
              f"| {peak:.0f} |")


if __name__ == "__main__":
    main()
