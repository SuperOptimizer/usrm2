"""Per-rung temperature calibration (docs/unified_design.md section 26, R4).

A dice-trained sigmoid is measurably overconfident (Mehrtash et al., arXiv:1911.13273), and ours is not
a probability *by construction* above the native rung: there the target is a pooled FRACTION of a binary
band, not a binary event. So one scalar temperature is fitted PER RUNG on the run's own held-out grid,
by minimising the same weighted BCE the training loss uses, and only on the rungs whose target really is
a binary band -- calibrating against a pooled fraction would be fitting a scale to a different quantity.

    p = sigmoid(logit / T)      T > 1 softens (the usual direction), T < 1 sharpens

The temperatures live in the checkpoint's own args (`args["temps"] = {"2": 1.13, ...}`); `predict.probs`
applies the one for the rung it is predicting at unless `--no-calib`, and `evalsurf` gets the same value
through `temp_for`. Nothing is retrained and no weight moves: `usrm2 calibrate CKPT` rewrites only
`args`, so an uncalibrated checkpoint and a calibrated one give the same logits.
"""
import json
import math

import torch
import torch.nn.functional as F

BINARY_FRAC = 0.5   # a rung counts as a BINARY band when at most this share of its weighted target mass
                    # is strictly between `BINARY_EPS` and 1 - `BINARY_EPS`
BINARY_EPS = 0.02
T_LO, T_HI = 0.2, 5.0


def temps_of(args):
    """{rung: T} of a checkpoint's args, ints for keys (json makes them strings)."""
    t = (args or {}).get("temps") or {}
    return {int(k): float(v) for k, v in t.items()}


def temp_for(args, rung, use=True):
    """The temperature to divide the logits by at `rung`: 1.0 when there is none or `use` is False."""
    if not use:
        return 1.0
    return float(temps_of(args).get(int(rung), 1.0))


def bce_at(logit, tgt, w, T):
    """Weighted BCE of `sigmoid(logit / T)` against `tgt` -- the metric `fit_temp` minimises."""
    b = F.binary_cross_entropy_with_logits(logit / float(T), tgt, reduction="none")
    return float((b * w).sum() / w.sum().clamp_min(1e-6))


def fit_temp(logit, tgt, w, lo=T_LO, hi=T_HI, iters=40):
    """The T in [lo, hi] minimising `bce_at`, by golden-section search on log T.

    The BCE of a sigmoid in one temperature is convex in log T for a fixed set of logits, so a
    golden-section search is exact to the bracket width and needs no gradients and no optimiser state.
    """
    g = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = bce_at(logit, tgt, w, math.exp(c)), bce_at(logit, tgt, w, math.exp(d))
    for _ in range(int(iters)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = bce_at(logit, tgt, w, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = bce_at(logit, tgt, w, math.exp(d))
    return float(math.exp((a + b) / 2))


def binary_frac(tgt, w):
    """Share of the weighted target mass strictly inside (eps, 1 - eps): 0 for a hard band, large for a
    pooled fraction."""
    mid = ((tgt > BINARY_EPS) & (tgt < 1 - BINARY_EPS)).to(tgt.dtype)
    return float((mid * w).sum() / w.sum().clamp_min(1e-6))


@torch.no_grad()
def collect(net, grid, dev, cout_t=None, norad=False, cascade=None):
    """{rung: (logit, tgt, weight)} over the validation grid, channel 0 only (the recto band is the one
    channel every val patch carries and the only one with a published meaning)."""
    from usrm2 import model as M, prep
    from usrm2.train import autocast
    out = {}
    net.eval()
    for item in grid:
        if isinstance(item, dict):
            ct, tg, ww = prep.prepare(prep.batch1(item), dev, norad=norad, cascade=cascade)
            ct, rung = ct.to(memory_format=M.memfmt()), int(item["rung"])
        else:
            ct, tg = item[0][None].to(dev), item[1][None].to(dev)
            ww = (item[2][None].to(dev) if len(item) > 2 else torch.ones_like(tg))
            rung = int(item[3]) if len(item) > 3 else 2
        with autocast(dev):
            lg = net(ct).float()
        if cout_t is not None:
            lg = lg[:, :int(cout_t)]
        a, b, c = lg[:, :1].cpu(), tg[:, :1].cpu(), ww[:, :1].cpu()
        if float(c.sum()) <= 0:
            continue
        out.setdefault(rung, []).append((a, b, c))
    return {k: tuple(torch.cat([q[i] for q in v]) for i in range(3)) for k, v in out.items()}


def run(ckpt, val=None, val_rungs=None, val_patches=None, device=None, all_rungs=False, write=True):
    """Fit one temperature per rung on the checkpoint's own validation grid and store them in its args.

    Returns the report dict (also printed as one json line per rung by `usrm2 calibrate`).
    """
    from usrm2 import data, model as M, prep
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.load(ckpt, map_location=dev)
    a = st["args"]
    cout, cin = int(a.get("cout", 1)), int(a.get("cin", 4))
    cout_t = int(a.get("cout_t", cout))
    net = M.build(a["size"], verbose=False, cout=cout, cin=cin, add_skip=a.get("add_skip", 0),
                  deep=a.get("deep", 0)).to(dev)
    net.load_state_dict({k: v.to(dev) for k, v in st["ema"].items()})
    net.eval()
    patch = a["patch"]
    rungs = a.get("rungs")
    assert rungs is not None, "calibrate needs a rung-ladder checkpoint (`train --rungs`)"
    lines = data.Patches(patch=patch, stores=a.get("stores"), stores_file=a.get("stores_file"),
                         rungs=True if rungs is True else set(rungs)).paths
    lines = [",".join(q) for q in lines]
    vr = [int(v) for v in (val_rungs or a.get("val_rungs") or data.VAL_RUNGS)]
    grid = data.val_grid_rungs(patch, lines, val or a.get("val") or data.VAL, rungs=vr,
                               limit=int(val_patches or a.get("val_patches", 32)),
                               ctx=tuple(a.get("ctx") or ()), cascade=str(a.get("cascade", "off") or "off"),
                               channels=(a.get("channels") or [])[:cout_t] or None,
                               verso=bool(a.get("verso")), verso_regions=a.get("verso_regions"))
    cas = None
    if str(a.get("cascade", "off") or "off") != "off":
        cas = prep.Cascade("self" if a["cascade"] in ("self", "mix") else "mask", self_p=1.0, drop=0.0,
                           noise=False, net=net)
    per = collect(net, grid, dev, cout_t=cout_t, norad=bool(a.get("no_radial")), cascade=cas)
    rep, temps = {}, {}
    for k in sorted(per):
        lg, tg, w = per[k]
        bf = binary_frac(tg, w)
        T = fit_temp(lg, tg, w)
        row = {"rung": k, "n_patches": int(lg.shape[0]), "binary_frac": round(bf, 4),
               "bce_T1": round(bce_at(lg, tg, w, 1.0), 6), "bce_T": round(bce_at(lg, tg, w, T), 6),
               "T": round(T, 4), "binary": bool(bf <= BINARY_FRAC)}
        rep[k] = row
        if row["binary"] or all_rungs:
            temps[str(k)] = round(T, 4)
        else:  # a pooled FRACTION is not a binary event: a temperature fitted there is not a calibration
            row["skipped"] = "target is a pooled fraction (use --all-rungs to fit it anyway)"
    if write and temps:
        st["args"]["temps"] = temps
        torch.save(st, str(ckpt) + ".tmp")
        import os
        os.replace(str(ckpt) + ".tmp", str(ckpt))
    return {"ckpt": str(ckpt), "temps": temps, "rungs": [rep[k] for k in sorted(rep)]}


def main(ckpt, **kw):
    r = run(ckpt, **kw)
    for row in r["rungs"]:
        print(json.dumps(row), flush=True)
    print(json.dumps({"ckpt": r["ckpt"], "temps": r["temps"]}), flush=True)
    return r
