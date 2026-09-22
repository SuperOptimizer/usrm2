"""The SIZE LADDER (experiment 12, docs/unified_design.md section 30).

`docs/research/synthesis_v2_with_literature.md` experiment 12: *"at rung 2 our unique-window supply means
params, not data, are the constraint"* -- run `15m` / `30m6` / `60m` at MATCHED STEPS on the SAME data in
the SAME order, then **fit `1 - dice` against `log(params)` PER RUNG** and call saturation only when the
slope flattens across >= 3 sizes *and* the train/val gap grows. Two halves:

  * `launch` prints the exact commands: one `stream-plan`, then one `usrm2 train` per rung, all of them
    differing ONLY in `--size`, `--out-dir` and `--stream-tag`.
  * `report` reads each run's `eval.jsonl` (and any `evalsurf --json` dumps beside it), takes the
    matched step, fits the per-rung slope in log-log and prints it next to the train/val gap trend.

**The data has to be identical, not merely identically distributed.** A scaling-law fit compares three
numbers that differ by a couple of points; three different window sequences move a number by more than
that. There are two honest ways to get one sequence into three trainers:

1. **One shared queue, one tag per rung** (the default here). `stream-plan` writes `queue.jsonl` ONCE and
   every trainer replays it. Nothing about a replay is destructive: a worker reads entry `i` when
   `i % GW == g`, and the file is append-only. What IS per-trainer is (a) the replay cursor
   `<queue>/progress/w<g>`, which says where a restart resumes, and (b) `<queue>/consumed`, the bound the
   planner evicts buffered chunks below. Shared, those two are exactly how two trainers eat each other's
   entries: rung A's cursor makes rung B skip windows, and rung A racing ahead lets the planner delete
   shards rung B has not read yet. `--stream-tag` moves both under the tag (`progress.<tag>/`,
   `consumed.<tag>`) and `stream.Planner.consumed()` takes the MINIMUM over the tags, so the buffer only
   advances at the pace of the slowest rung. That is the whole mechanism; it needs `--cache-gb` large
   enough to hold the spread between the fastest and the slowest rung (the 60m is ~2x the 15m's step
   time, so plan for ~2x the usual buffer), and every rung must use the same `--workers`, because the
   queue's `meta.json` fixes the stream count.
2. **One queue per rung, same seed.** The planner is deterministic in (stores, seed, patch, rungs,
   boost, region, walk, epochs) -- that tuple is literally its `fingerprint` -- so three queues planned
   with the same seed hold the same windows in the same order. It costs 3x the origin bandwidth and 3x
   the disk, which on the A100 box is the binding constraint, so it is the fallback (`--queues per-run`),
   not the default. It is also the ONLY option for rungs run one after another: a shared queue's buffer is
   a rolling window, so by the time a second rung started at index 0 the chunks of the first few thousand
   windows would have been evicted long ago. A shared queue means CONCURRENT rungs, one card each.

**muP.** `lit_optimisation_schedules.md` section 4 is explicit: do NOT adopt the muP reparametrisation
here (transformer-centric evidence, GroupNorm already normalises per-layer activation scale, and our
ladder warm-starts rather than fresh-init transfers). Its one cheap recommendation -- scale a WIDENED
layer's LR by ~1/sqrt(width ratio) and give it a short separate re-warmup -- is about warm starts, and
these rungs are fresh inits. So the ladder's default is **the same LR at every rung**, which is also the
experiment's own control; `--lr-scale mup` applies `lr_k = lr_base * sqrt(w_base / w_k)` for anyone who
wants that arm, and `--lr-sweep` prints the three-point sweep per rung the literature actually asks for
instead of a rule.
"""
import glob
import json
import os

import numpy as np

# The ladder, coarsest-first in parameters. `model.PRESETS` holds the widths; these three are the same six
# levels with every width scaled by 1/sqrt(2) and sqrt(2), which is a factor-2 ladder in parameters.
SIZES = ("15m", "30m6", "60m")


def params(size, args=None):
    """Parameters of a preset at a run's own `cin` / `cout` / `--add-skip` / `--deep`."""
    from usrm2 import model as M
    a = args or {}
    return M.params(size, cin=int(a.get("cin", 4) or 4), cout=int(a.get("cout", 1) or 1),
                    add_skip=int(a.get("add_skip", 0) or 0), deep=int(a.get("deep", 0) or 0))


def lr_for(size, base_size, lr, mode="same"):
    """The LR of one rung. "same" (the default and the experiment's control) or "mup" -- 1/sqrt of the
    width ratio, the one width-scaling rule `lit_optimisation_schedules` endorses."""
    from usrm2 import model as M
    if str(mode) == "same":
        return float(lr)
    assert str(mode) == "mup", f"--lr-scale {mode}: 'same' or 'mup'"
    return float(lr) * float(np.sqrt(M.PRESETS[base_size][0] / M.PRESETS[size][0]))


def launch(base, out_root, sizes=SIZES, queue=None, stores_file=None, lr=3e-4, lr_scale="same",
           base_size="30m6", queues="shared", tag_prefix="", log=print):
    """Print the commands that run the ladder. `base` is the rest of the `usrm2 train` flag line, copied
    verbatim to every rung -- the point of the experiment is that ONLY `--size` differs.

    Returns the list of (size, out_dir, command) so a caller (cloud/ladder.sh) can run them."""
    out = []
    plans = []
    if queue and stores_file:
        if queues == "shared":
            plans.append(f"usrm2 stream-plan {stores_file} --queue {queue} " + _plan_flags(base))
        else:
            for sz in sizes:
                plans.append(f"usrm2 stream-plan {stores_file} --queue {queue}_{sz} " + _plan_flags(base))
    for sz in sizes:
        d = os.path.join(str(out_root), f"{tag_prefix}{sz}")
        q = (str(queue) if queues == "shared" else f"{queue}_{sz}") if queue else None
        cmd = (f"usrm2 train {d} --size {sz} --lr {lr_for(sz, base_size, lr, lr_scale):g} "
               + (f"--stream {q} " + (f"--stream-tag {tag_prefix}{sz} " if queues == "shared" else "")
                  if q else "")
               + base).strip()
        out.append((sz, d, cmd))
    for p in plans:
        log(p)
    for sz, d, cmd in out:
        log(f"# {sz}: {params(sz):,} params at cin=4 cout=1 (the run's own count is printed at start)")
        log(cmd)
    return out


_PLAN_KEYS = ("--patch", "--ctx", "--rungs", "--rung-boost", "--cascade", "--planes", "--scan-meta",
              "--teacher-regions", "--verso-regions", "--workers", "--val", "--val-rungs")
_PLAN_FLAGS = ("--verso",)


def _plan_flags(base):
    """The subset of a train flag line that `stream-plan` also takes (the plan must build the same stem)."""
    toks = str(base).split()
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in _PLAN_FLAGS:
            out.append(t)
            i += 1
            continue
        if t in _PLAN_KEYS:
            out.append(t)
            i += 1
            while i < len(toks) and not toks[i].startswith("--"):
                out.append(toks[i])
                i += 1
            continue
        i += 1
    return " ".join(out)


# ------------------------------------------------------------------------------------ the report

def read_evals(run_dir):
    """[{step, metric: value, ...}] from a run's `eval.jsonl`, plus any `evalsurf --json` dumps beside it
    (`<run>/evalsurf/*.json` or `<run>/*.json`), merged by step: the surface metrics (recall@4, ERL,
    betti0_err) are what experiment 12's decision rule really wants and they live in those dumps."""
    rows = {}
    f = os.path.join(run_dir, "eval.jsonl")
    if os.path.exists(f):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("step") is not None:
                rows.setdefault(int(d["step"]), {}).update(d)
    for g in sorted(glob.glob(os.path.join(run_dir, "evalsurf", "*.json"))
                    + glob.glob(os.path.join(run_dir, "*.json"))):
        try:
            j = json.load(open(g))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(j, dict):
            continue
        st = j.get("step")                       # `evalsurf --json` puts the pooled metrics under
        d = j.get("pooled") if isinstance(j.get("pooled"), dict) else j   # "pooled" and the step at the top
        if st is None:
            st = d.get("step")
        if st is not None:
            rows.setdefault(int(st), {}).update({k: v for k, v in d.items()
                                                 if k != "step" and isinstance(v, (int, float))})
    return [dict(v, step=k) for k, v in sorted(rows.items())]


def train_gap(run_dir, step, metric="bce", win=10):
    """(train value, val value, gap) of `metric` at a step: the median of the last `win` `train.jsonl`
    records at or before the step, against the `eval.jsonl` value there. The gap is val - train, so it
    GROWS when the bigger model starts memorising -- the second half of experiment 12's decision rule
    ("the slope flattens AND the train/val gap grows")."""
    tr = []
    f = os.path.join(run_dir, "train.jsonl")
    if os.path.exists(f):
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("step") is not None and int(d["step"]) <= int(step) and metric in d:
                tr.append(float(d[metric]))
    ev = [r for r in read_evals(run_dir) if int(r["step"]) <= int(step) and metric in r]
    if not tr or not ev:
        return None
    a = float(np.median(tr[-int(win):]))
    b = float(ev[-1][metric])
    return {"train": a, "val": b, "gap": b - a}


def loglog_slope(par, loss):
    """Least-squares slope of log(loss) against log(params) -- the scaling exponent alpha in
    `loss ~ params^-alpha`. Returns (alpha, intercept, r2, n). A FLAT slope (alpha near 0) is the
    "params are no longer the constraint" reading; a steep one says the ladder is still paying."""
    p = np.asarray(par, float)
    y = np.asarray(loss, float)
    k = np.isfinite(p) & np.isfinite(y) & (p > 0) & (y > 0)
    p, y = np.log(p[k]), np.log(y[k])
    if len(p) < 2:
        return {"alpha": None, "n": int(len(p)), "error": "need at least 2 rungs"}
    A = np.stack([p, np.ones_like(p)], 1)
    (b, c), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([b, c])
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"alpha": float(-b), "intercept": float(c), "n": int(len(p)),
            "r2": float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else 1.0}


def report(runs, metric="dice", step=None, rungs=None, out=None, smooth=5, log=print):
    """The ladder report: per rung of the LADDER (the CT resolution rung, `dice_r2` / `dice_r3` / ...),
    the per-size loss at the matched step, the fitted log-log slope over the sizes, and the train/val gap
    trend; plus, per RUN, `evalsurf.fit_curve` on its own metric history so a rung that has not converged
    cannot be read as a plateau.

    `runs` are run directories. `step` is the matched step (default: the largest step EVERY run reached,
    which is the only honest comparison). `metric` is the eval.jsonl key whose 1 - value is the loss --
    `dice` (the mean over rungs), or any `dice_rK` / `recall@4` / `dice_verso`."""
    from usrm2 import evalsurf as E
    import torch
    info = []
    for d in runs:
        ck = os.path.join(d, "ckpt.pt")
        a = torch.load(ck, map_location="cpu", weights_only=False)["args"] if os.path.exists(ck) else {}
        ev = read_evals(d)
        info.append({"run": d, "size": a.get("size"), "args": a, "evals": ev,
                     "params": params(a["size"], a) if a.get("size") else None,
                     "last_step": int(ev[-1]["step"]) if ev else 0})
    assert info, "ladder-report: no runs"
    assert all(q["evals"] for q in info), \
        f"no eval.jsonl rows in {[q['run'] for q in info if not q['evals']]}"
    S = int(step) if step is not None else min(q["last_step"] for q in info)
    keys = ([f"dice_r{int(k)}" for k in rungs] if rungs else
            sorted({k for q in info for r in q["evals"] for k in r
                    if isinstance(k, str) and k.startswith("dice_r")}, key=lambda s: int(s[6:])))
    keys = keys or [metric]
    res = {"matched_step": S, "metric": metric, "runs": []}
    log(f"ladder report: {len(info)} runs, matched at step {S}")
    for q in info:
        rows = [r for r in q["evals"] if int(r["step"]) <= S]
        q["at"] = rows[-1] if rows else {}
        g = train_gap(q["run"], S)
        q["gap"] = g
        fit = E.fit_curve([r["step"] for r in q["evals"]],
                          [r.get(metric, np.nan) for r in q["evals"]], smooth=smooth) \
            if len(q["evals"]) >= 4 else {"model": None, "error": "fewer than 4 evals"}
        q["fit"] = fit
        res["runs"].append({"run": q["run"], "size": q["size"], "params": q["params"],
                            "step": int(q["at"].get("step", 0)), "metric": q["at"].get(metric),
                            "gap": g, "converged": fit})
        log(f"  {str(q['size']):>6} {q['params'] / 1e6 if q['params'] else float('nan'):8.2f} M params  "
            f"step {int(q['at'].get('step', 0)):>7}  {metric} {q['at'].get(metric, float('nan')):.4f}"
            + (f"  train/val {g['train']:.4f}/{g['val']:.4f} gap {g['gap']:+.4f}" if g else "")
            + (f"  [slope {fit['slope_per_10k']:+.4f}/10k, {fit['steps_to_95']:.0f} steps to 95 %]"
               if fit.get("model") else "  [not enough evals to fit a plateau]"))
    order = sorted(info, key=lambda q: (q["params"] or 0))
    log(f"\n  per-rung log-log fit of (1 - {metric.replace('dice', 'value')}) vs params")
    res["per_rung"] = {}
    for key in keys:
        par = [q["params"] for q in order if q["at"].get(key) is not None]
        val = [1.0 - float(q["at"][key]) for q in order if q["at"].get(key) is not None]
        if len(par) < 2:
            continue
        f = loglog_slope(par, val)
        gaps = [q["gap"]["gap"] for q in order if q["gap"]]
        f["gap_trend"] = (float(gaps[-1] - gaps[0]) if len(gaps) >= 2 else None)
        f["values"] = {str(q["size"]): float(q["at"][key]) for q in order if q["at"].get(key) is not None}
        res["per_rung"][key] = f
        log(f"    {key:>10}: alpha {f['alpha']:+.4f}  r2 {f['r2']:.3f}  over {f['n']} sizes  "
            + "  ".join(f"{k}={v:.4f}" for k, v in f["values"].items())
            + (f"  | train/val gap trend {f['gap_trend']:+.4f}" if f["gap_trend"] is not None else ""))
    log("\n  decision rule (experiment 12): saturation is BOTH a slope that flattens across >= 3 sizes\n"
        "  AND a train/val gap that grows. A flat alpha with a flat gap means the ladder is simply not\n"
        "  the binding constraint yet; a steep alpha with a growing gap means more params and more data.")
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res
