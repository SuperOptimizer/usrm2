"""Sequential augmentation-preset sweep: each preset trains in its own subdir of OUT, and OUT/
summary.jsonl + a preview PNG per preset (one val slice: CT | teacher | student) say what happened.
"""
import json
import statistics
from pathlib import Path

from usrm2 import data, model as M, train as T


def preview(ckpt, png, patch, ct, val, no_radial=False):
    """CT | teacher | student, middle z slice of the first val patch."""
    import numpy as np
    import torch
    from PIL import Image
    st = torch.load(ckpt, map_location="cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = M.build(st["args"]["size"], verbose=False, cout=st["args"].get("cout", 1), cin=st["args"].get("cin", 4), add_skip=st["args"].get("add_skip", 0)).to(dev)
    net.load_state_dict({k: v.to(dev) for k, v in st["ema"].items()})
    net.eval()
    x, t = data.val_grid(patch=patch, ct=ct, store=val, limit=1)[0]
    if no_radial:
        x[-3:] = 0
    with torch.no_grad(), T.autocast(dev):
        p = torch.sigmoid(net(x[None].to(dev)).float())[0, 0, patch // 2].cpu().numpy()
    c = x[0, patch // 2].numpy()
    c = (c - c.min()) / (c.max() - c.min() + 1e-6)
    im = np.concatenate([c, t[0, patch // 2].numpy(), p], 1)
    Image.fromarray((im * 255).astype("uint8")).save(png)


def sweep(out_dir, presets, **kw):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in presets:
        d = out / name.replace("+", "_")
        ck = T.train(d, aug=name, **kw)
        ev = [json.loads(l) for l in (d / "eval.jsonl").read_text().splitlines()]
        tr = [json.loads(l) for l in (d / "train.jsonl").read_text().splitlines() if "vox_s" in l]
        r = {"preset": name, **{k: round(ev[-1][k], 4) for k in ("bce", "dice", "mae")},
             "best_dice": round(max(e["dice"] for e in ev), 4),
             "vox_s": round(statistics.median(t["vox_s"] for t in tr)) if tr else None}
        try:
            preview(ck, d / "preview.png", kw.get("patch", 128), kw.get("ct", data.CT),
                    kw.get("val", data.VAL), no_radial="norad" in name)
        except Exception as e:  # a missing PIL must not lose the sweep
            r["preview_error"] = repr(e)
        rows.append(r)
        with open(out / "summary.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
        print("ablate", r, flush=True)
    cols = ["preset", "bce", "dice", "mae", "best_dice", "vox_s"]
    print("  ".join(c.ljust(14 if c == "preset" else 9) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(14 if c == "preset" else 9) for c in cols))
    return rows
