"""Continue a finished run for more steps on ALL paired boxes now on disk: same arguments (from its checkpoint),
new step budget, in a copy of the dir. usage: python continue_run.py SRC_RUN DST_RUN STEPS [VAL_EXCLUDE_FILE]"""
import glob, os, shutil, sys, torch
from usrm2 import train as T
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
hold = set(l.split(",")[0] for l in open(sys.argv[4])) if len(sys.argv) > 4 else set()  # validation boxes never train
shutil.copytree(src, dst, dirs_exist_ok=True)
a = dict(torch.load(f"{dst}/ckpt.pt", map_location="cpu")["args"])
for k in ("aug_cfg", "norm_stats", "cout", "continued_from"):
    a.pop(k, None)
T_ = "/vesuvius/usrm2/teacher"
S = [f"{T_}/a.zarr,{T_}/a_m7.zarr", f"{T_}/b.zarr,{T_}/b_m7.zarr"]
for d in sorted(glob.glob(f"{T_}/boxes*/")):
    d = d.rstrip("/")
    if d.endswith("_m7") or d.endswith("_tta4"):
        continue
    for b in sorted(glob.glob(f"{d}/box_*.zarr")):
        m = f"{d}_m7/{os.path.basename(b)}"
        if os.path.isdir(m) and b not in hold:
            S.append(f"{b},{m}")
a["stores"], a["steps"] = S, steps
print("continuing", src, "->", dst, "to", steps, "steps on", len(S), "stores", flush=True)
T.train(dst, resume=True, workers=6, eval_every=1000, **a)
