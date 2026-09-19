#!/bin/bash
# On the A100: rewrite ~/groups_raw.txt / ~/groups_skin.txt (4-way: recto, m7, verso x2) from ~/teacher every
# 10 min; the desk-made big stores (a/b) and boxes dirs share the desk's layout. Val = eval.zarr group + val_extra.
cd ~; . venv/bin/activate
while true; do
  python - <<'PY'
import glob, os
from usrm2.verso import verso_path
T = os.path.expanduser("~/teacher")
val_first = {l.split(",")[0].replace("/vesuvius/usrm2/teacher", T) for l in open(os.path.expanduser("~/val_extra.txt")) if l.strip()}
groups = [f"{T}/{s}.zarr,{T}/{s}_m7.zarr" for s in ("a", "b") if os.path.isdir(f"{T}/{s}_m7.zarr")]
for d in sorted(glob.glob(T + "/boxes*")):
    n = os.path.basename(d)
    if n.endswith(("_m7", "_v", "_vraw", "_tta4")) or not os.path.isdir(d):
        continue
    for b in sorted(glob.glob(d + "/box_*.zarr")):
        m = os.path.join(d + "_m7", os.path.basename(b))
        if os.path.exists(os.path.join(m, "zarr.json")) and b not in val_first:
            groups.append(b + "," + m)
def done(p):
    j = os.path.join(p, "zarr.json")
    return os.path.exists(j) and '"done": true' in open(j).read()
for mode in ("raw", "skin"):
    lines = [",".join(g.split(",") + [verso_path(p, mode) for p in g.split(",")]) for g in groups
             if all(done(verso_path(p, mode)) for p in g.split(","))]
    path = os.path.expanduser(f"~/groups_{mode}.txt"); new = "\n".join(lines) + "\n"
    old = open(path).read() if os.path.exists(path) else ""
    if new != old:
        open(path + ".tmp", "w").write(new); os.replace(path + ".tmp", path)
        print(f"{mode}: {len(lines)} groups (was {old.count(chr(10))})", flush=True)
PY
  [ "${ONCE:-0}" = 1 ] && break
  sleep 600
done
