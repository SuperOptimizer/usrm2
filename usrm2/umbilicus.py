"""Every scroll's own axis, in the format `data.axis` parses.

The radial channel of a sample is built from the scroll axis (`data.axis` -> `data.radial`), so a stores
file that mixes scrolls needs one axis PER SCROLL: taking Paris 4's for PHerc0139 would point the radial
vector at nothing. `data.source_groups` looks for `<UMBILICUS_DIR>/<scroll>/umbilicus-full-resolution.json`;
this module puts one there, from whichever of these is available:

- a published umbilicus, in either of the two formats in circulation: the loader's own
  `{"control_points": [{"z":..., "y":..., "x":...}, ...]}` and the volpkg `umbilicus.txt` ("x, y, z" per
  line, one point per z slice, 1-based -- the volpkg convention);
- otherwise DERIVED from the CT itself: the centroid of the non-air voxels of each z slice at a coarse rung
  (rung 7, 76.8 um, is a few MB for a whole scroll and the axis only has to be good to a winding). The
  scroll is a roll, so the centroid of its cross-section is the umbilicus to within the accuracy the radial
  channel needs.

UNITS. `data.axis_at(ax, k)` divides the control points by 2^(k - 2), i.e. the points are RUNG-2 voxels,
not the scroll's own level-0 voxels. For a scroll whose level 0 is rung 4 (a 9.6 um scan) a point read off
level 0 must therefore be multiplied by 4. Everything written here is in rung-2 voxels.
"""
import json
import os
import re

import numpy as np

from usrm2 import data


def parse(text):
    """Published umbilicus text -> [(z, y, x), ...] in the units of the file. Accepts the loader's json and
    the volpkg `umbilicus.txt` ("x, y, z" per line, 1-based)."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        j = json.loads(text)
        pts = j["control_points"] if isinstance(j, dict) else j
        if pts and isinstance(pts[0], dict):
            return [(float(p["z"]), float(p["y"]), float(p["x"])) for p in pts]
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    out = []
    for line in text.splitlines():
        v = [q for q in re.split(r"[,\s]+", line.strip()) if q]
        if len(v) >= 3:
            try:
                x, y, z = (float(v[0]), float(v[1]), float(v[2]))
            except ValueError:
                continue
            out.append((z - 1, y - 1, x - 1))  # umbilicus.txt is 1-based, x first
    assert out, "no control points in the umbilicus text"
    return out


def write(path, pts):
    """[(z, y, x), ...] in rung-2 voxels -> the loader's json."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump({"control_points": [{"z": float(z), "y": float(y), "x": float(x)} for z, y, x in pts]}, f)
    os.replace(tmp, path)
    return path


def derive(ct_base, rung=7, thresh=0, step=1):
    """The axis of a scroll from its own CT: per-z centroid of the non-air voxels at `rung`, returned in
    RUNG-2 voxels. A z slice with no papyrus is skipped (the ends of a scan); the result is smoothed by
    taking one control point per `step` slices."""
    pyr = data.rungs(ct_base)
    k = max(r for r in pyr if r <= rung) if rung not in pyr else rung
    a = data.full_level(pyr, k)
    if a is None:
        arr = pyr[k]
        a = np.asarray(arr[:] if arr.ndim == 3 else arr[0], np.uint8)
    f = 2.0 ** (k - 2)  # rung-k voxels -> rung-2 voxels
    pts = []
    for z in range(0, a.shape[0], step):
        sl = a[z]
        m = sl > thresh
        n = int(m.sum())
        if n < 16:
            continue
        ys, xs = np.nonzero(m)
        pts.append((z * f, float(ys.mean()) * f, float(xs.mean()) * f))
    assert len(pts) >= 2, f"{ct_base}: no non-air slices at rung {k}; cannot derive an axis"
    return pts


def ensure(ct_base, scroll=None, urls=(), rung=7, force=False):
    """Make sure this machine has the scroll's axis, and return its path. A published file is used when one
    of `urls` serves it; otherwise the axis is derived from the CT."""
    scroll = scroll or data.scroll_of(ct_base)
    assert scroll, f"{ct_base}: no scroll in the path, cannot place an umbilicus"
    path = data.umbilicus_path(scroll)
    if os.path.exists(path) and not force:
        return path
    for u in urls:
        try:
            import urllib.request
            with urllib.request.urlopen(u, timeout=60) as r:
                if r.status == 200:
                    return write(path, parse(r.read().decode("utf-8", "replace")))
        except Exception:  # noqa: BLE001  (no such file on the origin: try the next, then derive)
            continue
    return write(path, derive(ct_base, rung=rung))


def published_urls(ct_base, scroll=None):
    """Where a published umbilicus for this scroll might be, most specific first."""
    scroll = scroll or data.scroll_of(ct_base)
    base = f"{data.STREAM_VOLUMES}/{scroll}/representations/umbilicus"
    return [f"{base}/umbilicus-full-resolution.json", f"{base}/umbilicus.json", f"{base}/umbilicus.txt"]
