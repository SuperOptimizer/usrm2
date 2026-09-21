"""Evaluate a recto probability volume against the published tifxyz surfaces.

The meshes lie ON the recto face (usrm/docs/labels.md), so the probability band should peak at
offset 0 along the surface normal; everything here is measured at the published surface points.

`--head verso` (the unified model's second output channel, docs/unified_design.md section 23) runs and
prints the same numbers, but THEY DO NOT MEAN WHAT THEY MEAN FOR RECTO: there is no published verso
surface to score against. The verso band sits on the other face of the sheet, so measured at the recto
points it shows up as a large positive `offset_mean` (roughly the sheet thickness) and a low `recall@2`,
and scoring it against the recto surfaces SHIFTED by a guessed thickness would only measure the guess.
Until verso surfaces are published, treat a verso run of evalsurf as a smoke test plus a thickness
readout (`offset_mean` / `offset_std` over the points where a band was found at all), not as a score.
"""
import glob
import json
import os

import numpy as np

from usrm2 import data, predict as P

TIFXYZ = os.environ.get("USRM2_TIFXYZ", "/vesuvius/usrm/tifxyz/PHercParis4")
VAL_BOX = ((34432, 15104, 18432), (256, 1024, 1024))  # what /vesuvius/usrm2/teacher/eval.zarr covers


def read_surface(d):
    """(H,W,3) zyx level-0 points of one tifxyz dir, invalid ones NaN."""
    import tifffile
    g = np.stack([np.asarray(tifffile.imread(f"{d}/{c}.tif"), np.float32) for c in "zyx"], -1)
    return np.where((g > 0).all(-1)[..., None], g, np.nan)


def normals(g, ax):
    """(H,W,3) unit normals n = normalize(cross(t_v, t_u)), oriented so dot(n, radial) >= 0."""
    n = np.cross(np.gradient(g, axis=0), np.gradient(g, axis=1))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    cy, cx = np.interp(g[..., 0], ax[0], ax[1]), np.interp(g[..., 0], ax[0], ax[2])
    r = np.stack([np.zeros_like(cy), g[..., 1] - cy, g[..., 2] - cx], -1)
    return n * np.where((n * r).sum(-1, keepdims=True) < 0, -1.0, 1.0)


def sites(origin, size, tifxyz=TIFXYZ, ax=None, min_pts=200, cache=None):
    """All published surface points (+normals) inside the box: (pts (N,3), nrm (N,3), per-surface counts)."""
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["pts"], z["nrm"], json.loads(str(z["counts"]))
    o, s, ax = np.asarray(origin, np.float32), np.asarray(size, np.float32), ax if ax is not None else data.axis()
    pts, nrm, counts = [], [], {}
    for m in sorted(glob.glob(f"{tifxyz}/*/*/meta.json")) or sorted(glob.glob(f"{tifxyz}/*/meta.json")):
        b = np.asarray(json.load(open(m))["bbox"], np.float32)[:, ::-1]  # [[x,y,z]min,max] -> zyx
        if (b[1] < o).any() or (b[0] > o + s).any():
            continue
        g = read_surface(os.path.dirname(m))
        n = normals(g, ax)
        k = ((g >= o) & (g < o + s)).all(-1) & np.isfinite(n).all(-1)
        if k.sum() < min_pts:
            continue
        pts.append(g[k])
        nrm.append(n[k])
        counts[os.path.basename(os.path.dirname(m))] = int(k.sum())
    pts = np.concatenate(pts) if pts else np.zeros((0, 3), np.float32)
    nrm = np.concatenate(nrm) if nrm else np.zeros((0, 3), np.float32)
    if cache:
        np.savez(cache, pts=pts, nrm=nrm, counts=json.dumps(counts))
    return pts, nrm, counts


def trilerp(V, q):
    """V (Z,Y,X) float32 sampled at q (N,3) float, 0 outside."""
    f = np.floor(q).astype(np.int64)
    d = (q - f).astype(np.float32)
    sh = np.array(V.shape)
    out = np.zeros(len(q), np.float32)
    for c in range(8):
        e = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
        i = f + e
        w = np.prod(np.where(e, d, 1 - d), axis=1)
        ok = ((i >= 0) & (i < sh)).all(1)
        j = np.clip(i, 0, sh - 1)
        out += w * np.where(ok, V[j[:, 0], j[:, 1], j[:, 2]], 0)
    return out


def metrics(p_u8, origin, pts, nrm, thr=0.5, far=40, win=16):
    """recall@r / offset bias / precision proxy / merge count at the surface points."""
    V, q = np.asarray(p_u8, np.float32) / 255.0, (pts - np.asarray(origin, np.float32))
    ts = np.arange(-far, far + 1, dtype=np.float32)
    S = np.stack([trilerp(V, q + t * nrm) for t in ts])  # (2*far+1, N)
    c = far
    m = {f"recall@{r}": float((S[c - r:c + r + 1].max(0) >= thr).mean()) for r in (2, 4, 8)}
    w = S[c - win:c + win + 1]
    k = np.clip(w.argmax(0), 1, 2 * win - 1)
    y0, y1, y2 = (np.take_along_axis(w, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)  # the fit is only a peak where it is concave
    off = (k - win) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    hit = w.max(0) >= thr  # an argmax is only meaningful where there is a band at all
    o = off[hit]
    b = S >= thr
    runs = b[0].astype(np.int32) + (b[1:] & ~b[:-1]).sum(0)
    from scipy.spatial import cKDTree  # EDT of the rasterized points, exactly (points are float)
    pos = np.argwhere(V >= thr).astype(np.float32)
    d = cKDTree(q).query(pos, distance_upper_bound=6.0)[0] if len(pos) and len(q) else np.array([np.inf])
    m.update({"n_points": int(len(pts)), "offset_frac": float(hit.mean()),
              "offset_mean": float(o.mean()) if len(o) else float("nan"),
              "offset_std": float(o.std()) if len(o) else float("nan"),
              "offset_le3": float((np.abs(o) <= 3).mean()) if len(o) else float("nan"),
              "precision6": float((d <= 6.0).mean()), "pos_frac": float((V >= thr).mean()),
              "merge_runs": float(runs.mean()), "merge_frac": float((runs > 1).mean())})
    return m


def continuity(p_u8, origin, size, tifxyz=TIFXYZ, ax=None, thr=0.5, r=4, min_pts=200):
    """Along-sheet continuity of the band: for every published surface crossing the box, a grid cell is HIT when
    the probability along its normal reaches thr within +-r voxels; continuity = fraction of hit cells whose 8
    grid neighbours are all hit (a broken or fragmented band scores low even at high recall). Also the mean run
    length of hits along grid rows. Point-weighted over the surfaces."""
    from usrm2 import refine as R
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    V, ax = np.asarray(p_u8, np.float32) / 255.0, ax if ax is not None else data.axis()
    tot, cont, runs, n_hit = 0, 0.0, [], 0
    for d in R.surfaces_in(tifxyz, o, s, min_pts):
        g = read_surface(d)
        n = R.normals(g, ax)
        k = np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1) & np.isfinite(n).all(-1)
        if k.sum() < min_pts:
            continue
        S = R.profile(V, g[k] - o, n[k], r)
        hit = np.zeros(g.shape[:2], bool)
        hit[k] = S.max(0) >= thr
        inner = k.copy()
        inner[:1], inner[-1:], inner[:, :1], inner[:, -1:] = False, False, False, False
        nb = np.ones(g.shape[:2], bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb[1:-1, 1:-1] &= hit[1 + dy:hit.shape[0] - 1 + dy, 1 + dx:hit.shape[1] - 1 + dx]
        c = inner & hit
        cont += float(nb[c].sum()); n_hit += int(c.sum()); tot += int(k.sum())
        for row in hit & k:  # run lengths along grid rows
            if row.any():
                edges = np.diff(np.concatenate([[0], row.astype(int), [0]]))
                runs += (np.where(edges == -1)[0] - np.where(edges == 1)[0]).tolist()
    return {"continuity": cont / max(n_hit, 1), "hit_frac": n_hit / max(tot, 1), "mean_run": float(np.mean(runs)) if runs else 0.0,
            "n_points": tot}


def read_box(path, origin, size):
    """A box of a probability store (uint8), by global origin."""
    a = data.open_zarr(path)
    lo = np.asarray(origin, np.int64) - np.asarray(a.attrs["origin_zyx"], np.int64)
    assert (lo >= 0).all() and (lo + size <= np.array(a.shape[-3:])).all(), f"{path} does not cover the box"
    s = tuple(slice(int(l), int(l) + int(n)) for l, n in zip(lo, size))
    return a[(0,) + s] if a.ndim == 4 else a[s]


def png(path, ct, p_u8, origin, pts, thr=0.5):
    """CT z-slice (the one with most surface points) + published points (green) + P>=thr (red)."""
    from PIL import Image
    zi = np.bincount(np.clip(np.rint(pts[:, 0] - origin[0]).astype(int), 0, ct.shape[0] - 1), minlength=ct.shape[0]).argmax()
    g = np.repeat(np.asarray(ct[zi], np.uint8)[..., None], 3, -1)
    g[..., 0] = np.where(p_u8[zi] >= thr * 255, 255, g[..., 0])
    k = np.abs(pts[:, 0] - origin[0] - zi) <= 2  # a thin slab, the grid is 20 voxels coarse
    yx = np.rint(pts[k, 1:] - np.asarray(origin, np.float32)[1:]).astype(int)
    yx = np.clip(yx[..., None, None] + np.mgrid[-1:2, -1:2], 0, np.array(g.shape[:2])[:, None, None] - 1)
    g[yx[:, 0].ravel(), yx[:, 1].ravel()] = (0, 255, 0)
    Image.fromarray(g).save(path)
    return path, int(zi)


def run(origin=VAL_BOX[0], size=VAL_BOX[1], ckpt=None, store=None, teacher=None, tifxyz=TIFXYZ,
        volume=None, window=128, halo=16, device=None, png_path=None, cache=None, tta=0, luts=(), head=0,
        cascade=None, cascade_depth=3):
    o, s = tuple(origin), tuple(size)
    import hashlib
    key = hashlib.md5(f"{s}|{tifxyz}|{data.UMBILICUS}".encode()).hexdigest()[:8]
    pts, nrm, counts = sites(o, s, tifxyz, cache=cache or (store and store.rstrip("/") + f".sites_{o[0]}_{o[1]}_{o[2]}_{key}.npz"))
    ct = data.open_zarr(volume or data.CT)[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
    keep = ct[tuple(np.clip(np.rint(pts - o).astype(int), 0, np.array(s) - 1).T)] > 0  # points in masked CT can't be predicted
    pts, nrm = pts[keep], nrm[keep]
    print(json.dumps({"box": [*o, *s], "surfaces": counts, "masked_points_dropped": int((~keep).sum())}))
    if str(head) == "verso" or (isinstance(head, int) and head == 1):
        print(json.dumps({"note": "head verso is scored at the RECTO surface points: there is no published "
                                  "verso surface. offset_mean is then the sheet thickness, not a bias; "
                                  "recall/precision are not comparable with a recto run."}))
    if ckpt:
        prob, st = P.probs(ckpt, volume or data.CT, *o, *s, window=window, halo=halo, device=device, tta=tta, luts=luts, head=head,
                           cascade=cascade, cascade_depth=cascade_depth)
        p_u8, name = P.u8(prob), f"{ckpt}@{st.get('step')}" + (f"+tta{tta}" if tta > 1 else "") + (f"+lut{len(luts)}" if luts else "") + f"+head{head}"
    else:
        p_u8, name = read_box(store, o, s), store
    print(json.dumps({"source": name, **metrics(p_u8, o, pts, nrm), **continuity(p_u8, o, s, tifxyz)}))
    if teacher:
        pt = read_box(teacher, o, s)
        print(json.dumps({"source": teacher, **metrics(pt, o, pts, nrm), **continuity(pt, o, s, tifxyz)}))
    if png_path:
        print(json.dumps({"png": png(png_path, ct, p_u8, o, pts)[0]}))
