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


def metrics(p_u8, origin, pts, nrm, thr=0.5, far=40, win=16, precision=True):
    """recall@r / offset bias / precision proxy / merge count at the surface points.

    `precision=False` drops `precision6`/`pos_frac` only (their KD-tree is over the whole box and is not a
    per-surface quantity); every other number is bit-for-bit what it has always been."""
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
    m.update({"n_points": int(len(pts)), "offset_frac": float(hit.mean()),
              "offset_mean": float(o.mean()) if len(o) else float("nan"),
              "offset_std": float(o.std()) if len(o) else float("nan"),
              "offset_le3": float((np.abs(o) <= 3).mean()) if len(o) else float("nan"),
              "merge_runs": float(runs.mean()), "merge_frac": float((runs > 1).mean())})
    if precision:
        from scipy.spatial import cKDTree  # EDT of the rasterized points, exactly (points are float)
        pos = np.argwhere(V >= thr).astype(np.float32)
        d = cKDTree(q).query(pos, distance_upper_bound=6.0)[0] if len(pos) and len(q) else np.array([np.inf])
        m.update({"precision6": float((d <= 6.0).mean()), "pos_frac": float((V >= thr).mean())})
    return m


def continuity(p_u8, origin, size, tifxyz=TIFXYZ, ax=None, thr=0.5, r=4, min_pts=200, surfaces=None):
    """Along-sheet continuity of the band: for every published surface crossing the box, a grid cell is HIT when
    the probability along its normal reaches thr within +-r voxels; continuity = fraction of hit cells whose 8
    grid neighbours are all hit (a broken or fragmented band scores low even at high recall). Also the mean run
    length of hits along grid rows. Point-weighted over the surfaces.

    `surfaces`: an already-read `surface_list()`, to skip re-reading every tifxyz grid off disk. It is the
    same set this function would select itself, so the numbers are unchanged."""
    from usrm2 import refine as R
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    V, ax = np.asarray(p_u8, np.float32) / 255.0, ax if ax is not None else data.axis()
    tot, cont, runs, n_hit = 0, 0.0, [], 0
    for g, n, k in ([(g, n, k) for _, g, k, n in surfaces] if surfaces is not None else
                    ((lambda gg: (gg, R.normals(gg, ax), None))(read_surface(d)) for d in R.surfaces_in(tifxyz, o, s, min_pts))):
        if k is None:
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
    lo = np.asarray(origin, np.int64) - np.asarray(a.attrs.get("origin_zyx", (0, 0, 0)), np.int64)  # a whole-scroll pyramid starts at 0
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
        cascade=None, cascade_depth=3, ceil=None, json_out=None, boot=200, seed=0, betti=True,
        betti_margin=8, betti_band=6, betti_dilate=2.0, no_ceiling_cache=False):
    """Score one checkpoint or store on the box. `ceil` (a store path, or "" for the default published
    recto pyramid / the `--teacher` store) adds the noise ceiling; every headline number is then printed
    as "value (ceiling) [bootstrap CI]". See docs/unified_design.md section 25."""
    o, s = tuple(origin), tuple(size)
    import hashlib
    key = hashlib.md5(f"{s}|{tifxyz}|{data.UMBILICUS}".encode()).hexdigest()[:8]
    pts, nrm, counts = sites(o, s, tifxyz, cache=cache or (store and store.rstrip("/") + f".sites_{o[0]}_{o[1]}_{o[2]}_{key}.npz"))
    ct = data.open_zarr(volume or data.CT)[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
    keep = ct[tuple(np.clip(np.rint(pts - o).astype(int), 0, np.array(s) - 1).T)] > 0  # points in masked CT can't be predicted
    pts, nrm = pts[keep], nrm[keep]
    um = data.native_um(data.pyramid_base(volume or data.CT))
    print(json.dumps({"box": [*o, *s], "surfaces": counts, "masked_points_dropped": int((~keep).sum()), "voxel_um": um}))
    if str(head) == "verso" or (isinstance(head, int) and head == 1):
        print(json.dumps({"note": "head verso is scored at the RECTO surface points: there is no published "
                                  "verso surface. offset_mean is then the sheet thickness, not a bias; "
                                  "recall/precision are not comparable with a recto run."}))
    surfaces = surface_list(o, s, tifxyz)
    kw = dict(pts=pts, nrm=nrm, surfaces=surfaces, tifxyz=tifxyz, ct=ct, um=um, boot=boot, seed=seed,
              betti=betti, betti_margin=betti_margin, betti_band=betti_band, betti_dilate=betti_dilate,
              ref=mesh_reference(o, s, surfaces) if betti else None)  # rasterized once, shared by all sources
    cres = None
    if ceil is not None:
        cres = ceiling(o, s, ceil or teacher or CEILING_STORE, cache=not no_ceiling_cache, **kw)
        print(json.dumps({"ceiling": cres["source"], **{k: v for k, v in cres.items()
                                                        if k not in ("surfaces", "ci", "source")}}))
    if ckpt:
        prob, st = P.probs(ckpt, volume or data.CT, *o, *s, window=window, halo=halo, device=device, tta=tta, luts=luts, head=head,
                           cascade=cascade, cascade_depth=cascade_depth)
        p_u8, name = P.u8(prob), f"{ckpt}@{st.get('step')}" + (f"+tta{tta}" if tta > 1 else "") + (f"+lut{len(luts)}" if luts else "") + f"+head{head}"
        step = st.get("step")
    else:
        p_u8, name, step = read_box(store, o, s), store, None
    res = evaluate_all(p_u8, o, s, **kw)
    res["source"], res["step"], res["box"] = name, step, [*o, *s]
    print(json.dumps({"source": name, **{k: v for k, v in res.items() if k not in ("surfaces", "ci", "betti", "source", "step", "box")}}))
    print(table(res, cres))
    tres = None
    if teacher:
        tres = evaluate_all(read_box(teacher, o, s), o, s, **kw)
        tres["source"] = teacher
        print(json.dumps({"source": teacher, **{k: v for k, v in tres.items() if k not in ("surfaces", "ci", "betti", "source")}}))
    if png_path:
        print(json.dumps({"png": png(png_path, ct, p_u8, o, pts)[0]}))
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        json.dump({"box": [*o, *s], "voxel_um": um, "tifxyz": tifxyz, "surface_points": counts,
                   "result": res, "ceiling": cres, "teacher": tres}, open(json_out, "w"), indent=1)
        print(json.dumps({"json": json_out}))
    return res



# ---------------------------------------------------------------------------------------------------
# Evaluation v2 (docs/unified_design.md section 25): noise ceiling, ERL, Betti-0/1, bootstrap CIs.
# Everything below is ADDITIVE: `metrics()` and `continuity()` above are untouched, so every number this
# file printed before still means exactly what it meant.
# ---------------------------------------------------------------------------------------------------

# The published recto mask pyramid, the default noise ceiling: a human-verified-ish machine label whose
# own agreement with the meshes bounds what any student trained on it can score (lit_evaluation_metrics
# section 6). `--ceiling PATH` or USRM2_CEILING_STORE overrides it.
CEILING_STORE = os.environ.get(
    "USRM2_CEILING_STORE",
    "/vesuvius/usrm/volcomp/PHercParis4/representations/predictions/surfaces/"
    "20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr/2.4")
CEILING_CACHE = os.environ.get("USRM2_CEILING_CACHE", "")  # default: next to the eval box (data.VAL)


def surface_list(origin, size, tifxyz=TIFXYZ, ax=None, min_pts=200):
    """[(name, g (H,W,3), inside (H,W) bool, n (H,W,3))] for every published surface crossing the box.

    Same selection rule as `sites()`/`continuity()` (bbox test, then >= min_pts points inside the box), so
    the per-surface breakdown covers exactly the surfaces the pooled numbers are computed from."""
    from usrm2 import refine as R
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    ax = ax if ax is not None else data.axis()
    out = []
    for d in R.surfaces_in(tifxyz, o, s, min_pts):
        g = read_surface(d)
        n = R.normals(g, ax)
        k = np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1) & np.isfinite(n).all(-1)
        if k.sum() >= min_pts:
            out.append((os.path.basename(d), g, k, n))
    return out


def _runs_1d(ok, ln):
    """Total length of every maximal run of True edges: ok (M,) bool, ln (M,) edge lengths -> (R,)."""
    e = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    a, b = np.where(e == 1)[0], np.where(e == -1)[0]
    c = np.concatenate([[0.0], np.cumsum(np.where(ok, ln, 0.0))])
    return c[b] - c[a]


def _walk_axis(good, valid, g, axis, um):
    """Runs and path length along one grid axis. Returns (run lengths um, total path length um,
    per-vertex length share um (H,W))."""
    sl0 = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
    sl1 = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
    ev = valid[sl0] & valid[sl1]                                   # an edge exists between two in-box points
    ln = np.linalg.norm(g[sl1] - g[sl0], axis=-1).astype(np.float64) * um
    ln = np.where(ev & np.isfinite(ln), ln, 0.0)
    ok = ev & good[sl0] & good[sl1]
    share = np.zeros(valid.shape, np.float64)                      # half of each incident edge
    share[sl0] += 0.5 * ln
    share[sl1] += 0.5 * ln
    lines = ln.T if axis == 0 else ln                              # walk along the axis -> put it last
    oks = ok.T if axis == 0 else ok
    sep = np.zeros((lines.shape[0], 1))
    r = _runs_1d(np.concatenate([oks, sep.astype(bool)], 1).ravel(),
                 np.concatenate([lines, sep], 1).ravel())
    return r, float(ln.sum()), share


def erl(p_u8, origin, g, k, n, thr=0.5, r=4, far=40, um=2.4):
    """Expected run length (Januszewski et al. 2018) along one published surface, in MICROMETRES.

    The tifxyz UV grid is the walk graph: an edge between two neighbouring in-box grid points is
    traversable when BOTH its endpoints are good, where a vertex is good when the probability reaches
    `thr` within +-r voxels along its normal (no break) AND the ray crosses `thr` exactly once over
    +-far (no merge -- the same `merge_runs` test `metrics()` uses). ERL = sum(L_i^2)/sum(L_total): the
    expected length of the run containing a point drawn uniformly by length, so a 2 mm break costs far
    more than a 2 voxel one. `erl_break_um` / `erl_merge_um` repeat the walk with only one of the two
    stopping conditions, and `lost_break_frac` / `lost_merge_frac` split the surface length between them
    by giving every vertex half of each incident edge."""
    from usrm2 import refine as R
    V, o = np.asarray(p_u8, np.float32) / 255.0, np.asarray(origin, np.float32)
    S = R.profile(V, g[k] - o, n[k], far)                          # (2*far+1, M)
    c = far
    b = S >= thr
    hit = np.zeros(g.shape[:2], bool)
    hit[k] = b[c - r:c + r + 1].max(0)
    merged = np.zeros(g.shape[:2], bool)
    merged[k] = (b[0].astype(np.int32) + (b[1:] & ~b[:-1]).sum(0)) > 1
    gf = np.where(np.isfinite(g), g, 0.0)
    out, tot = {}, 0.0
    for name, good in (("", k & hit & ~merged), ("_break", k & hit), ("_merge", k & ~merged)):
        rl, tl = [], 0.0
        for a in (0, 1):
            ra, la, _ = _walk_axis(good, k, gf, a, um)
            rl.append(ra); tl += la
        rl = np.concatenate(rl) if rl else np.zeros(0)
        out["erl" + name + "_um"] = float((rl ** 2).sum() / tl) if tl > 0 else 0.0
        out["_runsq" + name] = float((rl ** 2).sum())
        tot = tl
    share = sum(_walk_axis(k, k, gf, a, um)[2] for a in (0, 1))
    out["path_um"] = tot
    out["break_um"] = float(share[k & ~hit].sum())
    out["merge_um"] = float(share[k & hit & merged].sum())
    out["lost_break_frac"] = out["break_um"] / tot if tot > 0 else 0.0
    out["lost_merge_frac"] = out["merge_um"] / tot if tot > 0 else 0.0
    return out


def surface_rows(p_u8, origin, size, tifxyz=TIFXYZ, ax=None, ct=None, thr=0.5, r=4, far=40, um=2.4,
                 min_pts=200, surfaces=None):
    """Per-surface metrics: the existing suite restricted to one surface, plus ERL. One row per surface,
    with the sufficient statistics (`_*` keys) that `pool()` needs to recombine them exactly."""
    o, s = np.asarray(origin, np.float32), np.asarray(size, np.float32)
    rows = []
    for name, g, k, n in (surfaces if surfaces is not None else surface_list(origin, size, tifxyz, ax, min_pts)):
        pts, nrm = g[k], n[k]
        if ct is not None:  # points inside masked CT cannot be predicted (same filter run() applies)
            keep = np.asarray(ct)[tuple(np.clip(np.rint(pts - o).astype(int), 0, np.asarray(size) - 1).T)] > 0
            pts, nrm = pts[keep], nrm[keep]
        m = metrics(p_u8, origin, pts, nrm, thr=thr, far=far, precision=False) if len(pts) else {}
        cy = continuity_one(p_u8, origin, g, k, n, thr=thr, r=r)
        e = erl(p_u8, origin, g, k, n, thr=thr, r=r, far=far, um=um)
        V = np.asarray(p_u8, np.float32) / 255.0
        q = (pts - o)
        off = _abs_offsets(V, q, nrm, far=far, win=16, thr=thr) if len(pts) else np.zeros(0, np.float32)
        rows.append({"surface": name, **{kk: vv for kk, vv in m.items() if kk not in ("precision6", "pos_frac")},
                     **cy, **e, "offset_hd95": float(np.percentile(off, 95)) if len(off) else float("nan"),
                     "offset_p99": float(np.percentile(off, 99)) if len(off) else float("nan"),
                     "_n": int(len(pts)), "_noff": int(round(len(pts) * m.get("offset_frac", 0.0))),
                     "_absoff": off})
    return rows


def _abs_offsets(V, q, nrm, far=40, win=16, thr=0.5):
    """|sub-voxel peak offset| at the points where a band is found -- the sample HD95/P99 are taken from."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    S = np.stack([trilerp(V, q + t * nrm) for t in ts])
    w = S[far - win:far + win + 1]
    k = np.clip(w.argmax(0), 1, 2 * win - 1)
    y0, y1, y2 = (np.take_along_axis(w, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    off = (k - win) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    return np.abs(off[w.max(0) >= thr]).astype(np.float32)


def continuity_one(p_u8, origin, g, k, n, thr=0.5, r=4):
    """`continuity()`'s numbers for ONE surface, with the same definitions (8-neighbour hit consistency,
    hit fraction, mean run length along grid rows) plus the counts `pool()` weights them by."""
    from usrm2 import refine as R
    V, o = np.asarray(p_u8, np.float32) / 255.0, np.asarray(origin, np.float32)
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
    runs = []
    for row in hit & k:
        if row.any():
            e = np.diff(np.concatenate([[0], row.astype(int), [0]]))
            runs += (np.where(e == -1)[0] - np.where(e == 1)[0]).tolist()
    return {"continuity": float(nb[c].sum()) / max(int(c.sum()), 1), "hit_frac": int(c.sum()) / max(int(k.sum()), 1),
            "mean_run": float(np.mean(runs)) if runs else 0.0,
            "_nhit": int(c.sum()), "_ncont": int(k.sum()), "_nruns": len(runs)}


POOL_W = {"recall@2": "_n", "recall@4": "_n", "recall@8": "_n", "offset_frac": "_n", "merge_runs": "_n",
          "merge_frac": "_n", "offset_le3": "_noff", "continuity": "_nhit", "hit_frac": "_ncont",
          "mean_run": "_nruns"}


def pool(rows):
    """Recombine per-surface rows into the pooled numbers, exactly (point-weighted means, pooled variance,
    length-weighted ERL, quantiles over the concatenated offsets)."""
    if not rows:
        return {}
    out = {"n_surfaces": len(rows), "n_points": int(sum(r["_n"] for r in rows))}
    for key, wk in POOL_W.items():
        v = np.array([r.get(key, np.nan) for r in rows], float)
        w = np.array([r.get(wk, 0) for r in rows], float)
        m = np.isfinite(v) & (w > 0)
        out[key] = float((v[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")
    v = np.array([r.get("offset_mean", np.nan) for r in rows], float)
    sd = np.array([r.get("offset_std", np.nan) for r in rows], float)
    w = np.array([r["_noff"] for r in rows], float)
    m = np.isfinite(v) & np.isfinite(sd) & (w > 0)
    if m.any():
        mu = float((v[m] * w[m]).sum() / w[m].sum())
        out["offset_mean"] = mu
        out["offset_std"] = float(np.sqrt(max(((sd[m] ** 2 + v[m] ** 2) * w[m]).sum() / w[m].sum() - mu ** 2, 0.0)))
    else:
        out["offset_mean"] = out["offset_std"] = float("nan")
    a = np.concatenate([r["_absoff"] for r in rows]) if any(len(r["_absoff"]) for r in rows) else np.zeros(0)
    out["offset_hd95"] = float(np.percentile(a, 95)) if len(a) else float("nan")
    out["offset_p99"] = float(np.percentile(a, 99)) if len(a) else float("nan")
    tot = sum(r["path_um"] for r in rows)
    for suf in ("", "_break", "_merge"):
        out["erl" + suf + "_um"] = float(sum(r["_runsq" + suf] for r in rows) / tot) if tot > 0 else 0.0
    out["path_um"] = float(tot)
    out["lost_break_frac"] = float(sum(r["break_um"] for r in rows) / tot) if tot > 0 else 0.0
    out["lost_merge_frac"] = float(sum(r["merge_um"] for r in rows) / tot) if tot > 0 else 0.0
    return out


def bootstrap(rows, n=200, seed=0, lo=2.5, hi=97.5):
    """Resample SURFACES with replacement (points inside one surface are far too correlated to resample
    individually) and repool; returns {metric: [lo, hi]} 95% percentile intervals."""
    if len(rows) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = [pool([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(int(n))]
    keys = [k for k in draws[0] if isinstance(draws[0][k], float)]
    return {k: [float(np.nanpercentile([d[k] for d in draws], lo)),
                float(np.nanpercentile([d[k] for d in draws], hi))] for k in keys}


def mesh_reference(origin, size, surfaces):
    """The mesh-derived binary reference sheet for the box (quads filled in; see topo.rasterize)."""
    from usrm2 import topo
    return topo.rasterize([g for _, g, _, _ in surfaces], tuple(int(x) for x in size), origin)


def betti_of(p_u8, origin, size, surfaces, thr=0.5, margin=8, band=6, dilate=2.0, ref=None):
    """Betti-0/1 error of the thresholded band against the mesh-rasterized reference, on the box interior."""
    from usrm2 import topo
    ref = mesh_reference(origin, size, surfaces) if ref is None else ref
    return topo.betti_error(np.asarray(p_u8) >= thr * 255, ref, margin=margin, band=band, dilate=dilate)


def evaluate_all(p_u8, origin, size, pts, nrm, surfaces, tifxyz=TIFXYZ, ax=None, ct=None, thr=0.5, r=4,
                 far=40, um=2.4, boot=200, seed=0, betti=True, betti_margin=8, betti_band=6, betti_dilate=2.0, ref=None):
    """The whole v2 suite on one probability box: the pooled legacy numbers (unchanged), ERL, Betti-0/1,
    per-surface rows and bootstrap CIs."""
    m = metrics(p_u8, origin, pts, nrm, thr=thr, far=far)
    c = continuity(p_u8, origin, size, tifxyz, ax=ax, thr=thr, r=r, surfaces=surfaces)
    rows = surface_rows(p_u8, origin, size, tifxyz, ax=ax, ct=ct, thr=thr, r=r, far=far, um=um, surfaces=surfaces)
    pooled = pool(rows)
    out = {**m, **c, **{k: v for k, v in pooled.items() if k not in m and k not in c}}
    out["ci"] = bootstrap(rows, n=boot, seed=seed)
    out["surfaces"] = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    if betti:
        out["betti"] = betti_of(p_u8, origin, size, surfaces, thr=thr, margin=betti_margin, band=betti_band,
                                dilate=betti_dilate, ref=ref)
    return out


HEADLINE = ("recall@2", "recall@4", "recall@8", "offset_le3", "offset_mean", "offset_std", "offset_hd95",
            "offset_p99", "precision6", "merge_runs", "merge_frac", "continuity", "hit_frac", "mean_run",
            "erl_um", "erl_break_um", "erl_merge_um", "lost_break_frac", "lost_merge_frac", "path_um")


def table(res, ceil=None):
    """`metric  value (ceiling)  [lo, hi]`, the form every number is meant to be quoted in from now on."""
    ci, out = res.get("ci") or {}, []
    for k in HEADLINE:
        if k not in res:
            continue
        v = res[k]
        s = f"{k:<18} {v:>10.4g}"
        s += f" ({ceil[k]:.4g})" if ceil and k in ceil and np.isfinite(ceil[k]) else " " * 9
        s += f"  [{ci[k][0]:.4g}, {ci[k][1]:.4g}]" if k in ci else ""
        out.append(s)
    b, bc = res.get("betti"), (ceil or {}).get("betti")
    if b:
        out.append(f"{'betti0_err':<18} {b['betti0_err']:>10d}" + (f" ({bc['betti0_err']:d})" if bc else "")
                   + f"   b0 {b['betti0']} vs ref {b['betti0_ref']}")
        out.append(f"{'betti1_err':<18} {b['betti1_err']:>10d}" + (f" ({bc['betti1_err']:d})" if bc else "")
                   + f"   b1 {b['betti1']} vs ref {b['betti1_ref']}")
    return "\n".join(out)


def ceiling_cache_path(origin, size, store, tifxyz=TIFXYZ):
    import hashlib
    d = CEILING_CACHE or os.path.dirname(data.VAL.rstrip("/")) or "."
    k = hashlib.md5(f"{store}|{tuple(origin)}|{tuple(size)}|{tifxyz}|{data.UMBILICUS}".encode()).hexdigest()[:10]
    return f"{d}/evalsurf_ceiling_{origin[0]}_{origin[1]}_{origin[2]}_{k}.json"


def ceiling(origin, size, store=None, cache=True, **kw):
    """The noise ceiling: the identical metric suite scored for `store` (the published recto mask pyramid by
    default) against the same meshes. Cached per (box, store) as json next to the eval box, so every later
    run prints "value (ceiling)" for free (lit_evaluation_metrics section 6)."""
    store = store or CEILING_STORE
    p = ceiling_cache_path(origin, size, store, kw.get("tifxyz", TIFXYZ))
    if cache and os.path.exists(p):
        return {**json.load(open(p)), "cached": p}
    res = evaluate_all(read_box(store, origin, size), origin, size, **kw)
    res["source"] = store
    if cache:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            json.dump(res, open(p, "w"))
        except OSError as e:  # a read-only mirror must not lose the eval
            print(json.dumps({"ceiling_cache_error": repr(e)}))
    return res


# ---------------------------------------------------------------------------------------------------
# Plateau fitting (`usrm2 evalsurf-curve`): is this metric still moving, or is it done?
# ---------------------------------------------------------------------------------------------------

def fit_curve(steps, vals, bounded=True, gain95=0.95, smooth=1, tail=1.0):
    """Fit a saturating curve to (step, metric) and say whether the run is done.

    Two forms are tried and the lower-RMSE one wins (Hestness et al. 2017 for the power law; the
    literature's own caveat is that a power law does not saturate below 1, so for a [0,1] metric a
    logistic in log-step is usually the honest fit -- but it needs a FLOOR, since a val dice starts near
    0.3, not 0, and a floorless logistic just pins its asymptote to the upper bound):
        power      y = c - a * step^-alpha
        logistic4  y = y0 + (c - y0) / (1 + exp(-k * (log10 step - m)))
    `smooth` is the width of a centred running median applied first (raw per-checkpoint numbers are not
    monotone and an unsmoothed fit is unstable); `tail` keeps only the last fraction of the points, which
    is what the scaling-law literature fits when only the plateau matters.

    Returns the fitted asymptote `c`, `step95` (where 95% of the gain still outstanding at the LAST
    measured step has been collected) and `slope_per_10k` (dy/dstep * 1e4 at the last step). Do not trust
    a saturation call from fewer than ~10-15 points, and never without the Phase-A0 bootstrap CI to
    compare the slope against. `asymptote_at_bound` means the fit ran into the ceiling of the parameter
    range and the asymptote is not to be believed."""
    s = np.asarray(steps, float)
    y = np.asarray(vals, float)
    k = np.isfinite(s) & np.isfinite(y) & (s > 0)
    s, y = s[k], y[k]
    o = np.argsort(s)
    s, y = s[o], y[o]
    if int(smooth) > 1 and len(y) > int(smooth):
        w, z = int(smooth) | 1, y.copy()          # FULL windows only: a truncated window at the end drags
        h = w // 2                                 # a rising curve down and fakes a plateau
        for i in range(h, len(y) - h):
            z[i] = np.median(y[i - h:i + h + 1])
        y = z
    if 0 < tail < 1:
        s, y = s[int(len(s) * (1 - tail)):], y[int(len(y) * (1 - tail)):]
    if len(s) < 4:
        return {"model": None, "n": int(len(s)), "error": "need at least 4 points"}
    hi = 1.0 if bounded else float(y.max() * 4 + 1)
    ymax, ymin = float(y.max()), float(y.min())
    from scipy.optimize import curve_fit

    def power(x, c, a, al):
        return c - a * x ** (-al)

    def logis(x, c, kk, m, y0):
        return y0 + (c - y0) / (1.0 + np.exp(-kk * (np.log10(x) - m)))

    fits = []
    for f, p0, bnd in (
            (power, [min(ymax * 1.05, hi), max(ymax - ymin, 1e-3) * s[0] ** 0.5, 0.5],
             ([ymax, 0.0, 1e-3], [hi, np.inf, 5.0])),
            (logis, [min(ymax * 1.05, hi), 2.0, float(np.log10(s.mean())), ymin],
             ([ymax, 0.05, -10.0, ymin - 1.0], [hi, 10.0, 12.0, ymax]))):  # k <= 10: a steeper
             # logistic in log-step is a STEP function, which fits any finished run with zero residual
             # slope and would declare every run saturated
        try:
            p, _ = curve_fit(f, s, y, p0=p0, bounds=bnd, maxfev=60000)
            fits.append((float(np.sqrt(np.mean((f(s, *p) - y) ** 2))), f.__name__, p))
        except Exception:
            pass
    if not fits:
        return {"model": None, "n": int(len(s)), "error": "no fit converged"}
    rmse, name, p = min(fits, key=lambda t: t[0])
    c, last = float(p[0]), float(s[-1])
    ylast = float(power(last, *p) if name == "power" else logis(last, *p))
    rem = c - ylast
    if name == "power":
        _, a, al = p
        slope = float(a * al * last ** (-al - 1) * 1e4)
        s95 = float(np.exp(min(np.log(a / max((1 - gain95) * rem, 1e-12)) / al, 700.0))) if rem > 1e-9 else last
    else:
        _, kk, m, y0 = p
        e = np.exp(-kk * (np.log10(last) - m))
        slope = float((c - y0) * kk * e / (1 + e) ** 2 / (last * np.log(10)) * 1e4)
        t = ylast + gain95 * rem                       # the value 95% of the way to the asymptote
        z = (c - y0) / max(t - y0, 1e-12) - 1
        s95 = float(10 ** min(m - np.log(max(z, 1e-12)) / kk, 300.0)) if rem > 1e-9 and z > 0 else last
    at_bound = [bool(abs(float(v) - b) < 1e-6 * max(abs(b), 1.0)) for v, b in
                zip(p, ([hi, np.inf, 5.0] if name == "power" else [hi, 10.0, 12.0, ymax]))]
    return {"model": name, "n": int(len(s)), "rmse": rmse, "params": [float(x) for x in p],
            "asymptote": c, "asymptote_at_bound": bool(c >= hi - 1e-6), "params_at_bound": at_bound,
            "last_step": last,
            "last_value": ylast, "remaining": float(rem), "smooth": int(smooth), "tail": float(tail),
            "step95": s95, "steps_to_95": float(max(s95 - last, 0.0)), "slope_per_10k": slope}


def curve(run_dir, metric="dice", bounded=True, out=None, smooth=5, tail=1.0):
    """Fit the plateau of one metric over a run: `run_dir/eval.jsonl` (val dice per step) when it exists,
    otherwise a directory of `evalsurf --json` dumps (each carrying `step` and its pooled metrics)."""
    pts = []
    f = os.path.join(run_dir, "eval.jsonl")
    if os.path.exists(f):
        src = f
        for line in open(f):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if metric in d and d.get("step") is not None:
                pts.append((float(d["step"]), float(d[metric])))
    else:
        src = run_dir
        for g in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
            d = json.load(open(g))
            if d.get("step") is not None and metric in d:
                pts.append((float(d["step"]), float(d[metric])))
    pts = [p for p in pts if np.isfinite(p[1])]
    res = {"source": src, "metric": metric, "points": len(pts),
           **(fit_curve([p[0] for p in pts], [p[1] for p in pts], bounded=bounded, smooth=smooth, tail=tail)
              if len(pts) >= 4
              else {"model": None, "error": f"only {len(pts)} points"})}
    print(json.dumps(res))
    if res.get("model"):
        print(f"{metric}: asymptote {res['asymptote']:.4f}{' AT THE BOUND -- do not believe it' if res['asymptote_at_bound'] else ''} ({res['model']} fit, rmse {res['rmse']:.4g}, "
              f"{res['points']} points)\n  at step {res['last_step']:.0f}: {res['last_value']:.4f}, "
              f"slope {res['slope_per_10k']:+.4f} / 10k steps\n  95% of the remaining "
              f"{res['remaining']:.4f} by step {res['step95']:.0f} ({res['steps_to_95']:.0f} more)")
    if out:
        json.dump(res, open(out, "w"), indent=1)
    return res
