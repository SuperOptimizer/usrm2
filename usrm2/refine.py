"""Prediction-guided surface refinement: move each point of a published tifxyz surface along its normal to the
nearest probability peak of a recto store, with a confidence-weighted smoothing of the displacement field over
the (H,W) grid so the sheet moves coherently. Points outside the store, in masked CT, or without a peak stay put.
All surfaces crossing the store's box are refined JOINTLY, lasagna-style: along each point's normal ray the
neighbouring sheets (nearest below and above) and the probability peaks are matched one-to-one in order, so
neighbouring sheets can neither merge onto one band nor cross, and a sheet that is a whole gap off still finds
its own band. Evaluate with a DIFFERENT store than the one refined on (--eval-store), or the gain is circular."""
import json
import os

import numpy as np

from usrm2 import data, evalsurf as E


def profile(V, q, n, far):
    """Probability sampled along the normal: (2*far+1, N)."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    return np.stack([E.trilerp(V, q + t * n) for t in ts])


def peaks(S, far, thr):
    """Sub-voxel peak offset per point (parabola through the argmax) and its confidence (peak height, 0 below thr)."""
    k = np.clip(S.argmax(0), 1, 2 * far - 1)
    y0, y1, y2 = (np.take_along_axis(S, k[None] + j, 0)[0] for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    off = (k - far) + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    conf = np.where(y1 >= thr, y1, 0.0).astype(np.float32)
    return off.astype(np.float32), conf


def local_maxima(S, far, thr, P=6):
    """Top-P local maxima of each profile: (positions (P,N) in voxels, strengths (P,N)); padding has strength -1."""
    ts = np.arange(-far, far + 1, dtype=np.float32)
    mid = S[1:-1]
    ismax = (mid >= S[:-2]) & (mid >= S[2:]) & (mid >= thr)
    st = np.where(ismax, mid, -1.0)
    idx = np.argsort(-st, axis=0)[:P]  # strongest first
    stren = np.take_along_axis(st, idx, 0)
    y0, y1, y2 = (np.take_along_axis(S, idx + 1 + j, 0) for j in (-1, 0, 1))
    den = np.minimum(y0 - 2 * y1 + y2, -1e-6)
    pos = ts[idx + 1] + np.clip(0.5 * (y0 - y2) / den, -1, 1)
    return pos.astype(np.float32), stren.astype(np.float32)


def assign(pos, stren, below, above, alpha=0.08):
    """Order-preserving one-to-one matching of the sheets on a ray (nearest other sheet below at offset `below`
    (<0, NaN if none), this sheet at 0, nearest above at `above` (>0, NaN if none)) to the peaks (pos, stren)
    (P,N). Cost = alpha * |offset - peak| - strength, summed over the matched sheets. Returns this sheet's
    displacement (NaN where no peak is available)."""
    from itertools import combinations
    P, N = pos.shape
    order = np.argsort(np.where(stren > 0, pos, np.inf), axis=0)  # peaks by position, padding last
    pos, stren = np.take_along_axis(pos, order, 0), np.take_along_axis(stren, order, 0)
    out = np.full(N, np.nan, np.float32)
    best = np.full(N, np.inf, np.float32)
    zero = np.zeros(N, np.float32)
    for K, sheets, me in ((3, (below, zero, above), 1), (2, (below, zero), 1), (2, (zero, above), 0), (1, (zero,), 0)):
        use = np.ones(N, bool)
        for s in sheets:
            use &= np.isfinite(s)
        use &= ~np.isfinite(best)  # only points not yet matched with more sheets
        if not use.any():
            continue
        offs = [np.nan_to_num(s) for s in sheets]
        for combo in combinations(range(P), K):
            valid = use & (stren[list(combo)] > 0).all(0)
            if not valid.any():
                continue
            cost = sum(alpha * np.abs(offs[k] - pos[c]) - stren[c] for k, c in enumerate(combo))
            better = valid & (cost < best)
            best[better], out[better] = cost[better], pos[combo[me]][better]
        best[use & np.isfinite(best)] = np.minimum(best[use & np.isfinite(best)], 1e6)  # matched at this K: done
    return out


def ray_neighbours(q, n, others, R, lateral=3.0):
    """Offsets along the normal of the nearest other-sheet points below (<0) and above (>0) each point, NaN if none."""
    from scipy.spatial import cKDTree
    below, above = np.full(len(q), np.nan, np.float32), np.full(len(q), np.nan, np.float32)
    if not len(others):
        return below, above
    tree = cKDTree(others)
    for i, nb in enumerate(tree.query_ball_point(q, r=R + lateral)):
        if not nb:
            continue
        d = others[nb] - q[i]
        t = d @ n[i]
        lat = np.linalg.norm(d - t[:, None] * n[i], axis=1)
        t = t[(lat <= lateral) & (np.abs(t) <= R)]
        if (t < -0.5).any():
            below[i] = t[t < -0.5].max()
        if (t > 0.5).any():
            above[i] = t[t > 0.5].min()
    return below, above


def smooth(field, w, sigma):
    """Confidence-weighted Gaussian smoothing over the grid (normalized convolution); NaN/zero-weight cells get
    the neighbourhood's value."""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.nan_to_num(field * w), sigma)
    den = gaussian_filter(w, sigma)
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)


def filled(g, sigma=1.0):
    """The grid with holes (NaN) filled from their neighbourhood, for normal estimation only."""
    w = np.isfinite(g).all(-1).astype(np.float32)
    return np.stack([smooth(np.where(w > 0, g[..., i], 0), w, sigma) for i in range(3)], -1)


def normals(g, ax):
    """Hole-tolerant unit normals (oriented outward from the axis): computed on the filled grid."""
    n = E.normals(filled(g), ax)
    return np.where(np.isfinite(g).all(-1)[..., None], n, np.nan)


def refine_many(grids, V, origin, ax, far=12, sigma=2.0, iters=3, thr=0.5, ct=None, chunk=2_000_000):
    """Joint refinement of several (H,W,3) zyx grids (NaN = hole) against V, a (Z,Y,X) probability in [0,1] at
    `origin`. Each iteration: normals; per point the peaks along the normal ray and the neighbouring sheets on
    it are matched in order (assign()); the matched peak's offset is smoothed over the grid (confidence-weighted)
    and the sheet moves. Returns (refined grids, per-iteration stats)."""
    o = np.asarray(origin, np.float32)
    grids = [g.copy() for g in grids]
    lo, hi = o, o + np.array(V.shape, np.float32) - 1
    insides = [np.isfinite(g).all(-1) & ((g >= lo) & (g <= hi)).all(-1) for g in grids]
    stats = []
    for it in range(iters):
        r = max(3, int(round(far / (1.5 ** it))))
        st = {"iter": it, "far": r, "points": 0, "with_peak": 0.0, "mean_abs_move": 0.0, "max_move": 0.0, "capped": 0.0}
        ns = [normals(g, ax) for g in grids]
        oks = [ins & np.isfinite(n).all(-1) for ins, n in zip(insides, ns)]
        pts = [g[ok] for g, ok in zip(grids, oks)]
        moves, tot = [], 0
        for j, (g, n, ok, q) in enumerate(zip(grids, ns, oks, pts)):
            if not ok.any():
                moves.append(np.zeros(g.shape[:2], np.float32)); continue
            qi = q - o
            others = np.concatenate([p for k, p in enumerate(pts) if k != j and len(p)] or [np.zeros((0, 3), np.float32)])
            nn = n[ok]
            below, above = ray_neighbours(q, nn, others, r)
            off, conf = np.zeros(len(q), np.float32), np.zeros(len(q), np.float32)
            for i in range(0, len(q), chunk):
                sl = slice(i, i + chunk)
                pos, stren = local_maxima(profile(V, qi[sl], nn[sl], r), r, thr)
                d = assign(pos, stren, below[sl], above[sl])
                hit = np.isfinite(d)
                off[sl] = np.where(hit, d, 0.0)
                conf[sl] = np.where(hit, stren.max(0), 0.0)
            if ct is not None:  # masked CT: no evidence
                idx = np.clip(np.rint(qi).astype(int), 0, np.array(V.shape) - 1)
                conf[ct[idx[:, 0], idx[:, 1], idx[:, 2]] == 0] = 0
            # never cross a neighbouring sheet: stay at least 1 voxel on this side of it
            hi = np.where(np.isfinite(above), above - 1.0, r).astype(np.float32)
            lo = np.where(np.isfinite(below), below + 1.0, -r).astype(np.float32)
            field, w = np.zeros(g.shape[:2], np.float32), np.zeros(g.shape[:2], np.float32)
            field[ok], w[ok] = np.clip(off, lo, hi), conf
            dd = smooth(field, w, sigma)
            lof, hif = np.full(g.shape[:2], -float(r), np.float32), np.full(g.shape[:2], float(r), np.float32)
            lof[ok], hif[ok] = lo, hi
            dd = np.clip(dd, lof, hif) * ok
            moves.append(dd)
            st["points"] += int(ok.sum()); tot += len(q)
            st["with_peak"] += float((conf > 0).sum()); st["capped"] += float((np.isfinite(below) | np.isfinite(above)).sum())
            st["mean_abs_move"] += float(np.abs(dd[ok]).sum()); st["max_move"] = max(st["max_move"], float(np.abs(dd).max()))
        for j, (g, n, dd) in enumerate(zip(grids, ns, moves)):  # move all sheets after all were measured
            grids[j] = g + dd[..., None] * np.where(np.isfinite(n), n, 0)
        for k in ("with_peak", "mean_abs_move", "capped"):
            st[k] = st[k] / max(tot, 1)
        stats.append(st)
    return grids, stats


def refine(g, V, origin, ax, **kw):
    gs, stats = refine_many([g], V, origin, ax, **kw)
    return gs[0], stats


def write_tifxyz(src_dir, out_dir, g, note):
    """A tifxyz directory like `src_dir` with the refined points (meta.json copied, bbox recomputed)."""
    import tifffile
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate("zyx"):
        tifffile.imwrite(f"{out_dir}/{c}.tif", np.where(np.isfinite(g[..., i]), g[..., i], 0).astype(np.float32))
    meta = json.load(open(f"{src_dir}/meta.json"))
    v = g[np.isfinite(g).all(-1)]
    meta["bbox"] = [v.min(0)[::-1].tolist(), v.max(0)[::-1].tolist()] if len(v) else meta.get("bbox")
    meta["refined"] = note
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
    for f in os.listdir(src_dir):  # any other per-surface files ride along untouched
        if f not in ("z.tif", "y.tif", "x.tif", "meta.json") and os.path.isfile(f"{src_dir}/{f}"):
            import shutil
            shutil.copy(f"{src_dir}/{f}", f"{out_dir}/{f}")
    return out_dir


def surfaces_in(tifxyz, origin, size, min_pts=200):
    """tifxyz directories whose points fall inside the box (bbox test, then a point count)."""
    import glob
    o, s, out = np.asarray(origin, np.float32), np.asarray(size, np.float32), []
    for m in sorted(glob.glob(f"{tifxyz}/*/*/meta.json")) or sorted(glob.glob(f"{tifxyz}/*/meta.json")):
        b = np.asarray(json.load(open(m))["bbox"], np.float32)[:, ::-1]
        if (b[1] < o).any() or (b[0] > o + s).any():
            continue
        d = os.path.dirname(m)
        g = E.read_surface(d)
        if (np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1)).sum() >= min_pts:
            out.append(d)
    return out


def run(surfaces, store, out_root, eval_store=None, far=12, sigma=2.0, iters=3, thr=0.5, volume=None, tifxyz=None):
    """Refine `surfaces` (tifxyz dirs; or every surface of `tifxyz` crossing the store's box) jointly with `store`,
    write them under out_root/<name>, and report metrics before/after on `eval_store` (default: `store`)."""
    a = data.open_zarr(store)
    o, s = data.box(a)
    if tifxyz:
        surfaces = list(surfaces or []) + surfaces_in(tifxyz, o, s)
    assert surfaces, "no surface crosses the store's box"
    V = np.asarray(a[(0,) + (slice(None),) * 3] if a.ndim == 4 else a[:], np.float32) / 255.0
    ct = data.open_zarr(volume or a.attrs.get("volume", data.CT))[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
    ax = data.axis(a.attrs.get("umbilicus", None))
    g0 = [E.read_surface(d) for d in surfaces]
    print(json.dumps({"box": [*o.tolist(), *s.tolist()], "surfaces": [os.path.basename(d.rstrip("/")) for d in surfaces]}))
    g1, stats = refine_many(g0, V, o, ax, far=far, sigma=sigma, iters=iters, thr=thr, ct=ct)
    for st in stats:
        print(json.dumps(st))
    note = {"store": str(store), "far": far, "sigma": sigma, "iters": iters, "thr": thr, "joint_with": len(surfaces)}
    outs = [write_tifxyz(d, os.path.join(out_root, os.path.basename(d.rstrip("/"))), g, note) for d, g in zip(surfaces, g1)]
    ev = eval_store or store
    if ev != store:
        b = data.open_zarr(ev)
        assert data.box(b)[0].tolist() == o.tolist() and tuple(b.shape[-3:]) == tuple(s), "eval store must cover the same box"
        V = np.asarray(b[(0,) + (slice(None),) * 3] if b.ndim == 4 else b[:], np.float32) / 255.0
    Vu = np.clip(np.rint(V * 255), 0, 255).astype(np.uint8)
    for d, ga, gb in zip(surfaces, g0, g1):
        rec = {"surface": os.path.basename(d.rstrip("/"))}
        for name, g in (("before", ga), ("after", gb)):
            k = np.isfinite(g).all(-1) & ((g >= o) & (g < o + s)).all(-1)
            n = normals(g, ax)
            k &= np.isfinite(n).all(-1)
            m = E.metrics(Vu, o, g[k], n[k]) if k.any() else {}
            rec[name] = {q: round(m[q], 4) for q in ("recall@2", "recall@4", "offset_mean", "offset_std", "offset_le3", "merge_frac") if q in m}
            rec["points"] = int(k.sum())
        print(json.dumps(rec))
    return outs
