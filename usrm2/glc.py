"""GLC-style per-source loss weights from the human-verified meshes (section 26, experiment 8).

Gold Loss Correction (Hendrycks et al., NeurIPS 2018) estimates a label-noise transition matrix from a
small TRUSTED set and corrects the loss with it. `lit_noisy_labels_self_training.md` explicitly downgrades
that to what actually scales here: one pass of COUNTS against the trusted set, then a fixed (not learned)
per-source weight -- no meta-gradient, no bilevel optimisation, and never finer than (source x coarse
bucket), because a few hundred verified boxes cannot support a per-voxel noise matrix.

Our trusted set is the published tifxyz surfaces: they lie ON the recto face, so at a mesh point a
correct source has a band and away from every mesh point it should not. For each source that is one
2x2 confusion over the box:

    fnr  = share of mesh points with NO band within +-r voxels along the normal        (missed sheet)
    fpr  = share of the source's positive voxels farther than `r` from every mesh point (spurious sheet)
    J    = 1 - fnr - fpr                                                   (Youden's J, 1 = perfect)
    w    = max(J, 0) / max_over_sources(J)                                 (the suggested source weight)

`fpr` is a PROXY, not a true false-positive rate: the published surfaces do not cover every sheet in the
box, so a band on an unlabelled sheet counts against the source. It is comparable ACROSS sources over the
same box, which is all a relative weight needs, and it is the same quantity `evalsurf`'s `precision6`
reports. The survey's pitfall applies in full: the verified boxes are where the labelling was easy, so
these rates do not transfer to the hard regions, and a weight fitted here should be treated as a prior,
not a measurement.
"""
import json

import numpy as np

from usrm2 import data

R_DEFAULT = 4.0   # voxels along the normal that count as "the source found this sheet"
THR = 0.5


def profile_max(V, origin, pts, nrm, r=R_DEFAULT):
    """Max probability within +-r voxels along the normal at each mesh point."""
    from usrm2 import evalsurf as E
    q = (pts - np.asarray(origin, np.float32))
    ts = np.arange(-float(r), float(r) + 1e-6, 1.0, dtype=np.float32)
    return np.stack([E.trilerp(V, q + t * nrm) for t in ts]).max(0)


def rates(p_u8, origin, pts, nrm, r=R_DEFAULT, thr=THR):
    """(fnr, fpr, n_points, pos_frac) of one source over the box."""
    V = np.asarray(p_u8, np.float32) / 255.0
    if not len(pts):
        return float("nan"), float("nan"), 0, float((V >= thr).mean())
    hit = profile_max(V, origin, pts, nrm, r) >= thr
    fnr = float(1.0 - hit.mean())
    pos = np.argwhere(V >= thr).astype(np.float32)
    if len(pos):
        from scipy.spatial import cKDTree
        q = (pts - np.asarray(origin, np.float32)).astype(np.float32)
        d = cKDTree(q).query(pos, distance_upper_bound=float(r) + 2.0)[0]
        fpr = float((d > float(r)).mean())
    else:
        fpr = 0.0
    return fnr, fpr, int(len(pts)), float((V >= thr).mean())


def weights(rows):
    """{name: weight} from the per-source rows, normalised so the best source weighs 1.0."""
    j = {r["source"]: max(1.0 - r["fnr"] - r["fpr"], 0.0) for r in rows}
    best = max(j.values()) if j else 0.0
    return {k: (round(v / best, 3) if best > 0 else 0.0) for k, v in j.items()}


def run(sources, origin=None, size=None, tifxyz=None, volume=None, r=R_DEFAULT, thr=THR):
    """`sources`: {name: probability-store path}. Returns the report dict.

    Every store is read over the same box by its own `origin_zyx`, so a region teacher store and an
    exported pyramid box are compared on exactly the same voxels and the same mesh points.
    """
    from usrm2 import evalsurf as E
    o = tuple(origin or E.VAL_BOX[0])
    s = tuple(size or E.VAL_BOX[1])
    pts, nrm, counts = E.sites(o, s, tifxyz or E.TIFXYZ)
    ct = data.open_zarr(volume or data.CT)[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]]
    keep = ct[tuple(np.clip(np.rint(pts - o).astype(int), 0, np.array(s) - 1).T)] > 0
    pts, nrm = pts[keep], nrm[keep]
    rows = []
    for name, path in sources.items():
        p = E.read_box(path, o, s)
        fnr, fpr, n, pf = rates(p, o, pts, nrm, r=r, thr=thr)
        rows.append({"source": name, "store": str(path), "n_points": n, "fnr": round(fnr, 4),
                     "fpr": round(fpr, 4), "pos_frac": round(pf, 4),
                     "J": round(max(1.0 - fnr - fpr, 0.0), 4)})
    w = weights(rows)
    return {"box": [*o, *s], "surfaces": counts, "sources": rows, "weights": w,
            "source_w": " ".join(f"{k}={v}" for k, v in w.items())}


def main(sources, **kw):
    rep = run(sources, **kw)
    for row in rep["sources"]:
        print(json.dumps(row), flush=True)
    print(json.dumps({"weights": rep["weights"]}), flush=True)
    print(f"suggested: usrm2 train ... --source-w {rep['source_w']}", flush=True)
    return rep
