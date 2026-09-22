"""Cheap global topology numbers for a thresholded probability band (docs/unified_design.md section 25).

Betti numbers of a binary volume, computed on the CUBICAL COMPLEX in which every foreground voxel is a
closed unit cube. That complex is 26-connected for the foreground and 6-connected for the background, so
`b0` uses a 3x3x3 structuring element and `b2` uses the 6-neighbourhood of the complement.

  b0 = connected components of the foreground (26-conn)
  b2 = connected components of the complement (6-conn, one zero layer padded on) minus the unbounded one
  chi = V - E + F - C over the cells of the cubical complex (exact, no approximation)
  b1 = b0 + b2 - chi                     [the approximation: see below]

`b1` is derived from the Euler characteristic rather than counted, which is exact for a complex whose
homology is torsion-free over Z (true for a cubical subcomplex of R^3, by Alexander duality) -- i.e. the
number itself is right, but unlike a persistence computation it says nothing about WHERE the loops are,
and one spurious handle can cancel one missing loop. That is the documented Betti-NUMBER error of
Hu et al. 2019; the spatially matched version (Betti matching, Stucki et al. ICML 2023 / arXiv:2407.04683)
is a TODO -- it needs a persistent-homology dependency we do not have, and the 2024 GPU implementation is
C++.

Everything here is pure numpy + scipy.ndimage and is chunked along z so a 256x1024x1024 box does not need
more than a slab of bool at a time.
"""
import numpy as np

CHUNK = 64  # z-slices per slab of the Euler pass


def _or_count(mp, spec, chunk=CHUNK):
    """Number of sliding-window positions of `mp` (bool, already zero-padded by 1 on BOTH sides of every
    axis) that contain a True, where spec[a] is 2 for a window of 2 over the whole padded axis and 1 for a
    window of 1 over `mp[1:-1]` of that axis. Chunked along z; z-chunks overlap by spec[0]-1."""
    n, z = 0, mp.shape[0]
    lo, hi = (0, z - 1) if spec[0] == 2 else (1, z - 1)  # first / last+1 valid window start in axis 0
    for a in range(lo, hi, chunk):
        b = mp[a:min(a + chunk + spec[0] - 1, z)]
        b = (b[:, :-1] | b[:, 1:]) if spec[1] == 2 else b[:, 1:-1]
        b = (b[:, :, :-1] | b[:, :, 1:]) if spec[2] == 2 else b[:, :, 1:-1]
        b = (b[:-1] | b[1:]) if spec[0] == 2 else b
        n += int(b.sum())
    return n


def euler(m, chunk=CHUNK):
    """Euler characteristic chi = V - E + F - C of the closed-unit-cube complex of the bool volume `m`."""
    mp = np.zeros(tuple(s + 2 for s in m.shape), bool)
    mp[1:-1, 1:-1, 1:-1] = m
    V = _or_count(mp, (2, 2, 2), chunk)
    E = sum(_or_count(mp, tuple(1 if i == a else 2 for i in range(3)), chunk) for a in range(3))
    F = sum(_or_count(mp, tuple(2 if i == a else 1 for i in range(3)), chunk) for a in range(3))
    C = _or_count(mp, (1, 1, 1), chunk)
    return V - E + F - C


def betti(m, chunk=CHUNK):
    """(b0, b1, b2, chi) of a bool volume. b1 is the Euler-derived count (see the module docstring)."""
    from scipy import ndimage as ndi
    m = np.ascontiguousarray(m, bool)
    if not m.any():
        return 0, 0, 0, 0
    b0 = int(ndi.label(m, structure=np.ones((3, 3, 3), np.uint8))[1])
    bg = np.ones(tuple(s + 2 for s in m.shape), bool)
    bg[1:-1, 1:-1, 1:-1] = ~m
    b2 = int(ndi.label(bg, structure=ndi.generate_binary_structure(3, 1))[1]) - 1
    chi = euler(m, chunk)
    return b0, b0 + b2 - chi, b2, int(chi)


def band_of(ref, radius):
    """Voxels within `radius` of the reference sheet (EDT of ~ref), the region topology is measured in."""
    from scipy import ndimage as ndi
    if radius <= 0 or not ref.any():
        return np.ones_like(ref)
    return ndi.distance_transform_edt(~ref) <= float(radius)


def betti_error(pred, ref, margin=8, band=6, dilate=2.0, chunk=CHUNK):
    """Betti-0/1 error of `pred` against `ref` (both bool, same box), measured on the box INTERIOR.

    `margin` voxels are cropped off every face first, so a sheet the box merely cuts through does not
    read as a component or a loop that the model invented. `band` restricts both volumes to voxels within
    that many voxels of the reference sheet: the published meshes cover only SOME of the sheets crossing
    the box, so an unrestricted count would charge the model for every correctly predicted sheet that has
    no mesh. The band must stay below half the sheet pitch (15-35 voxels here) or two sheets' bands fuse.

    `dilate` thickens the reference to that many voxels either side. A mesh rasterizes to a ONE-voxel
    staircase, and a 26-connected staircase traps a background voxel in every corner: measured raw, the
    val-box reference has b2 = 48679 cavities and b1 = 19786 loops that are pure rasterization artefacts
    (dilate 1 -> 11897 / 7422, 2 -> 4765 / 3351, 3 -> 1706 / 1612). A predicted band at thr 0.5 is 3-5
    voxels thick and traps none of them, so the two must be put on the same footing first.
    """
    assert pred.shape == ref.shape, (pred.shape, ref.shape)
    from scipy import ndimage as ndi
    s = tuple(slice(margin, -margin if margin else None) for _ in range(3))
    p, r = np.ascontiguousarray(pred[s], bool), np.ascontiguousarray(ref[s], bool)
    d = ndi.distance_transform_edt(~r) if r.any() and (band > 0 or dilate > 0) else None
    b = np.ones_like(r) if d is None or band <= 0 else (d <= float(band))
    if d is not None and dilate > 0:
        r = d <= float(dilate)
    p, r = p & b, r & b
    p0, p1, p2, pc = betti(p, chunk)
    r0, r1, r2, rc = betti(r, chunk)
    return {"betti0": p0, "betti1": p1, "betti2": p2, "euler": pc,
            "betti0_ref": r0, "betti1_ref": r1, "betti2_ref": r2, "euler_ref": rc,
            "betti0_err": abs(p0 - r0), "betti1_err": abs(p1 - r1),
            "betti0_err_norm": abs(p0 - r0) / max(r0, 1), "betti1_err_norm": abs(p1 - r1) / max(r1, 1),
            "betti_margin": int(margin), "betti_band": float(band), "betti_dilate": float(dilate),
            "betti_interior_vox": int(p.size)}
    # TODO: Betti MATCHING error (Stucki et al., ICML 2023; arXiv:2407.04683) -- the counts above cannot
    # tell a loop in the right place from a loop in the wrong place. It needs a 3D persistent-homology
    # implementation; the efficient one is C++/CUDA and is not a dependency we carry today.


def rasterize(grids, shape, origin, step=0.7, pad=64.0):
    """Bool volume of the published surfaces: every mesh quad whose four corners are finite is sampled on a
    regular (u x u) lattice dense enough that consecutive samples are `step` voxels apart, and the samples
    are rounded into the box. The tifxyz grid is many voxels coarse, so the quads MUST be filled in or the
    reference would be a cloud of disconnected specks with a meaningless b0.

    A published surface spans the whole scroll and the box is one 256x1024x1024 window of it, so quads
    with no corner within `pad` voxels of the box are dropped BEFORE sampling -- otherwise almost all the
    work goes into samples that are then clipped away."""
    out = np.zeros(shape, bool)
    o = np.asarray(origin, np.float32)
    lo, hi = o - pad, o + np.asarray(shape, np.float32) + pad
    for g in grids:
        v = np.isfinite(g).all(-1)
        near = v & ((g >= lo) & (g < hi)).all(-1)
        q = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
        q &= near[:-1, :-1] | near[1:, :-1] | near[:-1, 1:] | near[1:, 1:]
        if not q.any():
            continue
        C = np.stack([g[:-1, :-1][q], g[1:, :-1][q], g[:-1, 1:][q], g[1:, 1:][q]]).astype(np.float32)  # (4,M,3)
        sh = np.asarray(shape)
        # The sampling density is PER QUAD, bucketed to powers of two: one hole in the grid can leave a
        # single quad hundreds of voxels wide, and a global density taken from it would cost 10^4 times
        # more samples on every ordinary 20-voxel quad.
        d = np.maximum(np.abs(C[1] - C[0]).max(-1), np.abs(C[2] - C[0]).max(-1))
        ub = np.clip(1 << np.ceil(np.log2(np.maximum(np.ceil(d / step) + 1, 2))).astype(int), 2, 512)
        for u in np.unique(ub):
            Cu = C[:, ub == u]
            u = int(u)
            t = np.linspace(0, 1, u, dtype=np.float32)
            wa, wb = t[:, None, None, None], t[None, :, None, None]  # (u,1,1,1) x (1,u,1,1)
            blk = max(1, 4_000_000 // (u * u))
            for j in range(0, Cu.shape[1], blk):  # keep the sample block bounded
                c = Cu[:, j:j + blk]
                p = ((1 - wa) * (1 - wb) * c[0] + wa * (1 - wb) * c[1] + (1 - wa) * wb * c[2] + wa * wb * c[3])
                i = np.rint(p.reshape(-1, 3) - o).astype(np.int64)
                i = i[((i >= 0) & (i < sh)).all(1)]
                out[i[:, 0], i[:, 1], i[:, 2]] = True
    return out
