"""Phase B target pyramids: signed distance, midline and thickness (docs/unified_design.md section 29).

`dist_pyramid` turns an exported MASK pyramid (and, where there is one, the paired VERSO source) into a
second pyramid of the same shape whose voxels hold a DISTANCE instead of a probability. Three flavours,
all uint8, all with the same reserved no-data code:

    sdist      signed distance to the RECTO FACE, positive on the recto (radially outward) side
    midline    signed distance to the sheet MIDLINE, positive on the recto side
    thickness  the recto-to-verso separation along the normal, lower-bounded by `tmin`

Encoding (the tracer contract, docs/research/synthesis_v2_with_literature.md section 2):

    signed   code = 128 + round(d / 0.25),  clamped to 1..255  ->  d = (code - 128) * 0.25 voxels
    unsigned code =       round(t / 0.25),  clamped to 1..255  ->  t =  code        * 0.25 voxels
    code 0   = NO DATA (weight 0)

i.e. offset 128 and 0.25-voxel units, cap +-31.75 voxels, which is the +-32 cap of the contract with the
one endpoint given up so that 0 can be the no-data marker -- the contract's own rule, "0 is reserved for
no-data in every uint8 distance field". The units are WORKING VOXELS OF THAT RUNG, so every store records
`voxel_um`, and a rung's field is recomputed from that rung's own mask: distances are NEVER pooled. A 2x
mean pool of a distance field is not the distance field of the pooled mask (it is a distance in the FINE
rung's voxels, halved by nothing), so `data.read_rung`'s pooling fallback must never be reached -- which
is why the loader gives a distance channel weight 0 at any rung the store does not itself hold
(`data._rung_target`).

Weight-0 (code 0) is written wherever the field is not trustworthy:
  * the mask is absent (no level at that rung, or the block is all air) -- nothing to measure from;
  * above rung 4, and above the mask's own native rung + 1 -- higher up the exported "mask" is a pooled
    FRACTION, not a band, and its 0.5 level set is not a surface;
  * within `axis_r_um` microns of the umbilicus axis -- the core is crushed and the published masks put
    `recto_is_in` near 0.5 there (tsm measured ~0.47 and dropped those voxels; so do we);
  * where CT is 0 -- applied by the loader, not here (`_rung_target` masks every channel with CT > 0).

Cost: two scipy EDTs per block per rung, CPU only. This is the offline step Phase B is waiting on; it must
NOT be run on a card the training jobs are using.
"""
import json
import math
import os
import re

import numpy as np

from usrm2 import data

UNIT = 0.25          # voxels per code step
OFF = 128            # the code of distance 0
CAP = 31.75          # +-CAP voxels is the representable range (code 1 .. 255)
TMIN = 3.0           # voxels: the floor on a predicted / stored sheet thickness (see `pair_bands`)
AXIS_R_UM = 400.0    # microns around the umbilicus axis that are dropped
MAX_RUNG = 4         # no distance target above this rung
CHANNELS = {"face": "sdist", "midline": "midline", "thickness": "thickness"}
SUFFIX = {"face": "_sdist", "midline": "_midline", "thickness": "_thick"}


# ------------------------------------------------------------------------------------ the encoding

def encode_signed(d, valid, cap=CAP):
    """float voxels -> uint8, with `valid` False becoming code 0 (no data)."""
    c = np.rint(np.clip(d, -cap, cap) / UNIT) + OFF
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def decode_signed(u):
    """uint8 -> float voxels (code 0 decodes to 0; the CALLER must use the weight, not the value)."""
    return (np.asarray(u, np.float32) - OFF) * UNIT


def encode_unsigned(t, valid, cap=255 * UNIT):
    c = np.rint(np.clip(t, UNIT, cap) / UNIT)
    return np.where(valid, np.clip(c, 1, 255), 0).astype(np.uint8)


def decode_unsigned(u):
    return np.asarray(u, np.float32) * UNIT


# ------------------------------------------------------------------------------- geometry per block

def axis_offsets(ax, lo, shape):
    """(dy, dx, r) of every voxel of the block from the scroll axis, in rung voxels.

    `ax` is `data.axis_at(axis, k)` -- the umbilicus polyline resampled to rung k -- and `lo` the block
    corner. This is exactly what `data.radial` / `prep.radial_t` build the radial unit vector from; the
    radius comes out of the same expression, so the radial channel and the axis exclusion can never
    disagree about where the axis is."""
    Z, Y, X = (int(v) for v in shape)
    z = np.arange(Z, dtype=np.float64) + int(lo[0])
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = (np.arange(Y, dtype=np.float64) + int(lo[1]))[None, :, None] - cy[:, None, None]
    dx = (np.arange(X, dtype=np.float64) + int(lo[2]))[None, None, :] - cx[:, None, None]
    r = np.sqrt(dy * dy + dx * dx)
    return dy.astype(np.float32), dx.astype(np.float32), r.astype(np.float32)


def medial(band):
    """A one-voxel-thick medial surface of a binary band: the voxels of `band` that are a local maximum
    of the distance to background. The same construction as `losses.skeleton`, on the CPU with an exact
    Euclidean transform instead of a capped Chebyshev one."""
    from scipy import ndimage as ndi
    if not band.any():
        return np.zeros(band.shape, bool)
    d = ndi.distance_transform_edt(band)
    return band & (d >= ndi.maximum_filter(d, size=3, mode="nearest") - 1e-6)


def signed_to(surf, dy, dx, cap=CAP):
    """Signed distance (voxels) to the 1-voxel surface `surf`, POSITIVE on the radially outward side.

    `scipy.ndimage.distance_transform_edt(..., return_indices=True)` gives, per voxel, both the distance
    to the nearest surface voxel and WHICH voxel that is; the displacement from that voxel to this one,
    dotted with the (unnormalised) radial direction (dy, dx, z-component 0), is the side. The radial
    vector is the same one the model gets as an input channel, so "positive" means the same thing in the
    target, in `prep.radial_t` and in the exported normal (`dot(n, radial) > 0`).

    Where the displacement is exactly perpendicular to the radial direction the side is arbitrary; that
    is a measure-zero set on a sheet that is roughly perpendicular to the radius, and it is resolved to +.
    """
    from scipy import ndimage as ndi
    if not surf.any():
        return np.zeros(surf.shape, np.float32), np.zeros(surf.shape, bool)
    u, ix = ndi.distance_transform_edt(~surf, return_indices=True)
    gy = (np.arange(surf.shape[1], dtype=np.int32)[None, :, None] - ix[1]).astype(np.float32)
    gx = (np.arange(surf.shape[2], dtype=np.int32)[None, None, :] - ix[2]).astype(np.float32)
    s = gy * dy + gx * dx
    d = np.where(s < 0, -1.0, 1.0).astype(np.float32) * u.astype(np.float32)
    return np.clip(d, -cap, cap), np.ones(surf.shape, bool)


def block_fields(recto, verso, dy, dx, thr=0.5, cap=CAP, tmin=TMIN):
    """(sdist, midline, thickness, valid_rv) of one block.

    `recto` / `verso` are uint8 probability blocks (verso may be None). The recto face is the medial
    surface of the recto band and `d_r` is the signed distance to it; likewise `d_v` for the verso band.
    Along the radial direction a sheet sits between the two faces, so for a voxel at radial coordinate x,
    a recto face at a and a verso face at a - t,

        d_r = x - a           d_v = x - (a - t) = d_r + t

    and therefore, exactly,

        midline  m = (d_r + d_v) / 2        thickness  t = d_v - d_r

    Without a verso band there is no second face: the midline falls back to the recto face (m = d_r) and
    the thickness is NOT measurable, so its `valid_rv` is False and the loader gives it weight 0."""
    br = recto >= int(round(thr * 255))
    dr, ok = signed_to(medial(br), dy, dx, cap)
    if verso is None:
        return dr, dr, np.zeros_like(dr), np.zeros(dr.shape, bool)
    bv = verso >= int(round(thr * 255))
    if not bv.any():
        return dr, dr, np.zeros_like(dr), np.zeros(dr.shape, bool)
    dv, _ = signed_to(medial(bv), dy, dx, cap)
    m = 0.5 * (dr + dv)
    t = np.maximum(dv - dr, tmin)
    return dr, m, t, ok


# --------------------------------------------------------------------------------- the verso source

class VersoSource:
    """Whatever holds the verso band, read like a pyramid level.

    Two shapes are supported, because the verso target exists in two: a pyramid GROUP (a `verso` export,
    if one is ever made) and the REGION STORES the pod publishes (`<root>/verso/region_<z>_<y>_<x>.zarr`,
    1024^3 at rung 2, rung 3 their 2x mean pool -- `data.read_teacher`'s rule). `read` returns None where
    there is no finished store, which is what makes the midline fall back to the recto face."""

    def __init__(self, spec):
        self.spec, self.pyr, self.root = str(spec), None, None
        if os.path.isdir(os.path.join(self.spec, data.VERSO)) or re.search(r"region_\d+_\d+_\d+", self.spec):
            self.root = self.spec
        else:
            self.pyr = data.rungs(self.spec)

    def read(self, k, lo, shape):
        if self.pyr is not None:
            return None if k not in self.pyr else data.read_rung(self.pyr, k, lo, shape, dtype=np.uint8)
        if int(k) not in (2, 3):
            return None
        d, p = int(k) - 2, np.asarray(shape, np.int64)
        lo2, hi2 = np.asarray(lo, np.int64) << d, ((np.asarray(lo, np.int64) + p) << d) - 1
        a, b = lo2 // data.REGION, hi2 // data.REGION
        if not np.array_equal(a, b) or (lo2 < 0).any():
            return None                                  # the block straddles two region stores
        try:
            z = data.open_zarr(data.teacher_region_path(a * data.REGION, data.VERSO, self.root))
        except Exception:  # noqa: BLE001
            return None
        if not z.attrs.get("done"):
            return None
        v, ins = data.read_teacher(z, k, lo, p)
        return np.where(ins, v, np.uint8(0))


# ------------------------------------------------------------------------------------- the pyramid

def group_json(path, attrs):
    """A zarr v3 GROUP node beside the levels, so `data.rungs` / `data.source_groups` can open the store
    as a target pyramid (the levels are named by voxel size in microns, like the exported predictions)."""
    os.makedirs(path, exist_ok=True)
    tmp = os.path.join(path, "zarr.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"zarr_format": 3, "node_type": "group", "attributes": attrs}, f, indent=1)
    os.replace(tmp, os.path.join(path, "zarr.json"))


def out_name(mask, kind):
    """'<...>/x.zarr' -> '<...>/x_sdist.zarr' (face), '_midline.zarr', '_thick.zarr'."""
    b = data.pyramid_base(str(mask).rstrip("/"))
    root, ext = (b[:-5], ".zarr") if b.endswith(".zarr") else (b, "")
    return root + SUFFIX[kind] + ext


def pad128(shape):
    return tuple(-(-int(s) // 128) * 128 for s in shape)


def dist_pyramid(mask, out=None, verso=None, kinds=("face",), rungs=range(0, MAX_RUNG + 1),
                 volume=None, umbilicus=None, axis_r_um=AXIS_R_UM, tmin=TMIN, cap=CAP, thr=0.5,
                 block=128, halo=48, max_rung=MAX_RUNG, native_plus=1, box=None, dry_run=False,
                 log=print):
    """Write the distance pyramid(s) of a mask pyramid. Returns {kind: path}.

    One pass per rung per kind. A rung is written only when the MASK ITSELF has a level there (never a
    pooled one: `data.read_rung` would pool the mask, and a distance computed from a pooled fraction is
    not a distance), when the rung is at most `max_rung`, and when it is at most `native + native_plus`
    -- above that the exported level is an area fraction, not a band, and its 0.5 level set is not a
    surface. Every other rung simply has no level in the output, and the loader gives the channel weight
    0 there (`data._rung_target`).

    Blocks are `block`^3 with a `halo` of context on every side, so the EDT of a block agrees with the
    EDT of the whole level everywhere within `cap` voxels of a surface (halo >= cap + the band's own
    half-thickness). A block whose mask is entirely background is skipped: its shard is never written and
    reads back as code 0, i.e. no data.

    `box` ((z, y, x), (Z, Y, X)) at RUNG 2, as every box in this codebase is: only the blocks inside it
    are visited. A whole Paris 4 level is 10^12 rung-2 voxels, so the full pass is a region-by-region
    job; this is also how the held-out box alone is done, to check the numbers before committing a scroll.
    """
    kinds = tuple(kinds)
    assert set(kinds) <= set(SUFFIX), f"kinds {kinds}: a subset of {sorted(SUFFIX)}"
    assert halo >= cap + 8, f"--halo {halo} is below the clamp {cap} + the band half-width"
    pyr = data.rungs(mask)
    nat = min(pyr)
    vs = None if not verso else VersoSource(verso)
    ax0 = data.axis(umbilicus or data.group_attrs(mask).get("umbilicus") or None)
    outs = {k: (out if out and len(kinds) == 1 else out_name(mask, k)) for k in kinds}
    ks = sorted(k for k in rungs if k in pyr and k <= max_rung and k <= nat + native_plus)
    log(f"dist-pyramid {mask}: rungs {ks} (native {nat}, pyramid has {sorted(pyr)}) -> "
        + ", ".join(f"{k}:{v}" for k, v in outs.items()))
    if dry_run:
        return outs
    made = {k: [] for k in kinds}
    for k in ks:
        shape = tuple(int(v) for v in data.rung_shape(pyr, k))
        um = data.rung_um(k)
        ax = data.axis_at(ax0, k)
        rmin = float(axis_r_um) / um
        arrs = {}
        for kd in kinds:
            p = f"{outs[kd]}/{um:g}"
            arrs[kd] = _level(p, pad128(shape), um, k, volume, umbilicus)
            made[kd].append(f"{um:g}")
        nb = nw = 0
        for lo in _blocks(shape, block, _box_at(box, k)):
            olo = np.maximum(np.asarray(lo, np.int64) - halo, -halo)
            osh = tuple(int(min(block, shape[i] - lo[i])) + 2 * halo for i in range(3))
            rec = data.read_rung(pyr, k, olo, osh, dtype=np.uint8)
            nb += 1
            if not (rec >= int(round(thr * 255))).any():
                continue
            nw += 1
            ver = None if vs is None else vs.read(k, olo, osh)
            dy, dx, r = axis_offsets(ax, olo, osh)
            dr, m, t, okt = block_fields(rec, ver, dy, dx, thr=thr, cap=cap, tmin=tmin)
            ok = r >= rmin
            sl = tuple(slice(halo, halo + int(min(block, shape[i] - lo[i]))) for i in range(3))
            for kd in kinds:
                u = (encode_signed(dr, ok, cap) if kd == "face" else
                     encode_signed(m, ok, cap) if kd == "midline" else
                     encode_unsigned(t, ok & okt))
                _put(arrs[kd], u[sl], *lo)
            if nw % 200 == 0:
                log(f"  rung {k} ({um:g} um): {nw} blocks written / {nb} visited", flush=True)
        log(f"  rung {k} ({um:g} um) {shape}: {nw} blocks written / {nb} visited")
    for kd in kinds:
        # a second run of this tool in the same process must not read the previous run's array handles
        # NOR its decoded levels (`data.full_level` caches whole small levels by "<dir>#<rung>")
        for key in [q for q in list(data.CTX_CACHE) if str(q).startswith(str(outs[kd]))]:
            data.CTX_CACHE.pop(key, None)
        group_json(outs[kd], {
            "channel": CHANNELS[kd], "channels": [CHANNELS[kd]], "weight": 1.0,
            "levels": made[kd], "encoding": ("signed_u8_off128_q0.25" if kd != "thickness" else
                                             "unsigned_u8_q0.25"),
            "unit": "voxels_of_this_rung", "no_data": 0, "clamp_voxels": float(cap),
            "sign_convention": "recto_positive: d > 0 on the radially OUTWARD side of the surface "
                               "(dot(v - nearest_surface_point, radial) > 0), the same radial vector "
                               "usrm2.prep.radial_t feeds the model; the recto face of a sheet is its "
                               "outward face, so the recto side is positive and the verso side negative",
            "source_mask": str(mask), "source_verso": (str(verso) if verso else None),
            "axis_exclusion_um": float(axis_r_um), "tmin_voxels": float(tmin),
            "max_rung": int(max_rung), "native_rung": int(nat),
            "umbilicus": str(umbilicus) if umbilicus else None,
            "volcomp": {"rung_voxel_size_um": data.rung_um(nat)},
        })
    return outs


def _box_at(box, k):
    """A rung-2 ((origin), (size)) box expressed in rung-k voxels, snapped OUTWARDS."""
    if box is None:
        return None
    d = int(k) - 2
    o, s = np.asarray(box[0], np.int64), np.asarray(box[1], np.int64)
    return ((o >> d, np.maximum(-(-s >> d), 1)) if d >= 0 else (o << -d, s << -d))


def _blocks(shape, block, box=None):
    lo = np.zeros(3, np.int64) if box is None else np.maximum(box[0] // block * block, 0)
    hi = np.asarray(shape, np.int64) if box is None else np.minimum(box[0] + box[1], shape)
    for z in range(int(lo[0]), int(hi[0]), block):
        for y in range(int(lo[1]), int(hi[1]), block):
            for x in range(int(lo[2]), int(hi[2]), block):
                yield (z, y, x)


def _put(arr, u8, z, y, x):
    from usrm2 import predict as P
    P.put(arr, u8, z, y, x)


def _level(path, shape, um, k, volume, umbilicus):
    from usrm2 import predict as P
    # q=0, LOSSLESS. volcomp q8 -- the probability stores' setting -- rounds: a stored 0 can read back
    # as a 6, which would silently turn the no-data marker into a -30.5-voxel distance, and it compounds
    # under the partial-chunk writes a block smaller than 128 makes. A distance field is smooth, so the
    # lossless codec still compresses it well.
    a = P.out_array(path, shape, (0, 0, 0), volcomp=True, volume=volume, umbilicus=umbilicus,
                    rung=k, channels=(CHANNELS["face"],), q=0)
    a.attrs.update({"voxel_um": float(um), "rung": int(k)})
    return a
