"""CT + teacher patch sampling. All boxes are (z, y, x) level-0 voxels.

Teacher stores: uint8 = recto probability * 255, shape (Z,Y,X) (or (1,Z,Y,X)),
attrs origin_zyx = the level-0 index of element 0. CT: uint8, 0 = air/masked.
A store entry "a.zarr,a_m7.zarr" names several teachers over the SAME box: one target channel each
(a multi-head student); the first one supplies the box, volume and umbilicus.
"""
import itertools
import json
import math
import os
import re

import numpy as np
import torch

import os as _os
# defaults are the forlindesk2 layout; USRM2_* environment variables override them on other machines
CT = _os.environ.get("USRM2_CT", "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/0")
TRAIN = ["/vesuvius/usrm2/teacher/a.zarr", "/vesuvius/usrm2/teacher/b.zarr"]  # made by `usrm2 teacher`; plus
TRAIN += sorted(__import__("glob").glob("/vesuvius/usrm2/teacher/boxes/box_*.zarr"))  # `usrm2 teacher-boxes` output
VAL = _os.environ.get("USRM2_VAL", "/vesuvius/usrm2/teacher/eval.zarr")
UMBILICUS = _os.environ.get("USRM2_UMBILICUS", "/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json")
MARGIN = 16  # sliding-window predictions are worse at the teacher box edges


LOCAL_VOLUMES = "/vesuvius/usrm/volcomp"  # mirror of dl.ash2txt.org/community-uploads/forrest/volcomp/<scroll>/volumes/


STREAM_VOLUMES = "https://dl.ash2txt.org/community-uploads/forrest/volcomp"  # the mirror's origin


def local(path):
    """A streamed volume URL -> its local mirror when present (stores made in the cloud name the URL), and a
    mirror path that does not exist here -> the streamed URL (stores made on the desk, read in the cloud)."""
    import os
    if "://" in path and "/volcomp/" in path:
        scroll, _, rest = path.split("/volcomp/", 1)[1].partition("/volumes/")
        cand = f"{LOCAL_VOLUMES}/{scroll}/{rest}"
        return cand if os.path.exists(cand) else path
    if path.startswith(LOCAL_VOLUMES + "/") and not os.path.exists(path):
        scroll, _, rest = path[len(LOCAL_VOLUMES) + 1:].partition("/")
        return f"{STREAM_VOLUMES}/{scroll}/volumes/{rest}"
    return path


def open_zarr(path):
    import zarr
    try:
        import volcomp_zarr  # noqa: F401  (registers the "volcomp" codec)
    except Exception:
        pass
    path = local(str(path))
    if "://" in path:  # streamed: more chunk fetches in flight, and a stalled request fails instead of hanging
        import aiohttp
        zarr.config.set({"async.concurrency": 64})
        return zarr.open(path, mode="r", storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=300)}})
    return zarr.open(path, mode="r")


def box(arr):
    """(origin_zyx, shape_zyx) of a teacher store."""
    return np.array(arr.attrs["origin_zyx"], np.int64), np.array(arr.shape[-3:], np.int64)


def shape3(p):
    """A patch size: an int (cube) or a (Z, Y, X) triple -> np.int64[3]."""
    return np.array([p, p, p] if np.isscalar(p) else p, np.int64)


def read3(arr, o, p):
    """Read a patch (cube or (Z,Y,X)) at store-local offset o, tolerating (1,Z,Y,X) stores."""
    s = tuple(slice(int(a), int(a) + int(n)) for a, n in zip(o, shape3(p)))
    return arr[(0,) + s] if arr.ndim == 4 else arr[s]


def axis(path=None):
    """(z, y, x) arrays of the scroll axis control points, sorted by z (level-0 voxels).
    The default is read at call time, so `--umbilicus` (which rebinds UMBILICUS) is honoured."""
    import json, os
    path = path or UMBILICUS
    if not os.path.exists(path) and os.path.exists(UMBILICUS):
        # a store made on another machine records that machine's path: same scroll -> the configured file
        scroll = os.path.basename(os.path.dirname(path))
        if scroll and scroll in UMBILICUS:
            path = UMBILICUS
    assert os.path.exists(path), f"no umbilicus at {path}: every scroll needs one (create it, then pass --umbilicus)"
    pts = sorted((p["z"], p["y"], p["x"]) for p in json.load(open(path))["control_points"])
    return np.array(pts, np.float64).T


def axis_at(ax, rung):
    """The scroll axis control points (given in rung-2 / level-0 voxels) expressed in rung-k voxels."""
    return np.asarray(ax, np.float64) / (2.0 ** (int(rung) - 2))


def radial(ax, origin, shape):
    """Unit vectors (3,Z,Y,X) pointing away from the scroll axis in the xy plane (z component 0)."""
    z = np.arange(shape[0]) + origin[0]
    cy, cx = np.interp(z, ax[0], ax[1]), np.interp(z, ax[0], ax[2])
    dy = (np.arange(shape[1]) + origin[1])[None, :, None] - cy[:, None, None]
    dx = (np.arange(shape[2]) + origin[2])[None, None, :] - cx[:, None, None]
    n = np.sqrt(dy * dy + dx * dx) + 1e-6
    return np.stack([np.zeros(shape, np.float32), (dy / n).astype(np.float32), (dx / n).astype(np.float32)])


NORM = None  # None = per-patch z-score; (mean, std) = fixed scan-level normalization (set by `global_norm`)


def zscore(x):
    x = x.astype(np.float32)
    if NORM is not None:
        return (x - NORM[0]) / NORM[1]
    return (x - x.mean()) / (x.std() + 1e-3)


def global_norm(volume=None, n=200, seed=0):
    """(mean, std) of the non-air voxels of `volume` from n random 128^3 patches; sets NORM."""
    global NORM
    a, rng, acc = open_zarr(volume or CT), np.random.default_rng(seed), []
    while len(acc) < n:
        o = rng.integers(0, np.array(a.shape) - 128)
        c = a[o[0]:o[0] + 128, o[1]:o[1] + 128, o[2]:o[2] + 128]
        if (c > 0).mean() >= 0.5:
            acc.append(c[c > 0].astype(np.float32))
    v = np.concatenate(acc)
    NORM = (float(v.mean()), float(v.std()))
    return NORM


VOLCOMP_CHUNK = 128  # the codec encodes 128^3 uint8 blocks only; other shapes are padded and tiled


def volcomp_roundtrip(vol, q):
    """Encode+decode a uint8 volume with the real volcomp codec at quality q (ported from tsm)."""
    import volcomp_zarr as vc
    C = VOLCOMP_CHUNK
    pad = [(0, (-n) % C) for n in vol.shape]
    buf = np.pad(vol, pad, mode="edge") if any(h for _, h in pad) else vol.copy()
    for z in range(0, buf.shape[0], C):
        for y in range(0, buf.shape[1], C):
            for x in range(0, buf.shape[2], C):
                blk = np.ascontiguousarray(buf[z:z + C, y:y + C, x:x + C])
                out = np.frombuffer(bytes(vc.decode(vc.encode(blk.tobytes(), float(q)))), np.uint8)
                buf[z:z + C, y:y + C, x:x + C] = out.reshape((C, C, C))
    return buf[:vol.shape[0], :vol.shape[1], :vol.shape[2]]


def raw(rng, ct, cfg):
    """Raw-uint8 CT augs that have to happen before the z-score (see aug.py): the 8-bit export
    window (an affine remap is a no-op after re-z-scoring -- only its clipping is real) and the
    volcomp codec round trip (C code over bytes; skipped silently without the codec)."""
    k = cfg.get("window")
    if k and rng.random() < k["p"]:
        lo, hi = rng.uniform(k["lo_lo"], k["lo_hi"]), rng.uniform(k["hi_lo"], k["hi_hi"])
        ct = np.clip((ct.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    k = cfg.get("volcomp")
    if k and rng.random() < k["p"]:
        try:
            ct = volcomp_roundtrip(np.ascontiguousarray(ct, np.uint8), rng.uniform(*k["q"]))
        except Exception:
            pass
    return ct


def augment(rng, x, tg):
    """Random axis permutation + flips applied to the (C,Z,Y,X) input and (T,Z,Y,X) target; the radial
    vector channels 1..3 of x are permuted/negated to match. Only permutations that keep the patch shape are
    drawn (a 384x512x512 patch may swap y and x, not z)."""
    sh = np.array(x.shape[1:])
    perms = [q for q in itertools.permutations(range(3)) if (sh[list(q)] == sh).all()]
    perm, flip = np.array(perms[rng.integers(len(perms))]), rng.random(3) < 0.5
    sl = tuple(slice(None, None, -1 if f else 1) for f in flip)
    tg = np.ascontiguousarray(np.transpose(tg, (0,) + tuple(perm + 1))[(slice(None),) + sl])
    x = np.transpose(x, (0,) + tuple(perm + 1))[(slice(None),) + sl]
    ni = x.shape[0] - 3  # the radial vector is the last 3 channels
    x = np.concatenate([x[:ni], x[ni + perm] * np.where(flip, -1, 1).astype(np.float32)[:, None, None, None]])
    return np.ascontiguousarray(x), tg


# --------------------------------------------------------------------------------- the rung ladder
# Rung k has voxel size 0.6 * 2^k um (docs/unified_design.md section 1): 2.4 um = rung 2, 1228.8 um = rung 11.
# A pyramid group is either an exported prediction group (levels named by the exact voxel size in um, with
# OME multiscales in the group's zarr.json) or a CT mirror (integer level names: level l of a <native> um
# volume is rung l + um_rung(native)).

CTX_CACHE = {}  # {group base: {rung: array}} and {f"{base}#{rung}": ndarray} for arrays small enough to keep

RUNG0_UM = 0.6   # rung 0
NRUNGS = 12      # rungs 0 .. 11
NCTX = 9         # context cubes of a sample: rungs k+1 .. k+9
SMALL_RUNG = 96 ** 3  # a rung this small is read once and kept in memory (the top of every pyramid)


def rung_um(k):
    """Voxel size of rung k, in micrometres."""
    return RUNG0_UM * 2.0 ** int(k)


def um_rung(um):
    """The rung a voxel size belongs to (nearest in log2): 2.4 -> 2, 9.6 -> 4, 1228.8 -> 11."""
    return int(round(math.log2(float(um) / RUNG0_UM)))


def pyramid_base(path):
    """'<name>.zarr/0', '<name>.zarr/2.4' or '<name>.zarr' -> the group '<name>.zarr'."""
    p = str(path).rstrip("/")
    head, _, last = p.rpartition("/")
    return head if head and re.fullmatch(r"[0-9]+(\.[0-9]+)?", last) else p


def native_um(base):
    """The native voxel size of a pyramid, from its volume name ('...-2.400um-...'); 2.4 um by default."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)um", os.path.basename(str(base).rstrip("/")))
    return float(m.group(1)) if m else 2.4


def group_meta(base):
    """(multiscales dict | None, attributes dict) of a pyramid group, read straight off disk."""
    for name in ("zarr.json", ".zattrs"):
        p = f"{base}/{name}"
        if os.path.exists(p):
            j = json.load(open(p))
            at = j.get("attributes", j) if isinstance(j, dict) else {}
            ms = (at.get("ome") or at).get("multiscales") if isinstance(at, dict) else None
            return (ms[0] if ms else None), (at if isinstance(at, dict) else {})
    return None, {}


def rungs(base):
    """{rung: zarr array} of a pyramid group (a level path such as '.../x.zarr/0' names its group).
    Levels named by voxel size ('2.4', '1228.8': the exported prediction groups) are placed by that size;
    integer level names (the CT mirrors) by the group's native voxel size plus the level index. Only the
    levels actually on disk are returned, and nothing is ever streamed (data.local first)."""
    base = pyramid_base(local(str(base)))
    assert "://" not in base, f"{base}: the rung ladder is read from local mirrors only, never over HTTP"
    if base in CTX_CACHE:
        return CTX_CACHE[base]
    ms, at = group_meta(base)
    names = [str(d["path"]) for d in ms["datasets"]] if ms else sorted(os.listdir(base))
    nat = float((at.get("volcomp") or {}).get("rung_voxel_size_um") or native_um(base))
    out = {}
    for n in names:
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", n) or not os.path.isdir(f"{base}/{n}"):
            continue
        k = um_rung(float(n)) if "." in n else um_rung(nat) + int(n)  # um name vs integer level name
        try:
            out[k] = open_zarr(f"{base}/{n}")
        except Exception:
            pass
    assert out, f"{base}: no pyramid levels on disk"
    CTX_CACHE[base] = out
    return out


def base_rung(path):
    """The rung of the level a path names ('.../x.zarr/0'), or the finest rung of the group."""
    p = str(local(str(path))).rstrip("/")
    base = pyramid_base(p)
    pyr = rungs(base)
    last = p.rpartition("/")[2]
    if base != p and re.fullmatch(r"[0-9]+(\.[0-9]+)?", last):
        if "." in last:
            return um_rung(float(last))
        ms, at = group_meta(base)
        return um_rung(float((at.get("volcomp") or {}).get("rung_voxel_size_um") or native_um(base))) + int(last)
    return min(pyr)


def levels(volume):
    """Backward compatibility: {level offset: array} of the rungs ABOVE the level `volume` names."""
    k0, pyr = base_rung(volume), rungs(volume)
    return {k - k0: a for k, a in pyr.items() if k > k0}


def read_block(pyr, k, lo, hi):
    """pyr[k][lo:hi] as float32, keeping a small rung entirely in memory (CTX_CACHE)."""
    a = pyr[k]
    if int(np.prod(a.shape[-3:])) <= SMALL_RUNG:
        key = f"{a.store_path}#{k}"
        if key not in CTX_CACHE:
            CTX_CACHE[key] = np.asarray(a[:] if a.ndim == 3 else a[0])
        a = CTX_CACHE[key]
        return a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    s = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2])))
    return np.asarray(a[(0,) + s] if a.ndim == 4 else a[s], np.float32)


def read_rung(pyr, k, lo, p):
    """The (Z,Y,X) float32 cube of a pyramid at rung k, corner `lo`, size `p` (both in rung-k voxels).
    A rung above the top of the pyramid is made by 2x mean pooling the highest rung that exists -- the
    physical extent is the same, so the scroll simply shrinks inside the cube. Outside the array = 0."""
    src = max((r for r in pyr if r <= k), default=None)
    assert src is not None, f"pyramid has no rung at or below {k} (has {sorted(pyr)})"
    e = 1 << (k - src)
    p = shape3(p)
    n, slo = p * e, np.asarray(lo, np.int64) * e
    cube = np.zeros(tuple(n), np.float32)
    lo_c, hi = np.maximum(slo, 0), np.minimum(slo + n, np.array(pyr[src].shape[-3:], np.int64))
    if (hi > lo_c).all():
        blk = read_block(pyr, src, lo_c, hi)
        s = lo_c - slo
        cube[s[0]:s[0] + blk.shape[0], s[1]:s[1] + blk.shape[1], s[2]:s[2] + blk.shape[2]] = blk
    if e > 1:
        cube = cube.reshape(p[0], e, p[1], e, p[2], e).mean((1, 3, 5))
    return cube


def context(volume, origin, shape, ctx, rung=None):
    """Coarse cubes of the SAME size centred on the same point as the patch at `origin`/`shape`:
    one (Z,Y,X) uint8 cube per OFFSET in `ctx` (1 = one rung coarser, 2 = two rungs, ...), read from the
    pyramid of `volume`. `origin`/`shape` are voxels of rung `rung` (default: the rung `volume` names).
    A rung above the top of the pyramid is pooled from the highest one that exists; outside = 0 (air)."""
    pyr, out = rungs(volume), []
    k0 = base_rung(volume) if rung is None else int(rung)
    c0 = np.asarray(origin, np.int64) + shape3(shape) // 2  # centre, rung-k0 voxels
    for d in ctx:
        lo = c0 // (1 << int(d)) - shape3(shape) // 2  # the same centre, in rung-(k0+d) voxels
        out.append(read_rung(pyr, k0 + int(d), lo, shape).astype(np.uint8))
    return out


def scale_plane(rung, shape):
    """The constant scale channel of a sample at rung k: (k - 2) / 9, 0 at 2.4 um."""
    return np.full(tuple(shape3(shape)), (int(rung) - 2) / 9.0, np.float32)


def inputs(ct, rad, ctx=(), rung=None):
    """Model input (1 + len(ctx) + (rung is not None) + 3, Z,Y,X): z-scored CT, z-scored coarse context
    cubes, the constant scale plane (only when a rung is given) and the radial unit vector. The scale
    plane sits right before the radial channels, so warm-starting a 13-channel checkpoint zero-fills it."""
    sc = [scale_plane(rung, ct.shape)[None]] if rung is not None else []
    return np.concatenate([zscore(ct)[None]] + [zscore(c)[None] for c in ctx] + sc + [rad])


# ------------------------------------------------------------- which chunks of a level exist locally
# The desk's CT mirror has a PARTIAL level 0 (rung 2): only the chunks under earlier teacher boxes were
# pulled, and a missing chunk reads as zeros. Sampling must not land on one.

CHUNK_INDEX = {}


def array_dir(arr):
    p = str(getattr(arr.store, "root", "") or arr.store_path)
    p = p[len("file://"):] if p.startswith("file://") else p
    sub = getattr(arr, "path", "") or ""
    return os.path.join(p, sub) if sub and not p.endswith(sub) else p


def chunk_index(arr):
    """((gz, gy, gx) write-chunk size, present[bool array]) of a local array: which chunk (shard) keys are
    on disk. Scanned once per array and cached. Returns (grid, None) when every chunk is present."""
    d = array_dir(arr)
    if d in CHUNK_INDEX:
        return CHUNK_INDEX[d]
    g = np.array(getattr(arr, "shards", None) or arr.chunks, np.int64)[-3:]
    n = -(-np.array(arr.shape[-3:], np.int64) // g)
    pres = np.zeros(tuple(n), bool)
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), d).replace("\\", "/")
                parts = [q for q in rel.split("/") if q != "c"]
                if len(parts) == 1 and parts[0].count(".") == 2:  # zarr v2 "z.y.x"
                    parts = parts[0].split(".")
                if len(parts) == 3 and all(q.isdigit() for q in parts):
                    i = tuple(int(q) for q in parts)
                    if all(a < b for a, b in zip(i, n)):
                        pres[i] = True
    out = (g, None if pres.all() else pres)
    CHUNK_INDEX[d] = out
    return out


def covered(arr, lo, n):
    """Is every chunk of arr overlapping [lo, lo+n) present on disk?"""
    g, pres = chunk_index(arr)
    if pres is None:
        return True
    lo, n = np.asarray(lo, np.int64), np.asarray(n, np.int64)
    a = np.maximum(lo, 0) // g
    b = -(-np.minimum(lo + n, np.array(arr.shape[-3:], np.int64)) // g)
    if (b <= a).any():  # the window is entirely outside the array: nothing to read
        return True
    return bool(pres[a[0]:b[0], a[1]:b[1], a[2]:b[2]].all())


def coverage(arr):
    """The fraction of an array's chunks that are on disk."""
    _, pres = chunk_index(arr)
    return 1.0 if pres is None else float(pres.mean())


# --------------------------------------------------------------------------------- rung sources
# A source is one line of the stores file: `ct_base,target_group[,target_group...]`. The CT base is a
# pyramid group (or one of its levels); every target group is a whole-scroll (or boxed) prediction
# pyramid on the ladder, one output CHANNEL each. Both are read at the same rung and the same grid.


def group_attrs(base):
    return group_meta(pyramid_base(local(str(base))))[1]


def rung_shape(pyr, k):
    """The (Z,Y,X) shape a pyramid has at rung k (pooled from the highest rung below when k is above it)."""
    src = max(r for r in pyr if r <= k)
    return -(-np.array(pyr[src].shape[-3:], np.int64) // (1 << (k - src)))


def target_box(t, k):
    """(lo, size) of a target at rung k, in rung-k voxels: its `box` attr (given at its native rung) or
    the whole array."""
    if t["box"] is None:
        return np.zeros(3, np.int64), rung_shape(t["pyr"], k)
    o, s = np.array(t["box"][0], np.int64), np.array(t["box"][1], np.int64)
    d = k - t["native"]
    return (o >> d, np.maximum(-(-s >> d), 1)) if d >= 0 else (o << -d, s << -d)


def source_groups(lines):
    """Parse and open `ct_base,target_group[,target_group...]` lines. Each source:
    {ct, ct_pyr, targets: {channel: {pyr, weight, box, native}}, native, umbilicus, voxels}."""
    out = []
    for line in lines:
        parts = [q.strip() for q in str(line).split(",") if q.strip()]
        assert len(parts) >= 2, f"{line!r}: a rung source is 'ct_base,target_group[,target_group...]'"
        src = {"ct": parts[0], "ct_pyr": rungs(parts[0]), "line": ",".join(parts), "targets": {}}
        umb = None
        for g in parts[1:]:
            pyr, at = rungs(g), group_attrs(g)
            ch = at.get("channel") or (at.get("channels") or ["recto"])[0]
            assert ch not in src["targets"], f"{line!r}: two targets for channel {ch}"
            bx = at.get("box")
            src["targets"][ch] = {"path": g, "pyr": pyr, "native": min(pyr),
                                  "weight": float(at.get("weight", 1.0)),
                                  "box": (bx[0], bx[1]) if bx else None}
            umb = umb or at.get("umbilicus")
        src["native"] = min(t["native"] for t in src["targets"].values())
        src["umbilicus"] = umb or group_attrs(parts[0]).get("umbilicus") or UMBILICUS
        src["voxels"] = float(np.prod(target_box(next(iter(src["targets"].values())), src["native"])[1]))
        out.append(src)
    return out


def usable_rungs(src, allowed=None):
    """The rungs a source can be sampled at: its native rung .. NRUNGS - 1, intersected with `allowed`."""
    ks = list(range(src["native"], NRUNGS))
    return [k for k in ks if allowed is None or k in allowed]


def rung_probs(src, patch, allowed=None, boost=None):
    """{rung: probability} within a source: proportional to n_k ** 0.5 with n_k = (target voxels at rung k)
    / (voxels per patch), floored at 1, times an optional per-rung multiplier (`boost`)."""
    per = float(np.prod(shape3(patch)))
    ks = usable_rungs(src, allowed)
    assert ks, f"{src['line']}: no usable rung (native {src['native']}) among {sorted(allowed or [])}"
    t = next(iter(src["targets"].values()))
    w = np.array([max(float(np.prod(target_box(t, k)[1])) / per, 1.0) ** 0.5 *
                  float((boost or {}).get(k, 1.0)) for k in ks])
    return dict(zip(ks, w / w.sum()))


class Patches(torch.utils.data.IterableDataset):
    """Random (ct, teacher) patches; train patches never touch the val box."""

    def __init__(self, patch=128, ct=CT, stores=TRAIN, exclude=VAL, seed=0, air_keep=0.1, fg_min=0.05,
                 fg_keep=0.25, sym=True, aug=None, dense_pow=0.0, dense_ref=0.2, ctx=(), stores_file=None,
                 recheck=200, rungs=None, rung_boost=None, channels=None, require_targets=False):
        """dense_pow > 0 biases sampling towards sheet-dense patches: a patch whose target mean m is below
        dense_ref is kept with probability (m / dense_ref) ** dense_pow (crushed windings are where students fail).
        stores_file: instead of `stores`, a text file with one comma-joined group per line that is re-read
        whenever its mtime changes (checked every `recheck` patches): a training run picks up new stores as
        they are generated, and keeps sampling the old ones (by voxel count) when nothing new arrived.

        rungs: switches on the MULTI-RESOLUTION mode (docs/unified_design.md). Every line of `stores` /
        `stores_file` is then `ct_base,target_group[,target_group...]`: a CT pyramid plus one whole-scroll
        target pyramid per output channel, both addressed by rung (rung k = 0.6 * 2^k um). A sample is
        (source, rung k, corner): the CT and the targets are read at rung k, the context cubes at rungs
        k + ctx[0] .. k + ctx[-1], and the loader yields (x, target, weight, rung). `rungs` is the set of
        allowed rungs (None inside rung mode = every usable one), `rung_boost` {rung: multiplier} skews the
        rung mix, `channels` fixes the output channel order (default: order of first appearance).

        The rung mix within a source is p_k ~ n_k ** 0.5, n_k = target voxels at rung k / voxels per patch
        (floored at 1): n_k falls 8x per rung, so each rung is 2 * sqrt(2) times rarer than the one below
        until the floor flattens the top. Measured mix for the Paris 4 bootstrap at 256^3 (`usrm2 rung-mix`,
        whole scroll = 75784 x 32693^2 at rung 2), two sources -- the recto mask pulled from the published
        4.8 um level (native rung 3) and the m7 mask (native rung 4) -- weighted by their voxel counts
        (recto 88.9 %, m7 11.1 %):

            rung          3      4      5      6      7      8     9    10    11
            recto      64.6%  22.8%   8.1%   2.9%   1.0%   0.4%  0.1%  0.1%  0.1%
            m7            -   64.5%  22.8%   8.1%   2.9%   1.0%  0.4%  0.2%  0.2%
            overall    57.4%  27.5%   9.7%   3.4%   1.2%   0.4%  0.1%  0.1%  0.1%

        Rung 2 appears once a rung-2 target is added (the boxed level-0 recto mask); `--rung-boost 2=4`
        skews the mix by hand."""
        super().__init__()
        self.rungs, self.rung_boost, self.channels = rungs, dict(rung_boost or {}), channels
        self.norm, self.umbilicus = NORM, UMBILICUS  # module state the (spawned) workers must inherit explicitly
        self.stores_file, self.recheck, self.file_mtime = stores_file, recheck, None
        self.require_targets = require_targets  # only draw windows whose target chunks are on disk (a partial pull)
        if stores_file:
            stores = self.read_groups()
        self.patch, self.ct_path, self.paths, self.seed = shape3(patch), ct, [str(s).split(",") for s in stores], seed
        ex = exclude if isinstance(exclude, (list, tuple)) else [exclude]
        if len(ex) == 2 and not isinstance(ex[0], str) and np.ndim(ex[0]) == 1:
            ex = [exclude]  # a single (origin, size) box, not two excludes
        self.exclude = [e for e in ex if e is not None and (isinstance(e, tuple) or len(e))]
        self.dense_pow, self.dense_ref, self.ctx = dense_pow, dense_ref, tuple(ctx)  # coarse context levels
        self.sym = sym  # the 48 cube symmetries; the GPU augs are in aug.py
        self.aug = aug or {}  # the worker-side raw-uint8 stage: window / volcomp / blank
        self.air_keep, self.fg_min, self.fg_keep = air_keep, fg_min, fg_keep  # low-foreground patches are mostly skipped
        self.arrs = None

    def read_groups(self):
        import os
        self.file_mtime = os.path.getmtime(self.stores_file)
        return [l.strip() for l in open(self.stores_file) if l.strip() and not l.startswith("#")]

    def file_changed(self):
        import os
        try:
            return self.stores_file and os.path.getmtime(self.stores_file) != self.file_mtime
        except OSError:  # being rewritten
            return False

    def _open_rungs(self):
        """Rung mode: open the source lines (CT pyramid + target pyramids), their rung mixes and the
        held-out boxes (given at rung 2 and scaled to every rung)."""
        global NORM, UMBILICUS
        NORM, UMBILICUS = self.norm, self.umbilicus
        if self.stores_file:
            self.paths = [g.split(",") for g in self.read_groups()]
            assert self.paths, f"{self.stores_file} lists no store groups"
        self.srcs = source_groups([",".join(ps) for ps in self.paths])
        if self.channels is None:
            self.channels = list(dict.fromkeys(c for s in self.srcs for c in s["targets"]))
        allowed = None if self.rungs is True else set(self.rungs)
        for s in self.srcs:
            s["probs"] = rung_probs(s, self.patch, allowed, self.rung_boost)
            s["axis"] = axis(s["umbilicus"])
        self.w = np.array([s["voxels"] for s in self.srcs], np.float64)
        self.w /= self.w.sum()
        self.ex = [e if isinstance(e, (tuple, list)) and len(e) == 2 and not isinstance(e[0], str)
                   else val_box(e) for e in self.exclude]  # (origin, size) at rung 2
        self.arrs = self.srcs

    def _rung_sample(self, rng):
        """One (x, tgt, w, rung) draw, or None when the corner is rejected."""
        p = self.patch
        i = rng.choice(len(self.srcs), p=self.w)
        s = self.srcs[i]
        ks, pk = list(s["probs"]), np.array(list(s["probs"].values()))
        k = int(rng.choice(ks, p=pk))
        t0 = next(iter(s["targets"].values()))
        blo, bs = target_box(t0, k)
        lo_min, lo_max = np.minimum(blo, blo + bs - p), np.maximum(blo, blo + bs - p)
        lo = rng.integers(lo_min, lo_max + 1)
        for o, sz in self.ex:  # the held-out box, given at rung 2
            d = k - 2
            eo, es = (np.array(o) >> d, np.maximum(np.array(sz) >> d, 1)) if d >= 0 else (np.array(o) << -d, np.array(sz) << -d)
            if np.all(lo < eo + es) and np.all(lo + p > eo):
                return None
        cpyr = s["ct_pyr"]
        csrc = max(r for r in cpyr if r <= k)
        e = 1 << (k - csrc)
        if not covered(cpyr[csrc], lo * e, p * e):  # a partially mirrored level (level 0 of the desk's CT)
            return None
        if self.require_targets:  # a partially pulled export: an absent shard is "not yet exported", not air
            for t in s["targets"].values():
                tsrc = max(r for r in t["pyr"] if r <= k)
                te = 1 << (k - tsrc)
                if not covered(t["pyr"][tsrc], lo * te, p * te):
                    return None
        bl = self.aug.get("blank")
        if bl and rng.random() < bl["p"]:  # an all-air patch (CT 0 = air) with target 0
            ct = np.zeros(tuple(p), np.uint8)
            tg = np.zeros((len(self.channels),) + tuple(p), np.float32)
            w = np.zeros_like(tg)
        else:
            ct = read_rung(cpyr, k, lo, p).astype(np.uint8)
            if (ct == 0).mean() > 0.9 and rng.random() > self.air_keep:
                return None
            tg, w = self._rung_target(s, k, lo, ct)
            m = float((tg * (w > 0)).sum() / max((w > 0).sum(), 1))
            if m < self.fg_min and rng.random() > self.fg_keep:
                return None
            if self.dense_pow > 0 and m < self.dense_ref and rng.random() > (m / self.dense_ref) ** self.dense_pow:
                return None
            ct = raw(rng, ct, self.aug)
        cx = context(s["ct"], lo, ct.shape, self.ctx, rung=k) if self.ctx else ()
        rad = radial(axis_at(s["axis"], k), lo, ct.shape)
        x = inputs(ct, rad, cx, rung=k)
        if self.sym:
            x, tw = augment(rng, x, np.concatenate([tg, w]))
            tg, w = tw[:len(self.channels)], tw[len(self.channels):]
        return x, tg, w, k

    def _rung_target(self, s, k, lo, ct):
        """(target, weight) at rung k: one channel per output channel, 0 (weight 0) for a channel this
        source does not provide; weight 1 inside the target's box and where CT > 0, times its source weight."""
        p = self.patch
        tg = np.zeros((len(self.channels),) + tuple(p), np.float32)
        w = np.zeros_like(tg)
        inside_ct = (ct > 0).astype(np.float32)
        for c, chan in enumerate(self.channels):
            t = s["targets"].get(chan)
            if t is None:
                continue  # per-channel ignore: this source says nothing about that channel
            tg[c] = np.clip(read_rung(t["pyr"], k, lo, p) / 255.0, 0, 1)
            blo, bs = target_box(t, k)
            ins = np.zeros(tuple(p), np.float32)
            a = np.maximum(blo - lo, 0)
            b = np.minimum(blo + bs - lo, p)
            if (b > a).all():
                ins[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = 1.0
            w[c] = ins * inside_ct * t["weight"]
            tg[c] *= inside_ct  # masked CT (0) carries no surface
        return tg, w

    def _open(self):
        """Each teacher store names its CT volume and scroll axis (attrs), so stores from several scrolls can mix."""
        global NORM, UMBILICUS
        if self.rungs is not None:
            return self._open_rungs()
        NORM, UMBILICUS = self.norm, self.umbilicus  # forkserver workers start with fresh module globals
        if self.stores_file:
            self.paths = [g.split(",") for g in self.read_groups()]
            assert self.paths, f"{self.stores_file} lists no store groups"
        self.heads = [[open_zarr(q) for q in ps] for ps in self.paths]  # target channels
        for ps, h in zip(self.paths, self.heads):
            for q, a in zip(ps[1:], h[1:]):
                assert box(a)[0].tolist() == box(h[0])[0].tolist() and a.shape[-3:] == h[0].shape[-3:] \
                    and local(a.attrs.get("volume", self.ct_path)) == local(h[0].attrs.get("volume", self.ct_path)), \
                    f"{q} is not the same box as {ps[0]}"  # (a streamed URL and its local mirror are the same volume)
        self.arrs = [h[0] for h in self.heads]
        self.vols = [a.attrs.get("volume", self.ct_path) for a in self.arrs]
        cts = {v: open_zarr(v) for v in set(self.vols)}
        self.cts = [cts[v] for v in self.vols]
        self.axes = [axis(a.attrs.get("umbilicus", UMBILICUS)) for a in self.arrs]
        self.boxes = [box(a) for a in self.arrs]
        self.w = np.array([np.prod(s) for _, s in self.boxes], np.float64)
        self.w /= self.w.sum()
        exs = [open_zarr(str(e).split(",")[0]) for e in self.exclude]
        self.ex = [(box(e), local(e.attrs.get("volume", self.ct_path))) for e in exs]  # boxes never sampled

    def __iter__(self):
        if self.arrs is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + 1000 * (info.id if info else 0))
        p = self.patch
        rejected, served = 0, 0
        while True:
            assert rejected < 10000, "no acceptable patch in 10000 draws (stores all air, or all inside the val box?)"
            rejected += 1
            if served and served % self.recheck == 0 and self.file_changed():
                n0 = len(self.paths)
                self._open()  # the group list grew (or changed): re-open everything, new weights
                print(f"stores file changed: {n0} -> {len(self.paths)} groups", flush=True)
                served += 1
            if self.rungs is not None:
                got = self._rung_sample(rng)
                if got is None:
                    continue
                x, tg, w, k = got
                rejected, served = 0, served + 1
                yield torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(tg)), \
                    torch.from_numpy(np.ascontiguousarray(w)), k
                continue
            i = rng.choice(len(self.arrs), p=self.w)
            o, s = self.boxes[i]
            m = np.where(p > s - 2 * MARGIN, 0, MARGIN)  # a patch spanning a whole axis (384 z) may use the edges
            assert (p <= s).all(), f"patch {tuple(p)} does not fit store box {tuple(s)}"
            lo = rng.integers(m, s - m - p + 1)  # store-local corner
            g = o + lo  # global corner
            if any(local(self.vols[i]) == v and np.all(g < b[0] + b[1]) and np.all(g + p > b[0]) for b, v in self.ex):
                continue
            bl = self.aug.get("blank")
            if bl and rng.random() < bl["p"]:  # an all-air patch (CT 0 = air) with target 0
                ct, tg = np.zeros(tuple(p), np.uint8), np.zeros((len(self.heads[i]),) + tuple(p), np.float32)
            else:
                ct = self.cts[i][g[0]:g[0] + p[0], g[1]:g[1] + p[1], g[2]:g[2] + p[2]]
                if (ct == 0).mean() > 0.9 and rng.random() > self.air_keep:
                    continue
                tg = np.stack([read3(a, lo, p) for a in self.heads[i]]).astype(np.float32) / 255.0 * (ct > 0)  # masked CT -> no surface
                m = tg.mean()
                if m < self.fg_min and rng.random() > self.fg_keep:
                    continue
                if self.dense_pow > 0 and m < self.dense_ref and rng.random() > (m / self.dense_ref) ** self.dense_pow:
                    continue
                ct = raw(rng, ct, self.aug)
            cx = context(self.vols[i], g, ct.shape, self.ctx) if self.ctx else ()
            rejected, x = 0, inputs(ct, radial(self.axes[i], g, ct.shape), cx)
            if self.sym:
                x, tg = augment(rng, x, tg)
            served += 1
            yield torch.from_numpy(x), torch.from_numpy(tg)


def val_grid(patch=128, ct=CT, store=VAL, limit=32, ctx=()):
    """Deterministic non-overlapping tiling of the val box(es) -> list of (ct, tgt). `store` may be a list of
    boxes (each a comma-joined teacher group); `limit` patches are taken from each."""
    if isinstance(store, (list, tuple)):
        return [x for s in store for x in val_grid(patch, ct, s, limit, ctx)]
    heads = [open_zarr(q) for q in str(store).split(",")]
    tga = heads[0]
    cta, ax = open_zarr(tga.attrs.get("volume", ct)), axis(tga.attrs.get("umbilicus", UMBILICUS))
    o, s = box(tga)
    p3 = shape3(patch)
    corners = [(z, y, x) for z in range(0, s[0] - p3[0] + 1, p3[0])
               for y in range(0, s[1] - p3[1] + 1, p3[1])
               for x in range(0, s[2] - p3[2] + 1, p3[2])]
    if limit and len(corners) > limit:  # deterministic even subsample
        corners = [corners[i] for i in np.linspace(0, len(corners) - 1, limit).astype(int)]
    out = []
    for lo in corners:
        g = o + np.array(lo)
        c = cta[g[0]:g[0] + p3[0], g[1]:g[1] + p3[1], g[2]:g[2] + p3[2]]
        t = np.stack([read3(a, lo, patch) for a in heads]).astype(np.float32) / 255.0 * (c > 0)
        vol = tga.attrs.get("volume", ct)
        cx = context(vol, g, c.shape, ctx) if ctx else ()
        out.append((torch.from_numpy(inputs(c, radial(ax, g, c.shape), cx)), torch.from_numpy(t)))
    return out


def rung_mix(lines, patch=256, allowed=None, boost=None):
    """The sampling mix of a stores file: one row per (source, rung) with the target size at that rung,
    n_k = target voxels / patch voxels, the probability inside the source and overall, and how much of the
    CT level that rung reads is mirrored locally."""
    srcs = source_groups(lines)
    sw = np.array([s["voxels"] for s in srcs], np.float64)
    sw /= sw.sum()
    per, rows = float(np.prod(shape3(patch))), []
    for s, ws in zip(srcs, sw):
        for k, p in rung_probs(s, patch, allowed, boost).items():
            t = next(iter(s["targets"].values()))
            shp = target_box(t, k)[1]
            csrc = max(r for r in s["ct_pyr"] if r <= k)
            rows.append({"source": s["line"], "rung": k, "um": rung_um(k), "shape": tuple(int(v) for v in shp),
                         "n": max(float(np.prod(shp)) / per, 1.0), "p_src": float(p), "p": float(ws * p),
                         "ct_rung": csrc, "ct_coverage": coverage(s["ct_pyr"][csrc])})
    return rows


def format_rung_mix(rows):
    out = []
    for line in dict.fromkeys(r["source"] for r in rows):
        out.append(line)
        out.append(f"  {'rung':>4} {'um':>8} {'target at rung':>22} {'n_k':>12} {'p|src':>7} {'p':>7} "
                   f"{'CT rung':>7} {'CT local':>9}")
        for r in [q for q in rows if q["source"] == line]:
            out.append(f"  {r['rung']:>4} {r['um']:>8.1f} {str(r['shape']):>22} {r['n']:>12.1f} "
                       f"{100 * r['p_src']:>6.1f}% {100 * r['p']:>6.1f}% {r['ct_rung']:>7} {100 * r['ct_coverage']:>8.1f}%")
    return "\n".join(out)


def val_box(store=VAL):
    """(origin, size) of a held-out region in rung-2 voxels: a 'z0,y0,x0,Z,Y,X' string, a pair already,
    or a teacher store whose attrs carry origin_zyx and shape."""
    if isinstance(store, (tuple, list)):
        if len(store) == 1:  # argparse nargs="+" hands over a one-element list
            return val_box(store[0])
        if len(store) == 6 and not isinstance(store[0], str):
            return np.array(store[:3], np.int64), np.array(store[3:], np.int64)
        if len(store) == 2 and not isinstance(store[0], str):
            return np.array(store[0], np.int64), np.array(store[1], np.int64)
    s = str(store)
    if re.fullmatch(r"[-0-9]+(,[-0-9]+){5}", s):
        v = [int(q) for q in s.split(",")]
        return np.array(v[:3], np.int64), np.array(v[3:], np.int64)
    return box(open_zarr(s.split(",")[0]))


VAL_RUNGS = (2, 3, 4, 6)  # the rungs the held-out box is scored at


def val_grid_rungs(patch, stores, box2, rungs=VAL_RUNGS, limit=8, ctx=(), channels=None):
    """Per-rung validation: the held-out box (given at rung 2) read at each rung from the same pyramids
    as training. Returns [(x, target, weight, rung)] -- the same tuples the rung loader yields."""
    srcs = source_groups(stores)
    if channels is None:
        channels = list(dict.fromkeys(c for s in srcs for c in s["targets"]))
    ds = Patches(patch=patch, stores=stores, exclude=[], rungs=True, ctx=ctx, channels=channels, sym=False)
    ds._open_rungs()
    p3, out = shape3(patch), []
    o2, s2 = val_box(box2)
    for k in rungs:
        d = k - 2
        o = (o2 >> d, np.maximum(s2 >> d, 1)) if d >= 0 else (o2 << -d, s2 << -d)
        org, sz = o
        corners = [(z, y, x) for z in range(0, max(int(sz[0]) - int(p3[0]), 0) + 1, int(p3[0]))
                   for y in range(0, max(int(sz[1]) - int(p3[1]), 0) + 1, int(p3[1]))
                   for x in range(0, max(int(sz[2]) - int(p3[2]), 0) + 1, int(p3[2]))]
        if limit and len(corners) > limit:
            corners = [corners[i] for i in np.linspace(0, len(corners) - 1, limit).astype(int)]
        for s in ds.srcs[:1]:  # the first source supplies the validation CT and targets
            for c in corners:
                lo = org + np.array(c, np.int64)
                ct = read_rung(s["ct_pyr"], k, lo, p3).astype(np.uint8)
                tg, w = ds._rung_target(s, k, lo, ct)
                cx = context(s["ct"], lo, ct.shape, ctx, rung=k) if ctx else ()
                x = inputs(ct, radial(axis_at(s["axis"], k), lo, ct.shape), cx, rung=k)
                out.append((torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(tg),
                            torch.from_numpy(w), k))
    assert out, "the validation box is smaller than one patch at every rung"
    return out


def loader(patch, batch, workers, **kw):
    """Workers start as fresh processes (forkserver), never forks: the parent has usually opened zarr already
    (validation grid), whose asyncio loop thread does not survive a fork and breaks streamed (HTTP) reads."""
    ds = Patches(patch=patch, **kw)
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers,
                                       pin_memory=True, persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers else None,
                                       multiprocessing_context="forkserver" if workers else None)
