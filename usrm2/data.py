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
    mirror path that does not exist here -> the streamed URL (stores made on the desk, read in the cloud).
    Two subtrees live under a scroll: the CT volumes (URL `<scroll>/volumes/<rest>`, mirrored at
    `<scroll>/<rest>`) and the exported predictions (`<scroll>/representations/...` either way)."""
    import os
    if "://" in path and "/volcomp/" in path:
        scroll, _, rest = path.split("/volcomp/", 1)[1].partition("/")
        rest = rest[len("volumes/"):] if rest.startswith("volumes/") else rest
        cand = f"{LOCAL_VOLUMES}/{scroll}/{rest}"
        return cand if os.path.exists(cand) else path
    if path.startswith(LOCAL_VOLUMES + "/") and not os.path.exists(path):
        return remote(path)
    return path


def remote(path):
    """The mirror path -> its origin URL (the inverse of `local`), whether or not it exists here."""
    path = str(path)
    if "://" in path:
        return path
    assert path.startswith(LOCAL_VOLUMES + "/"), f"{path} is not under the local mirror {LOCAL_VOLUMES}"
    scroll, _, rest = path[len(LOCAL_VOLUMES) + 1:].partition("/")
    if rest.startswith("representations/"):  # the exported prediction pyramids keep their key
        return f"{STREAM_VOLUMES}/{scroll}/{rest}"
    return f"{STREAM_VOLUMES}/{scroll}/volumes/{rest}"


UMBILICUS_DIR = _os.environ.get("USRM2_UMBILICUS_DIR", "/vesuvius/usrm/umbilicus")


def scroll_of(path):
    """The scroll a mirror path or origin URL belongs to ('.../volcomp/PHerc0139/...' -> 'PHerc0139')."""
    p = str(path)
    if "/volcomp/" in p:
        return p.split("/volcomp/", 1)[1].split("/", 1)[0]
    if p.startswith(LOCAL_VOLUMES + "/"):
        return p[len(LOCAL_VOLUMES) + 1:].split("/", 1)[0]
    return None


def umbilicus_path(scroll):
    """Where this machine keeps a scroll's axis (usrm2.umbilicus writes it there)."""
    return f"{UMBILICUS_DIR}/{scroll}/umbilicus-full-resolution.json"


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


def raw_params(rng, cfg):
    """The rng draws of the raw-uint8 stage as a small JSON-able dict ({} = nothing to do). Split from
    `raw_apply` so the stream planner (usrm2/stream.py) makes exactly the draws the loader used to make
    and records the outcome in the queue, and the replaying worker applies it without an rng."""
    out = {}
    k = cfg.get("window")
    if k and rng.random() < k["p"]:
        out["window"] = [float(rng.uniform(k["lo_lo"], k["lo_hi"])), float(rng.uniform(k["hi_lo"], k["hi_hi"]))]
    k = cfg.get("volcomp")
    if k and rng.random() < k["p"]:
        out["volcomp"] = float(rng.uniform(*k["q"]))
    return out


def raw_apply(ct, prm):
    """`raw_params` applied to a uint8 CT cube."""
    if prm.get("window"):
        lo, hi = prm["window"]
        ct = np.clip((ct.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    if prm.get("volcomp"):
        try:
            ct = volcomp_roundtrip(np.ascontiguousarray(ct, np.uint8), prm["volcomp"])
        except Exception:
            pass
    return ct


def raw(rng, ct, cfg):
    """Raw-uint8 CT augs that have to happen before the z-score (see aug.py): the 8-bit export
    window (an affine remap is a no-op after re-z-scoring -- only its clipping is real) and the
    volcomp codec round trip (C code over bytes; skipped silently without the codec)."""
    return raw_apply(ct, raw_params(rng, cfg))


SYM_PERMS = tuple(itertools.permutations(range(3)))  # the 6 axis permutations


def sym_decode(sym):
    """A cube symmetry index 0..47 -> (axis permutation, flips): sym = 8 * permutation index + flip bits."""
    return np.array(SYM_PERMS[int(sym) // 8]), np.array([bool(int(sym) >> d & 1) for d in range(3)])


def draw_sym(rng, shape):
    """The cube symmetry of one patch, as an index 0..47. Only permutations that keep the patch shape are
    drawn (a 384x512x512 patch may swap y and x, not z); the rng draws are the ones `augment` has always
    made, so a seed gives the same stream as before. Index 0 is the identity."""
    sh = np.array(shape)
    perms = [q for q in SYM_PERMS if (sh[list(q)] == sh).all()]
    perm, flip = perms[rng.integers(len(perms))], rng.random(3) < 0.5
    return SYM_PERMS.index(perm) * 8 + int(flip[0]) + 2 * int(flip[1]) + 4 * int(flip[2])


def sym_apply(sym, x, tg):
    """Cube symmetry `sym` applied to the (C,Z,Y,X) input and the (T,Z,Y,X) target; the radial vector
    channels (the last 3 of x) are permuted/negated to match. `prep.sym_apply_t` is the same map on the
    GPU, and tests/test_prep.py checks the two agree for all 48 symmetries."""
    perm, flip = sym_decode(sym)
    sl = tuple(slice(None, None, -1 if f else 1) for f in flip)
    tg = np.ascontiguousarray(np.transpose(tg, (0,) + tuple(perm + 1))[(slice(None),) + sl])
    x = np.transpose(x, (0,) + tuple(perm + 1))[(slice(None),) + sl]
    ni = x.shape[0] - 3  # the radial vector is the last 3 channels
    x = np.concatenate([x[:ni], x[ni + perm] * np.where(flip, -1, 1).astype(np.float32)[:, None, None, None]])
    return np.ascontiguousarray(x), tg


def augment(rng, x, tg):
    """Random axis permutation + flips applied to the (C,Z,Y,X) input and (T,Z,Y,X) target."""
    return sym_apply(draw_sym(rng, x.shape[1:]), x, tg)


# --------------------------------------------------------------------------------- the rung ladder
# Rung k has voxel size 0.6 * 2^k um (docs/unified_design.md section 1): 2.4 um = rung 2, 1228.8 um = rung 11.
# A pyramid group is either an exported prediction group (levels named by the exact voxel size in um, with
# OME multiscales in the group's zarr.json) or a CT mirror (integer level names: level l of a <native> um
# volume is rung l + um_rung(native)).

CTX_CACHE = {}  # {group base: {rung: array}} and {f"{level dir}#{rung}": ndarray} for whole levels kept decoded

RUNG0_UM = 0.6   # rung 0
NRUNGS = 12      # rungs 0 .. 11
NCTX = 9         # context cubes of a sample: rungs k+1 .. k+9
CASCADE_MODES = ("off", "mask", "self", "mix")  # source of the cascade input channel (section 22)
CACHE_VOX = 48 << 20     # a level of at most this many voxels is decoded once and kept whole in the worker
CACHE_BUDGET = 192 << 20  # ... up to this many bytes of them per process
SMALL_RUNG = CACHE_VOX   # backward-compatible name


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


def group_native_um(at):
    """The native voxel size a pyramid group's attributes declare, or None. The exported prediction groups
    carry it as `volcomp.rung_voxel_size_um` (Paris 4, resampled onto the ladder) or, for the scrolls
    exported on their native grid, `native_voxel_size_um` / `volcomp.native_voxel_size_um`."""
    v = (at.get("volcomp") or {})
    for q in (v.get("rung_voxel_size_um"), v.get("native_voxel_size_um"), at.get("native_voxel_size_um")):
        if q:
            return float(q)
    return None


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
    names = [str(d["path"]) for d in ms["datasets"]] if ms else []
    names += [d for d in sorted(os.listdir(base)) if d not in names]  # levels built locally after the export (6-9)
    nat = float(group_native_um(at) or native_um(base))
    out, why = {}, []
    for n in names:
        if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", n) or not os.path.isdir(f"{base}/{n}"):
            continue
        k = um_rung(float(n)) if "." in n else um_rung(nat) + int(n)  # um name vs integer level name
        try:
            out[k] = open_zarr(f"{base}/{n}")
        except Exception as e:  # noqa: BLE001  (a level whose codec this build cannot open)
            why.append(f"{n}: {e!r}")
    assert out, f"{base}: no pyramid levels on disk" + (f" (tried {names}; {why[0]})" if why else "")
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
        return um_rung(float(group_native_um(at) or native_um(base))) + int(last)
    return min(pyr)


def levels(volume):
    """Backward compatibility: {level offset: array} of the rungs ABOVE the level `volume` names."""
    k0, pyr = base_rung(volume), rungs(volume)
    return {k - k0: a for k, a in pyr.items() if k > k0}


def read_block(pyr, k, lo, hi):
    """pyr[k][lo:hi] as uint8 (the whole level when it is cached)."""
    a = full_level(pyr, k)
    if a is None:
        a = pyr[k]
        s = (slice(int(lo[0]), int(hi[0])), slice(int(lo[1]), int(hi[1])), slice(int(lo[2]), int(hi[2])))
        return np.asarray(a[(0,) + s] if a.ndim == 4 else a[s], np.uint8)
    return a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def _cache_bytes():
    return sum(v.nbytes for v in CTX_CACHE.values() if isinstance(v, np.ndarray))


def pool2(v):
    """2x mean pooling of a uint8 volume, zero-padded to an even shape (as `read_rung` pools)."""
    s = np.array(v.shape, np.int64)
    n = -(-s // 2)
    if (s % 2).any():
        v = np.pad(v, [(0, int(q)) for q in n * 2 - s])
    return v.reshape(n[0], 2, n[1], 2, n[2], 2).astype(np.float32).mean((1, 3, 5)).astype(np.uint8)


def full_level(pyr, k):
    """The WHOLE level at rung k as a cached uint8 array, or None when it is too big to keep (CACHE_VOX /
    CACHE_BUDGET). Rungs above the top of the pyramid are pooled from the cached level below, so a sample's
    nine context cubes do not re-read (and re-decode) the top of the pyramid once per rung: for a scroll
    whose pyramid stops at rung 8 the coarse rungs 9..11 are a few MB each and become free after the first
    sample. Pooling in 2x steps truncates to uint8 at each step, so a cached coarse rung may differ from a
    direct pool by at most one grey level."""
    src = max((r for r in pyr if r <= k), default=None)
    if src is None:
        return None
    key = f"{array_dir(pyr[src])}#{k}"
    if key in CTX_CACHE:
        return CTX_CACHE[key]
    n = int(np.prod(rung_shape(pyr, k)))
    if n > CACHE_VOX or _cache_bytes() + n > CACHE_BUDGET:
        return None
    if k == src:
        a = pyr[src]
        v = np.ascontiguousarray(np.asarray(a[:] if a.ndim == 3 else a[0], np.uint8))
    else:
        below = full_level(pyr, k - 1)
        if below is None:  # the level below is too big to keep: pool it in z slabs, keeping only the result
            a, e = pyr[src], 1 << (k - 1 - src)
            S = np.array(a.shape[-3:], np.int64)
            below = np.zeros(tuple(-(-S // e)), np.uint8)
            step = max(e, (1 << 24) // max(int(S[1] * S[2]), 1) // e * e)
            for z in range(0, int(S[0]), step):
                blk = np.asarray(a[z:z + step] if a.ndim == 3 else a[0, z:z + step], np.uint8)
                if e > 1:
                    m = -(-np.array(blk.shape, np.int64) // e)
                    pad = m * e - np.array(blk.shape)
                    if pad.any():
                        blk = np.pad(blk, [(0, int(q)) for q in pad])
                    blk = blk.reshape(m[0], e, m[1], e, m[2], e).astype(np.float32).mean((1, 3, 5)).astype(np.uint8)
                below[z // e:z // e + blk.shape[0]] = blk
        v = pool2(below)
    CTX_CACHE[key] = v
    return v


def read_rung(pyr, k, lo, p, dtype=np.float32):
    """The (Z,Y,X) cube of a pyramid at rung k, corner `lo`, size `p` (both in rung-k voxels).
    A rung above the top of the pyramid is made by 2^d mean pooling the highest rung that exists -- the
    physical extent is the same, so the scroll simply shrinks inside the cube. Outside the array = 0.
    Only the part of the source that overlaps the array is read: a rung far above the top would otherwise
    address (p * 2^d)^3 voxels (a 256^3 cube five rungs above the top is 8192^3)."""
    src = max((r for r in pyr if r <= k), default=None)
    assert src is not None, f"pyramid has no rung at or below {k} (has {sorted(pyr)})"
    p, lo = shape3(p), np.asarray(lo, np.int64)
    cube = np.zeros(tuple(p), dtype)
    full = full_level(pyr, k)
    if full is not None:
        S = np.array(full.shape, np.int64)
        a, b = np.maximum(lo, 0), np.minimum(lo + p, S)
        if (b > a).all():
            s = a - lo
            blk = full[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            cube[s[0]:s[0] + blk.shape[0], s[1]:s[1] + blk.shape[1], s[2]:s[2] + blk.shape[2]] = blk
        return cube
    e = 1 << (k - src)
    S = np.array(pyr[src].shape[-3:], np.int64)
    jlo = np.maximum(-lo, 0)                      # the rung-k voxels of the cube that touch the array
    jhi = np.minimum(-(-S // e) - lo, p)
    if (jhi <= jlo).any():
        return cube
    a, b = (lo + jlo) * e, np.minimum((lo + jhi) * e, S)
    blk = read_block(pyr, src, a, b)
    if e > 1:
        n = jhi - jlo
        pad = n * e - (b - a)
        if pad.any():                             # past the end of the array: pooled against air
            blk = np.pad(blk, [(0, int(q)) for q in pad])
        blk = blk.reshape(n[0], e, n[1], e, n[2], e).astype(np.float32).mean((1, 3, 5))
    cube[jlo[0]:jhi[0], jlo[1]:jhi[1], jlo[2]:jhi[2]] = blk.astype(dtype, copy=False)
    return cube


def context(volume, origin, shape, ctx, rung=None, dtype=np.uint8):
    """Coarse cubes of the SAME size centred on the same point as the patch at `origin`/`shape`:
    one (Z,Y,X) uint8 cube per OFFSET in `ctx` (1 = one rung coarser, 2 = two rungs, ...), read from the
    pyramid of `volume`. `origin`/`shape` are voxels of rung `rung` (default: the rung `volume` names).
    A rung above the top of the pyramid is pooled from the highest one that exists; outside = 0 (air)."""
    pyr, out = rungs(volume), []
    k0 = base_rung(volume) if rung is None else int(rung)
    c0 = np.asarray(origin, np.int64) + shape3(shape) // 2  # centre, rung-k0 voxels
    for d in ctx:
        lo = c0 // (1 << int(d)) - shape3(shape) // 2  # the same centre, in rung-(k0+d) voxels
        out.append(read_rung(pyr, k0 + int(d), lo, shape, dtype=dtype))
    return out


def scale_plane(rung, shape):
    """The constant scale channel of a sample at rung k: (k - 2) / 9, 0 at 2.4 um."""
    return np.full(tuple(shape3(shape)), (int(rung) - 2) / 9.0, np.float32)


def inputs(ct, rad, ctx=(), rung=None, cascade=None):
    """Model input (1 + len(ctx) + (cascade is not None) + (rung is not None) + 3, Z,Y,X): z-scored CT,
    z-scored coarse context cubes, the CASCADE channel, the constant scale plane (only when a rung is
    given) and the radial unit vector.

    Channel order is [CT, ctx..., CASCADE, scale, radial(3)]: the image channels stay first and the radial
    vector stays last, so `train.warm_start` widens a stem by zero-filling the new planes. The cascade
    channel is the model's own rung-(k+1) prediction over the same field of view upsampled 2x -- a
    probability in 0..1, NOT z-scored (it is not an image channel), exactly like the scale plane."""
    sc = [scale_plane(rung, ct.shape)[None]] if rung is not None else []
    cs = [np.asarray(cascade, np.float32)[None]] if cascade is not None else []
    return np.concatenate([zscore(ct)[None]] + [zscore(c)[None] for c in ctx] + cs + sc + [rad])


def rung_item(ct, tg, w, k, lo, ax, sym=0, norm=None, cm=None, cx=None, lo1=None):
    """The compact sample the rung loader yields: everything uint8, so a 256^3 sample is ~200 MB instead
    of the ~1 GB of float32 `inputs` + `augment` used to build in the worker. `usrm2.prep.prepare` turns a
    collated batch of these into the model input on the GPU.

    ct  (1 + len(ctx), Z, Y, X) uint8: the CT cube and the context cubes as read, not z-scored
    tgt (channels, Z, Y, X) uint8: the decoded mask field, masked-CT zeroing applied
    w   (channels, Z, Y, X) uint8: the per-voxel weight, 255 = 1.0
    lo  (3,) int64: the corner, in rung-k voxels
    cyx (2, Z) float64: the scroll axis (y, x) at each z of the cube -- what `radial` interpolates
    sym (): the cube symmetry index drawn by the worker (0 = identity), applied on the GPU
    rung (), norm (2,): the rung k and the (mean, std) of the z-score (std 0 = per-patch)

    The CASCADE extras (`--cascade`, docs/unified_design.md section 22), all uint8/int64 like the rest:
    cm  (Z/2, Y/2, X/2) uint8: the rung-(k+1) target block over the patch footprint -- the `mask` source of
        the cascade channel, upsampled 2x on the GPU. Its presence is what makes the sample 15-channel.
    cx  (1, Z, Y, X) uint8: the TENTH context cube (rung k + ctx[-1] + 1), the one extra cube the
        rung-(k+1) input needs that the rung-k input does not (its own contexts are ctx_2..ctx_9).
    lo1 (3,) int64: the corner of that rung-(k+1) cube; `cyx1` (2, Z) is its scroll axis, so `prep` can
        recompute the radial vector at the coarser rung."""
    lo = np.asarray(lo, np.int64)
    a = axis_at(ax, k)
    z = np.arange(ct.shape[-3]) + lo[0]
    cyx = np.stack([np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])])
    nm = (0.0, 0.0) if (norm or NORM) is None else tuple(float(v) for v in (norm or NORM))
    out = {"ct": torch.from_numpy(np.ascontiguousarray(ct)),
           "tgt": torch.from_numpy(np.ascontiguousarray(tg)),
           "w": torch.from_numpy(np.ascontiguousarray(w)),
           "lo": torch.from_numpy(np.ascontiguousarray(lo)),
           "cyx": torch.from_numpy(np.ascontiguousarray(cyx)),
           "sym": torch.tensor(int(sym)), "rung": torch.tensor(int(k)),
           "norm": torch.tensor(nm, dtype=torch.float32)}
    if cm is not None:
        out["cm"] = torch.from_numpy(np.ascontiguousarray(np.asarray(cm, np.uint8)))
    if cx is not None:
        cx = np.asarray(cx, np.uint8)
        out["cx"] = torch.from_numpy(np.ascontiguousarray(cx if cx.ndim == 4 else cx[None]))
    if lo1 is not None:
        a1 = axis_at(ax, k + 1)
        lo1 = np.asarray(lo1, np.int64)
        z1 = np.arange(ct.shape[-3]) + lo1[0]
        out["lo1"] = torch.from_numpy(np.ascontiguousarray(lo1))
        out["cyx1"] = torch.from_numpy(np.ascontiguousarray(
            np.stack([np.interp(z1, a1[0], a1[1]), np.interp(z1, a1[0], a1[2])])))
    return out


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
    # <level>/mirror.json says what the mirror KNOWS (absent keys on the origin are air, so presence on disk
    # alone cannot tell air from not-mirrored): {"complete": true} = every chunk known; {"boxes": [[z,y,x,Z,Y,X],
    # ...]} (voxels of this level) = the chunks inside those boxes are known.
    mj = os.path.join(d, "mirror.json")
    if os.path.isfile(mj):
        try:
            m = json.load(open(mj))
        except Exception:
            m = {}
        if m.get("complete"):
            out = (g, None)
            CHUNK_INDEX[d] = out
            return out
        for b in m.get("boxes", []):
            o, sz = np.array(b[:3], np.int64), np.array(b[3:6], np.int64)
            a = np.maximum(o, 0) // g
            hi = np.minimum(-(-(o + sz) // g), n)
            if (hi > a).all():
                pres[a[0]:hi[0], a[1]:hi[1], a[2]:hi[2]] = True
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
        # a multi-scroll stores file must never fall back to another scroll's axis: the source's own
        # scroll directory comes first, and the configured default only serves a path without a scroll
        sc = scroll_of(parts[0])
        own = umbilicus_path(sc) if sc else None
        src["umbilicus"] = (umb or group_attrs(parts[0]).get("umbilicus")
                            or (own if own and os.path.exists(own) else UMBILICUS))
        # a source's draw weight is its PHYSICAL volume (um^3), not its voxel count: a 9.6 um-native scroll would
        # otherwise weigh 64x less than a 2.4 um one of the same size
        src["voxels"] = float(np.prod(target_box(next(iter(src["targets"].values())), src["native"])[1]))
        src["volume_um3"] = src["voxels"] * rung_um(src["native"]) ** 3
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


# ------------------------------------------------------------------------- the no-repeat region walk
# Region mode (below) draws a fresh (source, rung, region) for every visit, so over a long run the same
# region comes up again and again -- the user's "I don't want to train over the same data multiple times".
# The WALK enumerates instead: every region of every source at every usable rung, ONCE. A region is a
# shard-aligned region^3 tile of the target's box at that rung; a tile whose target is all air is dropped up
# front by a cheap check against a coarse level of the export. Each surviving region carries the draw weight
#
#     w(region) = (source's physical-volume share) * (its rung's probability) / (regions of that source+rung)
#
# so the source mix and the rung mix of `rung_probs` are honoured IN EXPECTATION while every region is
# visited exactly once; `walk_order` turns those weights into a visit ORDER (a weighted shuffle without
# replacement), and the planner walks that order with a persisted cursor.

OCC_VOX = 64 << 20  # the occupancy check reads the finest target level with at most this many voxels


def occupancy_rung(pyr, cap=OCC_VOX):
    """The finest rung of a pyramid whose whole level is at most `cap` voxels (the coarsest one if none is)."""
    for k in sorted(pyr):
        if int(np.prod(pyr[k].shape[-3:])) <= cap:
            return k
    return max(pyr)


def occupancy(pyr, k=None, cap=OCC_VOX):
    """(rung, bool array of the whole level): where a target export is not air. A few MB per scroll."""
    k = occupancy_rung(pyr, cap) if k is None else int(k)
    a = pyr[k]
    return k, np.asarray(a[:] if a.ndim == 3 else a[0], np.uint8) > 0


def shard_grid(pyr, k):
    """The shard (write-chunk) size, in rung-k voxels, of the level a pyramid is read from at rung k: what a
    region is snapped to, so that one region is one shard footprint."""
    src = max(r for r in pyr if r <= k)
    g = np.array(getattr(pyr[src], "shards", None) or pyr[src].chunks, np.int64)[-3:]
    return np.maximum(g >> (k - src), 1)


def region_tiles(s, k, region, patch):
    """The shard-aligned tiles covering a source's target box at rung k: (per-axis origins, tile size).
    The tile size is `region` rounded DOWN to a multiple of the shard grid (and never below one shard, nor
    above the box), so every tile origin is a shard boundary."""
    p, t = shape3(patch), next(iter(s["targets"].values()))
    blo, bs = target_box(t, k)
    g = shard_grid(s["ct_pyr"], k)
    R = np.maximum((shape3(region) // g) * g, g)
    R = np.minimum(R, -(-np.maximum(bs, p) // g) * g)  # a box smaller than a region: one tile, still aligned
    ax = [np.arange((blo[d] // g[d]) * g[d], int(blo[d] + bs[d]), int(R[d]), dtype=np.int64) for d in range(3)]
    return ax, R


def tiles_occupied(occ, ko, k, ax):
    """Which tiles hold any non-air target voxel: the block maximum of the coarse occupancy array `occ`
    (given at rung `ko`) over each tile's footprint at rung k. When a tile is smaller than one coarse voxel
    the blocks merge, which keeps a tile its neighbour occupies -- the check only ever errs towards keeping."""
    o = occ
    for d in range(3):
        st = (ax[d] >> (ko - k)) if ko >= k else (ax[d] << (k - ko))
        st = np.clip(st, 0, max(o.shape[d] - 1, 0))
        u, inv = np.unique(st, return_inverse=True)
        o = np.take(np.maximum.reduceat(o, u, axis=d), inv, axis=d)
    return o


def region_list(srcs, patch=256, region=1024, allowed=None, boost=None, exclude=(), cap=OCC_VOX, log=None):
    """Every (source, rung, shard-aligned region) worth visiting, with its draw weight. `exclude` is the
    held-out box(es) as (origin, size) at rung 2: a region entirely inside one is dropped. The weights sum
    to 1; the list is in enumeration order (`walk_order` gives the visit order)."""
    sw = np.array([s["volume_um3"] for s in srcs], np.float64)
    sw /= sw.sum()
    out = []
    for i, (s, ws) in enumerate(zip(srcs, sw)):
        ko, occ = occupancy(next(iter(s["targets"].values()))["pyr"], cap=cap)
        for k, pk in rung_probs(s, patch, allowed, boost).items():
            ax, R = region_tiles(s, k, region, patch)
            keep = tiles_occupied(occ, ko, k, ax)
            ex = []
            for o, sz in exclude:
                d = k - 2
                ex.append((np.array(o) >> d, np.maximum(np.array(sz) >> d, 1)) if d >= 0
                          else (np.array(o) << -d, np.array(sz) << -d))
            lo = [np.array([ax[0][a], ax[1][b], ax[2][c]], np.int64) for a, b, c in np.argwhere(keep)]
            lo = [q for q in lo if not any(np.all(q >= eo) and np.all(q + R <= eo + es) for eo, es in ex)]
            if not lo:
                continue
            w = float(ws * pk) / len(lo)
            out += [{"s": i, "k": int(k), "lo": [int(v) for v in q], "size": [int(v) for v in R], "w": w}
                    for q in lo]
            if log:
                log(f"  {s['line'].split(',')[0].split('/')[-1][:40]:40} rung {k:>2}: {len(lo):>7} regions "
                    f"of {int(np.prod([len(q) for q in ax])):>7} (tile {tuple(int(v) for v in R)})")
    tot = sum(q["w"] for q in out) or 1.0
    for q in out:
        q["w"] /= tot
    return out


# ------------------------------------------------------------- the region teacher stores
# A separate service (cloud/teacher_regions.py) walks the SAME region list in the same order and runs the
# upstream teacher over each rung-2 region, writing a probability store per region:
#
#     <TEACHER_REGIONS>/<channel>/region_<z>_<y>_<x>.zarr   (z, y, x = the region origin in RUNG-2 voxels,
#                                                            attrs origin_zyx and `done`)
#
# A rung-2 region origin is a multiple of REGION (1024): both the CT level and the target export are
# 1024^3-sharded there, so the walk's tiles are the shard grid. When a window lies inside ONE finished
# store, the loader takes its target from that store instead of the exported mask pyramid -- soft teacher
# probability instead of a thresholded mask -- and rung 3 is the 2x mean pool of it. Anything else (no
# store, not `done`, a window straddling two regions, any other rung) falls back to the mask pyramid.

TEACHER_REGIONS = _os.environ.get("USRM2_TEACHER_REGIONS", "/vesuvius/usrm2/teacher_regions")
REGION = 1024  # the rung-2 region edge the walk and the teacher service agree on

# The VERSO output channel (docs/unified_design.md section 23). The model has ONE recto output and one
# verso output -- channel 0 and channel 1 of the same 1x1x1 head, not a second branch. The verso target has
# no exported pyramid: it exists only as region stores, `<root>/verso/region_<z>_<y>_<x>.zarr`, written by
# the 5090 pod exactly like the recto ones (1024^3 uint8 probability * 255, volcomp q8, attrs `origin_zyx`,
# `channels: ["verso"]` and `done`). A voxel no finished verso store covers gets WEIGHT 0 in that channel,
# which is the per-channel ignore the loss already understands; at every rung but 2 and 3 the whole channel
# is weight 0 for now.
VERSO = "verso"
TSTORE_TTL = 1800.0  # seconds a MISSING region store stays missing in a loader's cache (the pod publishes
                     # continuously, so a negative result must expire; a found store is cached for good)


def teacher_region_path(lo2, channel="recto", root=None):
    """Where the region teacher service writes the store of the region at rung-2 origin `lo2`."""
    z, y, x = (int(v) for v in lo2)
    return f"{root or TEACHER_REGIONS}/{channel}/region_{z}_{y}_{x}.zarr"


# Where the 5090 pod PUBLISHES the verso region stores. A store is two objects -- `zarr.json` and the one
# 1024^3 shard `c/0/0/0` -- and it appears only once the pod has finished it (`done` in the attrs), so a
# 404 means "not published yet", never "no verso here" (see `stream.Planner.fetch_verso`).
VERSO_REGIONS_URL = ("https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/"
                     "representations/predictions/teacher_regions/verso-2.4um")
REGION_FILES = ("zarr.json", "c/0/0/0")


def verso_region_url(lo2, base=None):
    """The published URL of the verso region store at rung-2 origin `lo2` (no trailing slash)."""
    z, y, x = (int(v) for v in lo2)
    return f"{str(base or VERSO_REGIONS_URL).rstrip('/')}/region_{z}_{y}_{x}.zarr"


def read_teacher(a, k, lo, p):
    """(cube, inside) of a region teacher store read at rung k (2, or 3 = its 2x mean pool): the uint8
    probability over the window at corner `lo` (rung-k voxels) and the mask of the voxels the store
    actually covers. The store's own grid is rung 2, its corner `origin_zyx`."""
    d, p3 = int(k) - 2, shape3(p)
    o, S = np.array(a.attrs["origin_zyx"], np.int64), np.array(a.shape[-3:], np.int64)
    lo2, n2 = (np.asarray(lo, np.int64) << d) - o, p3 << d
    out = np.zeros(tuple(n2), np.uint8)
    aa, bb = np.maximum(lo2, 0), np.minimum(lo2 + n2, S)
    if (bb > aa).all():
        sl = tuple(slice(int(x), int(y)) for x, y in zip(aa, bb))
        blk = np.asarray(a[(0,) + sl] if a.ndim == 4 else a[sl], np.uint8)
        st = aa - lo2
        out[st[0]:st[0] + blk.shape[0], st[1]:st[1] + blk.shape[1], st[2]:st[2] + blk.shape[2]] = blk
    for _ in range(d):
        out = pool2(out)
    e = 1 << d                                   # a rung-k voxel counts only if its whole footprint is stored
    ins = np.zeros(tuple(p3), bool)
    jlo = np.clip(-(-np.maximum(-lo2, 0) // e), 0, p3)
    jhi = np.clip((np.minimum(S - lo2, n2)) // e, 0, p3)
    if (jhi > jlo).all():
        ins[jlo[0]:jhi[0], jlo[1]:jhi[1], jlo[2]:jhi[2]] = True
    return out, ins


def region_visits(regions, cap=64):
    """`region_list` expanded into VISITS, so the rung mix holds throughout the walk and not only in the
    expectation of a prefix.

    A weighted shuffle front-loads the heavy items, and a rung with few regions but a large weight (rung 9
    of a scroll is ONE region, and `--rung-boost 9=16` asks for a large share of it) is therefore used up in
    the first percent of the walk and never seen again. Giving that region v = round(w * R) visits, each of
    weight w / v, makes almost every entry weigh 1 / R: the order becomes near-uniform, the number of visits
    of a group is proportional to its intended share, and the coarse rungs are spread over the whole epoch.
    A visit draws its own windows (its own rng), so `v > 1` is a denser sampling of a region, not the same
    windows again -- and v > 1 only happens where the weight per region is above average, i.e. exactly where
    the ladder has almost no data to begin with. The fine rungs keep their one visit each."""
    out = []
    R = len(regions)
    for r in regions:
        v = int(min(max(round(r["w"] * R), 1), max(int(cap), 1)))
        for j in range(v):
            out.append(dict(r, w=r["w"] / v, v=j))
    return out


def walk_order(w, seed=0):
    """A weighted shuffle WITHOUT replacement (Efraimidis-Spirakis): key = Exp(1) / w, ascending. The first
    item is i with probability w_i / sum(w), and every item appears exactly once."""
    rng = np.random.default_rng(int(seed))
    w = np.asarray(w, np.float64)
    return np.argsort(rng.exponential(size=len(w)) / np.maximum(w, 1e-300), kind="stable")


class Patches(torch.utils.data.IterableDataset):
    """Random (ct, teacher) patches; train patches never touch the val box."""

    def __init__(self, patch=128, ct=CT, stores=TRAIN, exclude=VAL, seed=0, air_keep=0.1, fg_min=0.05,
                 fg_keep=0.25, sym=True, aug=None, dense_pow=0.0, dense_ref=0.2, ctx=(), stores_file=None,
                 recheck=200, rungs=None, rung_boost=None, channels=None, require_targets=False, stream=None,
                 region=0, windows_per_region=64, region_fails=0, teacher_regions=None, cascade="off",
                 verso=False, verso_regions=None):
        """dense_pow > 0 biases sampling towards sheet-dense patches: a patch whose target mean m is below
        dense_ref is kept with probability (m / dense_ref) ** dense_pow (crushed windings are where students fail).
        stores_file: instead of `stores`, a text file with one comma-joined group per line that is re-read
        whenever its mtime changes (checked every `recheck` patches): a training run picks up new stores as
        they are generated, and keeps sampling the old ones (by voxel count) when nothing new arrived.

        rungs: switches on the MULTI-RESOLUTION mode (docs/unified_design.md). Every line of `stores` /
        `stores_file` is then `ct_base,target_group[,target_group...]`: a CT pyramid plus one whole-scroll
        target pyramid per output channel, both addressed by rung (rung k = 0.6 * 2^k um). A sample is
        (source, rung k, corner): the CT and the targets are read at rung k, the context cubes at rungs
        k + ctx[0] .. k + ctx[-1], and the loader yields a COMPACT uint8 sample (`rung_item`) that
        `usrm2.prep.prepare` turns into (x, target, weight) on the GPU. `rungs` is the set of
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
        skews the mix by hand.

        stream: a queue directory written by `usrm2 stream-plan` (usrm2/stream.py). The workers then REPLAY
        that queue instead of sampling -- worker w takes entries w, w + W, w + 2W, ... and reads each window
        from the rolling local buffer the planner fills, waiting when the planner has not got there yet --
        and every sample carries `idx` (its queue index) and `wait` (milliseconds waited, ms), which
        train.py logs as `stream_wait_ms`."""
        super().__init__()
        self.stream = None if stream is None else str(stream)
        # REGION MODE: visit one 1024^3 region of one source at one rung, draw `windows_per_region` windows
        # inside it, then move on. The region is snapped to the shard grid of the level the CT is read from,
        # so a visit touches one shard footprint per level and the buffer serves the whole visit.
        self.region, self.windows_per_region = int(region or 0), int(windows_per_region)
        self.region_fails = int(region_fails or 8 * max(self.windows_per_region, 1))
        # the region teacher stores (see `read_teacher`): soft probability instead of the exported mask,
        # for any rung-2/3 window that lies inside one finished region store
        self.teacher_regions = None if teacher_regions in (None, "") else str(teacher_regions)
        # the VERSO output channel (section 23): an extra output channel whose only source is the verso
        # region stores under `verso_regions` (default: the same root as the recto ones). Everywhere no
        # finished verso store covers, the channel's WEIGHT is 0 and the sample trains recto alone.
        self.verso = bool(verso)
        self.verso_regions = (None if verso_regions in (None, "") else str(verso_regions)) or self.teacher_regions
        self._tstore = {}
        # CASCADE (docs/unified_design.md section 22): the rung-(k+1) prediction as an extra input channel.
        # "off" (nothing changes), "mask" (the rung-(k+1) target block), "self" (the model's own coarse
        # prediction, built on the GPU in prep) or "mix". The worker's job is only to READ the extras.
        self.cascade = str(cascade or "off")
        assert self.cascade in CASCADE_MODES, f"--cascade {cascade}: one of {CASCADE_MODES}"
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
        self.nrecto = None  # set by _open_rungs: the output channels that are NOT the verso one
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
        CTX_CACHE.clear(); CHUNK_INDEX.clear()  # a re-open (stores file changed) must see levels/shards added since
        self._tstore = {}                       # ... and region teacher stores finished since
        if self.stores_file:
            self.paths = [g.split(",") for g in self.read_groups()]
            assert self.paths, f"{self.stores_file} lists no store groups"
        self.srcs = source_groups([",".join(ps) for ps in self.paths])
        if self.channels is None:
            self.channels = list(dict.fromkeys(c for s in self.srcs for c in s["targets"]))
        if self.verso and VERSO not in self.channels:  # no source pyramid provides it: the stores do
            self.channels = list(self.channels) + [VERSO]
        # the verso channel is always LAST and never decides whether a window is worth training on: the
        # foreground / density rejection rules stay exactly what a recto-only run's were
        self.nrecto = len(self.channels) - (1 if self.channels and self.channels[-1] == VERSO else 0)
        allowed = None if self.rungs is True else set(self.rungs)
        for s in self.srcs:
            s["probs"] = rung_probs(s, self.patch, allowed, self.rung_boost)
            s["axis"] = axis(s["umbilicus"])
        self.w = np.array([s["volume_um3"] for s in self.srcs], np.float64)
        self.w /= self.w.sum()
        self.ex = [e if isinstance(e, (tuple, list)) and len(e) == 2 and not isinstance(e[0], str)
                   else val_box(e) for e in self.exclude]  # (origin, size) at rung 2
        self.arrs = self.srcs

    def region_state(self, region=None):
        """Per-iterator region-visit state. It cannot live on the dataset: the stream planner drives one
        dataset from several threads, one rng stream (or one active region) each, and each visits its own
        region. With `region` (a `region_list` record) the state is FIXED to that region: the draw never
        picks a new one, which is how the walk hands regions out."""
        if region is None:
            return {"left": 0, "fails": 0}
        return {"i": int(region["s"]), "k": int(region["k"]), "lo": np.array(region["lo"], np.int64),
                "size": np.array(region["size"], np.int64), "left": self.windows_per_region, "fails": 0,
                "fixed": True}

    def _rung_sample(self, rng, st=None):
        """One compact sample (see `rung_item`), or None when the corner is rejected."""
        return self._rung_draw(rng, st=st)[1]

    def region_grid(self, s, k):
        """The shard size, in rung-k voxels, of the level the CT is read from at rung k: what a region is
        snapped to so that one region is one shard footprint."""
        return shard_grid(s["ct_pyr"], k)

    def new_region(self, rng, st):
        """Pick the next (source, rung, 1024^3 region) to visit. One rng draw per quantity, as always."""
        i = rng.choice(len(self.srcs), p=self.w)
        s = self.srcs[i]
        ks, pk = list(s["probs"]), np.array(list(s["probs"].values()))
        k = int(rng.choice(ks, p=pk))
        blo, bs = target_box(next(iter(s["targets"].values())), k)
        size = np.minimum(shape3(self.region), np.maximum(bs, self.patch))
        hi = np.maximum(blo + bs - size, blo)
        lo = rng.integers(np.minimum(blo, hi), hi + 1)
        g = self.region_grid(s, k)
        lo = np.maximum((lo // g) * g, 0)  # snapped: the region is one shard footprint at this rung
        st.update(i=int(i), k=k, lo=lo, size=size, left=self.windows_per_region, fails=0)
        return st

    def _rung_draw(self, rng, hook=None, build=True, st=None):
        """Draw one candidate window, apply the rejection rules and (with `build`) read it.
        Returns (descriptor, sample); (None, None) when the window is rejected. The descriptor is what the
        stream queue stores: source index, rung, corner, cube symmetry, the blank-patch flag and the raw-aug
        draws -- everything `_rung_build` needs to reproduce the sample without an rng.

        `hook` (usrm2.stream.Hook) makes this the STREAM PLANNER: before each read it is asked to fetch the
        chunks that read will touch, and it replaces the local-coverage rules (nothing is mirrored up front;
        a key the origin does not serve is air). With build=False the accepted window is not turned into a
        sample -- the planner only needs the descriptor and the fetches."""
        p = self.patch
        if self.region:  # region mode: the windows of one visit come from one 1024^3 region
            st = self.region_state() if st is None else st
            if not st.get("fixed") and (st.get("left", 0) <= 0 or st.get("fails", 0) >= self.region_fails):
                self.new_region(rng, st)
            i, k, s = st["i"], st["k"], self.srcs[st["i"]]
            st["fails"] += 1
            hi = np.maximum(st["lo"] + st["size"] - p, st["lo"])
            lo = rng.integers(np.minimum(st["lo"], hi), hi + 1)
        else:
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
                return None, None
        cpyr = s["ct_pyr"]
        if hook is None:
            csrc = max(r for r in cpyr if r <= k)
            e = 1 << (k - csrc)
            if not covered(cpyr[csrc], lo * e, p * e):  # a partially mirrored level (level 0 of the desk's CT)
                return None, None
            if self.require_targets:  # a partially pulled export: an absent shard is "not yet exported", not air
                for t in s["targets"].values():
                    tsrc = max(r for r in t["pyr"] if r <= k)
                    te = 1 << (k - tsrc)
                    if not covered(t["pyr"][tsrc], lo * te, p * te):
                        return None, None
        elif not hook.data(s, k, lo):
            return None, None
        bl, blank, prm, tea = self.aug.get("blank"), False, {}, None
        if bl and rng.random() < bl["p"]:  # an all-air patch (CT 0 = air) with target 0
            blank = True
            ct = np.zeros(tuple(p), np.uint8)
            tg = np.zeros((len(self.channels),) + tuple(p), np.uint8)
            w = np.zeros_like(tg)
        else:
            ct = read_rung(cpyr, k, lo, p, dtype=np.uint8)
            if (ct == 0).mean() > 0.9 and rng.random() > self.air_keep:
                return None, None
            tea, ver = self._teacher_store(s, k, lo), self._verso_store(s, k, lo)
            tg, w = self._rung_target(s, k, lo, ct, teacher=tea, verso=ver)
            nr = self.nrecto or len(self.channels)
            sel = w[:nr] > 0          # the verso channel is ignored here: a cout=2 run draws the same
            m = float(np.sum(tg[:nr], where=sel, dtype=np.int64)) / 255.0 / max(int(sel.sum()), 1)  # windows
            if m < self.fg_min and rng.random() > self.fg_keep:
                return None, None
            if self.dense_pow > 0 and m < self.dense_ref and rng.random() > (m / self.dense_ref) ** self.dense_pow:
                return None, None
            if not sel.any():  # every voxel masked or outside the target box: no gradient, so no sample.
                return None, None  # (after the draws above, so the rng stream is the one it always was)
            prm = raw_params(rng, self.aug)
            ct = raw_apply(ct, prm)
        sym = int(draw_sym(rng, tuple(p))) if self.sym else 0  # applied on the GPU (usrm2.prep), not here
        if self.region and st is not None:
            st["left"], st["fails"] = st.get("left", 0) - 1, 0  # an accepted window ends the failure run
        desc = {"s": int(i), "k": k, "lo": [int(v) for v in lo], "y": sym, "b": int(blank), "r": prm}
        if not blank and tea:  # the replaying worker reads the same teacher store, not the mask pyramid
            desc["t"] = tea
        if not blank and ver:
            desc["v"] = ver
        if hook is not None and (self.ctx or self.cascade != "off"):
            hook.ctx(s, k, lo)
        if not build:
            return desc, None
        cx = context(s["ct"], lo, ct.shape, self.ctx, rung=k) if self.ctx else ()
        ex = self._cascade_extras(s, k, lo, ct.shape, blank=blank)
        return desc, rung_item(np.stack([ct] + list(cx)), tg, w, k, lo, s["axis"], sym, **ex)

    def _rung_build(self, d):
        """A queue descriptor (`_rung_draw`) -> the compact sample, read from the local buffer. No rng: the
        rejection rules and every random draw happened in the planner."""
        p, s, k = self.patch, self.srcs[int(d["s"])], int(d["k"])
        lo = np.array(d["lo"], np.int64)
        blank = bool(d.get("b"))
        if blank:
            ct = np.zeros(tuple(p), np.uint8)
            tg = np.zeros((len(self.channels),) + tuple(p), np.uint8)
            w = np.zeros_like(tg)
        else:
            ct = read_rung(s["ct_pyr"], k, lo, p, dtype=np.uint8)
            # the verso store may have been published AFTER the planner wrote this entry: look again
            tg, w = self._rung_target(s, k, lo, ct, teacher=d.get("t"),
                                      verso=d.get("v") or self._verso_store(s, k, lo))
            ct = raw_apply(ct, d.get("r") or {})
        cx = context(s["ct"], lo, ct.shape, self.ctx, rung=k) if self.ctx else ()
        ex = self._cascade_extras(s, k, lo, ct.shape, blank=blank)
        return rung_item(np.stack([ct] + list(cx)), tg, w, k, lo, s["axis"], int(d.get("y", 0)), **ex)

    def cascade_ctx(self):
        """The context offset of the TENTH cube: the one rung the coarse (rung k+1) input needs and the
        rung-k input does not. `--ctx 1..9` -> 10, so the coarse cube's own contexts are ctx_2..ctx_9
        (rungs k+2..k+9) plus rung k+10. Past the top of a pyramid it is pooled like any other."""
        return (int(self.ctx[-1]) + 1) if self.ctx else 1

    def _cascade_extras(self, s, k, lo, shape, blank=False):
        """What a cascade sample reads on top of the ordinary one (see `rung_item`): `cm`, the rung-(k+1)
        target block over the patch FOOTPRINT (half the patch on every axis, so 1/8 of a cube), and, for the
        self/mix modes, the tenth context cube `cx` and the corner `lo1` of the rung-(k+1) cube.

        Rung 11 is the top of the ladder: there is no rung 12, so its cascade channel is zero (which is also
        what `--cascade-drop` teaches the model to expect). A corner with an odd coordinate puts the coarse
        block half a rung-(k+1) voxel off the footprint; that is one rung-k voxel and is left as it is.
        `blank`: the all-air patch of the blank aug -- target 0 everywhere, so the coarse block is 0 too."""
        if self.cascade == "off":
            return {}
        p = shape3(shape)
        lo = np.asarray(lo, np.int64)
        hp = np.maximum(p // 2, 1)
        t = s["targets"].get(self.channels[0]) if self.channels else None
        if blank or int(k) + 1 >= NRUNGS or t is None:
            cm = np.zeros(tuple(hp), np.uint8)
        else:
            cm = read_rung(t["pyr"], int(k) + 1, lo // 2, hp, dtype=np.uint8)
        out = {"cm": cm}
        if self.cascade in ("self", "mix"):
            d, c0 = self.cascade_ctx(), lo + p // 2
            out["cx"] = (np.zeros(tuple(p), np.uint8) if blank else
                         read_rung(s["ct_pyr"], int(k) + d, c0 // (1 << d) - p // 2, p, dtype=np.uint8))
            out["lo1"] = c0 // 2 - p // 2
        return out

    def _region_store(self, s, k, lo, channel, root):
        """The finished region store of `channel` that covers this whole window, or None. Only rungs 2 and 3
        (the store's own grid is rung 2 and rung 3 is its 2x pool), only a window inside ONE region, and
        only for the FIRST source: the regions are named by rung-2 origin, which is one scroll's grid."""
        if not root or int(k) not in (2, 3) or s is not self.srcs[0]:
            return None
        d = int(k) - 2
        lo2, hi2 = np.asarray(lo, np.int64) << d, ((np.asarray(lo, np.int64) + self.patch) << d) - 1
        a, b = lo2 // REGION, hi2 // REGION
        if not np.array_equal(a, b) or (lo2 < 0).any():
            return None                                    # the window straddles two region stores
        p = teacher_region_path(a * REGION, channel, root)
        return p if self._teacher_arr(p) is not None else None

    def _teacher_store(self, s, k, lo):
        """The RECTO region teacher store covering this window (see `_region_store`)."""
        return self._region_store(s, k, lo, self.channels[0], self.teacher_regions)

    def _verso_store(self, s, k, lo):
        """The VERSO region store covering this window. The pod publishes these continuously, so a store
        that was missing when a window was drawn may exist later; `_teacher_arr` expires a miss."""
        if not self.verso:
            return None
        return self._region_store(s, k, lo, VERSO, self.verso_regions)

    def _teacher_arr(self, path):
        """The opened store, or None when it does not exist or the service has not finished it. A HIT is
        cached for the life of the loader; a MISS expires after `TSTORE_TTL` seconds, because the region
        services publish while the run trains (a verso store appearing mid-run must be picked up)."""
        import time as _t
        a = self._tstore.get(path)
        if a is None or (a[0] is False and _t.time() - a[1] > TSTORE_TTL):
            try:
                z = open_zarr(path)
                z = z if z.attrs.get("done") else False
            except Exception:  # noqa: BLE001  (not written yet)
                z = False
            a = self._tstore[path] = (z, _t.time())
        return a[0] or None

    def _rung_target(self, s, k, lo, ct, teacher=None, verso=None):
        """(target, weight) at rung k as uint8 (255 = 1.0): one channel per output channel, 0 (weight 0) for
        a channel this source does not provide; weight 1 inside the target's box and where CT > 0, times its
        source weight. `teacher`: a region teacher store whose soft probability replaces the exported mask
        for the first channel (the window is inside it; rung 3 is its 2x mean pool). `verso`: the VERSO
        region store, the only source of the verso channel -- without one that channel keeps weight 0
        everywhere, which is a perfectly good recto-only sample."""
        p = self.patch
        tg = np.zeros((len(self.channels),) + tuple(p), np.uint8)
        w = np.zeros_like(tg)
        inside_ct = ct > 0
        ta = self._teacher_arr(teacher) if teacher else None
        va = self._teacher_arr(verso) if verso else None
        for c, chan in enumerate(self.channels):
            if chan == VERSO:
                if va is not None:
                    v, ins = read_teacher(va, k, lo, p)
                    np.copyto(tg[c], v, where=inside_ct)
                    w[c] = np.where(ins & inside_ct, np.uint8(255), np.uint8(0))
                continue  # no store here: weight 0, i.e. the verso channel is ignored on this sample
            if ta is not None and chan == (ta.attrs.get("channel") or
                                           (list(ta.attrs.get("channels") or []) or [None])[0] or self.channels[0]):
                v, ins = read_teacher(ta, k, lo, p)
                np.copyto(tg[c], v, where=inside_ct)
                w[c] = np.where(ins & inside_ct, np.uint8(255), np.uint8(0))
                continue
            t = s["targets"].get(chan)
            if t is None:
                continue  # per-channel ignore: this source says nothing about that channel
            ins = np.zeros(tuple(p), bool)
            blo, bs = target_box(t, k)
            a = np.maximum(blo - lo, 0)
            b = np.minimum(blo + bs - lo, p)
            if (b > a).all():
                ins[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
            np.copyto(tg[c], read_rung(t["pyr"], k, lo, p, dtype=np.uint8), where=inside_ct)  # masked CT: no surface
            w[c] = np.where(ins & inside_ct, np.uint8(round(255 * t["weight"])), np.uint8(0))
        return tg, w

    def _open_stream(self):
        """Replay mode: the queue's meta.json fixes the store list, the patch, the context offsets and the
        channel order, so a queue can only be replayed by a run that matches the plan."""
        from usrm2 import stream as S
        m = self.smeta = S.read_meta(self.stream)
        assert [int(v) for v in m["patch"]] == [int(v) for v in self.patch], \
            f"--stream was planned at patch {m['patch']}, not {[int(v) for v in self.patch]}"
        assert tuple(m["ctx"]) == tuple(self.ctx), f"--stream was planned with --ctx {m['ctx']}"
        assert str(m.get("cascade", "off")) == self.cascade, \
            f"--stream was planned with --cascade {m.get('cascade', 'off')}, not {self.cascade} " \
            "(the planner fetches the coarse target block and the tenth context cube)"
        self.paths, self.stores_file = [g.split(",") for g in m["stores"]], None
        self.teacher_regions = self.teacher_regions or m.get("teacher_regions")
        self.verso = self.verso or bool(m.get("verso"))
        self.verso_regions = self.verso_regions or m.get("verso_regions") or self.teacher_regions
        if self.channels is None:
            self.channels = list(m["channels"])
        assert list(self.channels) == list(m["channels"]), f"--stream was planned for channels {m['channels']}"
        self._open_rungs()

    def _replay(self):
        """Stream g of GW yields the queue entries g, g + GW, g + 2GW, ... in order, waiting for the planner.

        A stream is (rank, loader worker): with DDP every rank replays the SAME queue, so the share has to be
        taken over `world * num_workers` streams -- otherwise every rank trains on every window and the run
        sees each one `world` times. `stream-plan --workers` is that total.

        The walk ends: when the planner has written `epoch_done` and the queue is exhausted the iterator
        STOPS instead of waiting forever, so `usrm2 train --stream` finishes its epoch cleanly. The queue
        always ends on a stream boundary (the planner writes it in groups of GW), so every stream sees the
        same number of windows and a DDP run's ranks stop together."""
        import time as _time
        from usrm2 import stream as S
        info = torch.utils.data.get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        g, GW = rank * W + w, world * W
        assert GW == int(self.smeta["workers"]), \
            f"--stream was planned for {self.smeta['workers']} streams, not {world} ranks x {W} workers"
        self._dirs = list(self.smeta["dirs"])
        os.makedirs(os.path.join(self.stream, S.PROGRESS), exist_ok=True)
        prog = os.path.join(self.stream, S.PROGRESS, f"w{g}")
        start = -1
        if os.path.exists(prog):
            try:
                start = int(open(prog).read().strip() or -1)  # a restart skips what this worker already served
            except ValueError:
                start = -1
        done = os.path.join(self.stream, S.EPOCH_DONE)
        wait = 0.0
        for line, waited in S.tail(os.path.join(self.stream, S.QUEUE), stop=lambda: os.path.exists(done)):
            wait += waited
            rec = json.loads(line)
            i = int(rec["i"])
            if i % GW != g or i <= start:
                continue
            yield self._replay_one(rec, prog, wait)
            wait = 0.0

    def _replay_one(self, rec, prog, wait):
        """One queue entry -> the compact sample, once every chunk it names is in the buffer."""
        import time as _time
        from usrm2 import stream as S
        t0, i = _time.time(), int(rec["i"])
        if any(di >= len(self._dirs) for di, _ in rec["c"]):  # a level the planner touched after meta was written
            self._dirs = list(S.read_meta(self.stream)["dirs"])
        for di, key in rec["c"]:
            back = 0.02
            while not S.have(f"{self._dirs[di]}/{key}"):
                _time.sleep(back)
                back = min(back * 1.5, 1.0)
        wait += _time.time() - t0   # the WAIT is the queue and the buffer, never this worker's own decode
        item = self._rung_build(rec)
        item["idx"], item["wait"] = torch.tensor(i), torch.tensor(wait * 1000.0)
        with open(prog, "w") as f:
            f.write(str(i))
        return item

    def _open(self):
        """Each teacher store names its CT volume and scroll axis (attrs), so stores from several scrolls can mix."""
        global NORM, UMBILICUS
        if self.stream:
            return self._open_stream()
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
        if self.stream:
            yield from self._replay()
            return
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + 1000 * (info.id if info else 0))
        p = self.patch
        rst = self.region_state()
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
                got = self._rung_sample(rng, st=rst)
                if got is None:
                    continue
                rejected, served = 0, served + 1
                yield got
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
    sw = np.array([s["volume_um3"] for s in srcs], np.float64)
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


def val_grid_rungs(patch, stores, box2, rungs=VAL_RUNGS, limit=8, ctx=(), channels=None, cascade="off",
                   verso=False, verso_regions=None):
    """Per-rung validation: the held-out box (given at rung 2) read at each rung from the same pyramids
    as training. Returns the same compact uint8 items the rung loader yields (`rung_item`), so 8 patches x
    3 rungs at 256^3 cost ~1.6 GB of host memory instead of the ~22 GB of 14-channel float32 they used to.

    `verso`: add the verso output channel. Its target comes from a published verso region store covering the
    val box (`verso_regions`) when there is one, and otherwise carries weight 0 -- so the grid still has two
    channels, `val_png` still shows both, and the verso metrics are simply empty until a store lands on the
    box. The RECTO channel is never taken from a region store here: the validation target stays the exported
    mask pyramid, so a run's recto numbers are comparable across the whole ladder."""
    srcs = source_groups(stores)
    if channels is None:
        channels = list(dict.fromkeys(c for s in srcs for c in s["targets"]))
        if verso and VERSO not in channels:
            channels = channels + [VERSO]
    ds = Patches(patch=patch, stores=stores, exclude=[], rungs=True, ctx=ctx, channels=channels, sym=False,
                 cascade=cascade, verso=verso, verso_regions=verso_regions)
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
                ct = read_rung(s["ct_pyr"], k, lo, p3, dtype=np.uint8)
                tg, w = ds._rung_target(s, k, lo, ct, verso=ds._verso_store(s, k, lo))
                cx = context(s["ct"], lo, ct.shape, ctx, rung=k) if ctx else ()
                ex = ds._cascade_extras(s, k, lo, ct.shape)
                out.append(rung_item(np.stack([ct] + list(cx)), tg, w, k, lo, s["axis"], **ex))
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
