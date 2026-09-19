"""CT + teacher patch sampling. All boxes are (z, y, x) level-0 voxels.

Teacher stores: uint8 = recto probability * 255, shape (Z,Y,X) (or (1,Z,Y,X)),
attrs origin_zyx = the level-0 index of element 0. CT: uint8, 0 = air/masked.
A store entry "a.zarr,a_m7.zarr" names several teachers over the SAME box: one target channel each
(a multi-head student); the first one supplies the box, volume and umbilicus.
"""
import itertools

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
    assert os.path.exists(path), f"no umbilicus at {path}: every scroll needs one (create it, then pass --umbilicus)"
    pts = sorted((p["z"], p["y"], p["x"]) for p in json.load(open(path))["control_points"])
    return np.array(pts, np.float64).T


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


CTX_CACHE = {}


def levels(volume):
    """The pyramid levels of a volume path '.../name.zarr/0' -> {level: zarr array} for the levels on disk
    (a streamed URL is mapped to its local mirror first: stores made in the cloud record the URL)."""
    import os
    base = str(local(str(volume))).rstrip("/").rsplit("/", 1)[0]
    if base not in CTX_CACHE:
        d = {}
        for l in range(1, 4):
            if "://" not in base and not os.path.isdir(f"{base}/{l}"):
                break  # a level missing from the local mirror is pooled from the one below, never streamed
            try:
                d[l] = open_zarr(f"{base}/{l}")
            except Exception:
                break
        CTX_CACHE[base] = d
    return CTX_CACHE[base]


def context(volume, origin, shape, ctx):
    """Coarse cubes of the SAME size centred on the same point as the level-0 patch at `origin`/`shape`:
    one (Z,Y,X) uint8 cube per level in `ctx` (1 = 4.8um, 2 = 9.6um, 3 = 19.2um), read from the pyramid;
    a level that is not on disk is made by 2x mean-pooling the level below. Outside the volume = 0 (air)."""
    lv, out = levels(volume), []
    c0 = np.asarray(origin, np.int64) + np.asarray(shape, np.int64) // 2  # centre, level-0 voxels
    for l in ctx:
        src = l if l in lv else max(lv) if lv else None
        assert src is not None, f"{volume} has no pyramid levels for context channels"
        f_read, extra = 2 ** src, 2 ** (l - src)  # read at `src`, pool by `extra`
        n = np.asarray(shape, np.int64) * extra
        lo = c0 // f_read - n // 2
        a = lv[src]
        hi = np.minimum(lo + n, a.shape[-3:])
        lo_c = np.maximum(lo, 0)
        cube = np.zeros(tuple(n), np.uint8)
        if (hi > lo_c).all():
            blk = a[lo_c[0]:hi[0], lo_c[1]:hi[1], lo_c[2]:hi[2]]
            s = lo_c - lo
            cube[s[0]:s[0] + blk.shape[0], s[1]:s[1] + blk.shape[1], s[2]:s[2] + blk.shape[2]] = blk
        if extra > 1:
            e = extra
            cube = cube.reshape(shape[0], e, shape[1], e, shape[2], e).mean((1, 3, 5)).astype(np.uint8)
        out.append(cube)
    return out


def inputs(ct, rad, ctx=()):
    """Model input (1+len(ctx)+3, Z,Y,X): z-scored CT, z-scored coarse context cubes, radial unit vector."""
    return np.concatenate([zscore(ct)[None]] + [zscore(c)[None] for c in ctx] + [rad])


class Patches(torch.utils.data.IterableDataset):
    """Random (ct, teacher) patches; train patches never touch the val box."""

    def __init__(self, patch=128, ct=CT, stores=TRAIN, exclude=VAL, seed=0, air_keep=0.1, fg_min=0.05,
                 fg_keep=0.25, sym=True, aug=None, dense_pow=0.0, dense_ref=0.2, ctx=(), stores_file=None,
                 recheck=200):
        """dense_pow > 0 biases sampling towards sheet-dense patches: a patch whose target mean m is below
        dense_ref is kept with probability (m / dense_ref) ** dense_pow (crushed windings are where students fail).
        stores_file: instead of `stores`, a text file with one comma-joined group per line that is re-read
        whenever its mtime changes (checked every `recheck` patches): a training run picks up new stores as
        they are generated, and keeps sampling the old ones (by voxel count) when nothing new arrived."""
        super().__init__()
        self.norm, self.umbilicus = NORM, UMBILICUS  # module state the (spawned) workers must inherit explicitly
        self.stores_file, self.recheck, self.file_mtime = stores_file, recheck, None
        if stores_file:
            stores = self.read_groups()
        self.patch, self.ct_path, self.paths, self.seed = shape3(patch), ct, [str(s).split(",") for s in stores], seed
        self.exclude = [e for e in (exclude if isinstance(exclude, (list, tuple)) else [exclude]) if e]
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

    def _open(self):
        """Each teacher store names its CT volume and scroll axis (attrs), so stores from several scrolls can mix."""
        global NORM, UMBILICUS
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
            i = rng.choice(len(self.arrs), p=self.w)
            o, s = self.boxes[i]
            lo = rng.integers(MARGIN, s - MARGIN - p + 1)  # store-local corner
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


def loader(patch, batch, workers, **kw):
    """Workers start as fresh processes (forkserver), never forks: the parent has usually opened zarr already
    (validation grid), whose asyncio loop thread does not survive a fork and breaks streamed (HTTP) reads."""
    ds = Patches(patch=patch, **kw)
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers,
                                       pin_memory=True, persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers else None,
                                       multiprocessing_context="forkserver" if workers else None)
