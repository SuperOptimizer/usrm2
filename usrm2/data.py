"""CT + teacher patch sampling. All boxes are (z, y, x) level-0 voxels.

Teacher stores: uint8 = recto probability * 255, shape (Z,Y,X) (or (1,Z,Y,X)),
attrs origin_zyx = the level-0 index of element 0. CT: uint8, 0 = air/masked.
"""
import numpy as np
import torch

CT = "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
TRAIN = ["/vesuvius/usrm2/teacher/a.zarr", "/vesuvius/usrm2/teacher/b.zarr"]  # made by `usrm2 teacher`; plus
TRAIN += sorted(__import__("glob").glob("/vesuvius/usrm2/teacher/boxes/box_*.zarr"))  # `usrm2 teacher-boxes` output
VAL = "/vesuvius/usrm2/teacher/eval.zarr"
UMBILICUS = "/vesuvius/usrm/umbilicus/PHercParis4/umbilicus-full-resolution.json"
MARGIN = 16  # sliding-window predictions are worse at the teacher box edges


def open_zarr(path):
    import zarr
    try:
        import volcomp_zarr  # noqa: F401  (registers the "volcomp" codec)
    except Exception:
        pass
    return zarr.open(path, mode="r")


def box(arr):
    """(origin_zyx, shape_zyx) of a teacher store."""
    return np.array(arr.attrs["origin_zyx"], np.int64), np.array(arr.shape[-3:], np.int64)


def read3(arr, o, p):
    """Read a p^3 patch at store-local offset o, tolerating (1,Z,Y,X) stores."""
    s = tuple(slice(int(a), int(a) + p) for a in o)
    return arr[(0,) + s] if arr.ndim == 4 else arr[s]


def axis(path=UMBILICUS):
    """(z, y, x) arrays of the scroll axis control points, sorted by z (level-0 voxels)."""
    import json
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


def zscore(x):
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-3)


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
    """Random axis permutation + flips applied to the (C,Z,Y,X) input and (Z,Y,X) target; the radial
    vector channels 1..3 of x are permuted/negated to match."""
    perm, flip = rng.permutation(3), rng.random(3) < 0.5
    sl = tuple(slice(None, None, -1 if f else 1) for f in flip)
    tg = np.ascontiguousarray(np.transpose(tg, perm)[sl])
    x = np.transpose(x, (0,) + tuple(perm + 1))[(slice(None),) + sl]
    x = np.concatenate([x[:1], x[1 + perm] * np.where(flip, -1, 1).astype(np.float32)[:, None, None, None]])
    return np.ascontiguousarray(x), tg


def inputs(ct, rad):
    """Model input (4,Z,Y,X): z-scored CT + radial unit vector."""
    return np.concatenate([zscore(ct)[None], rad])


class Patches(torch.utils.data.IterableDataset):
    """Random (ct, teacher) patches; train patches never touch the val box."""

    def __init__(self, patch=128, ct=CT, stores=TRAIN, exclude=VAL, seed=0, air_keep=0.1, fg_min=0.05,
                 fg_keep=0.25, sym=True, aug=None):
        super().__init__()
        self.patch, self.ct_path, self.paths, self.exclude, self.seed = patch, ct, list(stores), exclude, seed
        self.sym = sym  # the 48 cube symmetries; the GPU augs are in aug.py
        self.aug = aug or {}  # the worker-side raw-uint8 stage: window / volcomp / blank
        self.air_keep, self.fg_min, self.fg_keep = air_keep, fg_min, fg_keep  # low-foreground patches are mostly skipped
        self.arrs = None

    def _open(self):
        self.ct = open_zarr(self.ct_path)
        self.arrs = [open_zarr(p) for p in self.paths]
        self.boxes = [box(a) for a in self.arrs]
        self.w = np.array([np.prod(s) for _, s in self.boxes], np.float64)
        self.w /= self.w.sum()
        self.ex = box(open_zarr(self.exclude)) if self.exclude else None
        self.ax = axis()

    def __iter__(self):
        if self.arrs is None:
            self._open()
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + 1000 * (info.id if info else 0))
        p = self.patch
        rejected = 0
        while True:
            assert rejected < 10000, "no acceptable patch in 10000 draws (stores all air, or all inside the val box?)"
            rejected += 1
            i = rng.choice(len(self.arrs), p=self.w)
            o, s = self.boxes[i]
            lo = rng.integers(MARGIN, s - MARGIN - p + 1)  # store-local corner
            g = o + lo  # global corner
            if self.ex is not None and np.all(g < self.ex[0] + self.ex[1]) and np.all(g + p > self.ex[0]):
                continue
            bl = self.aug.get("blank")
            if bl and rng.random() < bl["p"]:  # an all-air patch (CT 0 = air) with target 0
                ct, tg = np.zeros((p, p, p), np.uint8), np.zeros((p, p, p), np.float32)
            else:
                ct = self.ct[g[0]:g[0] + p, g[1]:g[1] + p, g[2]:g[2] + p]
                if (ct == 0).mean() > 0.9 and rng.random() > self.air_keep:
                    continue
                tg = read3(self.arrs[i], lo, p).astype(np.float32) / 255.0 * (ct > 0)  # masked CT -> no surface
                if tg.mean() < self.fg_min and rng.random() > self.fg_keep:
                    continue
                ct = raw(rng, ct, self.aug)
            rejected, x = 0, inputs(ct, radial(self.ax, g, ct.shape))
            if self.sym:
                x, tg = augment(rng, x, tg)
            yield torch.from_numpy(x), torch.from_numpy(tg)[None]


def val_grid(patch=128, ct=CT, store=VAL, limit=32):
    """Deterministic non-overlapping tiling of the val box -> list of (ct, tgt)."""
    cta, tga, ax = open_zarr(ct), open_zarr(store), axis()
    o, s = box(tga)
    corners = [(z, y, x) for z in range(0, s[0] - patch + 1, patch)
               for y in range(0, s[1] - patch + 1, patch)
               for x in range(0, s[2] - patch + 1, patch)]
    if limit and len(corners) > limit:  # deterministic even subsample
        corners = [corners[i] for i in np.linspace(0, len(corners) - 1, limit).astype(int)]
    out = []
    for lo in corners:
        g = o + np.array(lo)
        c = cta[g[0]:g[0] + patch, g[1]:g[1] + patch, g[2]:g[2] + patch]
        t = read3(tga, lo, patch).astype(np.float32) / 255.0 * (c > 0)
        out.append((torch.from_numpy(inputs(c, radial(ax, g, c.shape))), torch.from_numpy(t)[None]))
    return out


def loader(patch, batch, workers, **kw):
    ds = Patches(patch=patch, **kw)
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers,
                                       pin_memory=True, persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers else None)
