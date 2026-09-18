"""Verso targets for the student, one per teacher lineage (recto, m7).

Neither teacher can label the verso face (mirroring the CT does not flip them), but a recto-trained student with
the radial vector NEGATED moves its band to the other face of the sheet (predict.probs(radial_sign=-1)). That
band is a broad blob filling the outer half of the sheet, so the target is its OUTER SKIN: blob voxels whose
outward neighbour along the blob's normal is outside the blob (a one-sided erosion), painted 3 voxels thick.
The skin is then anchored to the CT: from every voxel of the teacher's recto band the CT is marched outward
along the band normal until the papyrus ends (CT falls to the midpoint of the sheet's peak and the local air
level, before any further recto band); skin within `near` voxels of such an edge is written as 255 (loss weight
1), the rest of the skin (touching sheets, where the CT has no edge) as WEAK (weight ~0.3, see train.losses
`wtgt`), everything else 0. Where sheets touch there is deliberately NO air-contrast gate: the skin continues
through, only with less weight.

Boxes are processed in y/x tiles with a margin so GPU memory stays bounded; each tile runs the student (both
heads in one pass) and the geometry on the GPU."""
import os

import numpy as np
import torch
import torch.nn.functional as F

from usrm2 import data
from usrm2.predict import out_array, probs, put

WEAK = 77  # uint8 value of an unanchored skin voxel: target 1 with loss weight 77/255 = 0.3


MODES = ("skin", "raw")  # skin: anchored outer skin (above); raw: the flipped student's probability as it is


def verso_path(store, mode="skin"):
    """Where a store's verso target lives: boxesN/box.zarr -> boxesN_v/box.zarr, eval.zarr -> eval_v.zarr
    (suffix _vraw for the raw mode)."""
    suf = "_v" if mode == "skin" else f"_v{mode}"
    store = str(store).rstrip("/")
    d, n = os.path.split(store)
    if os.path.basename(d).startswith("boxes"):
        return os.path.join(d + suf, n)
    return store[:-5] + suf + ".zarr" if store.endswith(".zarr") else store + suf


def gauss3(x, sigma):
    """Separable Gaussian blur of a (Z,Y,X) float tensor (reflect padding)."""
    r = int(3 * sigma + 0.5)
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
    k /= k.sum()
    y = x[None, None]
    for d in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + d] = 2 * r + 1
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="replicate"), k.view(shape))
    return y[0, 0]


def normals(p, rad, sigma=2.0):
    """Unit normals (3,Z,Y,X) of a band/blob probability: gradient of its blur, oriented outward (dot radial >= 0).
    Inside a blob the gradient points inward, so the sign flip makes every voxel's normal point away from the axis."""
    ps = gauss3(p, sigma)
    g = torch.stack(torch.gradient(ps), 0)
    g = g * torch.where((g * rad).sum(0) < 0, -1.0, 1.0)
    return g / (g.norm(dim=0, keepdim=True) + 1e-6)


def sample(vol, q):
    """Trilinear sample of a (Z,Y,X) tensor at (N,3) zyx voxel coordinates (0 outside)."""
    sh = torch.tensor(vol.shape, device=vol.device, dtype=torch.float32)
    gq = ((q / (sh - 1)) * 2 - 1)[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
    return F.grid_sample(vol[None, None], gq, align_corners=True, mode="bilinear", padding_mode="zeros").view(-1)


def paint(mask, q, n, t, thick=(-1.0, 0.0, 1.0)):
    """Set mask at q + (t+dt) n for dt in thick (rounded, clipped)."""
    hi = torch.tensor(mask.shape, device=mask.device) - 1
    for dt in thick:
        v = torch.minimum((q + (t + dt)[:, None] * n).round().long().clamp(min=0), hi)
        mask[v[:, 0], v[:, 1], v[:, 2]] = True


def skin(blob_p, rad, thr=0.5, step=2.5, chunk=2_000_000):
    """Outer skin of the flipped-student blob: blob voxels whose outward neighbour (step voxels along the blob
    normal) is outside the blob, painted 3 voxels thick along the normal. Returns a bool (Z,Y,X) tensor."""
    n = normals(blob_p, rad)
    blob = blob_p >= thr
    idx = torch.nonzero(blob).float()
    out = torch.zeros(blob.shape, dtype=torch.bool, device=blob_p.device)
    for a in range(0, len(idx), chunk):
        q = idx[a:a + chunk]
        nq = n[:, q[:, 0].long(), q[:, 1].long(), q[:, 2].long()].T
        edge = sample(blob_p, q + step * nq) < thr
        paint(out, q[edge], nq[edge], torch.zeros(int(edge.sum()), device=q.device))
    return out


def ct_edge(band_p, ct, rad, thr=0.5, far=40, min_contrast=20, chunk=1_500_000):
    """Where the papyrus ends outward of the teacher's recto band: from every band voxel march the blurred CT
    along the band normal; the edge is the last voxel before the CT drops below the midpoint of the sheet's peak
    (first 12 voxels) and the profile's minimum, provided no further recto band is hit first. Bool (Z,Y,X)."""
    n = normals(band_p, rad)
    cts = gauss3(ct, 1.0)
    idx = torch.nonzero(band_p >= thr).float()
    out = torch.zeros(band_p.shape, dtype=torch.bool, device=band_p.device)
    for a in range(0, len(idx), chunk):
        q = idx[a:a + chunk]
        nq = n[:, q[:, 0].long(), q[:, 1].long(), q[:, 2].long()].T
        prof = torch.stack([sample(cts, q + t * nq) for t in range(far + 1)])  # (far+1, N)
        pb = torch.stack([sample(band_p, q + t * nq) for t in range(far + 1)])
        peak, air = prof[:12].max(0).values, prof.min(0).values
        below = prof < (0.5 * (peak + air))[None]
        dropped = torch.cumsum((pb < 0.2).float(), 0) > 0
        nxt = (pb >= thr) & dropped
        big = torch.full_like(peak, far + 1, dtype=torch.long)
        tE = torch.where(below.any(0), below.float().argmax(0), big)
        tN = torch.where(nxt.any(0), nxt.float().argmax(0), big)
        ok = (tE <= far) & (tE >= 3) & (tE < tN) & ((peak - air) > min_contrast)
        paint(out, q[ok], nq[ok], (tE[ok] - 1).float())
    return out


def dilate(mask, r):
    return F.max_pool3d(mask[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def targets(flip_p, band_p, ct, rad, near=3):
    """uint8 verso target for one lineage: 255 = skin anchored to a CT edge, WEAK = skin only, 0 = none."""
    s = skin(flip_p, rad)
    e = ct_edge(band_p, ct, rad)
    anchored = s & dilate(e, near)
    out = torch.where(anchored, 255, torch.where(s, WEAK, 0)).to(torch.uint8)
    out[ct <= 0] = 0  # masked CT: no surface
    return out


def read_tile(arr, z, y, x, Z, Y, X):
    s = (slice(z, z + Z), slice(y, y + Y), slice(x, x + X))
    return np.asarray(arr[(0,) + s] if arr.ndim == 4 else arr[s])


def run(stores, ckpt, window=128, halo=16, tile=512, margin=32, device=None, volume=None, force=False, batch=1,
        modes=("skin", "raw")):
    """Write the verso target store(s) of every teacher store in a comma-joined group (one lineage per store, head k
    of the student paired with store k), one per mode: "skin" (anchored outer skin, lossless 255/WEAK/0) and/or
    "raw" (the flipped student's probability, volcomp like a teacher store). Returns the output paths (mode ->
    list). Skips outputs already marked done."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    paths = [p for p in str(stores).split(",") if p]
    arrs = [data.open_zarr(p) for p in paths]
    origin, size = data.box(arrs[0])
    vol = data.local(volume or arrs[0].attrs.get("volume", data.CT))
    umb = arrs[0].attrs.get("umbilicus", data.UMBILICUS)
    ax = data.axis(umb if os.path.exists(umb) else None)  # a desk path read in the cloud -> the configured umbilicus
    outs = {m: [verso_path(p, m) for p in paths] for m in modes}
    todo = [(m, k) for m in modes for k, o in enumerate(outs[m])
            if force or not (os.path.exists(o) and data.open_zarr(o).attrs.get("done"))]
    if not todo:
        return outs
    ct_arr = data.open_zarr(vol)
    o_arrs = {}
    for m, k in todo:
        o = outs[m][k]
        os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
        o_arrs[m, k] = out_array(o, tuple(int(v) for v in size), origin, volcomp=(m == "raw"),  # skin: lossless 255 / WEAK / 0
                                 volume=arrs[0].attrs.get("volume", data.CT), umbilicus=arrs[0].attrs.get("umbilicus", data.UMBILICUS))
        o_arrs[m, k].attrs.update({"channels": ["verso"], "mode": m, "source": paths[k], "student": str(ckpt), "weak": WEAK})
    Z, Y, X = (int(v) for v in size)
    for y0 in range(0, Y, tile):
        for x0 in range(0, X, tile):
            ya, yb = max(y0 - margin, 0), min(y0 + tile + margin, Y)
            xa, xb = max(x0 - margin, 0), min(x0 + tile + margin, X)
            ct = read_tile(ct_arr, origin[0], origin[1] + ya, origin[2] + xa, Z, yb - ya, xb - xa)
            core = (slice(0, Z), slice(y0 - ya, min(y0 + tile, Y) - ya), slice(x0 - xa, min(x0 + tile, X) - xa))
            if not ct.any():
                for mk in todo:
                    put(o_arrs[mk], np.zeros(tuple(s.stop - s.start for s in core), np.uint8), 0, y0, x0)
                continue
            flip, _ = probs(ckpt, vol, int(origin[0]), int(origin[1] + ya), int(origin[2] + xa), Z, yb - ya, xb - xa,
                            window=window, halo=halo, device=dev, head="all", radial_sign=-1.0, batch=batch)
            if flip.ndim == 3:
                flip = flip[None]
            rad = torch.from_numpy(data.radial(ax, (origin[0], origin[1] + ya, origin[2] + xa), ct.shape)).to(dev)
            ct_t = torch.from_numpy(ct.astype(np.float32)).to(dev)
            for m, k in todo:
                fk = flip[min(k, len(flip) - 1)]
                if m == "raw":
                    put(o_arrs[m, k], np.clip(np.rint(fk[core] * 255), 0, 255).astype(np.uint8), 0, y0, x0)
                    continue
                band = torch.from_numpy(read_tile(arrs[k], 0, ya, xa, Z, yb - ya, xb - xa).astype(np.float32) / 255).to(dev)
                t = targets(torch.from_numpy(fk).to(dev), band, ct_t, rad)
                put(o_arrs[m, k], t[core].cpu().numpy(), 0, y0, x0)
                del band, t
            del ct_t, rad, flip
            torch.cuda.empty_cache() if dev.type == "cuda" else None
    for mk in todo:
        o_arrs[mk].attrs["done"] = True
    return outs
