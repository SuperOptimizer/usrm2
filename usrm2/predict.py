"""Sliding-window inference -> uint8 recto probability * 255 zarr (volcomp q=8)."""
import numpy as np
import torch

from usrm2 import data, model as M
from usrm2.train import autocast


def gauss(w):
    g = np.exp(-0.5 * ((np.arange(w) - (w - 1) / 2) / (w / 6)) ** 2).astype(np.float32)
    return g[:, None, None] * g[None, :, None] * g[None, None, :]


def starts(n, w, stride):
    s = list(range(0, max(n - w, 0) + 1, stride))
    if s[-1] != n - w:
        s.append(n - w)
    return s


def slide(fn, roi, window, halo, dev, prep=None):
    """Gaussian-blended sliding window over a uint8 ROI. fn: normalized (1,C,w,w,w) tensor -> prob (w,w,w).
    prep(ct_window, (z,y,x) window offset) -> (C,w,w,w) float input; default is z-scored CT alone."""
    prep = prep or (lambda c, o: data.zscore(c)[None])
    Z, Y, X = roi.shape
    acc, wsum, g = np.zeros(roi.shape, np.float32), np.zeros(roi.shape, np.float32), gauss(window)
    stride = window - 2 * halo
    with torch.no_grad():
        for z in starts(Z, window, stride):
            for y in starts(Y, window, stride):
                for x in starts(X, window, stride):
                    c = roi[z:z + window, y:y + window, x:x + window]
                    if not c.any():
                        continue
                    t = torch.from_numpy(prep(c, (z, y, x)))[None].to(dev).to(memory_format=torch.channels_last_3d)
                    with autocast(dev):
                        p = fn(t).float().cpu().numpy()
                    acc[z:z + window, y:y + window, x:x + window] += p * g
                    wsum[z:z + window, y:y + window, x:x + window] += g
    return np.where((wsum > 0) & (roi > 0), acc / np.maximum(wsum, 1e-6), 0)  # masked CT (0) -> no surface


def out_array(path, shape, origin, volcomp=True, volume=None, umbilicus=None):
    import zarr
    try:
        from volcomp_zarr import VolcompCodec
    except Exception:
        VolcompCodec = None
    kw = dict(shape=shape, chunks=(128, 128, 128), dtype="uint8", fill_value=0, overwrite=True)
    if volcomp and VolcompCodec is not None:  # volcomp requires exactly 128^3 chunks, hence 3D
        assert all(s % 128 == 0 for s in shape), "volcomp output needs box sizes that are multiples of 128"
        z = zarr.create_array(path, serializer=VolcompCodec(q=8), **kw)
    else:
        kw["shape"], kw["chunks"] = (1,) + tuple(shape), (1, 128, 128, 128)
        z = zarr.create_array(path, **kw)
    z.attrs.update({"channels": ["recto"], "voxel_um": 2.4, "origin_zyx": [int(v) for v in origin], "scale": 1.0,
                    "volume": volume or data.CT, "umbilicus": umbilicus or data.UMBILICUS})  # so loaders know the scroll
    return z


def u8(prob):
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def write(path, prob, origin, volcomp=True):
    a = out_array(path, prob.shape, origin, volcomp=volcomp)
    p = u8(prob)
    a[:] = p[None] if a.ndim == 4 else p


def write_ome(path, prob_u8, origin, levels=3, full_shape=None, meta=None):
    """Zarr v2 OME group, a positional drop-in for the vc3d tracer: level 0 has the FULL volume
    shape but only the box's chunks are written (sparse, fill 0). Values are probability * 255,
    no threshold. The origin is rounded down to the 256 chunk grid and the box zero-padded."""
    import json

    import zarr
    from numcodecs import Blosc
    o = np.asarray(origin, np.int64)
    pad = o % 256
    o, prob_u8 = o - pad, np.pad(prob_u8, [(int(p), 0) for p in pad]) if pad.any() else prob_u8
    assert not (o % 256).any()
    full = tuple(int(v) for v in (full_shape or data.open_zarr(data.CT).shape[-3:]))
    g, a = zarr.open_group(path, mode="w", zarr_format=2), prob_u8
    for l in range(levels):
        z = g.create_array(str(l), shape=tuple(-(-s >> l) for s in full), chunks=(256, 256, 256),
                           dtype="uint8", fill_value=0, compressors=Blosc(cname="zstd", clevel=1, shuffle=1),
                           chunk_key_encoding={"name": "v2", "separator": "/"})
        c = o >> l
        z[c[0]:c[0] + a.shape[0], c[1]:c[1] + a.shape[1], c[2]:c[2] + a.shape[2]] = a
        a = a[::2, ::2, ::2]  # nearest 2x downsample, box only
    g.attrs.update({
        "multiscales": [{"version": "0.4", "name": "recto",
                         "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
                         "datasets": [{"path": str(l), "coordinateTransformations":
                                       [{"type": "scale", "scale": [float(1 << l)] * 3}]} for l in range(levels)]}],
        "channels": ["recto"], "voxel_um": 2.4, "origin_zyx": [int(v) for v in o], "scale": 1.0})
    json.dump({**(meta or {}), "threshold": None, "origin_zyx": [int(v) for v in o],
               "shape_zyx": [int(v) for v in prob_u8.shape], "levels": levels, "voxel_um": 2.4},
              open(f"{path}/metadata.json", "w"), indent=1)
    return path


def probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=128, halo=16, device=None):
    """Sliding-window recto probability (float32) over a box; returns (prob, checkpoint state)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.load(ckpt, map_location=dev)
    net = M.build(st["args"]["size"], verbose=False).to(dev)
    net.load_state_dict(st["ema"])
    net.eval()
    roi, ax = data.open_zarr(volume)[z0:z0 + Z, y0:y0 + Y, x0:x0 + X], data.axis()
    r = 0.0 if st["args"].get("no_radial") else 1.0  # training zeroed the radial channels
    prep = lambda c, o: data.inputs(c, data.radial(ax, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape) * r)
    return slide(lambda t: torch.sigmoid(net(t))[0, 0], roi, window, halo, dev, prep), st


def predict(ckpt, volume, z0, y0, x0, Z, Y, X, out, window=128, halo=16, device=None, volcomp=True, ome=False):
    prob, st = probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=window, halo=halo, device=device)
    if ome:
        write_ome(out, u8(prob), (z0, y0, x0), meta={"checkpoint": str(ckpt), "step": int(st.get("step", 0))})
    else:
        write(out, prob, (z0, y0, x0), volcomp=volcomp)
    return out
