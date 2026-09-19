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
    """Gaussian-blended sliding window over a uint8 ROI. fn: normalized (B,C,w,w,w) tensor -> prob (B,w,w,w).
    prep(ct_window, (z,y,x) window offset) -> (C,w,w,w) float input; default is z-scored CT alone."""
    prep = prep or (lambda c, o: data.zscore(c)[None])
    if any(s < window for s in roi.shape):  # thinner than a window: pad with air, crop the result
        S = roi.shape
        roi = np.pad(roi, [(0, max(window - s, 0)) for s in S])
        return slide(fn, roi, window, halo, dev, prep)[:S[0], :S[1], :S[2]]
    Z, Y, X = roi.shape
    acc, wsum, g = None, np.zeros(roi.shape, np.float32), gauss(window)
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
                        p = fn(t)[0].float().cpu().numpy()  # (w,w,w) or (C,w,w,w): fn may return every head
                    if acc is None:
                        acc = np.zeros(p.shape[:-3] + roi.shape, np.float32)
                    acc[..., z:z + window, y:y + window, x:x + window] += p * g
                    wsum[z:z + window, y:y + window, x:x + window] += g
    if acc is None:
        return np.zeros(roi.shape, np.float32)
    return np.where((wsum > 0) & (roi > 0), acc / np.maximum(wsum, 1e-6), 0)  # masked CT (0) -> no surface


def slide_gpu(fn, roi, window, halo, dev, prep, batch=1, streams=1):
    """slide() with the ROI, the normalization and the (fp16) accumulators all on the GPU, `batch` windows per
    forward and `streams` CUDA streams (one thread + its own accumulators each, summed at the end) so the card
    never idles between kernel launches. A 384x2048x2048 box needs ~8 GB per stream + the model + batch x the
    activations. prep(ct_window uint8 tensor, (z,y,x)) -> (C,w,w,w) float tensor; masked CT (0) -> no surface."""
    import threading
    if any(s < window for s in roi.shape):
        S = roi.shape
        roi = np.pad(roi, [(0, max(window - s, 0)) for s in S])
        return slide_gpu(fn, roi, window, halo, dev, prep, batch, streams)[:S[0], :S[1], :S[2]]
    R = torch.from_numpy(np.ascontiguousarray(roi)).to(dev)
    g = torch.from_numpy(gauss(window)).to(dev).half()
    Z, Y, X = R.shape
    stride = window - 2 * halo
    todo = [(z, y, x) for z in starts(Z, window, stride) for y in starts(Y, window, stride) for x in starts(X, window, stride)
            if R[z:z + window, y:y + window, x:x + window].any()]
    accs, wsums = [], []

    def work(k):
        acc, wsum = None, torch.zeros(R.shape, dtype=torch.float16, device=dev)
        wsums.append(wsum)
        s = torch.cuda.Stream(dev) if dev.type == "cuda" else None
        with torch.no_grad(), (torch.cuda.stream(s) if s else contextlib.nullcontext()):
            mine = todo[k::streams]
            for i in range(0, len(mine), batch):
                offs = mine[i:i + batch]
                t = torch.stack([prep(R[z:z + window, y:y + window, x:x + window], o) for o in offs for z, y, x in [o]])
                with autocast(dev):
                    p = fn(t.contiguous(memory_format=torch.channels_last_3d))  # (B,w,w,w) or (B,C,w,w,w)
                if acc is None:
                    acc = torch.zeros(tuple(p.shape[1:-3]) + tuple(R.shape), dtype=torch.float16, device=dev)
                    accs.append(acc)
                for (z, y, x), pj in zip(offs, p):
                    acc[..., z:z + window, y:y + window, x:x + window] += (pj.float() * g.float()).half()
                    wsum[z:z + window, y:y + window, x:x + window] += g
            if s:
                s.synchronize()
    import contextlib
    if streams > 1:
        ths = [threading.Thread(target=work, args=(k,)) for k in range(streams)]
        [t.start() for t in ths], [t.join() for t in ths]
    else:
        work(0)
    if not accs:  # nothing but air
        return np.zeros(R.shape, np.float32)
    acc, wsum = accs[0], wsums[0]
    for a, w in zip(accs[1:], wsums[1:]):
        acc += a
        wsum += w
    del accs, wsums
    with torch.no_grad():
        out = np.empty(tuple(acc.shape), np.float32)
        for z in range(0, Z, 64):  # a slab at a time keeps the float32 temporaries small
            a, w, r = acc[..., z:z + 64, :, :].float(), wsum[z:z + 64].float(), R[z:z + 64]
            out[..., z:z + 64, :, :] = torch.where((w > 0) & (r > 0), a / w.clamp_min(1e-6), torch.zeros_like(a)).cpu().numpy()
    del R, acc, wsum
    return out


def zscore_t(c):
    x = c.float()
    return (x - x.mean()) / (x.std() + 1e-3)


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


def put(arr, u8, z=0, y=0, x=0):
    """Write a uint8 block into a store at a local offset, whatever its layout ((Z,Y,X) or (1,Z,Y,X))."""
    s = (slice(z, z + u8.shape[0]), slice(y, y + u8.shape[1]), slice(x, x + u8.shape[2]))
    arr[(0,) + s if arr.ndim == 4 else s] = u8


def u8(prob):
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def write(path, prob, origin, volcomp=True, volume=None):
    a = out_array(path, prob.shape, origin, volcomp=volcomp, volume=volume)
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


def flips_vec(fn, n=8):
    """Flip TTA for the student: flipping spatial axis d also negates radial component d (channel 1+d)."""
    fl = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)][:n]

    def go(t):
        out = 0
        for f in fl:
            x = torch.flip(t, [2 + d for d in f]).clone()
            ni = x.shape[1] - 3
            for d in f:
                x[:, ni + d] = -x[:, ni + d]
            out = out + torch.flip(fn(x), [1 + d for d in f])
        return out / len(fl)
    return go


HEADS = {"mean": lambda p: p.mean(1), "prod": lambda p: p.prod(1) ** (1 / p.shape[1]), "max": lambda p: p.max(1).values}


def probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=128, halo=16, device=None, tta=0, luts=(), head=0, radial_sign=1.0, batch=1):
    """Sliding-window recto probability (float32) over a box; returns (prob, checkpoint state).
    tta: number of axis flips to average; luts: intensity LUTs (uint8->float) whose predictions are averaged in;
    head: which head of a multi-teacher student (int), "mean" / "prod" / "max" over all heads, or "all" for a
    (heads, Z, Y, X) result. radial_sign=-1 negates the radial vector: a recto-trained student then places its
    band on the other face of the sheet (the verso; see verso.py). batch > 1: windows batched on the GPU
    (slide_gpu; CT + radial inputs only, no context channels / luts / tta)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.load(ckpt, map_location=dev)
    net = M.build(st["args"]["size"], verbose=False, cout=st["args"].get("cout", 1), cin=st["args"].get("cin", 4), add_skip=st["args"].get("add_skip", 0), deep=st["args"].get("deep", 0)).to(dev)
    net.load_state_dict(st["ema"])
    net.eval()
    roi, ax = data.open_zarr(volume)[z0:z0 + Z, y0:y0 + Y, x0:x0 + X], data.axis()
    r = 0.0 if st["args"].get("no_radial") else float(radial_sign)  # training zeroed the radial channels
    data.NORM = tuple(st["args"]["norm_stats"]) if st["args"].get("norm") == "global" else None  # as trained
    rad = lambda c, o: data.radial(ax, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape) * r
    ctx = tuple(st["args"].get("ctx") or ())
    cx = (lambda c, o: data.context(volume, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape, ctx)) if ctx else (lambda c, o: ())
    preps = [lambda c, o: data.inputs(c, rad(c, o), cx(c, o))] + [(lambda c, o, l=l: data.inputs(l[c], rad(c, o), cx(c, o))) for l in luts]
    pick = (lambda p: p) if head == "all" else HEADS[head] if isinstance(head, str) else (lambda p: p[:, int(head)])
    fn = lambda t: pick(torch.sigmoid(net(t)))  # (B,C,...) -> (B,...)  (or (B,C,...) for "all")
    if tta > 1:
        assert head != "all", "tta and head=all do not combine"
        fn = flips_vec(fn, tta)
    if batch > 1:
        assert not ctx and not luts and tta <= 1, "batched inference: CT + radial inputs only"
        R = torch.from_numpy(data.radial(ax, (z0, y0, x0), roi.shape) * r).to(dev)
        norm = data.NORM

        def prep_t(c, o):
            x = c.float()
            x = (x - norm[0]) / norm[1] if norm else (x - x.mean()) / (x.std() + 1e-3)
            z, y, xx = o
            return torch.cat([x[None], R[:, z:z + window, y:y + window, xx:xx + window]])
        return slide_gpu(fn, roi, window, halo, dev, prep_t, batch=batch), st
    return sum(slide(fn, roi, window, halo, dev, pr) for pr in preps) / len(preps), st


def predict(ckpt, volume, z0, y0, x0, Z, Y, X, out, window=128, halo=16, device=None, volcomp=True, ome=False, tta=0, luts=(), head=0, radial_sign=1.0):
    prob, st = probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=window, halo=halo, device=device, tta=tta, luts=luts, head=head, radial_sign=radial_sign)
    if ome:
        write_ome(out, u8(prob), (z0, y0, x0), full_shape=data.open_zarr(volume).shape[-3:],
                  meta={"checkpoint": str(ckpt), "step": int(st.get("step", 0)), "volume": volume})
    else:
        write(out, prob, (z0, y0, x0), volcomp=volcomp, volume=volume)
    return out
