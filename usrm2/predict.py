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
                    t = torch.from_numpy(prep(c, (z, y, x)))[None].to(dev).to(memory_format=M.memfmt())
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
                    p = fn(t.contiguous(memory_format=M.memfmt()))  # (B,w,w,w) or (B,C,w,w,w)
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


def shard_shape(shape, chunk=128, cap=1024):
    """Shard shape for a store: one shard per `cap`^3 box, a multiple of the chunk and covering the array
    (so a store smaller than `cap` on an axis is a single shard, i.e. ONE data file on disk)."""
    return tuple(min(cap, -(-int(s) // chunk) * chunk) for s in shape)


def out_array(path, shape, origin, volcomp=True, volume=None, umbilicus=None, rung=2):
    import zarr
    try:
        from volcomp_zarr import VolcompCodec
    except Exception:
        VolcompCodec = None
    # ONE data file per ~1024^3 of store: a zarr v3 shard holds the 128^3 inner chunks. 512 chunk files per
    # store cost ~590 sftp operations to publish to the mirror; a shard costs one put. Every store usrm2
    # writes is sharded this way, and its writers fill a shard region in one write (see teacher.run).
    sh = shard_shape(shape)
    kw = dict(shape=shape, chunks=(128, 128, 128), shards=sh, dtype="uint8", fill_value=0, overwrite=True)
    if volcomp and VolcompCodec is not None:  # volcomp requires exactly 128^3 chunks, hence 3D
        assert all(s % 128 == 0 for s in shape), "volcomp output needs box sizes that are multiples of 128"
        z = zarr.create_array(path, serializer=VolcompCodec(q=8), **kw)
    else:
        kw["shape"], kw["chunks"], kw["shards"] = (1,) + tuple(shape), (1, 128, 128, 128), (1,) + sh
        z = zarr.create_array(path, **kw)
    z.attrs.update({"channels": ["recto"], "voxel_um": data.rung_um(rung), "rung": int(rung),
                    "origin_zyx": [int(v) for v in origin], "scale": 1.0,
                    "volume": volume or data.CT, "umbilicus": umbilicus or data.UMBILICUS})  # so loaders know the scroll
    return z


def put(arr, u8, z=0, y=0, x=0):
    """Write a uint8 block into a store at a local offset, whatever its layout ((Z,Y,X) or (1,Z,Y,X))."""
    s = (slice(z, z + u8.shape[0]), slice(y, y + u8.shape[1]), slice(x, x + u8.shape[2]))
    arr[(0,) + s if arr.ndim == 4 else s] = u8


def u8(prob):
    return np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)


def write(path, prob, origin, volcomp=True, volume=None, rung=2):
    a = out_array(path, prob.shape, origin, volcomp=volcomp, volume=volume, rung=rung)
    p = u8(prob)
    a[:] = p[None] if a.ndim == 4 else p


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


CASCADE_HALO = 16  # rung-(k+1) voxels of margin around a coarse prediction's footprint


def crop_pad(a, off, shape):
    """`a[off : off + shape]` zero-padded where it runs past the end: `slide` pads an ROI thinner than one
    window and then asks for windows the cascade array does not cover."""
    out = np.zeros(tuple(int(v) for v in shape), np.float32)
    if a is None:
        return out
    s = tuple(slice(int(o), min(int(o) + int(n), int(d))) for o, n, d in zip(off, shape, a.shape))
    blk = a[s]
    out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
    return out


def up2x_np(a):
    """2x trilinear upsample of a (Z,Y,X) float32 array, the same `model.up2x` training uses."""
    t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None, None]
    return M.up2x(t, tuple(2 * int(q) for q in a.shape))[0, 0].numpy()


def probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=128, halo=16, device=None, tta=0, luts=(), head=0, radial_sign=1.0, batch=1, rung=None,
          cascade=None, cascade_depth=3):
    """Sliding-window recto probability (float32) over a box; returns (prob, checkpoint state).
    tta: number of axis flips to average; luts: intensity LUTs (uint8->float) whose predictions are averaged in;
    head: which head of a multi-teacher student (int), "mean" / "prod" / "max" over all heads, or "all" for a
    (heads, Z, Y, X) result. radial_sign=-1 negates the radial vector: a recto-trained student then places its
    band on the other face of the sheet (the verso; see verso.py). batch > 1: windows batched on the GPU
    (slide_gpu; CT + radial inputs only, no context channels / luts / tta).
    cascade: None = whatever the checkpoint was trained with (`args["cascade"]`), False / "off" = force the
    channel to zero. cascade_depth: how many rungs above k are predicted top-down to fill it (default 3;
    the cost is geometric, 1/8 per level, so three levels add ~14 %). A checkpoint trained without the
    cascade channel ignores both and runs on 14 channels exactly as before."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    st = torch.load(ckpt, map_location=dev)
    net = M.build(st["args"]["size"], verbose=False, cout=st["args"].get("cout", 1), cin=st["args"].get("cin", 4), add_skip=st["args"].get("add_skip", 0), deep=st["args"].get("deep", 0)).to(dev)
    net.load_state_dict(st["ema"])
    net.eval()
    scale = bool(st["args"].get("scale_plane"))
    k = int(rung) if rung is not None else (data.base_rung(volume) if scale else 2)
    ax0 = data.axis()
    if rung is None:
        roi = data.open_zarr(volume)[z0:z0 + Z, y0:y0 + Y, x0:x0 + X]
    else:  # read the CT at that rung of the pyramid (a rung above its top is pooled from the top)
        roi = data.read_rung(data.rungs(volume), k, (z0, y0, x0), (Z, Y, X)).astype(np.uint8)
    ax = data.axis_at(ax0, k)
    r = 0.0 if st["args"].get("no_radial") else float(radial_sign)  # training zeroed the radial channels
    data.NORM = tuple(st["args"]["norm_stats"]) if st["args"].get("norm") == "global" else None  # as trained
    ctx = tuple(st["args"].get("ctx") or ())
    kk = k if scale else None  # the scale plane only exists in checkpoints trained with it
    kr = k if (rung is not None or scale) else None  # legacy: let context() take the rung from the level path
    # CASCADE (docs/unified_design.md section 22): a checkpoint trained with --cascade takes a 15th channel,
    # the rung-(k+1) prediction over the same field of view. Inference is TOP-DOWN and recursive: the box's
    # footprint is predicted at rung k+1 first (half the size on every axis, so 1/8 of the work, plus a halo),
    # upsampled 2x and fed in. `cascade_depth` rungs above that the channel is zero, which is exactly the
    # `--cascade-drop` case the model was trained on. A checkpoint trained without it is unaffected.
    cmode = str(st["args"].get("cascade", "off") or "off")
    use_cas = cmode != "off"                       # the checkpoint HAS the channel: it must always be fed
    off = cascade is not None and (cascade is False or str(cascade) == "off")
    depth0 = 0 if (off or not use_cas) else max(int(cascade_depth), 0)  # 0 = the channel is fed as zeros
    pick = (lambda p: p) if head == "all" else HEADS[head] if isinstance(head, str) else (lambda p: p[:, int(head)])
    fn = lambda t: pick(torch.sigmoid(net(t)))  # (B,C,...) -> (B,...)  (or (B,C,...) for "all")
    if tta > 1:
        assert head != "all", "tta and head=all do not combine"
        fn = flips_vec(fn, tta)
    fn0 = lambda t: torch.sigmoid(net(t))[:, 0]  # the coarse passes only ever need head 0

    def cascade_for(kk_, o, s, depth):
        """The rung-(kk_+1) prediction over the footprint of the box (o, s), upsampled 2x onto that box's
        own grid; None when nothing above may be predicted (depth exhausted, or the top of the ladder)."""
        if depth <= 0 or kk_ + 1 >= data.NRUNGS:
            return None
        m = CASCADE_HALO
        o1 = [max(int(v) // 2 - m, 0) for v in o]
        s1 = [(int(v) + 1) // 2 + 2 * m for v in s]
        p1 = at_rung(kk_ + 1, o1, s1, depth - 1)
        up = up2x_np(p1)
        a = [int(o[i]) - 2 * o1[i] for i in range(3)]
        return np.ascontiguousarray(up[a[0]:a[0] + int(s[0]), a[1]:a[1] + int(s[1]), a[2]:a[2] + int(s[2])])

    def make_prep(kk_, o, casc, lut=None):
        axk = data.axis_at(ax0, kk_)

        def pr(c, off):
            g = (int(o[0]) + off[0], int(o[1]) + off[1], int(o[2]) + off[2])
            cc = crop_pad(casc, off, c.shape) if use_cas else None
            cxs = data.context(volume, g, c.shape, ctx, rung=kk_) if ctx else ()
            return data.inputs(c if lut is None else lut[c], data.radial(axk, g, c.shape) * r, cxs,
                               rung=(kk_ if scale else None), cascade=cc)
        return pr

    def at_rung(kk_, o, s, depth):
        """Head-0 probability over a box at rung kk_ (rung-kk_ voxels), cascading `depth` rungs above it."""
        sub = data.read_rung(data.rungs(volume), kk_, o, s).astype(np.uint8)
        return slide(fn0, sub, window, halo, dev, make_prep(kk_, o, cascade_for(kk_, o, s, depth)))

    casc = cascade_for(k, (z0, y0, x0), (Z, Y, X), depth0) if use_cas else None
    rad = lambda c, o: data.radial(ax, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape) * r
    cx = (lambda c, o: data.context(volume, (z0 + o[0], y0 + o[1], x0 + o[2]), c.shape, ctx, rung=kr)) if ctx else (lambda c, o: ())
    cas_at = (lambda c, o: crop_pad(casc, o, c.shape)) if use_cas else (lambda c, o: None)
    preps = [lambda c, o: data.inputs(c, rad(c, o), cx(c, o), rung=kk, cascade=cas_at(c, o))] + \
            [(lambda c, o, l=l: data.inputs(l[c], rad(c, o), cx(c, o), rung=kk, cascade=cas_at(c, o))) for l in luts]
    if batch > 1:
        assert not ctx and not luts and tta <= 1, "batched inference: CT + radial inputs only"
        R = torch.from_numpy(data.radial(ax, (z0, y0, x0), roi.shape) * r).to(dev)
        C = None if casc is None else torch.from_numpy(casc).to(dev)
        norm = data.NORM

        def prep_t(c, o):
            x = c.float()
            x = (x - norm[0]) / norm[1] if norm else (x - x.mean()) / (x.std() + 1e-3)
            z, y, xx = o
            sl = (slice(z, z + window), slice(y, y + window), slice(xx, xx + window))
            parts = [x[None]]
            if use_cas:  # the cascade channel sits before the scale plane; the compiled net never sees the recursion
                parts.append((torch.zeros_like(x) if C is None else C[sl]).float()[None])
            if scale:
                parts.append(torch.full_like(x, (k - 2) / 9.0)[None])
            parts.append(R[(slice(None),) + sl])
            return torch.cat(parts)
        return slide_gpu(fn, roi, window, halo, dev, prep_t, batch=batch), st
    return sum(slide(fn, roi, window, halo, dev, pr) for pr in preps) / len(preps), st


def predict(ckpt, volume, z0, y0, x0, Z, Y, X, out, window=128, halo=16, device=None, volcomp=True, tta=0, luts=(), head=0, radial_sign=1.0, rung=None,
            cascade=None, cascade_depth=3):
    prob, _ = probs(ckpt, volume, z0, y0, x0, Z, Y, X, window=window, halo=halo, device=device, tta=tta, luts=luts, head=head, radial_sign=radial_sign, rung=rung,
                    cascade=cascade, cascade_depth=cascade_depth)
    write(out, prob, (z0, y0, x0), volcomp=volcomp, volume=volume, rung=2 if rung is None else int(rung))
    return out
