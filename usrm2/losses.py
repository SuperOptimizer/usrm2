"""Phase-A auxiliary losses (docs/unified_design.md section 26).

Everything here is OFF by default: `aux_losses` returns an empty dict when no weight is set, and
`train.py` adds nothing to `bce + dice`, so a run whose flags are absent is byte-identical to one
built before this module existed.

The four terms, in the order of `docs/research/synthesis_v2_with_literature.md` section 4 (Phase A):

  L3  exclusivity      `relu(p_recto + p_verso - 1)`, masked by `w_recto * w_verso`  (`--loss-excl`)
  L4  self-consistency `|pool2(p_fine) - pool2(CASCADE)|` on the samples whose cascade channel came
                       from the model's own coarse forward                           (`--loss-selfcons`)
  L8  skeleton recall  `1 - mean(p) over the TARGET's medial surface`                (`--loss-skel`)
  O12 affinity         BCE on K extra output channels, "is the voxel d/2 back and the voxel d/2
                       forward along this axis the SAME sheet?"                      (`--affinity`)

All of them are computed from tensors that already exist at the loss site: no extra forward, no extra
store, no extra loader channel.
"""
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- L3 soft exclusivity

def exclusivity(p, wv):
    """`relu(p_recto + p_verso - 1)` averaged over the voxels where BOTH channels carry weight.

    p (B, >=2, Z, Y, X) probabilities, wv (B, >=2, ...) the loader's per-voxel weights. The mask is
    `wv[:, 0] * wv[:, 1]`, i.e. the L3 definition of the plan: only where a verso store actually
    covers the voxel is there anything to say about the pair. A sample with no verso store therefore
    contributes nothing (mask 0 everywhere) instead of pushing the untrained verso channel to 0.

    Zero for disjoint channels (p_r + p_v <= 1 everywhere) and zero-gradient where either weight is 0.
    """
    m = wv[:, 0] * wv[:, 1]
    return ((p[:, 0] + p[:, 1] - 1).clamp_min(0) * m).sum() / m.sum().clamp_min(1e-6)


# --------------------------------------------------------------------- L4 cascade self-consistency

def self_consistency(p_fine, cas, w=None, sel=None):
    """Mean |fine pooled 2x - coarse| at the COARSE resolution.

    `cas` is the CASCADE input channel this step already carries (`prep.Cascade.channel`): the
    rung-(k+1) prediction over the same field of view, upsampled 2x. Pooling it back by 2 undoes the
    upsample up to the trilinear kernel's residual smoothing (per axis 0.75/0.125/0.125 instead of a
    delta), which is the price of reusing the tensor instead of running a second coarse forward.
    The coarse side is detached; it comes from the EMA net under `no_grad` and has no graph anyway.

    `sel` (B,) is 1 for the samples whose channel came from the SELF source and 0 for the rest:
    the `mask` source is the rung-(k+1) TARGET, so scoring against it would be a second, blurrier
    copy of the supervised loss rather than a consistency term, and a dropped channel is all zeros.
    """
    pf = F.avg_pool3d(p_fine[:, :1], 2)
    pc = F.avg_pool3d(cas[:, :1], 2).detach()
    ww = torch.ones_like(pf) if w is None else F.avg_pool3d(w[:, :1], 2)
    if sel is not None:
        ww = ww * sel.to(ww.dtype).view(-1, 1, 1, 1, 1)
    return ((pf - pc).abs() * ww).sum() / ww.sum().clamp_min(1e-6)


# ------------------------------------------------------------------------------ L8 skeleton recall

def erode(x, k=3):
    """One 26-connected erosion of a 0/1 field, with the OUTSIDE treated as background (explicit zero
    padding, not `max_pool3d`'s -inf, which would treat it as foreground and leave the band uneroded
    at the patch face)."""
    return -F.max_pool3d(F.pad(-x, (1,) * 6, value=0.0), k, stride=1)


def skeleton(tgt, iters=4, thr=0.5):
    """A one-voxel-thick medial surface of the binary target, on the GPU, in `iters + 1` pooling ops.

    `d = fg + sum_j erode^j(fg)` is the Chebyshev distance to background, capped at `iters` (the band
    is a few voxels thick, so a small cap is exact where it matters and saves the rest); the skeleton
    is the set of foreground voxels that are a local maximum of `d` in their 3x3x3 neighbourhood.
    For a band of thickness t <= 2*iters+1 that is exactly its medial surface, and it is a SURFACE,
    not a curve -- which is what we want: the loss must reward a continuous sheet, not a centreline.
    """
    fg = (tgt >= thr).to(tgt.dtype)
    d, e = fg.clone(), fg
    for _ in range(int(iters)):
        e = erode(e)
        d = d + e
    return fg * (d >= F.max_pool3d(d, 3, stride=1, padding=1) - 1e-6).to(d.dtype)


def skel_recall(p, tgt, wv=None, iters=4, thr=0.5):
    """1 - (mean predicted probability along the target's skeleton), averaged over the channels that
    have any skeleton weight in the batch.

    This is the Skeleton Recall Loss (Kirchhoff/Isensee, ECCV 2024) with the hard skeleton computed
    here rather than offline: it is `iters + 1` pooling ops on a tensor the step already holds, which
    is cheaper than carrying a skeleton channel through the loader and the augmentation. Because it
    is a RECALL of a fixed point set it cannot be gamed by making the band wider, and because the
    point set is the target's, not the prediction's, it costs no differentiable skeletonisation.

    It is a GAPS term only: a merge bridge is itself thin and connected and scores well here (the
    survey is explicit about this), so `merge_frac` must be watched while it is on.
    """
    s = skeleton(tgt, iters=iters, thr=thr)
    w = s if wv is None else s * wv
    d = (0, 2, 3, 4)
    num, den = (p * w).sum(d), w.sum(d)
    live = (den > 0).to(p.dtype)
    rec = num / den.clamp_min(1e-6)
    return 1.0 - (rec * live).sum() / live.sum().clamp_min(1.0)


# ------------------------------------------------------------------------ O12 long-range affinity

def parse_offsets(spec):
    """`--affinity 16,32` -> (16, 32). Offsets are EVEN and in voxels at the sample's own rung."""
    if spec is None or spec is False or (not isinstance(spec, (int, float)) and len(spec) == 0):
        return ()
    if isinstance(spec, (list, tuple)):
        vs = [int(v) for v in spec]
    else:
        vs = [int(v) for v in str(spec).replace(" ", ",").split(",") if v]
    assert vs and all(v > 0 and v % 2 == 0 for v in vs), f"--affinity {spec}: positive EVEN offsets"
    return tuple(vs)


def n_affinity(offsets):
    """Number of extra output channels: one per (axis, offset)."""
    return 3 * len(parse_offsets(offsets))


def affinity_names(offsets):
    return [f"aff{d}_{'zyx'[a]}" for d in parse_offsets(offsets) for a in range(3)]


def _shift(x, axis, n):
    """`x` moved by `n` voxels along `axis` (2,3,4), zero-filled where it comes from outside."""
    if n == 0:
        return x
    pad = [0] * 6
    i = 2 * (4 - axis)                       # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
    pad[i + (0 if n > 0 else 1)] = abs(n)
    y = F.pad(x, pad, value=0.0)
    return y.narrow(axis, 0 if n > 0 else abs(n), x.shape[axis])


def affinity_targets(tgt, wv=None, offsets=(16, 32), thr=0.5, fg_only=True):
    """(target, weight) of the affinity channels, both (B, 3*len(offsets), Z, Y, X).

    Channel `3*i + a` of offset `d = offsets[i]` and axis `a` asks, AT THE MIDPOINT, whether the two
    voxels d/2 back and d/2 forward along that axis lie on the SAME sheet. Centring on the midpoint is
    what makes the channel set equivariant under the 48 cube symmetries: a flip maps the pair to
    itself and a permutation only permutes the three axis channels.

    Same-sheet test: the straight segment between the two voxels is entirely foreground, i.e.
        target = min over t in [-d/2, d/2] of fg(v + t e_a) = a centred 1D erosion of length d+1.
    This is connected-component labelling restricted to straight paths inside the patch. It is
    CONSERVATIVE: a pair on one strongly curved sheet can read as "different", never the other way
    round, so the negatives -- the merge signal, "these two are the next wrap apart" -- are clean,
    which is the direction that matters. The alternative, a real per-sample connected-component
    labelling, is a CPU pass of seconds per 256^3 sample and, precomputed in the loader, would not
    survive `aug.spatial`'s rotations; this form is three pooling ops on the augmented target.

    Weight: `w(v - d/2) * w(v + d/2)`, zero where the pair leaves the patch (the shifts are
    zero-padded), and with `fg_only` (the default) zero unless BOTH voxels are foreground -- the
    same/different question is only meaningful for a pair of band voxels, and without the restriction
    the channel is >95 % trivial "one end is air" zeros.

    `tgt` / `wv` are the RECTO channel (channel 0): it is the one channel every sample carries.
    """
    offs = parse_offsets(offsets)
    fg = (tgt[:, :1] >= thr).to(tgt.dtype)
    w0 = torch.ones_like(fg) if wv is None else wv[:, :1]
    ts, ws = [], []
    for d in offs:
        h = d // 2
        for a in (2, 3, 4):
            seg = -F.max_pool3d(F.pad(-fg, _pad_axis(a, h), value=0.0),
                                _k_axis(a, d + 1), stride=1)
            lo, hi = _shift(fg, a, h), _shift(fg, a, -h)
            wl, wh = _shift(w0, a, h), _shift(w0, a, -h)
            ts.append(seg)
            ws.append(wl * wh * (lo * hi if fg_only else 1.0))
    return torch.cat(ts, 1), torch.cat(ws, 1)


def _pad_axis(axis, h):
    pad = [0] * 6
    i = 2 * (4 - axis)
    pad[i], pad[i + 1] = h, h
    return pad


def _k_axis(axis, n):
    return tuple(n if 2 + j == axis else 1 for j in range(3))


def affinity_loss(logit, tgt, wv=None, offsets=(16, 32), thr=0.5, fg_only=True):
    """Weighted BCE of the affinity channels. `logit` is the affinity SLICE of the head's output."""
    at, aw = affinity_targets(tgt, wv, offsets, thr=thr, fg_only=fg_only)
    b = F.binary_cross_entropy_with_logits(logit, at, reduction="none")
    return (b * aw).sum() / aw.sum().clamp_min(1e-6)


# --------------------------------------------------------------------------------- the dispatcher

KEYS = ("excl", "selfcons", "skel", "affinity")


def aux_losses(logit, tgt, wv, w_excl=0.0, w_selfcons=0.0, w_skel=0.0, w_affinity=0.0,
               cascade=None, cascade_self=None, offsets=(), skel_iters=4, aff_fg_only=True):
    """{name: unweighted loss} for every term whose weight is non-zero, plus `aux` = their weighted sum.

    `logit` is the FULL head output (target channels then affinity channels), `tgt` / `wv` the
    loader's target and weight (target channels only). `cascade` is the CASCADE input channel of this
    step and `cascade_self` (B,) the mask of samples that took the SELF source.
    """
    out, total, nt = {}, None, tgt.shape[1]
    p = None
    if w_excl > 0 and nt >= 2:
        p = torch.sigmoid(logit[:, :nt])
        out["excl"] = exclusivity(p, wv)
    if w_selfcons > 0 and cascade is not None:
        p = torch.sigmoid(logit[:, :nt]) if p is None else p
        out["selfcons"] = self_consistency(p, cascade, wv, cascade_self)
    if w_skel > 0:
        p = torch.sigmoid(logit[:, :nt]) if p is None else p
        out["skel"] = skel_recall(p, tgt, wv, iters=skel_iters)
    if w_affinity > 0 and offsets:
        out["affinity"] = affinity_loss(logit[:, nt:nt + n_affinity(offsets)], tgt, wv,
                                        offsets=offsets, fg_only=aff_fg_only)
    for k, v in (("excl", w_excl), ("selfcons", w_selfcons), ("skel", w_skel), ("affinity", w_affinity)):
        if k in out:
            total = v * out[k] if total is None else total + v * out[k]
    if total is not None:
        out["aux"] = total
    return out
