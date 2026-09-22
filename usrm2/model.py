"""Tiny 3D U-Net: 4 input channels (z-scored CT + radial unit vector) -> one logit per teacher (head)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {"1m": (16, 32, 64, 128), "3m": (24, 48, 96, 192), "5m": (32, 64, 128, 256),
           # deeper nets for the 80 GB card and 256^3+ patches: one more level doubles the receptive field
           "12m": (32, 64, 128, 256, 384), "26m": (32, 64, 128, 256, 512), "45m": (48, 96, 192, 384, 640),
           "30m6": (32, 64, 128, 256, 384, 384),  # 6 levels: ~500-voxel theoretical receptive field
           # THE SIZE LADDER (experiment 12, docs/unified_design.md section 30). The same six levels as
           # `30m6`, the same depth, every width scaled by 1/sqrt(2) and by sqrt(2) and rounded to a
           # multiple of 8 (GroupNorm takes min(8, c) groups, and a width off the 8-grid costs tensor-core
           # alignment). Because a convolution's parameter count is quadratic in width, that is a clean
           # FACTOR-2 ladder in parameters -- 22.67 M / 44.92 M / 89.95 M at the canonical cin=4, cout=1,
           # deep=0 count -- which is what a log-log fit of val loss against log(params) needs: three
           # points evenly spaced in log(params). The names follow `30m6`'s own loose convention (it is
           # 44.9 M, not 30 M); `usrm2 ladder --print` reports the real counts for the run's own cin/cout.
           "15m": (24, 48, 88, 184, 272, 272),
           "60m": (48, 88, 184, 360, 544, 544),
           # narrow full-resolution level for 512^3 patches: the level-0 tensors (and the level-0 decoder cat)
           # are what does not fit in 80 GB; the depth and width live in the coarse levels
           "n16": (16, 32, 64, 128, 256, 512), "n24": (24, 48, 96, 192, 384, 512)}


def up2x(x, size):
    """Trilinear upsample of (B,C,Z,Y,X) to `size`, taking a fast path at an exact factor of 2.

    `F.interpolate(..., mode="trilinear", align_corners=False)` at 2x is, per axis,
        out[2m] = 0.75 in[m] + 0.25 in[m-1],  out[2m+1] = 0.75 in[m] + 0.25 in[m+1]
    with the out-of-range neighbour clamped to the edge. Written that way -- two weighted sums and an
    interleave per axis -- every operation is a GATHER, so the backward is a gather too. aten's
    upsample_trilinear3d_backward is a scatter-add (`_unsafe_index_put`, atomics) and at the decoder's
    widest stage (2 x 64 x 256^3, 2e9 elements) it costs about 1.1 s per call on an A100; this form,
    compiled, costs about 0.1 s (measured, cloud/a100_up.py). The values agree with `F.interpolate` to
    2e-7 relative in float32; under bf16 autocast the per-axis rounding differs by up to one bf16 ulp
    (9e-3 relative) because aten sums all eight corners in float before rounding once.

    Any other ratio (an odd patch size, where a stride-2 encoder level rounds up) falls back to
    `F.interpolate`, so the semantics are unchanged everywhere.
    """
    if tuple(int(v) for v in size) != tuple(2 * int(v) for v in x.shape[2:]):
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    for d in range(3):
        a, n = 2 + d, x.shape[2 + d]
        prev = torch.cat([x.narrow(a, 0, 1), x.narrow(a, 0, n - 1)], a)
        nxt = torch.cat([x.narrow(a, 1, n - 1), x.narrow(a, n - 1, 1)], a)
        x = torch.stack([0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt], a + 1).flatten(a, a + 1)
    return x


def block(cin, cout):
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv3d(c, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, widths=PRESETS["1m"], cin=4, cout=1, ckpt_act=0, add_skip=0, deep=0):
        """ckpt_act: recompute the activations of the blocks at the first `ckpt_act` levels (the full-resolution
        ones hold most of the memory) in the backward pass (torch.utils.checkpoint); True/-1 = every level.
        Trades compute for a much smaller activation footprint (large patches on one card).
        add_skip: at the first `add_skip` levels the decoder ADDS the skip to a 1x1 projection of the upsampled
        tensor instead of concatenating: the widest full-resolution tensor is then w0 channels, not w0 + w1,
        which is what keeps big patches under the ~2e9-element kernel cliff (352^3 at width 32).
        deep: also predict the cout maps at decoder levels 1..deep (2x, 4x, 8x coarser: 4.8/9.6/19.2 um for a
        2.4 um patch); forward returns [logits_level0, logits_level1, ...] while training, level 0 otherwise."""
        super().__init__()
        self.deep = min(int(deep), len(widths) - 2)  # a coarse head per decoder stage above the finest; a 4-level net has 2
        self.ckpt_act = len(widths) if ckpt_act is True or ckpt_act < 0 else int(ckpt_act)
        self.add_skip = int(add_skip)
        w = list(widths)
        self.enc = nn.ModuleList([block(cin if i == 0 else w[i - 1], w[i]) for i in range(len(w))])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 3, stride=2, padding=1) for c in w[:-1]])
        self.dec = nn.ModuleList([block(w[i] if i < self.add_skip else w[i] + w[i + 1], w[i]) for i in range(len(w) - 1)])
        self.proj = nn.ModuleList([nn.Conv3d(w[i + 1], w[i], 1) if i < self.add_skip else nn.Identity() for i in range(len(w) - 1)])
        self.head = nn.Conv3d(w[0], cout, 1)
        self.deep_heads = nn.ModuleList([nn.Conv3d(w[i], cout, 1) for i in range(1, self.deep + 1)])

    def _run(self, m, x, level):
        rg = any(t.requires_grad for t in (x if isinstance(x, tuple) else (x,)))
        if level < self.ckpt_act and self.training and rg:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(m, x, use_reentrant=False)
        return m(x)

    def forward(self, x):
        skips = []
        if self.ckpt_act and self.training and not x.requires_grad:
            x = x.requires_grad_()  # checkpointed blocks need a grad path through their input
        for i, e in enumerate(self.enc):
            x = self._run(e, x, i)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        outs = {}
        for i in range(len(self.dec) - 1, -1, -1):
            x = self._run(self._stage(i), (x, skips[i]), i)  # upsample + concat inside the checkpointed segment:
            if 1 <= i <= self.deep:                          # the (w_i + w_i+1)-channel concat is never stored
                outs[i] = self.deep_heads[i - 1](x)
        y = self.head(x)
        if self.deep and self.training:
            return [y] + [outs[i] for i in range(1, self.deep + 1)]
        return y

    def _stage(self, i):
        dec, proj, add = self.dec[i], self.proj[i], i < self.add_skip

        def f(pair):
            x, skip = pair
            x = up2x(x, skip.shape[2:])
            return dec(proj(x) + skip) if add else dec(torch.cat([x, skip], 1))
        return f


# Memory format of the weights and of the input. channels_last_3d gives the convolutions cudnn's NDHWC
# tensor-core kernels, but at 256^3 this net is bound by the normalisations and activations, and those
# are far slower in that layout: measured on an A100 (2 x 32 x 256^3, bf16, torch.compile), GroupNorm+SiLU
# forward+backward is 96 ms in channels_last_3d and 16 ms contiguous, and the whole 30m6 step is 1152 ms
# channels_last vs 679 ms contiguous (cloud/a100_gn.py). So the net runs in the plain NCDHW layout.
CHANNELS_LAST = False


def memfmt():
    return torch.channels_last_3d if CHANNELS_LAST else torch.contiguous_format


def params(size="1m", cin=4, cout=1, add_skip=0, deep=0):
    """Parameter count of a preset WITHOUT allocating it: the net is built on the `meta` device, so this
    costs microseconds and no memory. The size ladder fits val loss against log(params), and the count
    depends on `cin`, `cout`, `--add-skip` and `--deep`, so it is always taken from the run's own args
    rather than from the preset name."""
    with torch.device("meta"):
        m = UNet(PRESETS[size], cin=int(cin), cout=int(cout), add_skip=int(add_skip), deep=int(deep))
    return int(sum(p.numel() for p in m.parameters()))


def build(size="1m", verbose=True, cout=1, cin=4, ckpt_act=0, add_skip=0, deep=0):
    m = UNet(PRESETS[size], cin=cin, cout=cout, ckpt_act=ckpt_act, add_skip=add_skip, deep=deep).to(memory_format=memfmt())
    n = sum(p.numel() for p in m.parameters())
    if verbose:
        print(f"usrm2 UNet {size} widths={PRESETS[size]} in={cin} heads={cout} params={n / 1e6:.2f}M")
    return m
