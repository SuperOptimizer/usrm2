"""Tiny 3D U-Net: 4 input channels (z-scored CT + radial unit vector) -> one logit per teacher (head)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {"1m": (16, 32, 64, 128), "3m": (24, 48, 96, 192), "5m": (32, 64, 128, 256),
           # deeper nets for the 80 GB card and 256^3+ patches: one more level doubles the receptive field
           "12m": (32, 64, 128, 256, 384), "26m": (32, 64, 128, 256, 512), "45m": (48, 96, 192, 384, 640),
           "30m6": (32, 64, 128, 256, 384, 384),  # 6 levels: ~500-voxel theoretical receptive field
           # narrow full-resolution level for 512^3 patches: the level-0 tensors (and the level-0 decoder cat)
           # are what does not fit in 80 GB; the depth and width live in the coarse levels
           "n16": (16, 32, 64, 128, 256, 512), "n24": (24, 48, 96, 192, 384, 512)}


def block(cin, cout):
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv3d(c, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, widths=PRESETS["1m"], cin=4, cout=1, ckpt_act=0):
        """ckpt_act: recompute the activations of the blocks at the first `ckpt_act` levels (the full-resolution
        ones hold most of the memory) in the backward pass (torch.utils.checkpoint); True/-1 = every level.
        Trades compute for a much smaller activation footprint (large patches on one card)."""
        super().__init__()
        self.ckpt_act = len(widths) if ckpt_act is True or ckpt_act < 0 else int(ckpt_act)
        w = list(widths)
        self.enc = nn.ModuleList([block(cin if i == 0 else w[i - 1], w[i]) for i in range(len(w))])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 3, stride=2, padding=1) for c in w[:-1]])
        self.dec = nn.ModuleList([block(w[i] + w[i + 1], w[i]) for i in range(len(w) - 1)])
        self.head = nn.Conv3d(w[0], cout, 1)

    def _run(self, m, x, level):
        if level < self.ckpt_act and self.training and x.requires_grad:
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
        for i in range(len(self.dec) - 1, -1, -1):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            x = self._run(self.dec[i], torch.cat([x, skips[i]], 1), i)
        return self.head(x)


def build(size="1m", verbose=True, cout=1, cin=4, ckpt_act=0):
    m = UNet(PRESETS[size], cin=cin, cout=cout, ckpt_act=ckpt_act).to(memory_format=torch.channels_last_3d)
    n = sum(p.numel() for p in m.parameters())
    if verbose:
        print(f"usrm2 UNet {size} widths={PRESETS[size]} in={cin} heads={cout} params={n / 1e6:.2f}M")
    return m
