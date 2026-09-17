"""Tiny 3D U-Net: 4 input channels (z-scored CT + radial unit vector) -> one logit per teacher (head)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {"1m": (16, 32, 64, 128), "3m": (24, 48, 96, 192), "5m": (32, 64, 128, 256)}


def block(cin, cout):
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv3d(c, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU()]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, widths=PRESETS["1m"], cin=4, cout=1):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList([block(cin if i == 0 else w[i - 1], w[i]) for i in range(len(w))])
        self.down = nn.ModuleList([nn.Conv3d(c, c, 3, stride=2, padding=1) for c in w[:-1]])
        self.dec = nn.ModuleList([block(w[i] + w[i + 1], w[i]) for i in range(len(w) - 1)])
        self.head = nn.Conv3d(w[0], cout, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for i in range(len(self.dec) - 1, -1, -1):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


def build(size="1m", verbose=True, cout=1):
    m = UNet(PRESETS[size], cout=cout).to(memory_format=torch.channels_last_3d)
    n = sum(p.numel() for p in m.parameters())
    if verbose:
        print(f"usrm2 UNet {size} widths={PRESETS[size]} heads={cout} params={n / 1e6:.2f}M")
    return m
