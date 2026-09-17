"""Second teacher: scrollprize/surface_m7_nnunet (Kaggle surface-detection winner; stock nnU-Net ResEncL,
192^3 windows, CT normalization clip 0..212 then (x-87.5)/47.7, 2-way softmax, fg = "surface"). It works at
~8 um, so we run it on level 2 of the volume (9.6 um) and upsample its probability x4 to level 0, giving a
thin, well-localized band in the same store format as the 2.4 um teacher."""
import importlib
import os

import numpy as np
import torch
import torch.nn.functional as F

from usrm2 import data
from usrm2.predict import out_array, put, slide
from usrm2.teacher import flips

CKPT = os.environ.get("USRM2_M7_CKPT", "/vesuvius/tsm/models/surface_m7_nnunet.pth")
LEVEL = 2


def load(ckpt=CKPT, dev="cuda"):
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = dict(sd["init_args"]["plans"]["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"])
    for k in ("conv_op", "norm_op", "dropout_op", "nonlin"):  # nnU-Net stores class paths as strings
        a[k] = a[k] and getattr(importlib.import_module(a[k].rsplit(".", 1)[0]), a[k].rsplit(".", 1)[1])
    net = ResidualEncoderUNet(input_channels=1, num_classes=2, deep_supervision=False, **a)
    net.load_state_dict(sd["network_weights"], strict=True)
    p = sd["init_args"]["plans"]["foreground_intensity_properties_per_channel"]["0"]
    net.norm = (p["mean"], p["std"], p["percentile_00_5"], p["percentile_99_5"])
    return net.to(dev).eval()


def run(out, z0, y0, x0, Z, Y, X, volume=data.CT, window=192, halo=32, margin=64, ckpt=CKPT, device=None, tta=0, level=LEVEL,
        backend="torch"):
    """margin: level-`level` voxels of CT context read around the box (boxes are thin at 1/4 pitch)."""
    dev = torch.device(device or "cuda")
    net = load(ckpt, dev if backend == "torch" else "cpu")
    mean, std, lo, hi = net.norm
    prep = lambda c, _: ((np.clip(c.astype(np.float32), lo, hi) - mean) / std)[None]
    if backend == "trt":
        from usrm2 import trt
        net = trt.Engine(trt.plan("m7", window), dev)
    fn = lambda t: torch.softmax(net(t).float(), 1)[:, 1]
    if tta > 1:
        fn = flips(fn, tta)
    f = 1 << level
    assert all(v % f == 0 for v in (z0, y0, x0, Z, Y, X)), f"box must be a multiple of {f}"
    ct = data.open_zarr(volume.rstrip("/").rsplit("/", 1)[0] + f"/{level}")
    o, s = np.array([z0, y0, x0]) // f, np.array([Z, Y, X]) // f
    a, b = np.maximum(o - margin, 0), np.minimum(o + s + margin, ct.shape)
    roi = ct[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    r = np.pad(roi, [(0, max(window - n, 0)) for n in roi.shape])  # still thinner than a window: pad with air
    c = o - a
    prob = slide(fn, r, window, halo, dev, prep)[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]]
    up = F.interpolate(torch.from_numpy(prob)[None, None], size=(Z, Y, X), mode="trilinear", align_corners=False)[0, 0].numpy()
    air = np.repeat(np.repeat(np.repeat(roi[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] == 0, f, 0), f, 1), f, 2)
    arr = out_array(out, (Z, Y, X), (z0, y0, x0), volume=volume)
    put(arr, np.where(air, 0, np.clip(np.rint(up * 255), 0, 255)).astype(np.uint8))  # coarse air mask; the loader masks with the fine CT
    arr.attrs.update({"model": "m7", "level": level, "tta": tta, "done": True})
    return out
