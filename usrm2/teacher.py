"""Run the upstream teacher (scrollprize/surface_recto_3dunet, villa 3D U-Net, 256^3 windows,
per-window z-score, 2-way softmax) over a box -> uint8 recto prob * 255 volcomp zarr."""
import numpy as np
import torch

from usrm2 import data
from usrm2.predict import out_array, slide

CKPT = "/vesuvius/tsm/models/surface_recto_3dunet.pth"


def load(ckpt=CKPT, dev="cuda"):
    from vesuvius.models.build.build_network_from_config import NetworkFromConfig
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = sd["model_config"]
    mgr = type("Mgr", (), dict(model_config=mc, targets=mc["targets"], in_channels=mc["in_channels"],
                               train_patch_size=mc["train_patch_size"], train_batch_size=mc["train_batch_size"],
                               model_name=mc["model_name"], autoconfigure=False, spacing=(1, 1, 1),
                               enable_deep_supervision=False))()
    net = NetworkFromConfig(mgr)
    net.load_state_dict(sd["model"], strict=True)
    return net.to(dev).eval()


def run(out, z0, y0, x0, Z, Y, X, volume=data.CT, window=256, halo=32, tile=1536, margin=128, ckpt=CKPT, device=None):
    """Tiles over y/x so RAM stays bounded. Each tile is read with a `margin` (>= half a window: the teacher
    is poor within ~32 voxels of a window edge, and the crop boundary must be covered by an interior window)."""
    dev = torch.device(device or "cuda")
    net = load(ckpt, dev)
    fn = lambda t: torch.softmax(net(t)["surface"].float(), 1)[0, 1]
    ct, arr = data.open_zarr(volume), out_array(out, (Z, Y, X), (z0, y0, x0))
    for y in range(0, Y, tile):
        for x in range(0, X, tile):
            ya, yb, xa, xb = max(y - margin, 0), min(y + tile + margin, Y), max(x - margin, 0), min(x + tile + margin, X)
            roi = ct[z0:z0 + Z, y0 + ya:y0 + yb, x0 + xa:x0 + xb]
            prob = slide(fn, roi, window, halo, dev) if roi.any() else np.zeros(roi.shape, np.float32)
            prob = prob[:, y - ya:y - ya + tile, x - xa:x - xa + tile]
            arr[:, y:y + prob.shape[1], x:x + prob.shape[2]] = np.clip(np.rint(prob * 255), 0, 255).astype(np.uint8)
            print(f"tile y={y} x={x} done", flush=True)
    return out
