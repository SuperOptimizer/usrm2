import numpy as np
import torch

from usrm2 import verso
from usrm2.train import weighted


def sheet(Z=48, Y=48, X=64, lo=20, hi=36):
    """A flat sheet spanning x in [lo, hi) (recto face at lo, verso at hi), radial = +x."""
    ct = torch.zeros(Z, Y, X)
    ct[:, :, lo:hi] = 150.0
    rad = torch.zeros(3, Z, Y, X)
    rad[2] = 1.0
    band = torch.zeros(Z, Y, X)
    band[:, :, lo:lo + 3] = 1.0  # the teacher's recto band, 3 voxels thick at the inner face
    blob = torch.zeros(Z, Y, X)
    blob[:, :, lo + 8:hi] = 1.0  # the flipped student fills the outer half of the sheet
    return ct, rad, band, blob


def test_skin_is_the_outer_face():
    ct, rad, band, blob = sheet()
    s = verso.skin(blob, rad)
    xs = torch.nonzero(s[24, 24])[:, 0]
    assert 33 <= xs.min() and xs.max() <= 36, xs  # the outer face of the blob, ~3 thick
    assert not s[24, 24, :30].any()


def test_ct_edge_is_where_the_papyrus_ends():
    ct, rad, band, blob = sheet()
    e = verso.ct_edge(band, ct, rad)
    xs = torch.nonzero(e[24, 24])[:, 0]
    assert 32 <= xs.min() and xs.max() <= 37, xs
    assert not e[24, 24, :28].any()


def test_targets_anchor_the_skin():
    ct, rad, band, blob = sheet()
    t = verso.targets(blob, band, ct, rad)
    line = t[24, 24].numpy()
    assert (line[33:36] == 255).all(), line
    assert (line[:30] == 0).all()
    # no CT edge (papyrus continues): the skin is kept but weak
    ct2 = ct.clone()
    ct2[:, :, 36:] = 150.0
    t2 = verso.targets(blob, band, ct2, rad)
    line2 = t2[24, 24].numpy()
    assert (line2[33:36] == verso.WEAK).all(), line2
    assert (t2[24, 24, :30] == 0).all()


def test_weighted_targets():
    tgt = torch.zeros(1, 2, 2, 2, 2)
    tgt[0, 1, 0, 0, 0] = 1.0
    tgt[0, 1, 1, 1, 1] = verso.WEAK / 255
    t, w = weighted(tgt, (1,))
    assert t[0, 1, 1, 1, 1] == 1.0 and t[0, 1, 0, 0, 0] == 1.0
    assert abs(w[0, 1, 1, 1, 1] - verso.WEAK / 255) < 1e-6 and w[0, 1, 0, 0, 0] == 1.0 and w[0, 0].min() == 1.0
    assert weighted(tgt, ())[1] is None


def test_verso_path():
    assert verso.verso_path("/t/boxes7/box_1_2_3.zarr") == "/t/boxes7_v/box_1_2_3.zarr"
    assert verso.verso_path("/t/boxes7_m7/box_1_2_3.zarr") == "/t/boxes7_m7_v/box_1_2_3.zarr"
    assert verso.verso_path("/t/eval.zarr") == "/t/eval_v.zarr"
    assert verso.verso_path("/t/boxes7/box_1.zarr", "raw") == "/t/boxes7_vraw/box_1.zarr"
    assert verso.verso_path("/t/eval.zarr", "raw") == "/t/eval_vraw.zarr"
