"""The compact uint8 sample and the GPU-side `prepare`: it must build exactly what the old CPU path
(data.inputs + data.augment) built, for every one of the 48 cube symmetries and both normalizations."""
import numpy as np
import pytest
import torch

from usrm2 import data, prep

SHAPE = (6, 7, 8)  # deliberately not a cube: a permutation that swaps axes changes the shape
AX = np.array([[0.0, 64.0, 512.0], [11.3, 37.9, 22.1], [-5.5, 61.2, 40.0]])  # (3, N) control points
LO = np.array([3, 5, 7], np.int64)
K = 3


def sample(nctx=2, C=2, seed=0):
    rng = np.random.default_rng(seed)
    ct = rng.integers(0, 256, (1 + nctx,) + SHAPE, dtype=np.uint8)
    tg = rng.integers(0, 256, (C,) + SHAPE, dtype=np.uint8)
    w = rng.integers(0, 256, (C,) + SHAPE, dtype=np.uint8)
    return ct, tg, w


def cpu_path(ct, tg, w, sym):
    """What the worker used to build: z-scored cubes + scale plane + radial, then the cube symmetry."""
    rad = data.radial(data.axis_at(AX, K), LO, SHAPE)
    x = data.inputs(ct[0], rad, list(ct[1:]), rung=K)
    x, tw = data.sym_apply(sym, x, np.concatenate([tg, w]).astype(np.float32) / 255.0)
    return x, tw[:len(tg)], tw[len(tg):]


def gpu_path(ct, tg, w, sym):
    item = data.rung_item(ct, tg, w, K, LO, AX, sym)
    x, t, ww = prep.prepare(prep.batch1(item), torch.device("cpu"))
    return x[0].numpy(), t[0].numpy(), ww[0].numpy()


@pytest.fixture(autouse=True)
def no_norm(monkeypatch):
    monkeypatch.setattr(data, "NORM", None)


@pytest.mark.parametrize("sym", range(48))
def test_prepare_matches_the_cpu_path_for_every_cube_symmetry(sym):
    ct, tg, w = sample()
    xa, ta, wa = cpu_path(ct, tg, w, sym)
    xb, tb, wb = gpu_path(ct, tg, w, sym)
    assert xb.shape == xa.shape
    assert np.allclose(xb, xa, atol=1e-5), np.abs(xb - xa).max()
    assert np.allclose(tb, ta, atol=1e-6) and np.allclose(wb, wa, atol=1e-6)


@pytest.mark.parametrize("sym", [0, 9, 23, 47])
def test_prepare_matches_the_cpu_path_with_a_global_norm(monkeypatch, sym):
    monkeypatch.setattr(data, "NORM", (101.5, 37.25))
    ct, tg, w = sample(seed=1)
    xa = cpu_path(ct, tg, w, sym)[0]
    xb = gpu_path(ct, tg, w, sym)[0]
    assert np.allclose(xb, xa, atol=1e-5), np.abs(xb - xa).max()


def test_the_radial_channels_transform_as_a_vector():
    """The last three channels are a direction, not three more images: under a symmetry they are permuted
    AND negated. Comparing them against a permuted copy of the untransformed field would fail."""
    ct, tg, w = sample()
    x0 = gpu_path(ct, tg, w, 0)[0]
    for sym in range(48):
        perm, flip = data.sym_decode(sym)
        x = gpu_path(ct, tg, w, sym)[0]
        v = np.transpose(x0[-3:][perm], (0,) + tuple(perm + 1))
        v = v[(slice(None),) + tuple(slice(None, None, -1 if f else 1) for f in flip)]
        v = v * np.where(flip, -1.0, 1.0)[:, None, None, None]
        assert np.allclose(x[-3:], v, atol=1e-6)
        assert np.allclose(np.linalg.norm(x[-3:], axis=0), 1.0, atol=1e-3)  # still unit vectors


def test_radial_on_the_device_matches_data_radial():
    a = data.axis_at(AX, K)
    want = data.radial(a, LO, SHAPE)
    z = np.arange(SHAPE[0]) + LO[0]
    cyx = np.stack([np.interp(z, a[0], a[1]), np.interp(z, a[0], a[2])])
    got = prep.radial_t(torch.from_numpy(cyx)[None], torch.from_numpy(LO)[None], SHAPE).numpy()[0]
    assert np.allclose(got, want, atol=1e-5)


def test_the_zscore_is_per_cube_and_matches_data_zscore():
    ct, tg, w = sample(nctx=3, seed=2)
    x = gpu_path(ct, tg, w, 0)[0]
    for c in range(ct.shape[0]):
        assert np.allclose(x[c], data.zscore(ct[c]), atol=1e-5)
    assert np.allclose(x[ct.shape[0]], (K - 2) / 9.0)  # the scale plane sits before the radial vector


def test_collate_keeps_the_sample_compact():
    """The default collate is all the uint8 sample needs; the whole batch is one byte per channel voxel."""
    from torch.utils.data import default_collate
    cube, rng = (8, 8, 8), np.random.default_rng(3)  # a cube: every symmetry keeps the shape
    ct = rng.integers(0, 256, (10,) + cube, dtype=np.uint8)
    tg = rng.integers(0, 256, (1,) + cube, dtype=np.uint8)
    w = rng.integers(0, 256, (1,) + cube, dtype=np.uint8)
    syms = (0, 13)
    b = default_collate([data.rung_item(ct, tg, w, K, LO, AX, s) for s in syms])
    assert b["ct"].shape == (2, 10) + cube and b["ct"].dtype == torch.uint8
    assert b["tgt"].dtype == b["w"].dtype == torch.uint8
    assert b["lo"].shape == (2, 3) and b["cyx"].shape == (2, 2, cube[0])
    assert b["sym"].tolist() == list(syms) and b["rung"].tolist() == [K, K]
    nbytes = sum(v.numel() * v.element_size() for v in b.values())
    old = 2 * (14 + 2) * 4 * int(np.prod(cube))  # what (x, target, weight) used to cost as float32
    assert nbytes < old / 4
    x, t, ww = prep.prepare(b, torch.device("cpu"))
    assert x.shape == (2, 14) + cube and x.dtype == torch.float32
    assert t.shape == (2, 1) + cube and ww.shape == (2, 1) + cube
    for i, s in enumerate(syms):  # each sample keeps its own symmetry
        one = prep.prepare(prep.batch1(data.rung_item(ct, tg, w, K, LO, AX, s)), torch.device("cpu"))[0]
        assert np.allclose(x[i].numpy(), one[0].numpy(), atol=1e-6)


@pytest.mark.parametrize("weight", [0.0, 0.3, 0.5, 1.0])
def test_the_weight_quantisation_round_trips(weight):
    q = np.uint8(round(255 * weight))
    assert abs(float(q) / 255.0 - weight) <= 1.0 / 255
    t = prep.prepare(prep.batch1(data.rung_item(
        np.zeros((1,) + SHAPE, np.uint8), np.zeros((1,) + SHAPE, np.uint8),
        np.full((1,) + SHAPE, q, np.uint8), K, LO, AX, 0)), torch.device("cpu"))[2]
    assert abs(float(t.max()) - weight) <= 1.0 / 255


def test_norad_zeroes_the_radial_channels():
    ct, tg, w = sample()
    item = prep.batch1(data.rung_item(ct, tg, w, K, LO, AX, 5))
    x = prep.prepare(item, torch.device("cpu"), norad=True)[0]
    assert torch.count_nonzero(x[:, -3:]) == 0 and torch.count_nonzero(x[:, :-3]) > 0
