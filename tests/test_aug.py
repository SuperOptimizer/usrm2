import math

import numpy as np
import pytest
import torch

from usrm2 import aug as A, data as D


def batch(p=32, b=2):
    x = torch.randn(b, 4, p, p, p)
    v = torch.randn(b, 3, p, p, p)
    x[:, 1:] = v / v.norm(dim=1, keepdim=True)
    return x, torch.rand(b, 1, p, p, p)


@pytest.mark.parametrize("name", list(A.PRESETS))
def test_preset(name):
    x, t = batch()
    torch.manual_seed(0)
    y, u = A.apply(x.clone(), t.clone(), A.get(name))
    assert y.shape == x.shape and u.shape == t.shape
    assert y.dtype == torch.float32 and u.dtype == torch.float32
    assert torch.isfinite(y).all() and 0 <= u.min() and u.max() <= 1
    n = y[:, 1:].norm(dim=1)
    assert torch.allclose(n[n > 0.5], torch.ones_like(n[n > 0.5]), atol=1e-4)
    assert (name == "all_norad") == bool(n.max() < 0.5)


def test_bf16_input():
    x, t = batch(16)
    y, u = A.apply(x.bfloat16(), t.bfloat16(), A.get("all"))
    assert y.dtype == torch.float32 and torch.isfinite(y).all()


def test_rot90_rotates_the_vector_field():
    """A 90 deg rotation about z of a constant radial field is the matching permuted/negated constant,
    and the CT channel turns with it."""
    p = 16
    x, t = batch(p, 1)
    x[:, 1:] = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1, 1)  # (vz, vy, vx) = +y
    ramp = torch.linspace(-1, 1, p)
    x[:, 0] = ramp.view(1, p, 1)  # CT increases along +y
    M = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])[None]  # +90 deg about z, xyz order
    y, u = A.warp(x, t, M)
    c = p // 2
    v = y[0, 1:, c, c, c]
    assert torch.allclose(v, torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)  # +y -> +x
    assert float(y[0, 0, c, :, c].std()) < 1e-4  # ... and so does the CT ramp: flat along y now
    assert float(y[0, 0, c, c, 2]) < 0 < float(y[0, 0, c, c, p - 3])  # increasing along x
    assert math.isclose(float(y[0, 0, c, c, 2]), -float(y[0, 0, c, c, p - 3]), abs_tol=1e-3)


def test_tone_is_monotone():
    c = torch.linspace(-3, 3, 4096).view(1, 1, 16, 16, 16)
    torch.manual_seed(1)
    for _ in range(20):
        y = A._tone(c, A.TONE["tone"]).flatten()
        assert (y.diff() >= -1e-5).all() and math.isclose(float(y[0]), -3, abs_tol=1e-3)


def test_thick_kills_high_frequency_along_exactly_one_axis():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 32, 32, 32)

    def hf(t):
        return torch.stack([t.diff(dim=d).pow(2).mean() for d in (2, 3, 4)])

    r = (hf(A._thick(x, {"lo": 3.0, "hi": 4.0})) / hf(x)).sort().values
    # one axis loses almost all of it; the other two only what the interpolation blend costs
    assert r[0] < 0.2 and r[1] > 0.5 and r[1] > 5 * r[0]


def test_window_clips_on_the_raw_uint8():
    rng = np.random.default_rng(0)
    ct = rng.integers(0, 256, (16, 16, 16)).astype(np.uint8)
    out = D.raw(rng, ct, {"window": {"p": 1.0, "lo_lo": 40, "lo_hi": 40, "hi_lo": 200, "hi_hi": 200}})
    assert out.dtype == np.uint8 and out.shape == ct.shape
    assert (out[ct <= 40] == 0).all() and (out[ct >= 200] == 255).all()


def test_volcomp_roundtrip():
    try:
        import volcomp_zarr._lib  # noqa: F401  (needs a built libvolcomp.so or VOLCOMP_LIB)
    except Exception as e:
        pytest.skip(f"volcomp library unavailable: {e}")
    c = torch.from_numpy(np.random.default_rng(0).normal(112, 30, (8, 8, 8)).astype(np.float32))
    v = torch.nn.functional.interpolate(c[None, None], size=(128,) * 3, mode="trilinear",
                                        align_corners=False)[0, 0].numpy()
    v = np.clip(v, 0, 255).astype(np.uint8)  # structured data: the codec is not meant for noise
    out = D.volcomp_roundtrip(v, 8.0)
    assert out.dtype == np.uint8 and out.shape == v.shape
    assert 0 < np.abs(out.astype(np.float32) - v.astype(np.float32)).mean() < 20


def test_blank_yields_an_air_patch_with_target_zero(tmp_path, monkeypatch):
    from tests.test_smoke import make
    ct, (tr, va) = make(tmp_path)
    umb = tmp_path / "umb.json"
    umb.write_text('{"control_points": [{"z": 0, "y": 128, "x": 128}, {"z": 256, "y": 128, "x": 128}]}')
    monkeypatch.setattr(D, "UMBILICUS", str(umb))
    it = iter(D.Patches(patch=32, ct=ct, stores=[tr], exclude=va, sym=False, aug={"blank": {"p": 1.0}}))
    for _ in range(3):
        x, t = next(it)
        assert float(t.abs().max()) == 0 and float(x[0].abs().max()) == 0


def test_haze_only_touches_part_of_the_patch_and_flattens_it():
    torch.manual_seed(0)
    c = torch.randn(1, 1, 48, 48, 48)
    y = A._haze(c, A.HAZE["haze"])
    d = (y - c).abs().flatten()
    assert 0.02 < float((d > 1e-3).float().mean()) < 0.98  # some voxels hazed, some untouched
    hit = (d > 1e-2)
    assert float(y.flatten()[hit].std()) < float(c.flatten()[hit].std())  # local contrast shrank


def test_unsharp_is_nabus_form():
    c = torch.randn(1, 1, 16, 16, 16)
    k = dict(A.UNSHARP["unsharp"], a_lo=1.0, a_hi=1.0, s_lo=1.0, s_hi=1.0)
    y = A._unsharp(c, k)
    assert torch.allclose(y, 2 * c - A._blur1(c, 1.0), atol=1e-5)  # (1+a) I - a Gauss(I, s), a = 1


def test_quant_clips_and_rounds_to_256_levels():
    torch.manual_seed(0)
    c = torch.randn(2, 1, 24, 24, 24)
    y = A._quant(c, A.QUANT["quant"])
    assert y.min() >= c.min() - 1e-5 and y.max() <= c.max() + 1e-5
    for b in range(2):
        u = y[b].flatten()
        assert len(torch.unique(u)) <= 256 and (u.max() - u.min()) > 0


def test_cor_ghost_is_radial_and_zero_at_the_centre():
    p = 32
    x, t = batch(p, 1)
    x[:, 1:] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1, 1)  # radial = +x everywhere
    x[:, 0] = torch.linspace(-1, 1, p).view(1, 1, p)  # a ramp along x
    y = A._cor(x, dict(A.COR["cor"], lo=1.5, hi=1.5, a_lo=0.15, a_hi=0.15), torch.ones(1, 1, 1, 1, 1))
    d = (y[:, 0] - x[:, 0]).abs()
    c = p // 2
    assert float(d[0, c, c, c]) < 1e-3 < float(d[0, c, c, 2])  # zero on the axis, ghosting off it
    assert torch.equal(y[:, 1:], x[:, 1:])


@pytest.mark.parametrize("op", A.POOL_OPS)
def test_pool_ops_coarsen_and_keep_shape(op):
    torch.manual_seed(0)
    x = torch.randn(2, 1, 24, 24, 24)
    k = {**A.POOL["pool"], "ops": [op], "k_lo": 3, "k_hi": 3, "aniso": 0.0, "nearest": 1.0}
    y = A._pool(x, k)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (y[:, :, :3, :3, :3] == y[:, :, :1, :1, :1]).all()  # nearest-up: one value per 3^3 block
    if op == "max":
        assert (y[:, :, :3, :3, :3] >= x[:, :, :3, :3, :3]).all()
    if op == "min":
        assert (y[:, :, :3, :3, :3] <= x[:, :, :3, :3, :3]).all()
    if op == "avg":
        assert torch.allclose(y[:, 0, 0, 0, 0], x[:, 0, :3, :3, :3].mean((1, 2, 3)), atol=1e-5)
    if op == "median":
        assert torch.allclose(y[:, 0, 0, 0, 0], x[:, 0, :3, :3, :3].flatten(1).median(1).values)
    if op == "stride":
        assert (y[:, 0, 0, 0, 0] == x[:, 0, 0, 0, 0]).all()
