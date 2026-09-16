import math

import pytest
import torch

from usrm2 import aug as A


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
