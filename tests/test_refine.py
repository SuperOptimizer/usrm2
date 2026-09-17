import json

import numpy as np
import pytest

from usrm2 import refine as R


def band_volume(shape, y_center, width=2.0):
    """A soft band at y = y_center(x) inside a (Z,Y,X) volume."""
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(np.float32)
    return np.exp(-0.5 * ((y - y_center(x)) / width) ** 2).astype(np.float32)


def test_refine_moves_a_shifted_surface_onto_the_band():
    shape = (24, 64, 64)
    yc = lambda x: 30 + 4 * np.sin(x / 10.0)
    V = band_volume(shape, yc)
    # published surface: the band shifted by +3 voxels in y, with a bump of noise
    xs, zs = np.arange(4, 60, dtype=np.float32), np.arange(2, 22, dtype=np.float32)
    Z, X = np.meshgrid(zs, xs, indexing="ij")
    g = np.stack([Z, np.zeros_like(Z), X], -1)  # (H,W,3) zyx
    g[..., 1] = yc(g[..., 2]) + 3.0
    g[5:8, 10:14, 1] += 4.0
    ax = np.array([[0.0, 100.0], [-1000.0, -1000.0], [32.0, 32.0]])  # axis far below: outward = +y
    g1, stats = R.refine(g, V, (0, 0, 0), ax, far=8, sigma=1.5, iters=3, thr=0.3)
    err0 = np.abs(g[..., 1] - yc(g[..., 2])).mean()
    err1 = np.abs(g1[..., 1] - yc(g1[..., 2])).mean()
    assert err0 > 2.5 and err1 < 0.8, (err0, err1)
    assert np.isfinite(g1).all() and stats[-1]["with_peak"] > 0.9


def test_refine_leaves_holes_and_outside_points_alone():
    shape = (16, 32, 32)
    V = band_volume(shape, lambda x: 16 + 0 * x)
    g = np.zeros((6, 6, 3), np.float32)
    g[..., 0], g[..., 2] = np.arange(6)[:, None] * 2 + 2, np.arange(6)[None] * 4 + 4
    g[..., 1] = 19.0
    g[2, 2] = np.nan
    g[0, 0] = (8, 19, 200)  # outside the store
    g1, _ = R.refine(g, V, (0, 0, 0), np.array([[0.0, 100.0], [-1000.0, -1000.0], [16.0, 16.0]]), far=6, sigma=1.0, iters=2, thr=0.3)
    assert np.isnan(g1[2, 2]).all() and np.allclose(g1[0, 0], g[0, 0])
    inside = np.isfinite(g1).all(-1) & (g1[..., 2] < 100)
    assert np.abs(g1[inside][:, 1] - 16).mean() < 1.0


def test_joint_refinement_keeps_two_close_sheets_apart():
    """Two sheets 6 voxels apart, both published 4 voxels too high: alone, the upper one would jump onto the
    lower band; jointly, each stays on its own band."""
    shape = (16, 64, 32)
    V = np.maximum(band_volume(shape, lambda x: 30 + 0 * x, 1.2), band_volume(shape, lambda x: 36 + 0 * x, 1.2))
    def sheet(y):
        Z, X = np.meshgrid(np.arange(2, 14, dtype=np.float32), np.arange(2, 30, dtype=np.float32), indexing="ij")
        return np.stack([Z, np.full_like(Z, y), X], -1)
    ax = np.array([[0.0, 100.0], [-1000.0, -1000.0], [16.0, 16.0]])
    lower, upper = sheet(34.0), sheet(40.0)  # true 30 and 36, both +4
    (l1, u1), st = R.refine_many([lower, upper], V, (0, 0, 0), ax, far=8, sigma=1.0, iters=4, thr=0.3)
    assert abs(l1[..., 1].mean() - 30) < 1.0 and abs(u1[..., 1].mean() - 36) < 1.0, (l1[..., 1].mean(), u1[..., 1].mean())
    assert st[0]["capped"] > 0.5  # the sheets saw each other on the ray


def test_assign_prefers_one_peak_per_sheet_in_order():
    pos = np.array([[-4.0], [2.0], [0.0], [0.0], [0.0], [0.0]], np.float32)   # peaks at -4 and +2
    stren = np.array([[1.0], [1.0], [-1], [-1], [-1], [-1]], np.float32)
    # alone: the nearest peak (+2) wins
    assert abs(R.assign(pos, stren, np.array([np.nan]), np.array([np.nan]))[0] - 2.0) < 1e-4
    # with another sheet 6 above: that sheet takes +2, this one gets -4
    assert abs(R.assign(pos, stren, np.array([np.nan]), np.array([6.0]))[0] + 4.0) < 1e-4


def test_upsample_densifies_and_keeps_holes():
    Z, X = np.meshgrid(np.arange(0, 100, 20, dtype=np.float32), np.arange(0, 100, 20, dtype=np.float32), indexing="ij")
    g = np.stack([Z, 30 + 0.1 * X, X], -1)
    g[2, 2] = np.nan
    u = R.upsample(g, 5)
    assert u.shape == (21, 21, 3)
    ok = np.isfinite(u).all(-1)
    v = u[ok][:, 0] / 4
    assert np.abs(v - np.round(v)).max() < 1e-3 and np.abs(np.diff(u[0, :, 2])).mean() == pytest.approx(4.0, abs=1e-3)
    assert not ok[10, 10] and ok[0, 0] and ok[20, 20]
