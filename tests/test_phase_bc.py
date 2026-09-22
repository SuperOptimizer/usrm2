"""Phase B / C: distance targets, the regression heads, construction-based pairing, the topology pilot,
the metadata/radius planes and the tracer export (docs/unified_design.md section 29).

CPU and synthetic throughout. The load-bearing claims:
  * a distance target is computed per rung from THAT rung's mask and is never a pool of a finer one;
  * the encoding round-trips, and code 0 means no data everywhere;
  * growing the head leaves the recto/verso rows BIT-IDENTICAL after a warm start;
  * `--pair construct` cannot produce an overlapping pair, for any (m, t >= t_min);
  * every flag is off by default, so a run without them is what it always was.
"""
import json
import os

import numpy as np
import pytest
import torch

from usrm2 import data, losses as L, model as M, predict as P, prep, targets as TG, train as T
from tests.test_rungs import P32, ct_pyramid, plain_level, pred_pyramid, umbilicus


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)


# --------------------------------------------------------------------------------- the encoding

def test_the_signed_encoding_round_trips_and_code_0_is_no_data():
    d = np.array([-40.0, -31.75, -8.0, -0.25, 0.0, 0.25, 8.0, 31.75, 40.0], np.float32)
    u = TG.encode_signed(d, np.ones(d.shape, bool))
    assert u.dtype == np.uint8 and u.min() >= 1               # 0 is reserved
    back = TG.decode_signed(u)
    assert np.allclose(back, np.clip(d, -31.75, 31.75), atol=TG.UNIT / 2)
    assert TG.encode_signed(np.zeros(3), np.zeros(3, bool)).tolist() == [0, 0, 0]
    assert int(TG.encode_signed(np.zeros(1), np.ones(1, bool))[0]) == 128   # offset 128
    # the same constants on the torch side, which is what the loss decodes with
    t = torch.from_numpy(u.astype(np.float32) / 255.0)
    assert torch.allclose(L.decode_signed(t), torch.from_numpy(back), atol=1e-4)
    assert (L.UNIT, L.OFF, L.CAP) == (TG.UNIT, TG.OFF, TG.CAP)


def test_the_unsigned_thickness_encoding_round_trips():
    t = np.array([0.0, 3.0, 20.0, 63.75, 200.0], np.float32)
    u = TG.encode_unsigned(t, np.ones(t.shape, bool))
    assert u.min() >= 1
    assert np.allclose(TG.decode_unsigned(u), np.clip(t, TG.UNIT, 255 * TG.UNIT), atol=TG.UNIT / 2)
    assert torch.allclose(L.decode_unsigned(torch.from_numpy(u.astype(np.float32) / 255.0)),
                          torch.from_numpy(TG.decode_unsigned(u)), atol=1e-3)


# ------------------------------------------------------------------- the target on a synthetic slab

def slab(n=64, z0=28, z1=32, axis_y=-1000.0):
    """A band of `z1 - z0` voxels perpendicular to y... actually to z is wrong for a scroll: the sheets
    are radial, so the band is a plane of constant Y and the axis sits far away at small Y, which makes
    +Y the OUTWARD (recto) direction."""
    v = np.zeros((n, n, n), np.uint8)
    v[:, z0:z1] = 255
    return v


def offsets(n=64, cy=-1000.0):
    """(dy, dx) of a box whose axis is at y = cy, x = n/2: the radial direction is +y almost exactly."""
    dy = (np.arange(n, dtype=np.float32)[None, :, None] - cy) * np.ones((n, 1, n), np.float32)
    dx = (np.arange(n, dtype=np.float32)[None, None, :] - n / 2) * np.ones((n, n, 1), np.float32)
    return dy, dx


def test_signed_distance_on_a_slab_has_the_right_sign_units_and_clamp():
    v = slab()
    dy, dx = offsets()
    d, m, t, okt = TG.block_fields(v, None, dy, dx)
    assert not okt.any()                              # no verso source: the thickness is not measurable
    mid = 29.5                                        # the medial surface of z0=28..z1=32 along y
    y = np.arange(64, dtype=np.float32)
    want = np.clip(y - mid, -TG.CAP, TG.CAP)
    got = d[32, :, 32]
    assert np.allclose(got, want, atol=0.75)
    assert got[63] == pytest.approx(TG.CAP)           # clamped, not unbounded
    assert (got[:28] < 0).all() and (got[33:] > 0).all()   # verso side negative, recto side positive


def test_midline_and_thickness_on_a_two_face_slab():
    """recto band at y 40-42, verso band at y 20-22: the midline is at 31 and the thickness 20."""
    n = 64
    rec, ver = np.zeros((n, n, n), np.uint8), np.zeros((n, n, n), np.uint8)
    rec[:, 40:43] = 255
    ver[:, 20:23] = 255
    dy, dx = offsets(n)
    d, m, t, okt = TG.block_fields(rec, ver, dy, dx)
    assert okt.all()
    assert m[32, 31, 32] == pytest.approx(0.0, abs=0.75)          # the midline of the two faces
    assert t[32, 31, 32] == pytest.approx(20.0, abs=1.0)          # 41 - 21
    assert (t >= TG.TMIN).all()                                   # lower-bounded, everywhere
    # the identity the parameterisation rests on: recto face at m = +t/2, verso at m = -t/2
    assert (m - 0.5 * t)[32, 41, 32] == pytest.approx(0.0, abs=0.75)
    assert (m + 0.5 * t)[32, 21, 32] == pytest.approx(0.0, abs=0.75)


def dist_pyr(tmp_path, name="m.zarr", base=128, nlev=2):
    """A mask pyramid whose band is one slab per level, so each rung has a real 0.5 level set."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    lv = []
    for l in range(nlev):
        n, um = base >> l, data.rung_um(2 + l)
        lv.append(f"{um:g}")
        v = np.zeros((n, n, n), np.uint8)
        v[:, n // 2 - 2:n // 2 + 2] = 255
        a = plain_level(root / lv[-1], (n, n, n), 0)
        a[:] = v
    (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{"version": "0.5", "name": name, "type": "mean",
                "axes": [{"name": q, "type": "space", "unit": "micrometer"} for q in "zyx"],
                "datasets": [{"path": q, "coordinateTransformations":
                              [{"type": "scale", "scale": [float(q)] * 3}]} for q in lv]}]},
        "volcomp": {"rung_voxel_size_um": 2.4}}}))
    return str(root)


def test_dist_pyramid_recomputes_per_rung_instead_of_pooling(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mask = dist_pyr(tmp_path)
    out = TG.dist_pyramid(mask, kinds=("face",), rungs=(2, 3), umbilicus=umbilicus(tmp_path),
                          block=64, halo=48, axis_r_um=0.0, log=lambda *a, **k: None)["face"]
    pyr = data.rungs(out)
    assert sorted(pyr) == [2, 3]
    v2 = data.read_rung(pyr, 2, (32, 0, 32), (1, 128, 1), dtype=np.uint8).reshape(-1)
    v3 = data.read_rung(pyr, 3, (16, 0, 16), (1, 64, 1), dtype=np.uint8).reshape(-1)
    d2, d3 = TG.decode_signed(v2), TG.decode_signed(v3)
    # each rung is the distance in ITS OWN voxels, so the slope is 1 per voxel at BOTH rungs. A pooled
    # rung-2 field would have slope 2 per rung-3 voxel (and would still be in rung-2 units).
    ok2, ok3 = v2 != 0, v3 != 0
    s2 = np.diff(d2[ok2])[np.abs(np.diff(d2[ok2])) > 0]
    s3 = np.diff(d3[ok3])[np.abs(np.diff(d3[ok3])) > 0]
    assert np.median(np.abs(s2)) == pytest.approx(1.0, abs=0.05)
    assert np.median(np.abs(s3)) == pytest.approx(1.0, abs=0.05)
    at = data.group_attrs(out)
    assert at["channel"] == "sdist" and at["no_data"] == 0 and "recto_positive" in at["sign_convention"]


def test_dist_pyramid_marks_no_data_above_the_cap_and_near_the_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))  # the axis is at y = x = 128
    mask = dist_pyr(tmp_path, base=128, nlev=1)
    out = TG.dist_pyramid(mask, kinds=("face",), rungs=(2,), umbilicus=umbilicus(tmp_path),
                          block=64, halo=48, axis_r_um=1000.0, log=lambda *a, **k: None)["face"]
    v = data.read_rung(data.rungs(out), 2, (0, 0, 0), (128, 128, 128), dtype=np.uint8)
    # the far corner of this 128^3 box is 181 voxels from the axis at (y,x)=(128,128); 1000 um / 2.4 um
    # = 417 voxels, so the whole box is inside the exclusion radius and is written as no-data
    assert (v == 0).all()
    out2 = TG.dist_pyramid(mask, kinds=("face",), rungs=(2,), umbilicus=umbilicus(tmp_path),
                           block=64, halo=48, axis_r_um=0.0, log=lambda *a, **k: None)["face"]
    assert (data.read_rung(data.rungs(out2), 2, (0, 0, 0), (128, 128, 128), dtype=np.uint8) != 0).any()
    # nothing is written above the cap: rung 4 is above the mask's native + 1 AND above max_rung
    o3 = TG.dist_pyramid(mask, kinds=("face",), rungs=(2, 3, 4, 5), umbilicus=umbilicus(tmp_path),
                         block=64, halo=48, axis_r_um=0.0, log=lambda *a, **k: None)["face"]
    assert sorted(data.rungs(o3)) == [2]   # the mask pyramid only HAS rung 2 here


def test_a_distance_channel_has_weight_zero_at_a_rung_its_store_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    sd = pred_pyramid(tmp_path, name="sd.zarr", base=256, nlev=1, value=lambda k: 200,
                      attrs={"channel": "sdist"})   # ONE level, rung 2
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg},{sd}"], exclude=[], rungs={2, 3}, sym=False,
                      air_keep=1.0, fg_keep=1.0)
    ds._open_rungs()
    assert ds.channels == ["recto", "sdist"] and ds.nrecto == 1
    s = ds.srcs[0]
    for k, want in ((2, True), (3, False)):
        ctp = data.read_rung(s["ct_pyr"], k, (0, 0, 0), P32, dtype=np.uint8)
        t, w = ds._rung_target(s, k, np.array([0, 0, 0]), ctp)
        assert (w[1] > 0).any() == want, f"rung {k}: a distance must not be pooled"
    # code 0 in the store is no data, whatever the box says
    sd0 = pred_pyramid(tmp_path, name="sd0.zarr", base=256, nlev=1, value=lambda k: 0,
                       attrs={"channel": "sdist"})
    ds2 = data.Patches(patch=P32, stores=[f"{ct},{tg},{sd0}"], exclude=[], rungs={2}, sym=False)
    ds2._open_rungs()
    ctp = data.read_rung(ds2.srcs[0]["ct_pyr"], 2, (0, 0, 0), P32, dtype=np.uint8)
    assert not (ds2._rung_target(ds2.srcs[0], 2, np.array([0, 0, 0]), ctp)[1][1] > 0).any()


# ---------------------------------------------------------------------------- the regression losses

def true_distance_field(n=24):
    """A field that IS a distance: d(v) = y - c, so |grad d| == 1 everywhere."""
    y = torch.arange(n, dtype=torch.float32) - (n - 1) / 2
    return y.view(1, 1, 1, n, 1).expand(1, 1, n, n, n).contiguous()


def test_eikonal_is_zero_for_a_true_distance_field_and_positive_otherwise():
    d = true_distance_field()
    w = torch.ones_like(d)
    assert float(L.eikonal(d, w, band=1e9)) == pytest.approx(0.0, abs=1e-6)
    assert float(L.eikonal(0.5 * d, w, band=1e9)) == pytest.approx(0.25, abs=1e-4)
    assert float(L.eikonal(torch.zeros_like(d), w, band=1e9)) == pytest.approx(1.0, abs=1e-6)


def test_derived_normals_are_unit_and_point_from_verso_to_recto():
    d = true_distance_field()
    n = L.normals_from(d)
    inner = n[:, :, 2:-2, 2:-2, 2:-2]
    assert torch.allclose(inner.norm(dim=1), torch.ones(1), atol=1e-4)
    # d grows along +y, which is the RECTO (radially outward) side, so the normal is +y: verso -> recto
    assert float(inner[0, 1].mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(inner[0, 0].abs().max()) == pytest.approx(0.0, abs=1e-4)


def test_the_sdist_huber_is_zero_at_the_target_and_the_hetero_form_is_finite():
    n = 8
    tgt = torch.full((1, 1, n, n, n), 160 / 255.0)      # code 160 -> (160-128)*0.25 = +8 voxels
    w = torch.ones_like(tgt)
    pred = torch.full_like(tgt, 8.0)
    assert float(L.sdist_loss(pred, tgt, w)) == pytest.approx(0.0, abs=1e-6)
    assert float(L.sdist_loss(torch.zeros_like(pred), tgt, w)) > 1.0
    lv = torch.zeros_like(pred)
    assert np.isfinite(float(L.sdist_loss(pred, tgt, w, logvar=lv)))
    # a weight below WMIN is DROPPED, not down-weighted: an interpolated distance is a wrong distance
    assert float(L.dist_weight(torch.tensor([0.0, 0.5, 0.94, 1.0])).sum()) == pytest.approx(1.0)


def test_thickness_never_falls_below_tmin():
    raw = torch.linspace(-50, 50, 101)
    assert float(L.soft_thickness(raw).min()) >= L.TMIN
    assert float(L.soft_thickness(torch.tensor(0.0))) == pytest.approx(L.TMIN + np.log(2), abs=1e-5)


# --------------------------------------------------------------------- construction-based pairing

@pytest.mark.parametrize("half,tau", [(1.5, 0.5), (3.0, 1.0), (1.0, 0.25)])
def test_construct_pairing_can_never_overlap(half, tau):
    m = torch.linspace(-60, 60, 601).view(1, 1, 1, 1, -1)
    for raw in (-1e3, -1.0, 0.0, 5.0, 1e3):
        t = L.soft_thickness(torch.full_like(m, float(raw)), tmin=2 * half)
        pr, pv = L.pair_bands(m, t, half=half, tau=tau)
        assert float((pr + pv - 1).clamp_min(0).max()) == pytest.approx(0.0, abs=1e-7)
        assert float(L.exclusivity(torch.cat([pr, pv], 1), torch.ones(1, 2, 1, 1, 601))) \
            == pytest.approx(0.0, abs=1e-7)
    # ... and the bands really do sit at m = +- t/2
    t = torch.full_like(m, 20.0)
    pr, pv = L.pair_bands(m, t, half=half, tau=tau)
    assert float(m.reshape(-1)[pr.reshape(-1).argmax()]) == pytest.approx(10.0, abs=0.3)
    assert float(m.reshape(-1)[pv.reshape(-1).argmax()]) == pytest.approx(-10.0, abs=0.3)
    assert torch.allclose(torch.sigmoid(torch.cat(L.pair_logits(m, t, half, tau), 1)),
                          torch.cat([pr, pv], 1), atol=1e-6)


# ------------------------------------------------------------------------------- the ECT pilot

def test_ect_is_finite_zero_for_identical_inputs_and_positive_otherwise():
    p = torch.zeros(1, 1, 48, 48, 48)
    p[:, :, 12:36, 20:24, 12:36] = 1.0
    kw = dict(dirs=4, res=8, margin=8, block=16, nblocks=2)
    assert float(L.ect_loss(p, p, **kw)) == pytest.approx(0.0, abs=1e-9)
    q = p.clone()
    q[:, :, 20:28, 20:24, 20:28] = 0.0              # punch a hole: the Euler characteristic moves
    v = float(L.ect_loss(q, p, **kw))
    assert np.isfinite(v) and v > 0
    # the Euler characteristic of a solid block is 1, whatever the direction
    b = torch.ones(1, 1, 8, 8, 8)
    assert float(L.ect(b, L.fib_dirs(3), 6)[0, 0, -1]) == pytest.approx(1.0, abs=1e-5)
    # a patch too small for one interior sub-block gives exactly 0, not an error
    assert float(L.ect_loss(torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8), **kw)) == 0.0


def test_ect_is_differentiable():
    lg = torch.zeros(1, 1, 40, 40, 40, requires_grad=True)
    t = torch.zeros(1, 1, 40, 40, 40)
    t[:, :, 10:30, 18:22, 10:30] = 1.0
    L.ect_loss(torch.sigmoid(lg), t, dirs=3, res=6, margin=8, block=16, nblocks=1).backward()
    assert torch.isfinite(lg.grad).all() and float(lg.grad.abs().sum()) > 0


# ----------------------------------------------------------------------------------- the planes

def test_scan_planes_are_normalised_and_zero_when_the_metadata_is_missing():
    from usrm2 import scanmeta as SM
    assert np.allclose(data.scan_planes(SM.load(None)), 0.0)     # missing -> all zero, never the defaults
    assert np.allclose(data.scan_planes(None), 0.0)
    m = SM.flatten(json.load(open("docs/example_metadata_paris4_2.4um_78keV.json")))
    v = data.scan_planes(m)
    assert v.shape == (5,) and ((v >= 0) & (v <= 1)).all()
    assert v[0] == pytest.approx((78.0 - 30.0) / 90.0, abs=1e-5)                    # 78 keV
    assert v[4] == pytest.approx(np.log2(2.4 / 0.6) / np.log2(1228.8 / 0.6), abs=1e-5)   # rung 2 of 11
    assert v[1] > 0 and v[2] > 0 and v[3] > 0
    # a field the file does not supply stays 0 even though `flatten` fills a documented default
    m2 = dict(m, defaulted=["energy_kev"])
    assert data.scan_planes(m2)[0] == 0.0 and data.scan_planes(m2)[4] == v[4]


def test_the_planes_are_built_in_the_canonical_order_and_land_between_cascade_and_scale(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "NORM", (0.0, 1.0))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    assert data.parse_planes("meta,radius") == data.parse_planes("radius,meta") == ("radius", "meta")
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2}, sym=False,
                      air_keep=1.0, fg_keep=1.0, planes="meta,radius")
    item = next(iter(ds))
    assert "rmax" in item and "meta" in item
    x = prep.prepare(prep.batch1(item), torch.device("cpu"))[0][0]
    assert x.shape[0] == 1 + 6 + 1 + 3                       # CT + 6 planes + scale + radial
    assert prep.shapes(item)[0] == x.shape[0]
    r = x[1]
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0   # the radius plane, normalised
    assert float(r.std()) > 0                                # ... and a FIELD, not a constant
    for j in range(2, 7):                                    # the five scan planes are constant
        assert float(x[j].std()) == pytest.approx(0.0, abs=1e-7)
    assert np.allclose(x[2:7, 0, 0, 0].numpy(), item["meta"].numpy())
    assert torch.allclose(x[7], torch.zeros(1))              # the scale plane, rung 2
    assert torch.allclose(x[8:].norm(dim=0), torch.ones(1), atol=1e-5)   # radial last, still unit
    # this CT has no metadata.json beside it, so every scan plane is zero
    assert float(x[2:7].abs().max()) == 0.0


def test_warm_start_keeps_the_scale_plane_aligned_across_a_plane_growth():
    old = M.build("1m", verbose=False, cin=15, cout=2)       # [CT, 9 ctx, CASCADE, scale, radial(3)]
    w0 = old.state_dict()["enc.0.0.weight"]
    src = T.warm_start(old.state_dict(), cin=21, cout=2, cascade=True, src_scale=True,
                       src_cascade=True, planes=6, src_planes=0)
    w1 = src["enc.0.0.weight"]
    assert w1.shape[1] == 21
    assert torch.allclose(w1[:, :11], w0[:, :11])            # CT + 9 context + the cascade channel
    assert torch.allclose(w1[:, 11:17], torch.zeros(1))      # the six NEW planes start at zero
    assert torch.allclose(w1[:, 17], w0[:, 11])              # the scale plane stayed the scale plane
    assert torch.allclose(w1[:, 18:], w0[:, 12:])            # radial last
    # the step-0 output is the source's: the new planes contribute nothing whatever they hold
    net = M.build("1m", verbose=False, cin=21, cout=2)
    net.load_state_dict(src)
    net.eval(), old.eval()
    x = torch.randn(1, 15, 16, 16, 16)
    xx = torch.cat([x[:, :11], torch.randn(1, 6, 16, 16, 16), x[:, 11:]], 1)
    with torch.no_grad():
        a, b = old(x), net(xx)
    assert float((a - b).abs().max() / a.abs().max()) < 1e-5


# ------------------------------------------------------------------------------- the head growth

def test_growing_the_head_leaves_the_probability_rows_bit_identical():
    old = M.build("1m", verbose=False, cin=15, cout=2, deep=1)
    st = old.state_dict()
    # cout 2 -> 2 probabilities + sdist + thickness + 3 normals + logvar
    src = T.warm_start(st, cin=15, cout=8, ncopy=2)
    for k in ("head.weight", "deep_heads.0.weight"):
        assert torch.equal(src[k][:2], st[k][:2])            # BIT-identical, not close
        assert torch.equal(src[k][2:], torch.zeros_like(src[k][2:]))
        b = k[:-6] + "bias"
        assert torch.equal(src[b][:2], st[b][:2]) and torch.equal(src[b][2:], torch.zeros(6))
    net = M.build("1m", verbose=False, cin=15, cout=8, deep=1)
    net.load_state_dict(src)
    net.eval(), old.eval()
    x = torch.randn(1, 15, 16, 16, 16)
    with torch.no_grad():
        a, b = old(x), net(x)
    # the head is a 1x1x1 convolution, so the probability rows are the SAME arithmetic on the same
    # inputs; only the kernel's output-channel blocking differs, which the doc measures at ~1e-7
    assert float((a - b[:, :2]).abs().max() / a.abs().max()) < 1e-6
    assert float(b[:, 2:].abs().max()) == 0.0                # a zero row says distance 0 / p = 0.5


# ----------------------------------------------------------------------------- the tracer export

def test_the_tracer_encodings_round_trip():
    d = np.linspace(-40, 40, 81).astype(np.float32)
    u = P.enc_signed(d, np.ones(d.shape, bool))
    assert u.min() >= 1
    assert np.allclose((u.astype(np.float32) - 128) * 0.25, np.clip(d, -31.75, 31.75), atol=0.125)
    n = np.linspace(-1, 1, 41).astype(np.float32)
    v = P.enc_normal(n, np.ones(n.shape, bool))
    assert v.min() >= 1 and np.allclose(P.dec_normal(v), n, atol=1.0 / 127)
    assert int(P.enc_normal(np.zeros(1), np.ones(1, bool))[0]) == 128
    assert P.enc_signed(np.zeros(2), np.zeros(2, bool)).tolist() == [0, 0]
    assert P.enc_normal(np.zeros(2), np.zeros(2, bool)).tolist() == [0, 0]


def test_tracer_fields_subtract_the_half_thickness_and_derive_an_outward_normal():
    n = 32
    m = (np.arange(n, dtype=np.float32)[None, :, None] - 16) * np.ones((n, 1, n), np.float32)
    th = np.full((n, n, n), 10.0, np.float32)
    d, nrm, mag, ok = P.tracer_fields(m, th)
    assert d[16, 21, 16] == pytest.approx(0.0, abs=1e-4)          # the recto face is at m = +t/2 = 5
    inner = nrm[:, 4:-4, 4:-4, 4:-4]
    assert np.allclose(np.linalg.norm(inner, axis=0), 1.0, atol=1e-3)
    assert inner[1].mean() == pytest.approx(1.0, abs=1e-3)        # +y: verso -> recto
    assert np.allclose(mag[4:-4, 4:-4, 4:-4], 1.0, atol=1e-3)     # the Eikonal ideal
    assert P.tracer_fields(m)[0][16, 16, 16] == pytest.approx(0.0, abs=1e-4)   # no thickness: as is


def test_marching_cubes_writes_one_obj_per_shard(tmp_path):
    pytest.importorskip("skimage", reason="--marching-cubes needs scikit-image (the `dev` extra)")
    n = 48
    d = (np.arange(n, dtype=np.float32)[None, :, None] - 23.5) * np.ones((n, 1, n), np.float32)
    out = P.mesh_shards(d, np.ones(d.shape, bool), (100, 200, 300), str(tmp_path / "mesh"),
                        shard=32, log=lambda *a, **k: None)
    objs = sorted(os.listdir(out))
    assert objs and all(q.endswith(".obj") for q in objs)
    v = [q.split()[1:] for q in open(os.path.join(out, objs[0])).read().splitlines() if q.startswith("v ")]
    assert v
    ys = np.array([float(q[1]) for q in v])
    assert np.allclose(ys, 200 + 23.5, atol=1.0)   # the zero level, in GLOBAL ZYX voxels


# --------------------------------------------------------------------------- end to end, all flags on

def sdist_pyr(tmp_path, name, channel, base=256, value=200):
    return pred_pyramid(tmp_path, name=name, base=base, nlev=1, value=lambda k: value,
                        attrs={"channel": channel})


def test_a_cpu_training_step_with_every_phase_bc_flag_on(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    md = sdist_pyr(tmp_path, "mid.zarr", "midline", value=140)
    th = sdist_pyr(tmp_path, "th.zarr", "thickness", value=60)
    lines = [f"{ct},{tg},{md},{th}"]
    out = tmp_path / "run"
    ckpt = T.train(out, size="1m", steps=20, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=20, val_patches=1, device="cpu", ctx=(1,), rungs={2}, val_rungs=(2,),
                   stores=lines, val=((0, 0, 0), (32, 32, 32)),
                   verso=True, teacher_regions=str(tmp_path / "tr"), cout=None,
                   sdist="midline", thickness=True, normals="head", sdist_hetero=True,
                   loss_sdist=1.0, loss_eikonal=0.1, loss_normals=0.1,
                   pair="construct", loss_excl=0.1,
                   loss_ect=0.01, ect_dirs=3, ect_res=6, ect_margin=4, ect_block=8, ect_n=1, ect_rung=2,
                   planes="meta,radius")
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = st["args"]
    assert a["channels"][:4] == ["recto", "verso", "midline", "thickness"]
    assert a["channels"][4:] == ["nz", "ny", "nx", "logvar"]
    assert a["cout"] == 8 and a["cout_t"] == 4 and a["cout_p"] == 2
    assert a["cin"] == 1 + 1 + 6 + 1 + 3          # CT + 1 context + 6 planes + scale + radial
    assert a["sdist"] == "midline" and a["pair"] == "construct" and a["planes"] == ["radius", "meta"]
    rows = [json.loads(q) for q in (out / "train.jsonl").read_text().splitlines() if "loss" in q]
    assert rows, "no training step was logged"
    for k in ("sdist", "eikonal", "thick", "nrm", "pair_bce", "pair_dice", "ect"):
        assert k in rows[-1] and np.isfinite(rows[-1][k]), k
    ev = json.loads((out / "eval.jsonl").read_text().splitlines()[-1])
    assert "mae_midline" in ev and "mae_thickness" in ev and np.isfinite(ev["dice"])
    # and the fields come back out of `predict`
    for hd in ("sdist", "thickness", "conf", "normals"):
        v = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu",
                    rung=2, head=hd)[0]
        assert np.isfinite(v).all()
        assert v.shape == ((3,) + (P32,) * 3 if hd == "normals" else (P32,) * 3)
    tr = P.export_tracer(ckpt, ct, 0, 0, 0, P32, P32, P32, str(tmp_path / "tracer"), window=P32,
                         halo=8, device="cpu", rung=2, volcomp=False, log=lambda *a, **k: None)
    assert {"recto", "verso", "surf_sdist", "nz", "ny", "nx", "gmag", "conf", "thickness"} <= set(tr)
    import zarr
    z = zarr.open(tr["surf_sdist"], mode="r")
    assert z.attrs["axis_order"] == "ZYX" and z.attrs["no_data"] == 0
    assert "VERSO face towards the RECTO" in z.attrs["sign_convention"]


def test_the_flags_off_leave_the_checkpoint_args_untouched(tmp_path, monkeypatch):
    """The Phase B/C contract: a run without the flags records nothing new and builds the stem it always did."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    out = tmp_path / "run"
    ckpt = T.train(out, size="1m", steps=1, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=1, val_patches=1, device="cpu", ctx=(1,), rungs={2}, val_rungs=(2,),
                   stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)))
    a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    for k in ("sdist", "thickness", "normals", "sdist_hetero", "pair", "loss_ect", "planes",
              "loss_eikonal", "cout_t", "cout_p"):
        assert k not in a, f"{k} leaked into the args of a run that did not ask for it"
    assert a["cin"] == 1 + 1 + 1 + 3 and a["cout"] == 1
