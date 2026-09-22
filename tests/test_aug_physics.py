"""Physics augmentation v2: the Paganin jitter, micron-based sigmas, the shuffled artefact order and
the scan-metadata loader (docs/unified_design.md section 27)."""
import json
import math

import pytest
import torch

from usrm2 import aug as A, scanmeta as S

META = "docs/example_metadata_paris4_2.4um_78keV.json"


def _cube(b=2, p=32, seed=0):
    """A papyrus-ish cube: smooth sheets plus noise, in the tone range a z-scored CT has."""
    torch.manual_seed(seed)
    z = torch.linspace(0, 6 * math.pi, p).view(1, 1, p, 1, 1)
    c = z.sin().expand(b, 1, p, p, p).clone() + 0.15 * torch.randn(b, 1, p, p, p)
    return c


def _own(**over):
    """The paganin cfg with the sampled range collapsed onto the scan's own parameters."""
    k = dict(A.PAGANIN["paganin"])
    k.update(db_lo=k["db"], db_hi=k["db"], a_lo=k["a"], a_hi=k["a"], s_lo=k["s_um"], s_hi=k["s_um"])
    k.update(over)
    return k


def test_paganin_is_identity_at_the_scans_own_parameters():
    c = _cube()
    y = A._paganin_jitter(c, _own())
    assert (y - c).abs().max() < 1 / 255  # an 8-bit level: the identity is inside the sampled range


@pytest.mark.parametrize("db", [250.0, 500.0, 2000.0])
def test_paganin_moves_the_cube_and_stays_in_the_tone_range(db):
    c = _cube()
    y = A._paganin_jitter(c, _own(db_lo=db, db_hi=db))
    lo, hi = c.amin((2, 3, 4), True), c.amax((2, 3, 4), True)
    m = 0.25 * (hi - lo)  # the `keep` guard
    assert torch.isfinite(y).all() and (y >= lo - m - 1e-4).all() and (y <= hi + m + 1e-4).all()
    assert (y - c).abs().max() > 1 / 255
    # a smaller delta/beta is the sharper reconstruction, a larger one the smoother one
    hf = lambda v: (v - A._blur1(v, 2.0)).std()  # noqa: E731
    assert (hf(y) > hf(c)) == (db < A.PAGANIN["paganin"]["db"])


def test_paganin_is_scale_aware():
    """The same physical delta/beta is a different filter in VOXELS at a different pitch: one rung
    coarser, the voxel-space cutoff halves, so the cube changes less."""
    c = _cube()
    k = _own(db_lo=250.0, db_hi=250.0)
    d2 = (A._paganin_jitter(c, {**k, "vox_um": A.rung_um(2)}) - c).abs().mean()
    d3 = (A._paganin_jitter(c, {**k, "vox_um": A.rung_um(3)}) - c).abs().mean()
    assert d3 < d2


def test_for_rung_halves_the_voxel_sigmas_one_rung_coarser():
    cfg = A.get("full2")
    assert A.rung_um(2) == 2.4 and A.rung_um(3) == 4.8
    two, three = A.for_rung(cfg, 2), A.for_rung(cfg, 3)
    for name, keys in A.SIGMA_KEYS.items():
        for q in keys:
            if q in cfg.get(name, {}):
                assert two[name][q] == pytest.approx(cfg[name][q])          # rung 2 = today, exactly
                assert three[name][q] == pytest.approx(cfg[name][q] / 2)    # sigma_um / voxel_um
    assert three["paganin"]["vox_um"] == 4.8
    # not converted: detector-pixel widths, geometric smoothing, paganin's own micron sigmas
    for name, q in (("ring", "w_lo"), ("stripe", "w_lo"), ("sheetcomp", "smooth"), ("elastic", "sigma"),
                    ("paganin", "s_lo")):
        assert three[name][q] == cfg[name][q]


def test_for_rung_takes_a_per_sample_rung():
    cfg = A.for_rung(A.get("full2"), [2, 3])
    assert cfg["paganin"]["vox_um"] == [2.4, 4.8]
    assert cfg["blur"]["hi"] == pytest.approx(A.PRESETS["full2"]["blur"]["hi"] / 2)  # the median rung


def test_rung_2_is_exactly_todays_behaviour():
    """`full` at rung 2 is bit-identical to `full` with no rung at all."""
    x = torch.randn(2, 4, 32, 32, 32)
    v = torch.randn(2, 3, 32, 32, 32)
    x[:, 1:] = v / v.norm(dim=1, keepdim=True)
    t = torch.rand(2, 1, 32, 32, 32)
    torch.manual_seed(3)
    a, _ = A.apply(x.clone(), t.clone(), A.get("full"))
    torch.manual_seed(3)
    b, _ = A.apply(x.clone(), t.clone(), A.get("full"), rung=2)
    assert torch.equal(a, b)


def test_full_is_unchanged_and_full2_is_full_plus_the_new_ops():
    full, full2 = A.PRESETS["full"], A.PRESETS["full2"]
    assert "paganin" not in full and "shuffle" not in full
    assert set(full2) - set(full) == {"paganin", "shuffle"}
    assert all(full2[k] == full[k] for k in full)


def test_shuffle_covers_every_order_of_every_op(monkeypatch):
    """With `shuffle`, every op is applied and reaches every position of the pipeline; without it the
    order is the fixed INTENS one, every time."""
    ops = ["gamma", "bright", "contrast", "noise"]
    cfg = {"shuffle": True, **{o: dict(A.INTENSITY[o], p=1.0) for o in ops}}
    log = []
    monkeypatch.setattr(A, "INTENS", [(n, (lambda c, k, n=n: (log.append(n), c)[1])) for n, _ in A.INTENS])
    c = torch.randn(1, 1, 8, 8, 8)
    seen = {o: set() for o in ops}
    torch.manual_seed(0)
    for _ in range(40):
        log.clear()
        A.intensity(c, cfg)
        assert sorted(log) == sorted(ops)  # p=1: every op fires exactly once, whatever the order
        for i, n in enumerate(log):
            seen[n].add(i)
    assert all(v == set(range(len(ops))) for v in seen.values())
    log.clear()
    A.intensity(c, {**cfg, "shuffle": False})
    assert log == [n for n, _ in A.INTENS if n in cfg]  # unshuffled: the fixed pipeline order
    # p=0 everywhere is still the exact identity, shuffled or not
    log.clear()
    assert torch.equal(A.intensity(c, {**{o: dict(cfg[o], p=0.0) for o in ops}, "shuffle": True}), c)
    assert log == []


def test_shuffle_applies_a_different_order_to_different_samples():
    """Two samples, one op each of a non-commuting pair (bright then gamma vs gamma then bright):
    over many draws the shuffled pipeline produces both compositions."""
    cfg = {"shuffle": True, "gamma": {"p": 1.0, "max": 1.8}, "bright": {"p": 1.0, "max": 0.3}}
    outs = set()
    for s in range(30):
        torch.manual_seed(s)
        c = torch.rand(1, 1, 8, 8, 8)
        torch.manual_seed(100)  # same parameter draws, so only the ORDER can differ
        y = A.intensity(c, cfg)
        outs.add(round(float(y.mean()), 6))
    assert len(outs) > 1


def test_scanmeta_parses_the_example():
    m = S.load(META)
    assert not m["missing"] and m["defaulted"] == ["mosaic", "mosaic_tiles"]
    assert (m["energy_kev"], m["pixel_um"], m["distance_mm"]) == (78.0, 2.4, 220.0)
    assert (m["delta_beta"], m["unsharp_coeff"], m["unsharp_sigma_px"]) == (1000.0, 4.0, 1.2)
    assert m["unsharp_sigma_um"] == pytest.approx(2.88) and m["rung"] == 2
    assert m["phase_method"] == "Paganin" and m["helical"] is True
    assert (m["win_f32_lo"], m["win_f32_hi"]) == (-0.04, 0.22)
    assert m["hist_p998"] == pytest.approx(0.13065, abs=1e-4)
    assert set(S.DEFAULTS) <= set(m)


def test_scanmeta_tolerates_a_missing_file(tmp_path):
    for p in (None, tmp_path / "nope.zarr", tmp_path / "nope.json", tmp_path):
        m = S.load(p)
        assert m["missing"] and sorted(m["defaulted"]) == sorted(S.DEFAULTS)
        assert m["energy_kev"] == S.DEFAULTS["energy_kev"] and m["rung"] == 2


def test_scanmeta_tolerates_a_broken_file(tmp_path):
    (tmp_path / "metadata.json").write_text("{not json")
    assert S.load(tmp_path)["missing"]
    (tmp_path / "metadata.json").write_text(json.dumps({"scan": {"tomo": {"acquisition": {"energy": 137.0}}}}))
    m = S.load(tmp_path)
    assert not m["missing"] and m["energy_kev"] == 137.0 and "delta_beta" in m["defaulted"]


def test_ranges_for_centres_the_jitter_on_the_scan():
    """The 1.1 um mosaic's parameters: the scan's own values become the filter's reference AND stay
    inside the sampled range, so the identity is always reachable."""
    m = dict(S.DEFAULTS, pixel_um=1.1, delta_beta=500.0, unsharp_coeff=4.0, unsharp_sigma_px=2.5,
             unsharp_sigma_um=2.75, energy_kev=137.0)
    r = S.ranges_for(m)["paganin"]
    assert r["db"] == 500.0 and r["s_um"] == pytest.approx(2.75)
    assert r["db_lo"] <= r["db"] <= r["db_hi"] and r["s_lo"] <= r["s_um"] <= r["s_hi"]
    assert r["a_lo"] <= r["a"] <= r["a_hi"]
    assert S.ranges_for(m)["bias"]["max"] < S.ranges_for(dict(S.DEFAULTS))["bias"]["max"]  # 137 keV cups less


def test_get_with_meta_only_touches_ops_the_preset_has():
    m = S.load(META)
    assert "paganin" not in A.get("full", meta=m)          # `full` does not grow the new op
    assert A.get("full", meta=m)["bias"]["max"] == A.PRESETS["full"]["bias"]["max"]  # 78 keV = the calibration
    g = A.get("full2", meta=m)
    assert g["paganin"]["db"] == 1000.0 and g["paganin"]["p"] == A.PAGANIN["paganin"]["p"]
    assert A.PRESETS["full2"]["paganin"]["db"] == 1000.0   # the preset itself is not mutated
    assert A.get("full2", meta=m, rung=3)["paganin"]["vox_um"] == 4.8


def test_full2_runs_end_to_end_on_a_rung_batch():
    B, ni = 2, 10
    x = torch.randn(B, ni + 4, 32, 32, 32)
    x[:, ni] = 0.0
    r = torch.randn(B, 3, 32, 32, 32)
    x[:, ni + 1:] = r / r.norm(dim=1, keepdim=True)
    tg = torch.rand(B, 2, 32, 32, 32)
    torch.manual_seed(0)
    y, t = A.apply(x, tg, A.get("full2", meta=S.load(META)), rung=[2, 3])
    assert y.shape == x.shape and torch.isfinite(y).all() and 0 <= t.min() and t.max() <= 1
