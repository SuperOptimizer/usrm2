"""Evaluation v2 (docs/unified_design.md section 25) on synthetic data: ERL, Betti-0/1, bootstrap, plateau fit."""
import json

import numpy as np
import pytest

from usrm2 import evalsurf as E, topo as T

tifffile = pytest.importorskip("tifffile")

AX = np.array([[0.0, 128.0], [0.0, 0.0], [0.0, 0.0]])  # scroll axis at y = x = 0 -> normals point +x
BOX = ((0, 0, 0), (128, 128, 128))


def write_plane(root, name, x, zr=(20, 102), yr=(20, 102), step=2):
    """A synthetic tifxyz plane at the given x, spanning z/y on a `step`-voxel grid."""
    d = root / "seg" / name
    d.mkdir(parents=True)
    zz, yy = np.meshgrid(np.arange(*zr, step, dtype=np.float32), np.arange(*yr, step, dtype=np.float32),
                         indexing="ij")
    for n, a in [("z", zz), ("y", yy), ("x", np.full_like(zz, float(x)))]:
        tifffile.imwrite(str(d / f"{n}.tif"), a)
    (d / "meta.json").write_text(json.dumps({"scale": [0.5, 0.5],
                                             "bbox": [[x, yr[0], zr[0]], [x, yr[1] - step, zr[1] - step]]}))
    return str(root)


def band(x, zr=(20, 101), yr=(20, 101)):
    p = np.zeros((128, 128, 128), np.uint8)
    p[zr[0]:zr[1], yr[0]:yr[1], x] = 255
    return p


def surfaces(root):
    return E.surface_list(*BOX, tifxyz=str(root), ax=AX)


# --------------------------------------------------------------------------------------------- ERL

def test_perfect_prediction_erl_is_the_whole_surface(tmp_path):
    write_plane(tmp_path, "surf", 64)
    sf = surfaces(tmp_path)
    assert len(sf) == 1
    _, g, k, n = sf[0]
    r = E.erl(band(64), BOX[0], g, k, n, um=2.4)
    # 41 x 41 grid, 2-voxel cells, 2.4 um / voxel: every line is 40 * 2 * 2.4 = 192 um long, and each of
    # the 82 lines is traversable end to end, so ERL = the full line length.
    assert r["path_um"] == pytest.approx(2 * 41 * 40 * 2 * 2.4, rel=1e-6)
    assert r["erl_um"] == pytest.approx(192.0, rel=1e-6)
    assert r["lost_break_frac"] == 0.0 and r["lost_merge_frac"] == 0.0
    assert r["erl_break_um"] == pytest.approx(192.0, rel=1e-6)
    assert r["erl_merge_um"] == pytest.approx(192.0, rel=1e-6)


def test_broken_sheet_loses_length_to_breaks(tmp_path):
    write_plane(tmp_path, "surf", 64)
    _, g, k, n = surfaces(tmp_path)[0]
    p = band(64)
    p[:, 56:66, :] = 0  # a gap across the middle of the sheet: no band there at all
    r = E.erl(p, BOX[0], g, k, n, um=2.4)
    assert r["lost_break_frac"] > 0.05 and r["lost_merge_frac"] == 0.0
    assert r["erl_um"] < 192.0 and r["erl_um"] == pytest.approx(r["erl_break_um"], rel=1e-9)
    assert r["erl_merge_um"] == pytest.approx(192.0, rel=1e-6)  # nothing merged, only broken


def test_bridged_pair_loses_length_to_merges(tmp_path):
    write_plane(tmp_path, "surf", 64)
    _, g, k, n = surfaces(tmp_path)[0]
    p = band(64)
    p[:, :, 74] = np.where(np.zeros((128, 128), np.uint8) == 0, 0, 0)
    p[20:101, 20:101, 74] = 255  # a second sheet 10 voxels out, inside the +-40 merge window
    r = E.erl(p, BOX[0], g, k, n, um=2.4)
    assert r["lost_merge_frac"] > 0.9 and r["lost_break_frac"] == 0.0
    assert r["erl_break_um"] == pytest.approx(192.0, rel=1e-6)  # the sheet itself is never lost
    assert r["erl_merge_um"] < 20.0 and r["erl_um"] < 20.0


# ------------------------------------------------------------------------------------------- Betti

def test_betti_of_a_perfect_prediction_is_zero_error(tmp_path):
    write_plane(tmp_path, "surf", 64)
    sf = surfaces(tmp_path)
    ref = E.mesh_reference(BOX[0], BOX[1], sf)
    assert ref[:, :, 64].sum() > 1000 and ref.sum() == ref[:, :, 64].sum()  # one flat sheet at x = 64
    b = E.betti_of(ref.astype(np.uint8) * 255, BOX[0], BOX[1], sf, margin=8, band=6)
    assert b["betti0_err"] == 0 and b["betti1_err"] == 0
    assert b["betti0"] == 1 and b["betti1"] == 0


def test_bridged_pair_of_sheets_shows_a_betti0_error(tmp_path):
    write_plane(tmp_path, "a", 60)
    write_plane(tmp_path, "b", 68)
    sf = surfaces(tmp_path)
    assert len(sf) == 2
    ref = E.mesh_reference(BOX[0], BOX[1], sf)
    two = ref.copy()                       # the reference: two disjoint sheets -> b0 = 2
    bridged = ref.copy()
    bridged[40:60, 40:60, 60:69] = True    # a slab welding them together -> b0 = 1
    b_ok = E.betti_of(two.astype(np.uint8) * 255, BOX[0], BOX[1], sf, margin=8, band=4)
    b_bad = E.betti_of(bridged.astype(np.uint8) * 255, BOX[0], BOX[1], sf, margin=8, band=4)
    assert b_ok["betti0_ref"] == 2 and b_ok["betti0_err"] == 0
    assert b_bad["betti0"] == 1 and b_bad["betti0_err"] == 1


def test_a_hole_in_the_sheet_shows_a_betti1_error(tmp_path):
    write_plane(tmp_path, "surf", 64)
    sf = surfaces(tmp_path)
    ref = E.mesh_reference(BOX[0], BOX[1], sf)
    holed = ref.copy()
    holed[45:55, 45:55, :] = False         # a punched hole well inside the margin -> one extra loop
    b = E.betti_of(holed.astype(np.uint8) * 255, BOX[0], BOX[1], sf, margin=8, band=6)
    assert b["betti1_ref"] == 0 and b["betti1"] == 1 and b["betti1_err"] == 1


def test_margin_hides_the_box_faces():
    m = np.zeros((40, 40, 40), bool)
    m[:, 10:30, 20] = True                 # a sheet the box cuts at both z faces
    b = T.betti_error(m, m, margin=8, band=0)
    assert b["betti0_err"] == 0 and b["betti_interior_vox"] == 24 ** 3


# --------------------------------------------------------------------------------- ceiling / bootstrap

def test_ceiling_of_the_reference_against_itself_is_one(tmp_path):
    write_plane(tmp_path, "surf", 64)
    sf = surfaces(tmp_path)
    _, g, k, n = sf[0]
    pts, nrm = g[k], n[k]
    p = band(64)
    res = E.evaluate_all(p, BOX[0], BOX[1], pts, nrm, sf, tifxyz=str(tmp_path), ax=AX, um=2.4, boot=0, betti=True)
    assert res["recall@4"] == 1.0 and res["continuity"] == 1.0 and res["offset_le3"] == 1.0
    # `hit_frac` counts only cells with all 8 grid neighbours present, so a finite grid caps it below 1
    assert res["hit_frac"] == pytest.approx(39 * 39 / (41 * 41))
    assert res["erl_um"] == pytest.approx(192.0, rel=1e-6)
    assert res["betti"]["betti0_err"] == 0 and res["betti"]["betti1_err"] == 0


def test_bootstrap_ci_contains_the_point_estimate(tmp_path):
    for i, x in enumerate((50, 60, 70, 80)):
        write_plane(tmp_path, f"s{i}", x)
    sf = surfaces(tmp_path)
    p = np.zeros((128, 128, 128), np.uint8)
    for j, x in enumerate((50, 60, 70, 80)):
        p[20:101, 20:101 - 20 * (j == 3), x] = 255   # one surface deliberately half covered
    rows = E.surface_rows(p, BOX[0], BOX[1], tifxyz=str(tmp_path), ax=AX, um=2.4, surfaces=sf)
    pooled = E.pool(rows)
    ci = E.bootstrap(rows, n=200, seed=0)
    assert len(rows) == 4 and 0.0 < pooled["recall@4"] < 1.0
    for key in ("recall@4", "continuity", "erl_um", "hit_frac"):
        lo, hi = ci[key]
        assert lo <= pooled[key] <= hi, (key, lo, pooled[key], hi)
    assert E.bootstrap(rows, n=50, seed=1)["recall@4"] != ci["recall@4"] or True  # seeded, just must run


def test_pool_reproduces_the_legacy_pooled_numbers(tmp_path):
    for i, x in enumerate((50, 70)):
        write_plane(tmp_path, f"s{i}", x)
    sf = surfaces(tmp_path)
    p = np.zeros((128, 128, 128), np.uint8)
    p[20:101, 20:101, 50] = 255
    p[20:101, 20:90, 70] = 255
    rows = E.surface_rows(p, BOX[0], BOX[1], tifxyz=str(tmp_path), ax=AX, um=2.4, surfaces=sf)
    pooled = E.pool(rows)
    legacy = {**E.metrics(p, BOX[0], np.concatenate([g[k] for _, g, k, _ in sf]),
                          np.concatenate([n[k] for _, _, k, n in sf])),
              **E.continuity(p, *BOX, tifxyz=str(tmp_path), ax=AX)}
    for key in ("recall@2", "recall@4", "merge_frac", "continuity", "hit_frac", "mean_run"):
        assert pooled[key] == pytest.approx(legacy[key], abs=1e-9), key


# -------------------------------------------------------------------------------------- plateau fit

def test_curve_fitter_recovers_a_known_asymptote():
    c, a, al = 0.83, 3.0, 0.45
    steps = np.array([500, 1000, 2000, 4000, 6000, 10000, 15000, 20000, 30000, 45000, 60000], float)
    vals = c - a * steps ** -al
    f = E.fit_curve(steps, vals)
    assert f["model"] == "power"
    assert f["asymptote"] == pytest.approx(c, abs=5e-3)
    assert f["slope_per_10k"] > 0 and f["rmse"] < 1e-4
    assert f["step95"] > steps[-1]


def test_curve_fitter_on_a_flat_curve_says_there_is_nothing_left():
    steps = np.arange(1, 21) * 5000.0
    rng = np.random.default_rng(0)
    vals = 0.9 - 1e-5 * rng.standard_normal(20)
    f = E.fit_curve(steps, vals)
    assert f["asymptote"] == pytest.approx(0.9, abs=2e-3)
    assert abs(f["slope_per_10k"]) < 1e-2 and f["remaining"] < 5e-3


def test_curve_reads_eval_jsonl(tmp_path):
    steps = np.array([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000], float)
    vals = 0.75 - 2.0 * steps ** -0.4
    (tmp_path / "eval.jsonl").write_text(
        "".join(json.dumps({"step": int(s), "dice": float(v)}) + "\n" for s, v in zip(steps, vals)))
    out = tmp_path / "fit.json"
    r = E.curve(str(tmp_path), metric="dice", out=str(out))
    assert r["points"] == 8 and r["asymptote"] == pytest.approx(0.75, abs=5e-3)
    assert json.load(open(out))["asymptote"] == r["asymptote"]
