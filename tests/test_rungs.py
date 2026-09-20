"""The rung ladder: pyramids, sampling, weights, the scale plane, predict --rung.

Everything here is synthetic: tiny pyramids (a CT mirror with integer level names and exported prediction
groups with micron level names, some written with the real volcomp mask codec) and 32^3 patches.
"""
import json
import pathlib

import numpy as np
import pytest
import torch
import zarr

from usrm2 import data, model as M, predict as P, prep, train as T

P32 = 32  # patch


def umbilicus(tmp_path):
    p = tmp_path / "umb.json"
    p.write_text(json.dumps({"control_points": [{"z": 0, "y": 128, "x": 128},
                                                {"z": 512, "y": 128, "x": 128}]}))
    return str(p)


def plain_level(path, shape, value):
    a = zarr.create_array(str(path), shape=shape, chunks=(32, 32, 32), dtype="uint8", fill_value=0,
                          overwrite=True)
    a[:] = np.full(shape, value, np.uint8)
    return a


def ct_pyramid(root, base=256, nlev=4, value=lambda l: 100 + 10 * l, partial=False):
    """A CT mirror: '<...-2.400um-...>.zarr/<integer level>', level l = rung l + 2."""
    root = root / "20260101000000-2.400um-0.2m-78keV-masked.zarr"
    root.mkdir(parents=True, exist_ok=True)
    for l in range(nlev):
        n = base >> l
        if l == 0 and partial:  # only the first half of z is mirrored locally (the desk's partial level 0)
            a = zarr.create_array(str(root / "0"), shape=(n, n, n), chunks=(32, 32, 32), dtype="uint8",
                                  fill_value=0, overwrite=True)
            a[:n // 2] = np.full((n // 2, n, n), value(0), np.uint8)
        else:
            plain_level(root / str(l), (n, n, n), value(l))
    return str(root)


def pred_pyramid(root, name="pred.zarr", base=256, nlev=4, value=lambda k: 220 - 30 * k, attrs=None,
                 mask=False, k0=2):
    """An exported prediction group: levels named by the exact voxel size, OME multiscales in zarr.json."""
    root = root / name
    root.mkdir(parents=True, exist_ok=True)
    lv = []
    for l in range(nlev):
        n, um = base >> l, data.rung_um(k0 + l)
        lv.append({"path": f"{um:g}" if "." in f"{um:g}" else f"{um:.1f}", "um": um})
        if mask:
            import volcomp_zarr as vc
            a = zarr.create_array(str(root / lv[-1]["path"]), shape=(n, n, n), chunks=(128, 128, 128),
                                  dtype="uint8", fill_value=0, serializer=vc.VolcompCodec(mode="mask"),
                                  overwrite=True)
            v = np.zeros((n, n, n), np.uint8)
            v[n // 4:3 * n // 4] = 255  # a slab of "surface"
            a[:] = v
        else:
            plain_level(root / lv[-1]["path"], (n, n, n), value(k0 + l))
    meta = {"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{"version": "0.5", "name": name, "type": "mean",
                "axes": [{"name": q, "type": "space", "unit": "micrometer"} for q in "zyx"],
                "datasets": [{"path": q["path"], "coordinateTransformations":
                              [{"type": "scale", "scale": [q["um"]] * 3}]} for q in lv]}]},
        "volcomp": {"encoding": "mask" if mask else "ramp", "rung_voxel_size_um": data.rung_um(k0)}}}
    meta["attributes"].update(attrs or {})
    (root / "zarr.json").write_text(json.dumps(meta))
    return str(root)


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)


# ------------------------------------------------------------------------------------ the ladder

def test_rung_um_and_um_rung():
    assert [data.rung_um(k) for k in (0, 2, 4, 11)] == [0.6, 2.4, 9.6, 1228.8]
    assert [data.um_rung(u) for u in (0.6, 2.4, 9.6, 1228.8)] == [0, 2, 4, 11]


def test_rungs_on_both_naming_schemes(tmp_path):
    ct = ct_pyramid(tmp_path, nlev=4)
    assert sorted(data.rungs(ct)) == [2, 3, 4, 5]          # integer levels of a 2.400 um volume
    assert sorted(data.rungs(ct + "/0")) == [2, 3, 4, 5]   # a level path names its group
    assert data.base_rung(ct + "/0") == 2 and data.base_rung(ct + "/2") == 4
    tg = pred_pyramid(tmp_path, nlev=4)
    assert sorted(data.rungs(tg)) == [2, 3, 4, 5]          # levels named "2.4" .. "19.2"
    assert data.rungs(tg)[4].shape == (64, 64, 64)
    assert data.rungs(ct)[3].shape == (128, 128, 128)


def test_a_rung_above_the_top_is_pooled_from_the_highest_one(tmp_path):
    ct = ct_pyramid(tmp_path, base=128, nlev=2, value=lambda l: 100)  # rungs 2 (128^3) and 3 (64^3)
    pyr = data.rungs(ct)
    assert sorted(pyr) == [2, 3]
    c = data.read_rung(pyr, 4, (0, 0, 0), 32)  # rung 4 does not exist: pooled 2x from rung 3 (64^3 -> 32^3)
    assert c.shape == (32, 32, 32) and np.allclose(c, 100)
    c5 = data.read_rung(pyr, 5, (0, 0, 0), 32)  # two rungs above the top: the scroll shrinks inside the cube
    assert np.allclose(c5[:16, :16, :16], 100) and np.allclose(c5[16:], 0)


@pytest.mark.parametrize("nctx", [1, 3])
def test_context_cubes_come_from_the_right_rungs(tmp_path, nctx):
    ct = ct_pyramid(tmp_path, nlev=4)  # rung k holds the constant 100 + 10 * (k - 2)
    # centred at 128 so that even the rung-5 cube (the top of this pyramid, 32^3) fits exactly
    cx = data.context(ct + "/0", (112, 112, 112), (32, 32, 32), tuple(range(1, nctx + 1)), rung=2)
    assert len(cx) == nctx
    for d, c in enumerate(cx, start=1):
        assert c.shape == (32, 32, 32) and np.allclose(c, 100 + 10 * d)
    cx3 = data.context(ct, (48, 48, 48), (16, 16, 16), (1, 2), rung=3)  # anchored at rung 3
    assert np.allclose(cx3[0], 120) and np.allclose(cx3[1], 130)


def test_scale_plane_value():
    assert np.allclose(data.scale_plane(2, (4, 4, 4)), 0.0)
    assert np.allclose(data.scale_plane(11, (4, 4, 4)), 1.0)
    assert np.allclose(data.scale_plane(5, (2, 2, 2)), 3 / 9)
    x = data.inputs(np.full((4, 4, 4), 7, np.uint8), np.zeros((3, 4, 4, 4), np.float32), (), rung=4)
    assert x.shape == (5, 4, 4, 4) and np.allclose(x[1], 2 / 9)  # CT, scale, radial(3)


def test_mask_mode_chunks_decode_to_a_0_255_field(tmp_path):
    tg = pred_pyramid(tmp_path, name="m.zarr", base=128, nlev=1, mask=True)
    a = data.rungs(tg)[2]
    v = np.asarray(a[:])
    assert v.dtype == np.uint8 and v.max() == 255 and v.min() == 0
    assert v[64, 64, 64] == 255 and v[0, 0, 0] == 0
    assert 0 < ((v > 0) & (v < 255)).sum()  # the trilinear ramp across the boundary


# ------------------------------------------------------------------------------------ sampling

def sources(tmp_path, **kw):
    ct = ct_pyramid(tmp_path, **kw)
    tg = pred_pyramid(tmp_path)
    return ct, tg, [f"{ct},{tg}"]


def test_a_sample_reads_ct_target_and_context_from_the_rung(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "NORM", (0.0, 1.0))  # z-score off: the channels are the planted constants
    ct, tg, lines = sources(tmp_path)
    for k in (2, 3):
        ds = data.Patches(patch=P32, stores=lines, exclude=[], rungs={k}, ctx=(1, 2), sym=False,
                          air_keep=1.0, fg_keep=1.0)
        item = next(iter(ds))
        assert item["ct"].dtype == item["tgt"].dtype == item["w"].dtype == torch.uint8
        x, t, w = prep.prepare(prep.batch1(item), torch.device("cpu"))
        x, t, w, rung = x[0], t[0], w[0], int(item["rung"])
        assert rung == k and x.shape == (1 + 2 + 1 + 3, P32, P32, P32)
        assert torch.allclose(x[0], torch.tensor(100.0 + 10 * (k - 2)))        # CT at rung k
        for d in (1, 2):  # the context cubes come from rungs k+1, k+2 (0 where the cube leaves the volume)
            u = set(np.unique(x[d].numpy()).tolist())
            assert u <= {0.0, 100.0 + 10 * (k - 2 + d)} and max(u) == 100.0 + 10 * (k - 2 + d)
        assert torch.allclose(x[3], torch.tensor((k - 2) / 9))                 # the scale plane
        assert torch.allclose(t[0], torch.tensor((220.0 - 30 * k) / 255), atol=1e-6)  # target at rung k
        assert torch.allclose(w[0], torch.ones(1))


def test_rung_probabilities_follow_sqrt_of_the_patch_count(tmp_path):
    ct, tg, lines = sources(tmp_path)
    s = data.source_groups(lines)[0]
    assert data.usable_rungs(s) == list(range(2, 12))
    p = data.rung_probs(s, P32)
    assert abs(sum(p.values()) - 1) < 1e-9
    assert p[2] > p[3] > p[4] > p[5]                      # n_k halves per axis: sqrt(n) halves per rung
    assert abs(p[2] / p[3] - 2 * np.sqrt(2)) < 1e-6
    assert p[8] == pytest.approx(p[11])                   # n_k floored at 1 flattens the top
    b = data.rung_probs(s, P32, boost={2: 10.0})
    assert b[2] > p[2] and b[3] < p[3]
    assert list(data.rung_probs(s, P32, allowed={4, 5, 6})) == [4, 5, 6]


def test_weights_are_zero_outside_the_box_in_masked_ct_and_for_a_missing_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    # a boxed target (the box attr is given at the native rung) and a second source for another channel
    tg = pred_pyramid(tmp_path, name="boxed.zarr", attrs={"box": [[0, 0, 0], [64, 64, 64]], "weight": 0.5})
    vs = pred_pyramid(tmp_path, name="verso.zarr", attrs={"channel": "verso"})
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}", f"{ct},{vs}"], exclude=[], rungs={2}, sym=False,
                      air_keep=1.0, fg_keep=1.0)
    ds._open_rungs()
    assert ds.channels == ["recto", "verso"]
    s = ds.srcs[0]
    ctp = data.read_rung(s["ct_pyr"], 2, (48, 0, 0), P32, dtype=np.uint8)
    t, w = ds._rung_target(s, 2, np.array([48, 0, 0]), ctp)
    assert t.dtype == w.dtype == np.uint8  # 255 = 1.0; the weight quantisation round trips to 1/255
    assert np.allclose(w[0, :16] / 255, 0.5, atol=1 / 255) and np.all(w[0, 16:] == 0)  # the box ends at z = 64
    assert np.all(w[1] == 0) and np.all(t[1] == 0)                     # this source has no verso channel
    ctp[:, :8] = 0  # masked CT
    t, w = ds._rung_target(s, 2, np.array([0, 0, 0]), ctp)
    assert np.all(w[0][:, :8] == 0) and np.all(t[0][:, :8] == 0)
    assert np.allclose(w[0][:, 8:] / 255, 0.5, atol=1 / 255)


def test_the_partial_level_index_keeps_sampling_off_missing_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, partial=True)  # only z < 128 of level 0 is on disk
    tg = pred_pyramid(tmp_path)
    a0 = data.rungs(ct)[2]
    assert data.coverage(a0) == pytest.approx(0.5)
    assert data.covered(a0, (0, 0, 0), (32, 32, 32)) and not data.covered(a0, (160, 0, 0), (32, 32, 32))
    assert data.coverage(data.rungs(ct)[3]) == 1.0  # the coarse levels are whole
    monkeypatch.setattr(data, "NORM", (0.0, 1.0))  # z-score off: x[0] is the raw CT
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2}, sym=False,
                      air_keep=1.0, fg_keep=1.0)
    it = iter(ds)
    for _ in range(20):
        assert (next(it)["ct"][0] > 0).all()  # a missing chunk reads as zeros (air) and must never be sampled


def test_the_val_box_is_excluded_at_every_rung(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg, lines = sources(tmp_path)
    ds = data.Patches(patch=P32, stores=lines, rungs={2, 3}, sym=False, air_keep=1.0, fg_keep=1.0,
                      exclude=[((0, 0, 0), (256, 256, 128))])  # the whole x < 128 half, at rung 2
    rng = np.random.default_rng(0)
    ds._open_rungs()
    for _ in range(200):
        got = ds._rung_sample(rng)
        if got is None:
            continue
    # every accepted corner must sit in the other half: check the rule directly
    for k, lo, want in [(2, np.array([0, 0, 0]), False), (2, np.array([0, 0, 200]), True),
                        (3, np.array([0, 0, 0]), False), (3, np.array([0, 0, 100]), True)]:
        d = k - 2
        eo, es = np.array([0, 0, 0]) >> d, np.maximum(np.array([256, 256, 128]) >> d, 1)
        ok = not (np.all(lo < eo + es) and np.all(lo + P32 > eo))
        assert ok == want


def test_val_grid_rungs_scores_every_rung(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg, lines = sources(tmp_path)
    grid = data.val_grid_rungs(P32, lines, ((0, 0, 0), (64, 64, 64)), rungs=(2, 3, 4), limit=2, ctx=(1,))
    assert {int(g["rung"]) for g in grid} == {2, 3, 4}
    assert all(g["ct"].dtype == torch.uint8 for g in grid)  # compact: 8 patches x 3 rungs is ~1.6 GB at 256^3
    for g in grid:
        x, t, w = prep.prepare(prep.batch1(g), torch.device("cpu"))
        assert x.shape == (1, 1 + 1 + 1 + 3, P32, P32, P32) and t.shape == (1, 1, P32, P32, P32)
        assert torch.allclose(x[0, 2], torch.tensor((int(g["rung"]) - 2) / 9))


def test_rung_mix_report(tmp_path, capsys):
    ct, tg, lines = sources(tmp_path)
    rows = data.rung_mix(lines, patch=P32)
    assert {r["rung"] for r in rows} == set(range(2, 12))
    assert abs(sum(r["p"] for r in rows) - 1) < 1e-9
    assert all(r["ct_coverage"] == 1.0 for r in rows)
    assert [r["ct_rung"] for r in rows][:6] == [2, 3, 4, 5, 5, 5]  # above the CT top, the top rung is pooled
    txt = data.format_rung_mix(rows)
    assert "rung" in txt and "CT local" in txt


# ------------------------------------------------------------------------------------ augmentation

def test_augment_keeps_the_cubes_aligned_and_the_scale_plane_untouched():
    rng = np.random.default_rng(0)
    x = np.zeros((7, 8, 8, 8), np.float32)  # CT + 2 context + scale + radial(3)
    x[0] = np.arange(8 * 8 * 8).reshape(8, 8, 8)
    x[1], x[2] = x[0] * 2, x[0] * 3
    x[3] = 4 / 9
    x[6] = 1.0
    t = np.zeros((2, 8, 8, 8), np.float32)  # target + weight, augmented together
    t[0] = x[0]
    for _ in range(20):
        y, tt = data.augment(rng, x, t)
        assert np.allclose(y[1], y[0] * 2) and np.allclose(y[2], y[0] * 3)  # cubes permuted together
        assert np.allclose(y[3], 4 / 9)                                     # the scale plane is constant
        assert np.allclose(tt[0], y[0])                                     # target follows the CT
        assert np.allclose(np.linalg.norm(y[4:], axis=0), 1.0)


# ------------------------------------------------------------------------------------ losses

def test_a_zero_weight_voxel_carries_no_gradient():
    torch.manual_seed(0)
    logit = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    tgt = torch.zeros(1, 1, 4, 4, 4)
    tgt[..., :2] = 1.0
    w = torch.ones_like(tgt)
    w[..., :2, :, :] = 0.0  # ignore the first two z slices
    bce, dice = T.losses(logit, tgt, w=w)
    (bce + dice).backward()
    g = logit.grad
    assert torch.allclose(g[..., :2, :, :], torch.zeros(1)) and g[..., 2:, :, :].abs().sum() > 0
    # the unweighted loss does see them
    logit2 = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    T.losses(logit2, tgt)[0].backward()
    assert logit2.grad[..., :2, :, :].abs().sum() > 0


def test_deep_losses_pool_targets_and_weights():
    logits = [torch.zeros(1, 1, 8, 8, 8), torch.zeros(1, 1, 4, 4, 4)]
    tgt, w = torch.ones(1, 1, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8)
    w[..., :4, :, :] = 1.0
    bce, dice = T.deep_losses(logits, tgt, w=w)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    w0 = torch.zeros_like(w)
    b0, d0 = T.deep_losses(logits, tgt, w=w0)  # everything ignored: no BCE signal
    assert float(b0) == pytest.approx(0.0, abs=1e-5)


def test_evaluate_reports_dice_per_rung(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg, lines = sources(tmp_path)
    grid = data.val_grid_rungs(P32, lines, ((0, 0, 0), (64, 64, 64)), rungs=(2, 3), limit=1, ctx=())
    net = M.build("1m", verbose=False, cin=prep.shapes(grid[0])[0], cout=1)
    out = T.evaluate(net, grid, torch.device("cpu"))
    assert "dice_r2" in out and "dice_r3" in out
    assert out["dice"] == pytest.approx((out["dice_r2"] + out["dice_r3"]) / 2)


# ------------------------------------------------------------------------------------ warm start

def test_warm_start_widens_13_to_14_channels_and_keeps_head_0():
    old = M.build("1m", verbose=False, cin=13, cout=4, deep=1)
    src = T.warm_start(old.state_dict(), cin=14, cout=1)
    w0, w1 = old.state_dict()["enc.0.0.weight"], src["enc.0.0.weight"]
    assert w1.shape[1] == 14
    assert torch.allclose(w1[:, :10], w0[:, :10])          # CT + 9 context cubes
    assert torch.allclose(w1[:, 10], torch.zeros(1))       # the new scale plane starts at zero
    assert torch.allclose(w1[:, 11:], w0[:, 10:])          # radial vector last
    assert src["head.weight"].shape[0] == 1
    assert torch.allclose(src["head.weight"][0], old.state_dict()["head.weight"][0])
    assert torch.allclose(src["deep_heads.0.weight"][0], old.state_dict()["deep_heads.0.weight"][0])
    new = M.build("1m", verbose=False, cin=14, cout=1, deep=1)
    assert not new.load_state_dict(src, strict=False).unexpected_keys


# ------------------------------------------------------------------------------------ train + predict

def test_train_at_rungs_and_predict_reproduces_the_loader_input(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    # a box exactly one patch wide at rung 3: the loader then has a single legal corner
    tg = pred_pyramid(tmp_path, attrs={"box": [[64, 64, 64], [64, 64, 64]]})
    lines = [f"{ct},{tg}"]
    out = tmp_path / "run"
    ckpt = T.train(out, size="1m", steps=2, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=2, val_patches=2, device="cpu", ctx=(1, 2), rungs={3},
                   val_rungs=(3,), stores=lines, val=((0, 0, 0), (32, 32, 32)))
    rec = json.loads((out / "eval.jsonl").read_text().splitlines()[-1])
    assert "dice_r3" in rec and np.isfinite(rec["bce"])
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st["args"]["cin"] == 7 and st["args"]["cout"] == 1 and st["args"]["scale_plane"]
    logs = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    assert any("rung" in q for q in logs) or st["step"] < 20  # the rung histogram is logged every 20 steps

    ds = data.Patches(patch=P32, stores=lines, exclude=[], rungs={3}, ctx=(1, 2), sym=False,
                      air_keep=1.0, fg_keep=1.0)
    x_loader = prep.prepare(prep.batch1(next(iter(ds))), torch.device("cpu"))[0][0].numpy()

    seen = []
    real_inputs = data.inputs
    monkeypatch.setattr(data, "inputs", lambda *a, **k: seen.append(real_inputs(*a, **k)) or seen[-1])
    P.probs(ckpt, ct, 32, 32, 32, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3)
    assert seen, "predict built no input"
    assert np.allclose(seen[0], x_loader, atol=1e-5)  # same CT, context, scale plane and radial vector


def test_predict_writes_the_rung_in_the_attrs(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    out = tmp_path / "run"
    ckpt = T.train(out, size="1m", steps=1, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=1, val_patches=1, device="cpu", ctx=(1,), rungs={4},
                   val_rungs=(4,), stores=[f"{ct},{tg}"], val=((0, 0, 0), (64, 64, 64)))
    o = P.predict(ckpt, ct, 0, 0, 0, P32, P32, P32, str(tmp_path / "pred.zarr"), window=P32, halo=8,
                  device="cpu", volcomp=False, rung=4)
    a = zarr.open(o, mode="r")
    assert a.attrs["rung"] == 4 and a.attrs["voxel_um"] == pytest.approx(9.6)


def test_require_targets_skips_unpulled_windows(tmp_path, monkeypatch):
    """A partially pulled export: with require_targets a window whose target shard is missing is never drawn."""
    import shutil
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    lvl = tmp_path / "pred.zarr" / "2.4"
    zs = sorted(lvl.glob("c/*"), key=lambda q: int(q.name))
    for c in zs[len(zs) // 2:]:  # drop the upper half of the finest level's z chunk rows
        shutil.rmtree(c)
    data.CHUNK_INDEX.clear()
    assert 0.0 < data.coverage(data.rungs(tg)[2]) < 1.0
    kw = dict(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2}, sym=False, air_keep=1.0, fg_keep=1.0, fg_min=0.0)
    seen_empty = False
    it = iter(data.Patches(**kw))
    for _ in range(60):
        seen_empty |= float(next(it)["tgt"].max()) == 0.0
    assert seen_empty  # without the flag the missing shards read as zeros
    it = iter(data.Patches(require_targets=True, **kw))
    for _ in range(60):
        assert float(next(it)["tgt"].max()) > 0.0


def test_mirror_json_marks_known_chunks(tmp_path, monkeypatch):
    """mirror.json: {"complete": true} makes a sparse level whole; {"boxes": [...]} marks those chunks known."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, partial=True)
    a0 = data.rungs(ct)[2]
    d = data.array_dir(a0)
    assert data.coverage(a0) == pytest.approx(0.5)
    (pathlib.Path(d) / "mirror.json").write_text(json.dumps({"boxes": [[128, 0, 0, 64, 256, 256]]}))
    data.CHUNK_INDEX.clear()
    assert data.coverage(a0) > 0.5 and data.covered(a0, (128, 0, 0), (32, 32, 32))
    (pathlib.Path(d) / "mirror.json").write_text(json.dumps({"complete": True}))
    data.CHUNK_INDEX.clear()
    assert data.coverage(a0) == 1.0


def test_rungs_include_levels_built_after_the_export(tmp_path, monkeypatch):
    """A CT mirror whose group metadata lists levels 0..3 but has a level 4 directory on disk exposes rung 6."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, nlev=4)
    base = data.pyramid_base(ct)
    plain_level(pathlib.Path(base) / "4", (16, 16, 16), 7)
    data.CTX_CACHE.pop(base, None)
    assert 6 in data.rungs(ct)


def test_a_window_with_no_weighted_voxel_is_not_sampled(tmp_path, monkeypatch):
    """A window whose CT is all masked (or which falls outside the target's box) carries no gradient: the
    loss is exactly 0. Region mode makes those arrive 64 at a time, so they are rejected outright."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=128, nlev=2, value=lambda l: 0)  # the whole CT is air -> weight 0
    tg = pred_pyramid(tmp_path, base=128, nlev=2)
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2}, sym=False,
                      air_keep=1.0, fg_keep=1.0, fg_min=0.0)
    ds._open_rungs()
    rng = np.random.default_rng(0)
    assert all(ds._rung_draw(rng, build=False)[0] is None for _ in range(50))


def test_source_weight_is_physical_volume(tmp_path, monkeypatch):
    """Two scrolls of the same physical size draw equally even when one is stored 4x coarser (64x fewer voxels)."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct_a = ct_pyramid(tmp_path / "a", base=256)
    tg_a = pred_pyramid(tmp_path / "a", base=256)                       # native rung 2, 256^3 voxels
    ct_b = ct_pyramid(tmp_path / "b", base=64)
    tg_b = pred_pyramid(tmp_path / "b", base=64, k0=4)  # native 9.6 um
    srcs = data.source_groups([f"{ct_a},{tg_a}", f"{ct_b},{tg_b}"])
    assert srcs[1]["native"] == 4
    assert srcs[0]["voxels"] == 64 * srcs[1]["voxels"]
    assert srcs[0]["volume_um3"] == pytest.approx(srcs[1]["volume_um3"])
