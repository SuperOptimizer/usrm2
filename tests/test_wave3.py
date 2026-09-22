"""Wave 3 (docs/unified_design.md section 30): the size ladder, the multi-head export pass, the
label-free loader, the distance targets on the stores line, and the pod's v2 region writer.

CPU and synthetic throughout. The load-bearing claims:
  * the ladder presets are the SAME six levels at a factor 2 in parameters, so a log-log fit has three
    evenly spaced points;
  * one multi-head pass returns BIT-IDENTICAL planes to one pass per head -- that is what makes replacing
    `export-tracer`'s five passes with one a pure speed change;
  * a shared stream queue keeps one replay cursor and one eviction bound PER TAG, and the planner evicts
    only what the slowest tag has passed;
  * the label-free loader samples where the CT is, not where a label is, and takes a bare CT line;
  * a distance channel is read at every rung its store HOLDS and has weight 0 at every other one;
  * the pod's v2 core is v1 for a checkpoint without field heads, and its compat shims are copies.
"""
import json
import os

import numpy as np
import pytest
import torch

from usrm2 import data, ladder as LD, model as M, predict as P, stream as S, train as T
from tests.test_rungs import P32, ct_pyramid, pred_pyramid, umbilicus
from tests.test_stream import origin  # noqa: F401  (the http origin fixture, for the label-free plan)


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)


# ------------------------------------------------------------------------------- 1. the size ladder

def test_the_ladder_is_the_same_depth_at_a_factor_two_in_parameters():
    """Experiment 12 fits val loss against log(params), so the three rungs must be evenly spaced in
    log(params) and must differ ONLY in width -- a deeper net is a different experiment."""
    w = [M.PRESETS[s] for s in LD.SIZES]
    assert {len(q) for q in w} == {6}, "every rung is the six-level net"
    n = [M.params(s) for s in LD.SIZES]
    assert n == sorted(n)
    for a, b in zip(n, n[1:]):
        assert 1.9 < b / a < 2.1, f"{b / a}: the ladder must be a factor 2 in parameters"
    for q in w:  # GroupNorm takes min(8, c) groups and the tensor cores want the 8-grid
        assert all(c % 8 == 0 for c in q), q


def test_the_ladder_commands_differ_only_in_size_lr_and_tag():
    base = "--patch 256 --batch 2 --steps 20000 --rungs 2-11 --ctx 1..9 --workers 8"
    cmds = LD.launch(base, "/runs/L", queue="/q", stores_file="/s.txt", log=lambda *a: None)
    assert [c[0] for c in cmds] == list(LD.SIZES)
    seen = set()
    for sz, d, cmd in cmds:
        assert base in cmd and f"--size {sz}" in cmd and f"--stream-tag {sz}" in cmd
        seen.add(cmd.replace(f"--size {sz}", "").replace(f"--stream-tag {sz} ", "").replace(d, "OUT"))
    assert len(seen) == 1, "the rungs differ by more than --size / --stream-tag"


def test_mup_lite_scales_the_lr_by_one_over_sqrt_width_only_when_asked():
    assert LD.lr_for("60m", "30m6", 3e-4) == 3e-4                    # the default is the control
    assert LD.lr_for("60m", "30m6", 3e-4, "mup") == pytest.approx(3e-4 * np.sqrt(32 / 48))
    assert LD.lr_for("15m", "30m6", 3e-4, "mup") == pytest.approx(3e-4 * np.sqrt(32 / 24))


def test_the_loglog_slope_recovers_a_planted_exponent():
    par = np.array([1e6, 2e6, 4e6])
    f = LD.loglog_slope(par, 0.3 * par ** -0.2)
    assert f["alpha"] == pytest.approx(0.2, abs=1e-6) and f["r2"] == pytest.approx(1.0, abs=1e-9)
    flat = LD.loglog_slope(par, [0.1, 0.1, 0.1])
    assert abs(flat["alpha"]) < 1e-9, "a flat ladder must read as a flat slope, not as noise"


def fake_run(d, size, dice, steps=20, gap=0.0):
    """A run directory with the eval.jsonl / train.jsonl / ckpt.pt a report reads."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "eval.jsonl"), "w") as f:
        for i in range(1, steps + 1):
            v = dice * (1 - 0.5 / i)
            f.write(json.dumps({"step": 100 * i, "dice": v, "dice_r2": v, "bce": 1 - v}) + "\n")
    with open(os.path.join(d, "train.jsonl"), "w") as f:
        for i in range(1, steps + 1):
            f.write(json.dumps({"step": 100 * i, "bce": 1 - dice * (1 - 0.5 / i) - gap}) + "\n")
    torch.save({"args": {"size": size, "cin": 4, "cout": 1}, "step": 100 * steps},
               os.path.join(d, "ckpt.pt"))
    return d


def test_the_ladder_report_fits_per_rung_and_matches_the_step(tmp_path, capsys):
    runs = [fake_run(str(tmp_path / s), s, v, gap=g)
            for s, v, g in (("15m", 0.80, 0.00), ("30m6", 0.84, 0.02), ("60m", 0.87, 0.05))]
    res = LD.report(runs, metric="dice", log=lambda *a: None)
    assert res["matched_step"] == 2000, "the matched step is the largest step EVERY run reached"
    f = res["per_rung"]["dice_r2"]
    assert f["n"] == 3 and f["alpha"] > 0, "1 - dice falls with params, so alpha > 0"
    assert f["gap_trend"] > 0, "the planted train/val gap grows with size and must be reported"
    assert [r["params"] for r in res["runs"]] == [M.params(s) for s in ("15m", "30m6", "60m")]


def test_the_ladder_report_refuses_a_run_with_no_evals(tmp_path):
    fake_run(str(tmp_path / "a"), "15m", 0.8)
    os.makedirs(tmp_path / "b", exist_ok=True)
    with pytest.raises(AssertionError, match="eval.jsonl"):
        LD.report([str(tmp_path / "a"), str(tmp_path / "b")], log=lambda *a: None)


# --------------------------------------------------------------- 2. one pass, every head

def phase_b_ckpt(tmp_path, monkeypatch, **kw):
    """A tiny CPU-trained Phase-B checkpoint: recto + verso + midline + thickness + normals + logvar."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    md = pred_pyramid(tmp_path, name="mid.zarr", base=256, nlev=1, value=lambda k: 140,
                      attrs={"channel": "midline"})
    th = pred_pyramid(tmp_path, name="th.zarr", base=256, nlev=1, value=lambda k: 60,
                      attrs={"channel": "thickness"})
    ckpt = T.train(tmp_path / "run", size="1m", steps=2, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=2, val_patches=1, device="cpu", ctx=(1,), rungs={2}, val_rungs=(2,),
                   stores=[f"{ct},{tg},{md},{th}"], val=((0, 0, 0), (32, 32, 32)),
                   verso=True, teacher_regions=str(tmp_path / "tr"), cout=None,
                   sdist="midline", thickness=True, normals="head", sdist_hetero=True, **kw)
    return str(ckpt), ct


def test_head_names_lists_every_head_a_checkpoint_can_serve():
    a = {"channels": ["recto", "verso", "midline", "thickness", "nz", "ny", "nx", "logvar"], "cout_p": 2}
    assert P.head_names(a) == ["recto", "verso", "midline", "thickness", "normals", "conf"]
    assert P.head_names({"channels": ["recto", "verso"], "cout_p": 2}) == ["recto", "verso"]
    assert P.head_names({"channels": ["recto"], "cout_p": 1}) == ["recto"]


def test_one_multi_head_pass_is_bit_identical_to_one_pass_per_head(tmp_path, monkeypatch):
    """The whole justification for replacing export-tracer's five passes with one: every head is a
    POINTWISE function of the same raw output and the Gaussian blend is linear in it, so the planes must
    agree EXACTLY -- `array_equal`, not `allclose`."""
    ckpt, ct = phase_b_ckpt(tmp_path, monkeypatch)
    kw = dict(window=P32, halo=8, device="cpu", rung=2)
    got, st = P.probs_multi(ckpt, ct, 0, 0, 0, P32, P32, P32, **kw)
    assert list(got) == ["recto", "verso", "midline", "thickness", "nz", "ny", "nx", "conf"]
    for hd in ("recto", "verso", "midline", "thickness", "conf"):
        one = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, head=hd, **kw)[0]
        assert np.array_equal(one, got[hd]), hd
    n = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, head="normals", **kw)[0]
    for j, nm in enumerate(("nz", "ny", "nx")):
        assert np.array_equal(n[j], got[nm]), nm


def test_head_midline_returns_the_distance_not_the_normals(tmp_path, monkeypatch):
    """`field_head` returns the NAME asked for, and a midline checkpoint's distance channel is called
    "midline"; before wave 3 that name fell through to the normals branch and came back (3,Z,Y,X)."""
    ckpt, ct = phase_b_ckpt(tmp_path, monkeypatch)
    kw = dict(window=P32, halo=8, device="cpu", rung=2)
    a = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, head="midline", **kw)[0]
    b = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, head="sdist", **kw)[0]
    assert a.shape == (P32,) * 3 and np.array_equal(a, b)


def test_flips_chan_negates_a_normal_component_per_flipped_axis():
    """TTA over a VECTOR field: flipping the world along axis d negates component d. `flips_vec` would
    have flipped the channel axis as if it were z and cancelled the field."""
    v = torch.zeros(1, 3, 4, 4, 4)
    v[:, 0] = 1.0                       # a constant nz = 1 field, whatever the input
    out = P.flips_chan(lambda t: v.clone(), n=8, vec=[(0, 1, 2)])(torch.zeros(1, 7, 4, 4, 4))
    assert float(out[0, 0].mean()) == pytest.approx(0.0, abs=1e-6), \
        "four of the eight flips negate nz, so a constant field averages to zero"
    plain = P.flips_chan(lambda t: v.clone(), n=8, vec=[])(torch.zeros(1, 7, 4, 4, 4))
    assert float(plain[0, 0].mean()) == pytest.approx(1.0, abs=1e-6)


def test_export_tracer_still_writes_the_whole_contract_off_one_pass(tmp_path, monkeypatch):
    ckpt, ct = phase_b_ckpt(tmp_path, monkeypatch)
    calls = []
    real = P.probs

    def counting(*a, **k):
        calls.append(k.get("head"))
        return real(*a, **k)
    monkeypatch.setattr(P, "probs", counting)
    tr = P.export_tracer(ckpt, ct, 0, 0, 0, P32, P32, P32, str(tmp_path / "trc"), window=P32, halo=8,
                         device="cpu", rung=2, volcomp=False, log=lambda *a, **k: None)
    assert {"recto", "verso", "surf_sdist", "nz", "ny", "nx", "gmag", "conf", "thickness"} <= set(tr)
    assert len(calls) == 1 and isinstance(calls[0], list), \
        f"export-tracer must make ONE sliding-window pass, not {len(calls)}"


# ------------------------------------------------------- 3. the shared stream queue

def test_a_shared_queue_keeps_one_cursor_and_one_bound_per_tag(tmp_path):
    q = tmp_path / "q"
    q.mkdir()
    for tag, i in (("15m", 500), ("30m6", 300), ("60m", 120)):
        json.dump({"i": i, "margin": 20}, open(q / f"consumed.{tag}", "w"))
    pl = S.Planner.__new__(S.Planner)
    pl.dir = str(q)
    assert pl.consumed() == 100, "the eviction bound is the SLOWEST tag, minus its margin"
    json.dump({"i": 90, "margin": 0}, open(q / "consumed", "w"))
    assert pl.consumed() == 90, "an untagged trainer counts too"
    os.utime(q / "consumed.60m", (0, 0))           # a rung that died an hour ago
    assert pl.consumed() == 90
    os.remove(q / "consumed")
    assert pl.consumed() == 280, "a stale tag must not pin the buffer for ever"


def test_the_replay_cursor_is_per_tag(tmp_path):
    ds = data.Patches(patch=P32, stores=["a,b"], stream=str(tmp_path / "q"), stream_tag="60m")
    assert ds.stream_tag == "60m"
    assert data.Patches(patch=P32, stores=["a,b"], stream=str(tmp_path / "q")).stream_tag is None


def test_the_tag_reaches_the_checkpoint_args_only_when_it_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg = ct_pyramid(tmp_path), pred_pyramid(tmp_path)
    ck = T.train(tmp_path / "r", size="1m", steps=1, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                 eval_every=1, val_patches=1, device="cpu", ctx=(1,), rungs={2}, val_rungs=(2,),
                 stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)))
    assert "stream_tag" not in torch.load(ck, map_location="cpu", weights_only=False)["args"]


# ------------------------------------------------------- 4. the label-free loader

def test_a_bare_ct_line_is_a_source_only_in_the_label_free_path(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    with pytest.raises(AssertionError, match="label-free"):
        data.source_groups([ct])
    s = data.source_groups([ct], label_free=True)[0]
    assert s["targets"] == {} and s["label_free"] and s["bounds"]["pyr"] is s["ct_pyr"]


def test_label_free_sampling_leaves_the_target_box_behind(tmp_path, monkeypatch):
    """The point of the flag: a window is drawn where the CT is, not where a label is. The target here
    covers an eighth of the CT, so a labelled loader can never draw past it and a label-free one must."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256)
    tg = pred_pyramid(tmp_path, base=256, nlev=1, attrs={"box": [[0, 0, 0], [64, 64, 64]]})
    rng = np.random.default_rng(0)
    for lf, want in ((False, False), (True, True)):
        ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2}, sym=False,
                          air_keep=1.0, fg_keep=1.0, label_free=lf)
        ds._open_rungs()
        los = [d["lo"] for d in (ds._rung_draw(rng, build=False)[0] for _ in range(60)) if d]
        assert los, "nothing was drawn at all"
        assert any(max(q) >= 64 for q in los) == want, \
            f"label_free={lf}: drawn outside the target box = {any(max(q) >= 64 for q in los)}"


def test_label_free_keeps_a_window_a_labelled_loader_would_reject(tmp_path, monkeypatch):
    """A window with no weighted voxel carries no gradient and a labelled loader drops it. A label-free
    one has no gradient to protect and must keep it -- otherwise a bare CT line yields nothing at all."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256)
    ds = data.Patches(patch=P32, stores=[ct], exclude=[], rungs={2}, sym=False, air_keep=1.0,
                      label_free=True)
    ds._open_rungs()
    assert ds.channels == [] and ds.nrecto == 0
    rng = np.random.default_rng(0)
    d, item = ds._rung_draw(rng)
    assert d is not None and item is not None
    assert tuple(item["ct"].shape) == (1, P32, P32, P32) and item["tgt"].shape[0] == 0


def test_label_free_pretraining_runs_end_to_end_on_a_bare_ct_line(tmp_path, monkeypatch):
    from usrm2 import pretrain as PT
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256)
    ck = PT.pretrain(tmp_path / "pt", size="1m", steps=2, patch=P32, batch=1, lr=1e-3, workers=0,
                     warmup=1, eval_every=2, val_patches=1, device="cpu", ctx=(1,), rungs={2},
                     stores=[ct], val=((0, 0, 0), (32, 32, 32)), label_free=True)
    a = torch.load(ck, map_location="cpu", weights_only=False)["args"]
    assert a["label_free"] is True and a["cin"] == 1 + 1 + 1 + 1 + 3


def test_stream_plan_can_plan_label_free_windows(tmp_path, origin, monkeypatch):
    """The planner is the sampler, so the flag has to reach it: the plan records it, the walk comes off
    the CT pyramid and a replaying dataset picks it up from meta.json."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    sf = tmp_path / "stores.txt"
    sf.write_text(f"{mct}\n")                       # a BARE CT line: no target group at all
    q = tmp_path / "q"
    S.plan(stores_file=str(sf), queue=str(q), patch=P32, rungs={2, 3}, seed=0, workers=1, ahead=10 ** 6,
           cache_gb=10 ** 6, ctx=(1,), limit=6, jobs=8, report=10 ** 6, val=None, label_free=True)
    m = S.read_meta(str(q))
    assert m["label_free"] is True and m["channels"] == []
    recs = [json.loads(l) for l in open(q / S.QUEUE)]
    assert len(recs) == 6 and [r["i"] for r in recs] == list(range(6))
    ds = data.Patches(patch=P32, stores=[], stream=str(q), ctx=(1,), rungs={2, 3}, exclude=[])
    ds._open()
    assert ds.label_free is True


def test_label_free_and_require_targets_are_refused_together(tmp_path):
    with pytest.raises(AssertionError, match="contradict"):
        S.Planner(str(tmp_path / "s.txt"), str(tmp_path / "q"), label_free=True, require_targets=True)


# ------------------------------------------- 5. the distance targets on the stores line

def dist_group(tmp_path, name, channel, nlev, value=200):
    """A distance pyramid with `nlev` levels starting at rung 2, as `usrm2 dist-pyramid` writes."""
    return pred_pyramid(tmp_path, name=name, base=256, nlev=nlev, value=lambda k: value,
                        attrs={"channel": channel, "no_data": 0,
                               "sign_convention": "recto_positive (radially outward)"})


def test_the_dist_groups_join_the_stores_line_and_are_read_at_every_rung_they_hold(tmp_path, monkeypatch):
    """Section 29.8's Paris 4 line, `<ct>,<recto mask>,<M>_sdist.zarr,<M>_midline.zarr,<M>_thick.zarr`:
    the loader reads a distance channel at rungs 2-4 (the levels the store HOLDS) and gives it weight 0
    at rung 5, because a distance is never pooled."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256, nlev=5)
    tg = pred_pyramid(tmp_path, base=256, nlev=5)
    sd = dist_group(tmp_path, "m_sdist.zarr", "sdist", nlev=3, value=200)       # rungs 2, 3, 4
    md = dist_group(tmp_path, "m_midline.zarr", "midline", nlev=3, value=140)
    th = dist_group(tmp_path, "m_thick.zarr", "thickness", nlev=3, value=60)
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg},{sd},{md},{th}"], exclude=[], rungs={2, 3, 4, 5},
                      sym=False, air_keep=1.0, fg_keep=1.0)
    ds._open_rungs()
    assert ds.channels == ["recto", "sdist", "midline", "thickness"]
    assert ds.nrecto == 1, "a distance channel never decides whether a window is worth training on"
    s = ds.srcs[0]
    assert sorted(s["targets"]["sdist"]["pyr"]) == [2, 3, 4]
    for k in (2, 3, 4, 5):
        ctp = data.read_rung(s["ct_pyr"], k, (0, 0, 0), P32, dtype=np.uint8)
        tgt, w = ds._rung_target(s, k, np.array([0, 0, 0]), ctp)
        for c, chan in enumerate(ds.channels):
            if chan in data.DIST_CHANNELS:
                assert (w[c] > 0).any() == (k in (2, 3, 4)), f"{chan} at rung {k}"
        # and the VALUES are the store's codes, not a pool of a finer level
        if k in (2, 3, 4):
            assert tgt[1].max() == 200 and tgt[2].max() == 140 and tgt[3].max() == 60


def test_a_no_data_code_in_a_dist_store_is_weight_zero_not_a_distance(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256)
    tg = pred_pyramid(tmp_path)
    sd = dist_group(tmp_path, "z_sdist.zarr", "sdist", nlev=1, value=0)         # code 0 = NO DATA
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg},{sd}"], exclude=[], rungs={2}, sym=False)
    ds._open_rungs()
    ctp = data.read_rung(ds.srcs[0]["ct_pyr"], 2, (0, 0, 0), P32, dtype=np.uint8)
    assert not (ds._rung_target(ds.srcs[0], 2, np.array([0, 0, 0]), ctp)[1][1] > 0).any()


def test_a_dist_group_does_not_change_the_sampling_bounds(tmp_path, monkeypatch):
    """`source_groups` bounds a source by its FIRST target group, so adding distance groups after the
    recto mask must leave the rung mix and the region walk exactly where they were."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256, nlev=5)
    tg = pred_pyramid(tmp_path, base=256, nlev=5)
    sd = dist_group(tmp_path, "b_sdist.zarr", "sdist", nlev=3)
    a = data.source_groups([f"{ct},{tg}"])[0]
    b = data.source_groups([f"{ct},{tg},{sd}"])[0]
    assert b["bounds"] is b["targets"]["recto"] and a["voxels"] == b["voxels"]
    assert data.rung_probs(a, P32) == data.rung_probs(b, P32)


# ------------------------------------------------------- 6. the pod's v2 region writer

def pod_mod(name):
    import importlib.util
    import sys
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cloud", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_the_pod_v2_compat_shims_match_usrm2():
    """`cloud/verso_core_v2.py` carries copies of the encoders because the pod runs a usrm2 snapshot from
    before Phase B and refreshing it under a live 70-hour job is not worth the risk. This is the test that
    keeps the copies honest -- here BOTH are importable."""
    from usrm2 import losses as L
    VC = pod_mod("verso_core_v2")
    a = {"channels": ["recto", "verso", "midline", "thickness", "nz", "ny", "nx", "logvar"], "cout_p": 2}
    assert VC._head_names(a) == P.head_names(a)
    assert (VC.TRACER_UNIT, VC.TRACER_OFF, VC.TRACER_CAP, VC.NORMAL_SCALE) == \
        (P.TRACER_UNIT, P.TRACER_OFF, P.TRACER_CAP, P.NORMAL_SCALE)
    assert VC.TMIN == L.TMIN
    rng = np.random.default_rng(0)
    d = rng.normal(0, 10, (6, 6, 6)).astype(np.float32)
    v = rng.random((6, 6, 6)) > 0.2
    assert np.array_equal(VC._enc_signed(d, v), P.enc_signed(d, v))
    assert np.array_equal(VC._enc_normal(d / 40, v), P.enc_normal(d / 40, v))
    assert np.array_equal(VC._u8(np.clip(d / 40 + 0.5, 0, 1)), P.u8(np.clip(d / 40 + 0.5, 0, 1)))
    assert np.allclose(VC._scharr3(d), P.scharr3(d))
    t = torch.from_numpy(d)[None, None]
    assert torch.allclose(VC._normals_from(t), L.normals_from(t))
    assert torch.allclose(VC._soft_thickness(t), L.soft_thickness(t))


def test_the_pod_v2_core_is_v1_for_a_checkpoint_without_field_heads(tmp_path, monkeypatch):
    """The production checkpoint is cout 2 with no distance channel, so v2 must take v1's code path and
    write the same bytes -- which is what the pod's `--compare-v1` run asserted (bit-identical, 27
    windows, region_4352_21760_15616)."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg = ct_pyramid(tmp_path), pred_pyramid(tmp_path)
    ck = T.train(tmp_path / "r", size="1m", steps=1, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                 eval_every=1, val_patches=1, device="cpu", ctx=(1,), rungs={2}, val_rungs=(2,),
                 stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)), verso=True,
                 teacher_regions=str(tmp_path / "tr"))
    VC = pod_mod("verso_core_v2")
    monkeypatch.setattr(VC, "VOL", ct)
    net = VC.Net(str(ck), torch.device("cpu"), window=P32)
    assert net.planes == [] and net.nplanes == 0 and net.plane_names == []
    out = net(torch.zeros(1, net.cin, P32, P32, P32))
    assert out.shape == (1, P32, P32, P32), "v1 returns ONE plane with no channel axis"


def test_the_pod_v2_core_emits_every_head_for_a_phase_b_checkpoint(tmp_path, monkeypatch):
    ckpt, ct = phase_b_ckpt(tmp_path, monkeypatch)
    VC = pod_mod("verso_core_v2")
    monkeypatch.setattr(VC, "VOL", ct)
    net = VC.Net(ckpt, torch.device("cpu"), window=P32)
    assert net.plane_names == ["recto", "verso", "midline", "thickness", "nz", "ny", "nx", "conf"]
    y = net(torch.zeros(1, net.cin, P32, P32, P32))
    assert y.shape == (1, 8, P32, P32, P32)
    i = net.plane_names.index("thickness")
    assert float(y[:, i].min()) >= VC.TMIN, "thickness is TMIN + softplus, so it cannot fall below TMIN"
    j = [net.plane_names.index(q) for q in ("nz", "ny", "nx")]
    assert torch.allclose(y[:, j].norm(dim=1), torch.ones(1, P32, P32, P32), atol=1e-3)


def test_the_pod_v2_field_stores_use_the_tracer_encodings_and_q(tmp_path, monkeypatch):
    """Every field store is q=0 (LOSSLESS) because its code 0 means NO DATA and its other codes are a
    distance or a normal component; the recto/verso probability stores stay q=8 and byte-comparable with
    every other prediction store (docs/unified_design.md 29.1)."""
    RV = pod_mod("verso_run_v2")
    rng = np.random.default_rng(0)
    n = 16
    fields = {"recto": rng.random((n, n, n)).astype(np.float32),
              "verso": rng.random((n, n, n)).astype(np.float32),
              "midline": rng.normal(0, 4, (n, n, n)).astype(np.float32),
              "thickness": np.full((n, n, n), 4.0, np.float32),
              "conf": rng.random((n, n, n)).astype(np.float32)}
    out = RV.field_stores(fields, "ckpt.pt", 15000, 1024, 256, 32)
    assert set(out) == {"recto", "verso", "surf_sdist", "nz", "ny", "nx", "gmag", "thickness", "conf"}
    for k, (v, enc, q, extra) in out.items():
        assert v.dtype == np.uint8 and v.shape == (n, n, n)
        assert q == (8 if k in ("recto", "verso") else 0), f"{k}: q must be 0 for a field"
        assert extra["done"] is True and extra["step"] == 15000
    # the distance is exported in the RECTO-FACE convention: d = m - t/2, then encoded
    d = fields["midline"] - 0.5 * fields["thickness"]
    want = P.enc_signed(*P.tracer_fields(fields["midline"], fields["thickness"])[:1],
                        valid=np.ones((n, n, n), bool) & (fields["recto"] > 0))
    assert np.array_equal(out["surf_sdist"][0], want)
    assert np.allclose(P.TRACER_UNIT * (out["surf_sdist"][0].astype(np.int16) - P.TRACER_OFF), d,
                       atol=P.TRACER_UNIT)


def test_the_pod_v2_writer_names_the_stores_beside_the_verso_one():
    RV = pod_mod("verso_run_v2")
    assert RV.name((1024, 2048, 3072)) == "region_1024_2048_3072"
    assert "VERSO face towards the RECTO" in RV.SIGN_WORDS
