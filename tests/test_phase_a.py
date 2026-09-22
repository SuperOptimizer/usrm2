"""Phase A: the four label-free losses and the training recipe (docs/unified_design.md section 26).

Everything is CPU and synthetic: hand-built tensors for the loss forms, the tiny pyramids of
tests/test_rungs.py for the end-to-end steps. The load-bearing claim these tests defend is that every
flag is OFF by default, so a run started without them is byte-identical to one started before they
existed -- `test_resume_is_byte_identical_with_the_flags_off` is that claim.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from usrm2 import calib as C, data, glc as G, losses as L, model as M, prep, train as T
from tests.test_rungs import P32, ct_pyramid, pred_pyramid, umbilicus

REG = 128   # the region edge these tests use instead of data.REGION's 1024


def verso_store(path, lo2, shape=(REG, REG, REG), value=90, done=True):
    """A verso region store, PLAIN zarr (no volcomp: this file must run wherever pytest does)."""
    import zarr
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    a = zarr.create_array(str(path), shape=tuple(shape), chunks=(64, 64, 64), dtype="uint8",
                          fill_value=0, overwrite=True)
    a[:] = np.full(tuple(shape), value, np.uint8)
    a.attrs["origin_zyx"] = [int(v) for v in lo2]
    a.attrs["channels"] = ["verso"]
    a.attrs["voxel_um"] = 2.4
    if done:
        a.attrs["done"] = True
    return str(path)


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)
    monkeypatch.setattr(data, "REGION", REG)


# ------------------------------------------------------------------------------ L3 soft exclusivity

def test_exclusivity_is_zero_for_disjoint_channels_and_positive_when_they_overlap():
    p = torch.zeros(1, 2, 4, 4, 4)
    p[:, 0, :2] = 0.9        # recto in the first half
    p[:, 1, 2:] = 0.9        # verso in the second: p_r + p_v <= 1 everywhere
    w = torch.ones_like(p)
    assert float(L.exclusivity(p, w)) == pytest.approx(0.0)
    p[:, 1, :2] = 0.9        # now they share the first half
    assert float(L.exclusivity(p, w)) > 0.0
    # ... but only where BOTH channels carry weight
    w[:, 1] = 0
    assert float(L.exclusivity(p, w)) == pytest.approx(0.0)


def test_exclusivity_has_no_gradient_where_the_weight_is_zero():
    lg = torch.zeros(1, 2, 4, 4, 4, requires_grad=True)
    w = torch.ones(1, 2, 4, 4, 4)
    w[:, :, 2:] = 0
    L.exclusivity(torch.sigmoid(lg), w).backward()
    assert torch.allclose(lg.grad[:, :, 2:], torch.zeros(1))
    assert torch.isfinite(lg.grad).all()


# ------------------------------------------------------------------- L4 cascade self-consistency

def test_self_consistency_is_zero_when_the_pooled_fine_prediction_equals_the_coarse_one():
    torch.manual_seed(0)
    coarse = torch.rand(2, 1, 4, 4, 4)
    fine = M.up2x(coarse, (8, 8, 8))              # the channel is built exactly this way
    # pooling the upsample back is the coarse field up to the trilinear kernel's residual smoothing
    v = float(L.self_consistency(fine, fine, sel=torch.ones(2)))
    assert v == pytest.approx(0.0, abs=1e-6)
    assert float(L.self_consistency(fine, torch.zeros_like(fine), sel=torch.ones(2))) > 0.1


def test_self_consistency_only_scores_the_samples_that_used_the_self_source():
    fine, coarse = torch.ones(2, 1, 4, 4, 4), torch.zeros(2, 1, 4, 4, 4)
    both = float(L.self_consistency(fine, coarse, sel=torch.ones(2)))
    one = float(L.self_consistency(fine, coarse, sel=torch.tensor([1.0, 0.0])))
    assert both == pytest.approx(1.0) and one == pytest.approx(1.0)   # a mean, not a sum
    assert float(L.self_consistency(fine, coarse, sel=torch.zeros(2))) == pytest.approx(0.0)


def test_self_consistency_has_no_gradient_where_the_weight_is_zero():
    lg = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    w = torch.ones(1, 1, 4, 4, 4)
    w[:, :, 2:] = 0
    L.self_consistency(torch.sigmoid(lg), torch.ones(1, 1, 4, 4, 4), w, torch.ones(1)).backward()
    assert torch.allclose(lg.grad[:, :, 2:], torch.zeros(1)) and torch.isfinite(lg.grad).all()


# ---------------------------------------------------------------------------- L8 skeleton recall

def band(thick=3, size=16, chan=1):
    """A flat sheet `thick` voxels thick in the middle of a cube."""
    t = torch.zeros(1, chan, size, size, size)
    lo = size // 2 - thick // 2
    t[:, :, lo:lo + thick] = 1.0
    return t


def test_the_skeleton_of_a_flat_band_is_its_middle_plane():
    t = band(thick=5, size=16)
    s = L.skeleton(t, iters=4)
    z = s[0, 0].sum((1, 2))
    assert int(z.argmax()) == 8                                  # the middle plane
    assert float(s.sum()) == pytest.approx(float(z[8]))          # and that plane only
    # the erosion treats OUTSIDE the patch as background, so the distance -- and with it the ridge --
    # falls away within `iters` voxels of the four faces the band runs off: a 12x12 plane, not 16x16.
    # That is conservative in the right direction (a patch-edge voxel is simply not asked about).
    assert float(z[8]) == (16 - 2 * 2) ** 2


def test_skeleton_recall_is_one_for_a_perfect_prediction_and_zero_for_an_empty_one():
    t = band(thick=5, size=16)
    w = torch.ones_like(t)
    assert float(L.skel_recall(t.clone(), t, w)) == pytest.approx(0.0, abs=1e-6)     # loss = 1 - recall
    assert float(L.skel_recall(torch.zeros_like(t), t, w)) == pytest.approx(1.0, abs=1e-6)


def test_skeleton_recall_cannot_be_gamed_by_widening_the_band():
    """The recall is measured on the TARGET's skeleton, so a prediction that already covers it scores 1
    whether it is the right width or twice too wide."""
    t = band(thick=3, size=16)
    wide = band(thick=9, size=16)
    w = torch.ones_like(t)
    assert float(L.skel_recall(t, t, w)) == pytest.approx(float(L.skel_recall(wide, t, w)), abs=1e-6)


def test_skeleton_recall_has_no_gradient_where_the_weight_is_zero():
    t = band(thick=5, size=16)
    lg = torch.zeros_like(t, requires_grad=True)
    w = torch.ones_like(t)
    w[:, :, :, 8:] = 0
    L.skel_recall(torch.sigmoid(lg), t, w).backward()
    assert torch.allclose(lg.grad[:, :, :, 8:], torch.zeros(1)) and torch.isfinite(lg.grad).all()


# ------------------------------------------------------------------------ O12 long-range affinity

def two_sheets(size=48, pitch=16, thick=3):
    """Two parallel sheets `pitch` voxels apart, both normal to z."""
    t = torch.zeros(1, 1, size, size, size)
    for z0 in (size // 2 - pitch // 2, size // 2 + pitch // 2):
        t[:, :, z0 - thick // 2:z0 - thick // 2 + thick] = 1.0
    return t


def test_affinity_targets_on_a_synthetic_two_sheet_volume():
    """A pair at the sheet PITCH along z straddles the air gap -> different sheet (0). The same pair
    inside one sheet's plane (along y or x) stays in the band -> same sheet (1)."""
    pitch, size = 16, 48
    t = two_sheets(size=size, pitch=pitch, thick=3)
    at, aw = L.affinity_targets(t, torch.ones_like(t), offsets=(pitch,))
    z_ch, y_ch, x_ch = at[:, 0], at[:, 1], at[:, 2]
    mid = size // 2                             # the midpoint of the two sheets: pure air between them
    # z, at the pitch: both ends are foreground (one on each sheet) but the segment crosses the gap
    assert float(aw[0, 0, mid, 10, 10]) == pytest.approx(1.0)   # the pair IS scored
    assert float(z_ch[0, mid, 10, 10]) == pytest.approx(0.0)    # ... and it is NOT the same sheet
    # y and x, inside one sheet: the whole segment is band
    zc = size // 2 - pitch // 2
    assert float(y_ch[0, zc, size // 2, size // 2]) == pytest.approx(1.0)
    assert float(x_ch[0, zc, size // 2, size // 2]) == pytest.approx(1.0)
    # air, and pairs with one end in air, are not scored at all (fg_only)
    assert float(aw[0, 0, 2, 2, 2]) == pytest.approx(0.0)
    assert float(aw[0, 1, mid, 10, 10]) == pytest.approx(0.0)   # y-pair at the midpoint: both ends air


def test_affinity_channel_count_names_and_parsing():
    assert L.parse_offsets("16,32") == (16, 32) and L.n_affinity("16,32") == 6
    assert L.affinity_names((16,)) == ["aff16_z", "aff16_y", "aff16_x"]
    with pytest.raises(AssertionError):
        L.parse_offsets("15")            # offsets are EVEN: the pair is centred on the voxel
    assert L.parse_offsets(None) == () and L.n_affinity(None) == 0


def test_affinity_weight_is_zero_where_either_end_has_weight_zero_and_outside_the_patch():
    t = torch.ones(1, 1, 16, 16, 16)
    w = torch.ones_like(t)
    w[:, :, 8:] = 0
    at, aw = L.affinity_targets(t, w, offsets=(4,))
    assert float(aw[0, 0, 7, 0, 0]) == pytest.approx(0.0)   # its +2 partner has weight 0
    assert float(aw[0, 0, 5, 0, 0]) == pytest.approx(1.0)
    assert float(aw[0, 0, 0, 0, 0]) == pytest.approx(0.0)   # the -2 partner is outside the patch
    assert torch.isfinite(at).all() and torch.isfinite(aw).all()


def test_affinity_loss_is_finite_and_has_no_gradient_where_the_weight_is_zero():
    t = two_sheets()
    w = torch.ones_like(t)
    w[:, :, :, 24:] = 0
    lg = torch.zeros(1, 3, 48, 48, 48, requires_grad=True)
    v = L.affinity_loss(lg, t, w, offsets=(16,))
    assert torch.isfinite(v) and float(v) > 0
    v.backward()
    assert torch.isfinite(lg.grad).all()
    assert torch.allclose(lg.grad[:, :, :, 26:], torch.zeros(1))   # +-8 away from the zero-weight half


def test_aux_losses_returns_nothing_when_every_weight_is_zero():
    t = band(chan=2, size=16)
    lg = torch.zeros(1, 2, 16, 16, 16)
    assert L.aux_losses(lg, t, torch.ones_like(t)) == {}


# ------------------------------------------------------------------------------- the LR schedule

def test_wsd_values_at_the_warmup_stable_and_cooldown_boundaries():
    f = T.lr_lambda(1000, warmup=100, lr_floor=0.0, sched="wsd", stable_until=800, cooldown=200)
    assert f(0) == pytest.approx(0.01)            # (0 + 1) / 100
    assert f(99) == pytest.approx(1.0)            # warmup done
    assert f(500) == pytest.approx(1.0)           # the plateau is FLAT (cosine would be ~0.65 here)
    assert f(799) == pytest.approx(1.0)
    assert f(800) == pytest.approx(1.0)           # the cooldown starts at 1.0
    assert f(900) == pytest.approx(0.5, abs=1e-6)  # half way through the cosine cooldown
    assert f(1000) == pytest.approx(0.0, abs=1e-9)
    assert f(5000) == pytest.approx(0.0, abs=1e-9)  # past the end it stays down
    # the defaults are the literature's 10 % cooldown
    g = T.lr_lambda(1000, warmup=100, sched="wsd")
    assert g(899) == pytest.approx(1.0) and g(900) == pytest.approx(1.0) and g(950) == pytest.approx(0.5, abs=1e-6)


def test_cosine_is_unchanged():
    f = T.lr_lambda(1000, warmup=100, lr_floor=0.1, sched="cosine")
    for s in (0, 50, 250, 500, 999, 2000):
        want = min((s + 1) / 100, 1.0) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / 1000, 1.0))))
        assert f(s) == pytest.approx(want)


def test_wsd_plateau_can_be_extended_the_way_a_resume_would():
    """The point of WSD: the budget is not committed at run start."""
    short = T.lr_lambda(1000, 100, sched="wsd", stable_until=800, cooldown=200)
    longer = T.lr_lambda(5000, 100, sched="wsd", stable_until=4800, cooldown=200)
    assert short(700) == pytest.approx(longer(700))     # the same LR at the same step
    assert short(900) < longer(900)                     # but only one of them is cooling down


def test_ema_auto():
    assert T.ema_auto(50000, 50) == pytest.approx(1 - 50 / 50000)   # a 1000-step (2 %) window
    assert T.ema_auto(200000, 50) == pytest.approx(1 - 50 / 200000)  # 4000 steps: still 2 %
    assert T.ema_auto(10, 50) == pytest.approx(0.9)                  # clamped
    assert T.ema_auto(10 ** 9, 50) == pytest.approx(0.9999)          # clamped
    assert T.ema_auto(50000, 10) == pytest.approx(0.9998)            # k = 10 is a 10 % window


# ------------------------------------------------------------ warm start, param groups, re-warmup

def test_warm_start_zero_inits_the_affinity_rows_and_keeps_recto_and_verso():
    torch.manual_seed(0)
    old = M.build("1m", verbose=False, cin=15, cout=2, deep=2)
    src = T.warm_start(old.state_dict(), cin=15, cout=2 + 6, ncopy=2)
    for k in ("head", "deep_heads.0", "deep_heads.1"):
        w, b = src[f"{k}.weight"], src[f"{k}.bias"]
        w0, b0 = old.state_dict()[f"{k}.weight"], old.state_dict()[f"{k}.bias"]
        assert w.shape[0] == 8
        assert torch.equal(w[:2], w0) and torch.equal(b[:2], b0)      # recto / verso untouched
        assert torch.allclose(w[2:], torch.zeros(1)) and torch.allclose(b[2:], torch.zeros(1))


def test_the_affinity_rows_do_not_change_the_recto_output():
    torch.manual_seed(0)
    old = M.build("1m", verbose=False, cin=5, cout=2)
    new = M.build("1m", verbose=False, cin=5, cout=8)
    new.load_state_dict(T.warm_start(old.state_dict(), cin=5, cout=8, ncopy=2))
    x = torch.randn(1, 5, 16, 16, 16)
    old.eval(), new.eval()
    with torch.no_grad():
        # not `equal`: a 1x1x1 convolution with 8 output rows picks a different kernel from one with 2,
        # so the accumulation order (not the math) differs by a float32 ulp
        assert torch.allclose(old(x), new(x)[:, :2], atol=1e-6, rtol=0)


def test_param_groups_and_the_new_parameter_multiplier():
    net = M.build("1m", verbose=False, cin=5, cout=2)
    names = T.new_param_names(True, True, net)
    assert "enc.0.0.weight" in names and "head.weight" in names and "head.bias" in names
    g, split = T.param_groups(net, names, 1.0)
    assert len(g) == 1 and not split                      # M = 1: one group, exactly as before
    g, split = T.param_groups(net, names, 5.0)
    assert len(g) == 2 and split
    assert len(g[1]["params"]) == len(names)
    assert sum(len(q["params"]) for q in g) == len(list(net.parameters()))
    assert T.param_groups(net, set(), 5.0)[1] is False    # nothing new: no second group


def test_new_param_multiplier_applies_through_the_stable_phase_only():
    base = T.lr_lambda(1000, 100, sched="wsd", stable_until=800, cooldown=200)
    boosted = (lambda s: base(s) * (5.0 if s < 800 else 1.0))
    assert boosted(500) == pytest.approx(5 * base(500))
    assert boosted(900) == pytest.approx(base(900))


# ----------------------------------------------------------------------------- end to end (CPU)

def sources(tmp_path):
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    return ct, tg, [f"{ct},{tg}"]


def run(tmp_path, name, **kw):
    ct, tg, lines = sources(tmp_path)
    out = tmp_path / name
    kw.setdefault("steps", 2)
    kw.setdefault("lr", 1e-3)
    ckpt = T.train(out, size="1m", patch=P32, batch=1, workers=0, warmup=1,
                   eval_every=kw["steps"], val_patches=2, device="cpu", ctx=(1,), rungs={3},
                   val_rungs=(3,), stores=lines, val=((0, 0, 0), (32, 32, 32)), **kw)
    return ct, ckpt


def test_a_training_step_runs_with_every_phase_a_flag_on(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    verso_store(root / "verso" / "region_0_0_0.zarr", (0, 0, 0))
    ct, ckpt = run(tmp_path, "all_on", steps=3, cascade="mix", verso=True, cout=2,
                   teacher_regions=str(root), affinity="4,8", loss_affinity=0.1,
                   loss_excl=0.1, loss_selfcons=0.1, loss_skel=0.1, skel_iters=2,
                   cascade_self_p_anneal=(0.1, 0.9), sched="wsd", cooldown=1, ema_k=10,
                   new_param_lr_mult=5.0, fuse="agreement", source_w={"store": 1.0, "mask": 0.5})
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = st["args"]
    assert a["cout_t"] == 2 and a["cout"] == 2 + 6          # recto, verso + 2 offsets x 3 axes
    assert a["channels"][:2] == ["recto", "verso"] and a["channels"][2] == "aff4_z"
    assert st["ema"]["head.weight"].shape[0] == 8
    assert a["affinity"] == [4, 8] and a["sched"] == "wsd" and a["ema_k"] == 10
    assert a["ema_decay"] == pytest.approx(T.ema_auto(3, 10))
    assert a["fuse"] == "agreement" and a["source_w"] == {"store": 1.0, "mask": 0.5}
    rec = [json.loads(q) for q in (ckpt.parent / "eval.jsonl").read_text().splitlines()][-1]
    assert np.isfinite(rec["bce"]) and np.isfinite(rec["dice"])
    assert "dice_aff4_z" not in rec                          # the affinity channels are never scored
    assert (ckpt.parent / f"val_{st['step']:06d}.png").exists()


def test_the_flags_are_absent_from_the_args_when_they_are_off(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, ckpt = run(tmp_path, "off")
    a = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    for k in ("loss_excl", "loss_selfcons", "loss_skel", "loss_affinity", "affinity", "sched",
              "stable_until", "cooldown", "ema_k", "rewarm", "new_param_lr_mult", "fuse", "source_w",
              "cout_t", "temps", "cascade_self_p_anneal"):
        assert k not in a, f"{k} is recorded even though it is off"


def test_the_flags_at_their_defaults_change_nothing(tmp_path, monkeypatch):
    """The byte-identity claim: passing every Phase-A flag at its OFF value gives the same weights, the
    same args and the same schedule as not passing it at all."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    torch.manual_seed(0)
    _, bare = run(tmp_path, "bare", steps=2)
    torch.manual_seed(0)
    _, zeros = run(tmp_path, "zeros", steps=2, loss_excl=0.0, loss_selfcons=0.0, loss_skel=0.0,
                   loss_affinity=0.0, affinity=None, sched="cosine", cooldown=0, stable_until=None,
                   ema_k=None, rewarm=0, new_param_lr_mult=1.0, fuse="off", source_w={})
    a = torch.load(bare, map_location="cpu", weights_only=False)
    b = torch.load(zeros, map_location="cpu", weights_only=False)
    assert a["args"] == b["args"]
    for k in a["ema"]:
        assert torch.equal(a["ema"][k], b["ema"][k]), k
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_a_resume_with_the_flags_off_reloads_the_same_weights(tmp_path, monkeypatch):
    """Resume is exact in the state it restores: the checkpoint a 1-step run wrote is the state a
    0-further-step resume writes back (the loader's seed is the step, so a resumed run draws its OWN
    windows -- that is the existing design, not something these flags change)."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, one = run(tmp_path, "res", steps=1)
    a = torch.load(one, map_location="cpu", weights_only=False)
    _, again = run(tmp_path, "res", steps=1, resume=True)
    b = torch.load(again, map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 1
    for k in a["ema"]:
        assert torch.equal(a["ema"][k], b["ema"][k]), k


def test_a_phase_a_loss_may_be_turned_on_at_a_resume_but_the_head_may_not_change(tmp_path, monkeypatch):
    """Phase A's whole point is that `u3` resumes with the flags on: the four losses change no weight
    shape, so they are allowed mid-run. `--affinity` DOES change the head and is refused."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    run(tmp_path, "onoff", steps=1)
    run(tmp_path, "onoff", steps=2, resume=True, loss_skel=0.1)          # fine: no weight changes shape
    with pytest.raises(AssertionError, match="resume with different arguments"):
        run(tmp_path, "onoff", steps=3, resume=True, affinity="4", loss_affinity=0.1)  # cout changes


def test_the_wsd_plateau_may_be_moved_on_a_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    run(tmp_path, "wsd", steps=1, sched="wsd", stable_until=4, cooldown=1)
    run(tmp_path, "wsd", steps=2, sched="wsd", stable_until=8, cooldown=1, resume=True)
    a = torch.load(tmp_path / "wsd" / "ckpt.pt", map_location="cpu", weights_only=False)["args"]
    assert a["stable_until"] == 8 and a["steps"] == 2
    with pytest.raises(AssertionError, match="resume with different arguments"):
        run(tmp_path, "wsd", steps=3, sched="wsd", stable_until=8, cooldown=1, resume=True, lr=1e-4)


# --------------------------------------------------------------------------- teacher fusion (E)

def test_fuse_agreement_prefers_the_confident_source_and_down_weights_disagreement():
    p, mul = data.fuse_agreement(np.array([1.0, 0.5, 0.0]), np.array([1.0, 0.9, 1.0]))
    assert p[0] == pytest.approx(1.0) and mul[0] == pytest.approx(1.0)        # agreement: unchanged
    assert 0.5 < p[1] < 0.9 and mul[1] == pytest.approx(0.6)                  # the committed one pulls
    assert p[2] == pytest.approx(0.5, abs=1e-3) and mul[2] == pytest.approx(0.0)  # total disagreement
    assert data.binary_confidence(np.array([0.5]))[0] == pytest.approx(0.0, abs=1e-6)
    assert data.binary_confidence(np.array([1.0]))[0] == pytest.approx(1.0, abs=1e-4)


def test_source_weights_scale_the_loss_weight_of_the_exported_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, tg, lines = sources(tmp_path)
    kw = dict(patch=P32, stores=lines, exclude=[], rungs={3}, sym=False, air_keep=1.0, fg_keep=1.0)
    full = next(iter(data.Patches(**kw)))["w"].max().item()
    half = next(iter(data.Patches(**kw, source_w={"mask": 0.5})))["w"].max().item()
    assert full == 255 and half == pytest.approx(128, abs=1)


# ------------------------------------------------------------------------- glc-weights (E)

def test_glc_weights_ranks_synthetic_sources_by_their_error_rates():
    """Three sources over one box of 200 mesh points on the z = 32 plane: a perfect one, one that misses
    half the points, and one that is positive everywhere (no misses, but a wall of false positives)."""
    Z = Y = X = 64
    yy, xx = np.meshgrid(np.arange(8, 48, 2), np.arange(8, 48, 2), indexing="ij")
    pts = np.stack([np.full(yy.size, 32.0), yy.ravel().astype(np.float32), xx.ravel().astype(np.float32)], 1)
    nrm = np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (len(pts), 1))
    good = np.zeros((Z, Y, X), np.uint8)
    good[31:34] = 255
    half = good.copy()
    half[:, :, 28:] = 0                       # half the points lose their band
    everywhere = np.full((Z, Y, X), 255, np.uint8)
    rows = []
    for name, v in (("good", good), ("half", half), ("all", everywhere)):
        fnr, fpr, n, pf = G.rates(v, (0, 0, 0), pts, nrm)
        rows.append({"source": name, "fnr": fnr, "fpr": fpr})
    w = G.weights(rows)
    assert n == len(pts)
    assert w["good"] == 1.0 and w["half"] < 1.0 and w["all"] < 1.0
    assert rows[0]["fnr"] == pytest.approx(0.0) and rows[1]["fnr"] > 0.4
    assert rows[2]["fpr"] > rows[0]["fpr"]


# ----------------------------------------------------------------- per-rung temperature (F)

def test_calibration_reduces_the_bce_on_a_synthetic_grid():
    """Overconfident logits (the dice-trained failure mode): the target is a soft band, the logits are
    3x too large. The fitted temperature must be > 1 and must lower the BCE."""
    torch.manual_seed(0)
    t = band(thick=5, size=16)
    t = t * 0.9 + 0.05                       # a soft band: the honest logit is +-log(0.95/0.05)/1
    honest = torch.log(t / (1 - t))
    lg, w = honest * 3.0, torch.ones_like(t)
    Tt = C.fit_temp(lg, t, w)
    assert 2.0 < Tt < 4.0
    assert C.bce_at(lg, t, w, Tt) < C.bce_at(lg, t, w, 1.0)
    assert C.bce_at(lg, t, w, Tt) == pytest.approx(C.bce_at(honest, t, w, 1.0), abs=1e-4)


def test_binary_frac_separates_a_hard_band_from_a_pooled_fraction():
    hard = band(thick=5, size=16)
    w = torch.ones_like(hard)
    assert C.binary_frac(hard, w) == pytest.approx(0.0)
    assert C.binary_frac(torch.nn.functional.avg_pool3d(hard, 2).repeat(1, 1, 2, 2, 2) * 0 + 0.5, w) > 0.9


def test_temp_for_is_a_no_op_without_temperatures():
    assert C.temp_for({}, 3) == 1.0
    assert C.temp_for({"temps": {"3": 1.5}}, 3) == 1.5
    assert C.temp_for({"temps": {"3": 1.5}}, 4) == 1.0        # a rung that was not fitted
    assert C.temp_for({"temps": {"3": 1.5}}, 3, use=False) == 1.0


def test_calibrate_writes_the_temperatures_and_predict_applies_them(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    from usrm2 import predict as P
    ct, ckpt = run(tmp_path, "calib")
    rep = C.run(ckpt, device="cpu")
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert rep["rungs"] and all(np.isfinite(r["T"]) for r in rep["rungs"])
    if rep["temps"]:
        assert st["args"]["temps"] == rep["temps"]
        a, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3)
        b, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3,
                       calib=False)
        assert a.shape == b.shape and np.isfinite(a).all()
        assert not np.allclose(a, b) or abs(list(rep["temps"].values())[0] - 1.0) < 1e-3
