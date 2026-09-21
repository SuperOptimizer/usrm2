"""The CASCADE input channel (docs/unified_design.md section 22).

Channel order is [CT, ctx_1..ctx_n, CASCADE, scale, radial(3)]: the cascade channel sits right after the
context cubes and before the scale plane, so a 14-channel checkpoint warm-starts to bit-identical outputs.
Everything here is synthetic (the tiny pyramids of tests/test_rungs.py, 32^3 patches, CPU).
"""
import json

import numpy as np
import pytest
import torch

from usrm2 import data, model as M, predict as P, prep, train as T
from tests.test_rungs import P32, ct_pyramid, pred_pyramid, umbilicus


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)


def sources(tmp_path):
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    return ct, tg, [f"{ct},{tg}"]


def sample(tmp_path, cascade="mask", k=3, ctx=(1, 2), **kw):
    ds = data.Patches(patch=P32, stores=sources(tmp_path)[2], exclude=[], rungs={k}, ctx=ctx, sym=False,
                      air_keep=1.0, fg_keep=1.0, cascade=cascade, **kw)
    return ds, next(iter(ds))


# ------------------------------------------------------------------------------- the channel itself

def test_the_sample_carries_the_coarse_blocks_and_the_tenth_context_cube(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, item = sample(tmp_path, "mask")
    assert item["cm"].shape == (P32 // 2,) * 3 and item["cm"].dtype == torch.uint8
    assert "cx" not in item  # the mask source needs no extra cube
    _, item = sample(tmp_path, "self")
    assert item["cx"].shape == (1, P32, P32, P32) and item["cx"].dtype == torch.uint8
    assert item["lo1"].shape == (3,) and item["cyx1"].shape == (2, P32)


def test_mask_mode_is_the_2x_upsampled_coarse_target(tmp_path, monkeypatch):
    """Without noise the channel is exactly `model.up2x` of the rung-(k+1) target block."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, item = sample(tmp_path, "mask", k=3)
    b = prep.batch1(item)
    x = prep.prepare(b, torch.device("cpu"), cascade=prep.Cascade("mask", drop=0.0, noise=False))[0]
    assert x.shape[1] == 1 + 2 + 1 + 1 + 3  # CT + 2 ctx + CASCADE + scale + radial
    want = M.up2x(item["cm"][None, None].float() / 255.0, (P32,) * 3)[0, 0]
    assert torch.allclose(x[0, 3], want, atol=1e-6)     # channel 3 = right after CT + 2 context cubes
    assert torch.allclose(x[0, 4], torch.tensor((3 - 2) / 9))  # the scale plane still follows it
    # the coarse block is the pyramid's own rung-(k+1) level over the footprint
    tg = data.rungs(pred_pyramid(tmp_path))
    lo = item["lo"].numpy()
    ref = data.read_rung(tg, 4, lo // 2, np.array([P32 // 2] * 3), dtype=np.uint8)
    assert np.array_equal(item["cm"].numpy(), ref)


def test_the_channel_is_zero_at_rung_11_and_under_dropout(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, item = sample(tmp_path, "mask", k=3)
    b = prep.batch1(item)
    b["rung"] = torch.tensor([11])  # the top of the ladder: nothing above it to cascade from
    x = prep.prepare(b, torch.device("cpu"), cascade=prep.Cascade("mask", drop=0.0, noise=False))[0]
    assert torch.allclose(x[0, 3], torch.zeros(1))
    b["rung"] = torch.tensor([3])
    x = prep.prepare(b, torch.device("cpu"), cascade=prep.Cascade("mask", drop=1.0, noise=False))[0]
    assert torch.allclose(x[0, 3], torch.zeros(1))


def test_cascade_off_is_the_14_channel_path(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, item = sample(tmp_path, "off")
    assert "cm" not in item and "cx" not in item
    assert prep.shapes(item) == (1 + 2 + 4, 1)
    x = prep.prepare(prep.batch1(item), torch.device("cpu"))[0]
    assert x.shape[1] == 1 + 2 + 4


# ------------------------------------------------------------------------------- the cube symmetry

def test_sym_apply_t_handles_15_channels_and_still_rotates_the_radial_vector():
    torch.manual_seed(0)
    x = torch.zeros(1, 15, 8, 8, 8)
    x[0, :11] = torch.arange(11 * 512, dtype=torch.float32).reshape(11, 8, 8, 8)  # cubes + cascade + scale
    v = torch.nn.functional.normalize(torch.randn(3), dim=0)
    x[0, 12:] = v[:, None, None, None]
    tg = torch.zeros(1, 1, 8, 8, 8)
    tg[0, 0] = x[0, 0]
    for sym in range(48):
        y, t = prep.sym_apply_t(sym, x, tg)
        perm, flip = data.sym_decode(sym)
        # every non-vector channel is permuted and flipped exactly like the target
        assert torch.allclose(t[0, 0], y[0, 0])
        for c in range(1, 12):
            assert torch.equal(y[0, c], x[0, c].permute(tuple(int(q) for q in perm)).flip(
                [i for i, f in enumerate(flip) if f]))
        # the radial VECTOR is permuted and negated with the axes
        want = torch.tensor([(-1.0 if flip[i] else 1.0) * float(v[int(perm[i])]) for i in range(3)])
        assert torch.allclose(y[0, 12:, 0, 0, 0], want, atol=1e-6)
        assert torch.allclose(y[0, 12:].flatten(1).std(1), torch.zeros(3), atol=1e-6)


# ------------------------------------------------------------------------------- self mode

def test_self_mode_builds_the_rung_k_plus_1_input(tmp_path, monkeypatch):
    """The coarse input is what `prepare` would build for the rung-(k+1) cube over the same centre:
    CT = ctx_1, contexts = ctx_2..ctx_n plus the tenth cube, scale plane one rung up, its own cascade 0."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "NORM", (0.0, 1.0))  # z-score off: the channels are the planted constants
    ctx = (1, 2)
    _, item = sample(tmp_path, "self", k=3, ctx=ctx)
    b = prep.batch1(item)
    c = prep.Cascade("self", net=None)
    xc = c.coarse_input(b, 0, torch.device("cpu"), torch.float32)
    assert xc.shape == (1, 1 + len(ctx) + 1 + 1 + 3, P32, P32, P32)
    # the CT cube of the coarse input is ctx_1 of the fine one, and its contexts are the rest
    for j in range(len(ctx)):
        assert torch.allclose(xc[0, j], item["ct"][1 + j].float())
    assert torch.allclose(xc[0, len(ctx)], item["cx"][0].float())       # the tenth cube closes the stack
    assert torch.allclose(xc[0, len(ctx) + 1], torch.zeros(1))          # its own cascade channel: truncated
    assert torch.allclose(xc[0, len(ctx) + 2], torch.tensor((4 - 2) / 9.0))  # rung k+1 scale plane
    r = prep.radial_t(item["cyx1"][None], item["lo1"][None], (P32,) * 3)
    assert torch.allclose(xc[0, len(ctx) + 3:], r[0], atol=1e-6)

    # ... and it equals the input a rung-(k+1) SAMPLE over the same cube would produce (the training path)
    lo1 = item["lo1"].numpy()
    ct1 = data.read_rung(data.rungs(ct_pyramid(tmp_path)), 4, lo1, np.array([P32] * 3), dtype=np.uint8)
    assert np.array_equal(ct1, item["ct"][1].numpy())


def test_self_mode_runs_and_feeds_the_nets_own_coarse_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ctx = (1, 2)
    _, item = sample(tmp_path, "self", k=3, ctx=ctx)
    cin = 1 + len(ctx) + 1 + 1 + 3
    net = M.build("1m", verbose=False, cin=cin, cout=1)
    net.eval()
    c = prep.Cascade("self", drop=0.0, net=net)
    x = prep.prepare(prep.batch1(item), torch.device("cpu"), cascade=c)[0]
    with torch.no_grad():
        y = torch.sigmoid(net(c.coarse_input(prep.batch1(item), 0, torch.device("cpu"), torch.float32)))
    q = P32 // 4
    want = M.up2x(y[:, :1, q:q + P32 // 2, q:q + P32 // 2, q:q + P32 // 2], (P32,) * 3)[0, 0]
    assert torch.allclose(x[0, 1 + len(ctx)], want, atol=1e-5)
    assert float(x[0, 1 + len(ctx)].min()) >= 0 and float(x[0, 1 + len(ctx)].max()) <= 1


# ------------------------------------------------------------------------------- warm start

def test_warm_start_14_to_15_is_bit_identical_with_a_zero_cascade_channel():
    torch.manual_seed(0)
    old = M.build("1m", verbose=False, cin=14, cout=1, deep=1)
    src = T.warm_start(old.state_dict(), cin=15, cout=1, cascade=True, src_scale=True)
    w0, w1 = old.state_dict()["enc.0.0.weight"], src["enc.0.0.weight"]
    assert w1.shape[1] == 15
    assert torch.allclose(w1[:, :10], w0[:, :10])      # CT + 9 context cubes
    assert torch.allclose(w1[:, 10], torch.zeros(1))   # the new CASCADE channel starts at zero
    assert torch.allclose(w1[:, 11], w0[:, 10])        # the scale plane keeps its own weights
    assert torch.allclose(w1[:, 12:], w0[:, 11:])      # radial vector last
    new = M.build("1m", verbose=False, cin=15, cout=1, deep=1)
    assert not new.load_state_dict(src, strict=False).unexpected_keys
    old.eval(), new.eval()
    x = torch.randn(1, 14, 16, 16, 16)
    x15 = torch.cat([x[:, :10], torch.zeros(1, 1, 16, 16, 16), x[:, 10:]], 1)  # cascade channel zero
    with torch.no_grad():
        a, b = old(x), new(x15)
    # the zero-filled weights make the extra channel contribute nothing: the only difference is the order
    # the stem convolution accumulates its 15 (rather than 14) products in, a float32 ulp
    assert (a - b).abs().max() <= 1e-5 * a.abs().max()


def test_the_generic_warm_start_is_unchanged_for_13_to_14():
    old = M.build("1m", verbose=False, cin=13, cout=1)
    a = T.warm_start(old.state_dict(), cin=14, cout=1)["enc.0.0.weight"]
    b = T.warm_start(old.state_dict(), cin=14, cout=1, cascade=True, src_scale=False)["enc.0.0.weight"]
    assert torch.equal(a, b)


# ------------------------------------------------------------------------------- inference

def train_tiny(tmp_path, cascade, steps=2, ctx=(1,), k=3, **kw):
    ct, tg, lines = sources(tmp_path)
    out = tmp_path / f"run_{cascade}"
    ckpt = T.train(out, size="1m", steps=steps, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=steps, val_patches=2, device="cpu", ctx=ctx, rungs={k}, val_rungs=(k,),
                   stores=lines, val=((0, 0, 0), (32, 32, 32)), cascade=cascade, **kw)
    return ct, tg, ckpt


@pytest.mark.parametrize("mode", ["mask", "self", "mix"])
def test_a_training_step_runs_in_every_mode(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    _, _, ckpt = train_tiny(tmp_path, mode)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st["args"]["cin"] == 1 + 1 + 1 + 1 + 3  # CT + 1 ctx + CASCADE + scale + radial
    assert st["args"]["cascade"] == mode and st["args"]["cascade_drop"] == 0.1
    rec = json.loads((ckpt.parent / "eval.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(rec["bce"])


def test_top_down_inference_with_depth_0_equals_the_plain_path(tmp_path, monkeypatch):
    """`--cascade-depth 0` feeds a zero channel, which is what `--cascade off` at inference does too."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, _, ckpt = train_tiny(tmp_path, "mask")
    kw = dict(window=P32, halo=8, device="cpu", rung=3)
    a, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, cascade=False, **kw)
    b, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, cascade=True, cascade_depth=0, **kw)
    assert np.allclose(a, b, atol=1e-6)
    c, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, cascade=True, cascade_depth=2, **kw)
    assert c.shape == a.shape and np.isfinite(c).all()
    assert not np.allclose(a, c)  # the coarse pass actually reached the channel


def test_inference_follows_the_checkpoint_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, _, ckpt = train_tiny(tmp_path, "mask")
    kw = dict(window=P32, halo=8, device="cpu", rung=3)
    auto, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, **kw)                       # cascade=None
    on, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, cascade=True, **kw)
    assert np.allclose(auto, on)


def test_a_cascade_checkpoints_window_input_is_15_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct, _, ckpt = train_tiny(tmp_path, "mask", ctx=(1,))
    seen = []
    real = data.inputs
    monkeypatch.setattr(data, "inputs", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3, cascade_depth=1)
    assert seen and seen[0].shape[0] == 1 + 1 + 1 + 1 + 3
