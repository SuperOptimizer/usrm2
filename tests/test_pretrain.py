"""Masked-cube pretraining (usrm2/pretrain.py, docs/unified_design.md section 28).

Everything is synthetic and runs on the CPU: tiny pyramids (the `test_rungs` helpers), 32^3 patches, a few
steps. What is asserted is the contract with `train`: the checkpoint's trunk keys are the ones a fine-tuning
run builds, `train.warm_start` accepts it, the reconstruction head is NOT loaded into the segmentation head,
and the outputs after the warm start are finite.
"""
import json

import numpy as np
import pytest
import torch

from usrm2 import data, model as M, pretrain as PT, prep, train as T

from test_rungs import ct_pyramid, pred_pyramid, umbilicus  # noqa: F401  (same synthetic pyramids)

P32 = 32


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)


# --------------------------------------------------------------------------------- the masking

def test_block_mask_ratio_and_grid():
    """The achieved masked fraction lands in [lo, hi], and the mask is made of whole 8^3 blocks."""
    rng = np.random.default_rng(0)
    ct = torch.from_numpy(rng.normal(size=(4, 1, 32, 32, 32)).astype(np.float32))
    m, ratios, sheet = PT.block_mask(ct, block=8, lo=0.5, hi=0.75, sheet_p=0.5)
    assert m.shape == ct.shape and set(torch.unique(m).tolist()) <= {0.0, 1.0}
    frac = m.mean((1, 2, 3, 4))
    assert ((frac >= 0.5 - 1e-6) & (frac <= 0.75 + 1e-6)).all(), frac
    assert torch.allclose(frac, ratios.to(frac.dtype), atol=1e-6)  # whole blocks: block ratio == voxel ratio
    blocks = torch.nn.functional.avg_pool3d(m, 8)
    assert set(torch.unique(blocks).tolist()) <= {0.0, 1.0}  # every 8^3 block is all-masked or all-kept


def test_sheet_masking_prefers_foreground():
    """With sheet_p=1 the masked blocks sit on the foreground proxy; with sheet_p=0 they do not."""
    ct = torch.zeros(1, 1, 32, 32, 32)
    ct[:, :, :, :8] = 3.0  # a "sheet" slab: a quarter of the volume, well above any quantile of the rest
    sheet = PT.block_mask(ct, block=8, lo=0.25, hi=0.25, sheet_p=1.0)[0]
    plain = torch.stack([PT.block_mask(ct, block=8, lo=0.25, hi=0.25, sheet_p=0.0)[0] for _ in range(8)]).mean(0)
    on_sheet = float((sheet[:, :, :, :8]).sum() / sheet.sum())
    assert on_sheet > 0.9, on_sheet                       # structure-aware: the sheet is what disappears
    assert float((plain[:, :, :, :8]).sum() / plain.sum()) < 0.5   # uniform: a quarter of the mask, give or take


def test_masking_blanks_the_context_footprint():
    """The context channels must not hand the model a coarse copy of the masked CT."""
    x = torch.ones(1, 3, 16, 16, 16)          # CT + 2 context cubes (offsets 1, 2), no radial here
    x, tgt, m, _, _ = PT.mask_input(x.clone(), ctx=(1, 2), nimg=3, block=16, lo=1.0, hi=1.0, sheet_p=0.0)
    assert float(m.mean()) == 1.0 and float(x[:, 0].abs().max()) == 0.0
    assert float(x[:, 1, 4:12, 4:12, 4:12].abs().max()) == 0.0   # the central half of ctx_1
    assert float(x[:, 1, :4].abs().max()) == 1.0                 # ... and only that
    assert float(x[:, 2, 6:10, 6:10, 6:10].abs().max()) == 0.0   # the central quarter of ctx_2
    assert float(tgt.mean()) == 1.0                              # the target is the UNMASKED cube


def test_recon_loss_scores_masked_voxels_only():
    p, t = torch.zeros(1, 1, 4, 4, 4), torch.ones(1, 1, 4, 4, 4)
    m = torch.zeros_like(t)
    m[..., :2] = 1.0
    assert float(PT.recon_loss(p, t, m)) == pytest.approx(1.0)
    assert float(PT.recon_loss(p * 0 + 1, t, m)) == pytest.approx(0.0)  # right where it is scored


# ------------------------------------------------------------------- rungs that do not exist

def test_available_rungs_drops_rungs_no_scan_is_native_at(tmp_path, monkeypatch, capsys):
    """A 2.4 um mirror is native at rung 2: `--rungs 0-4` on it is rungs 2-4, said out loud."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path)
    lines, ks = PT.available_rungs([f"{ct},{tg}"], {0, 1, 2, 3, 4})
    assert ks == [2, 3, 4] and len(lines) == 1
    assert "not on disk anywhere" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        PT.available_rungs([f"{ct},{tg}"], {0, 1})


# ------------------------------------------------------------------------- the smoke run

def run3(tmp_path, **kw):
    ct = ct_pyramid(tmp_path)
    tg = pred_pyramid(tmp_path, attrs={"box": [[0, 0, 0], [128, 128, 128]]})
    out = tmp_path / "pre"
    ck = PT.pretrain(out, size="1m", steps=3, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                     eval_every=3, val_patches=1, device="cpu", ctx=(1, 2), rungs={2, 3},
                     stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)), **kw)
    return ck, out, ct, tg


def test_three_steps_and_the_checkpoint_warm_starts_train(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ck, out, ct, tg = run3(tmp_path)
    logs = [json.loads(l) for l in (out / "pretrain.jsonl").read_text().splitlines()]
    assert logs and logs[0]["cin"] == 1 + 2 + 1 + 4  # CT + 2 context + the zero cascade slot + scale + radial
    ev = [json.loads(l) for l in (out / "eval.jsonl").read_text().splitlines()]
    assert ev and np.isfinite(ev[-1]["recon"]) and ev[-1]["recon"] >= 0

    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["step"] == 3 and st["args"]["pretrain"] and st["args"]["scale_plane"]
    assert set(st["model"]) == set(st["ema"])
    # the head is renamed, the trunk is not
    assert not [k for k in st["ema"] if k.startswith("head.")]
    assert [k for k in st["ema"] if k.startswith("recon_head.")] == ["recon_head.weight", "recon_head.bias"]

    # the warm start a fine-tuning run does: same size, same cin, one segmentation head
    cin, cout = st["args"]["cin"], 1
    net = M.build("1m", verbose=False, cout=cout, cin=cin)
    src = T.warm_start(st["ema"], cin, cout, cascade=True, src_scale=True)
    miss = net.load_state_dict(src, strict=False)
    trunk = [k for k in net.state_dict() if k.split(".")[0] in ("enc", "down", "dec", "proj")]
    assert not [k for k in miss.missing_keys if k in trunk], miss.missing_keys
    assert set(miss.missing_keys) == {"head.weight", "head.bias"}        # the segmentation head starts fresh
    assert set(miss.unexpected_keys) == {"recon_head.weight", "recon_head.bias"}
    for k in trunk:  # the trunk really came from the pretrained checkpoint
        assert torch.equal(net.state_dict()[k], st["ema"][k])
    y = net(torch.randn(1, cin, P32, P32, P32))
    assert torch.isfinite(y).all() and y.shape == (1, cout, P32, P32, P32)


def test_loss_is_finite_and_the_masking_ratio_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ck, out, _, _ = run3(tmp_path, mask_lo=0.6, mask_hi=0.6, sheet_p=1.0)
    # step 3 is not a multiple of 20, so the per-step record is not written; the eval one always is
    ev = [json.loads(l) for l in (out / "eval.jsonl").read_text().splitlines()]
    assert all(np.isfinite(v) for k, v in ev[-1].items() if k.startswith("recon"))
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["args"]["mask_lo"] == 0.6 and st["args"]["sheet_p"] == 1.0
    assert all(torch.isfinite(v).all() for v in st["ema"].values() if v.is_floating_point())


def test_no_cascade_slot_gives_the_14_channel_stem(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ck, out, _, _ = run3(tmp_path, cascade_slot=False)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["args"]["cin"] == 1 + 2 + 4 and st["ema"]["enc.0.0.weight"].shape[1] == 7
    # and it still widens to a 15-channel cascade run, the new slot zeroed
    src = T.warm_start(st["ema"], 8, 1, cascade=True, src_scale=True)
    w = src["enc.0.0.weight"]
    assert w.shape[1] == 8 and float(w[:, 3].abs().max()) == 0.0   # [img x3, CASCADE(0), scale, radial]
    assert torch.equal(w[:, :3], st["ema"]["enc.0.0.weight"][:, :3])


def test_resume_continues_from_the_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ck, out, ct, tg = run3(tmp_path)
    ck = PT.pretrain(out, size="1m", steps=5, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                     eval_every=5, val_patches=1, device="cpu", ctx=(1, 2), rungs={2, 3}, resume=True,
                     stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)))
    st = torch.load(ck, map_location="cpu", weights_only=False)
    assert st["step"] == 5 and st["args"]["continued_from"] == 3
