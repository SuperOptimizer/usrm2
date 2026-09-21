"""The VERSO OUTPUT CHANNEL (docs/unified_design.md section 23).

ONE model, one head: the final 1x1x1 convolution grows from cout 1 to cout 2 -- channel 0 recto, channel 1
verso -- and the deep-supervision heads follow. The verso target has no pyramid; it exists only as region
stores (`<root>/verso/region_<z>_<y>_<x>.zarr`), so wherever no finished store covers a voxel the verso
channel's WEIGHT is 0 and the sample trains recto alone.

Everything here is synthetic: the tiny pyramids of tests/test_rungs.py, a 128^3 region (data.REGION
monkeypatched), 32^3 patches, CPU.
"""
import asyncio
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from usrm2 import data, model as M, predict as P, prep, stream as S, train as T
from tests.test_rungs import P32, ct_pyramid, pred_pyramid, umbilicus

REG = 128  # the region edge these tests use instead of 1024


@pytest.fixture(autouse=True)
def clean_caches(monkeypatch):
    monkeypatch.setattr(data, "CTX_CACHE", {})
    monkeypatch.setattr(data, "CHUNK_INDEX", {})
    monkeypatch.setattr(data, "NORM", None)
    monkeypatch.setattr(data, "REGION", REG)


def verso_store(path, lo2, shape=(REG, REG, REG), value=None, done=True):
    """A verso region store exactly as the pod writes it: `predict.out_array` with channels ["verso"],
    volcomp q8, one shard (so the published objects are `zarr.json` and `c/0/0/0`), `done` in the attrs."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    a = P.out_array(str(path), tuple(shape), tuple(int(v) for v in lo2), volcomp=True, rung=2,
                    channels=["verso"])
    v = np.full(shape, 90, np.uint8) if value is None else np.asarray(value, np.uint8)
    a[:] = v
    if done:
        a.attrs["done"] = True
    return str(path)


def patches(tmp_path, root, rungs={2, 3}, verso=True, **kw):
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs=rungs, sym=False,
                      verso=verso, verso_regions=str(root), **kw)
    ds._open_rungs()
    return ds


# --------------------------------------------------------------------------------- the warm start

def test_warm_start_1_to_2_copies_the_recto_filter_into_the_verso_channel():
    """head j <- source head j mod n: with one source head both output channels start as the recto one, and
    the recto channel's output is unchanged (a 1x1x1 head: channel 0 does not see channel 1 at all)."""
    torch.manual_seed(0)
    old = M.build("1m", verbose=False, cin=14, cout=1, deep=2)
    src = T.warm_start(old.state_dict(), cin=14, cout=2)
    for k in ("head", "deep_heads.0", "deep_heads.1"):
        w, b = src[f"{k}.weight"], src[f"{k}.bias"]
        w0, b0 = old.state_dict()[f"{k}.weight"], old.state_dict()[f"{k}.bias"]
        assert w.shape[0] == 2 and b.shape[0] == 2
        assert torch.equal(w[0], w0[0]) and torch.equal(w[1], w0[0])   # verso starts as a copy of recto
        assert torch.equal(b[0], b0[0]) and torch.equal(b[1], b0[0])
    new = M.build("1m", verbose=False, cin=14, cout=2, deep=2)
    assert not new.load_state_dict(src, strict=False).unexpected_keys
    old.eval(), new.eval()
    x = torch.randn(1, 14, 16, 16, 16)
    with torch.no_grad():
        a, b = old(x), new(x)
    assert b.shape[1] == 2
    # nothing on the recto path changed. The head is a 1x1x1 convolution, so channel 0 is the same dot
    # product either way, but a 2-output convolution accumulates it in a different order: the difference is
    # one float32 ulp (measured 1.2e-7 relative), not the ~1e-3 a changed weight would give.
    assert (a[:, 0] - b[:, 0]).abs().max() <= 1e-6 * a.abs().max()
    assert torch.equal(b[:, 0], b[:, 1])          # and verso starts out saying exactly what recto says


def test_warm_start_grows_the_cascade_channel_and_the_verso_output_together():
    """The A100 restart does both in one go: 14 -> 15 input channels and cout 1 -> 2."""
    torch.manual_seed(1)
    old = M.build("1m", verbose=False, cin=14, cout=1, deep=1)
    src = T.warm_start(old.state_dict(), cin=15, cout=2, cascade=True, src_scale=True)
    w0, w1 = old.state_dict()["enc.0.0.weight"], src["enc.0.0.weight"]
    assert w1.shape[1] == 15 and src["head.weight"].shape[0] == 2
    assert torch.allclose(w1[:, 10], torch.zeros(1))   # the cascade slot is zero
    assert torch.allclose(w1[:, 11], w0[:, 10])        # the scale plane keeps its weights
    new = M.build("1m", verbose=False, cin=15, cout=2, deep=1)
    assert not new.load_state_dict(src, strict=False).unexpected_keys
    old.eval(), new.eval()
    x = torch.randn(1, 14, 16, 16, 16)
    x15 = torch.cat([x[:, :10], torch.zeros(1, 1, 16, 16, 16), x[:, 10:]], 1)
    with torch.no_grad():
        a, b = old(x), new(x15)
    # a float32 ulp: the stem convolution accumulates 15 products instead of 14 (see section 22)
    assert (a[:, 0] - b[:, 0]).abs().max() <= 1e-5 * a.abs().max()
    assert torch.equal(b[:, 0], b[:, 1])  # noqa: E501


# --------------------------------------------------------------------------------- targets and weights

def test_without_a_verso_store_the_verso_channel_is_weight_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ds = patches(tmp_path, tmp_path / "treg")
    assert ds.channels[-1] == data.VERSO and ds.nrecto == len(ds.channels) - 1
    s = ds.srcs[0]
    assert ds._verso_store(s, 2, (0, 0, 0)) is None
    ct = data.read_rung(s["ct_pyr"], 2, (0, 0, 0), ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(s, 2, (0, 0, 0), ct, verso=None)
    assert tg.shape[0] == 2 and w.shape[0] == 2
    assert int(w[0].min()) > 0                       # recto is unchanged
    assert not w[1].any() and not tg[1].any()        # verso says nothing at all


def test_a_published_verso_store_fills_the_channel_and_its_inside_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    verso_store(root / "verso" / "region_0_0_0.zarr", (0, 0, 0))
    ds = patches(tmp_path, root)
    s = ds.srcs[0]
    p = ds._verso_store(s, 2, (0, 0, 0))
    assert p and p.endswith("verso/region_0_0_0.zarr")
    ct = data.read_rung(s["ct_pyr"], 2, (0, 0, 0), ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(s, 2, (0, 0, 0), ct, verso=p)
    assert int(tg[1].min()) == int(tg[1].max()) == 90 and bool((w[1] > 0).all())
    assert int(w[0].min()) > 0 and int(tg[0].max()) == 160  # recto still the exported mask pyramid
    # the store only covers region 0: a window past its far edge has no verso store at all
    assert ds._verso_store(s, 2, (REG, 0, 0)) is None
    # ... and a window straddling the region boundary falls back too
    assert ds._verso_store(s, 2, (REG - P32 // 2, 0, 0)) is None
    # a store covering only part of a window: `inside` is what carries the weight
    ds2 = patches(tmp_path, root)
    q = verso_store(tmp_path / "half" / "verso" / "region_0_0_0.zarr", (0, 0, 0), shape=(REG, REG, REG))
    a = data.open_zarr(q)
    v, ins = data.read_teacher(a, 2, (REG - P32 // 2, 0, 0), ds2.patch)
    assert ins[:P32 // 2].all() and not ins[P32 // 2:].any()
    assert int(v[:P32 // 2].max()) == 90 and int(v[P32 // 2:].max()) == 0


def test_rung_3_verso_is_the_2x_pool_of_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    g = np.zeros((REG, REG, REG), np.uint8)
    g[:] = (np.arange(REG, dtype=np.int64)[None, None, :] % 4 * 60).astype(np.uint8)  # not constant
    verso_store(root / "verso" / "region_0_0_0.zarr", (0, 0, 0), value=g)
    ds = patches(tmp_path, root, rungs={3})
    s = ds.srcs[0]
    lo3 = np.array([8, 8, 8])
    p = ds._verso_store(s, 3, lo3)
    assert p, "rung 3 must come from the same store (its 2x pool)"
    ct = data.read_rung(s["ct_pyr"], 3, lo3, ds.patch, dtype=np.uint8)
    tg, w = ds._rung_target(s, 3, lo3, ct, verso=p)
    lo2 = lo3 * 2
    a = data.open_zarr(p)   # read the block back from the store: volcomp q8 is lossy, g is not what is there
    blk = np.asarray(a[lo2[0]:lo2[0] + 2 * P32, lo2[1]:lo2[1] + 2 * P32, lo2[2]:lo2[2] + 2 * P32], np.uint8)
    assert blk.min() != blk.max()
    assert np.array_equal(tg[1], data.pool2(blk)) and bool((w[1] > 0).all())
    assert ds._verso_store(s, 4, (4, 4, 4)) is None, "no verso source above rung 3 yet"


def test_a_missing_store_is_re_probed_after_the_ttl(tmp_path, monkeypatch):
    """The pod publishes while the run trains: a store that was not there must be picked up later."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    ds = patches(tmp_path, root)
    s = ds.srcs[0]
    assert ds._verso_store(s, 2, (0, 0, 0)) is None
    verso_store(root / "verso" / "region_0_0_0.zarr", (0, 0, 0))
    assert ds._verso_store(s, 2, (0, 0, 0)) is None, "the miss is cached for TTL seconds"
    monkeypatch.setattr(data, "TSTORE_TTL", 0.0)
    assert ds._verso_store(s, 2, (0, 0, 0)) is not None


def test_the_verso_channel_never_changes_which_windows_are_drawn(tmp_path, monkeypatch):
    """The foreground / density rejection rules look at the recto channels only, so a cout=2 run draws
    exactly the windows a cout=1 run would (and a stream queue stays replayable by either)."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    for z in (0, REG):
        for y in (0, REG):
            for x in (0, REG):
                verso_store(root / "verso" / f"region_{z}_{y}_{x}.zarr", (z, y, x),
                            value=np.zeros((REG,) * 3, np.uint8))
    a = patches(tmp_path, root, verso=False)
    b = patches(tmp_path, root, verso=True)
    ra, rb = np.random.default_rng(3), np.random.default_rng(3)
    da = [a._rung_draw(ra, build=False)[0] for _ in range(40)]
    db = [b._rung_draw(rb, build=False)[0] for _ in range(40)]
    assert [None if d is None else (d["s"], d["k"], d["lo"]) for d in da] == \
           [None if d is None else (d["s"], d["k"], d["lo"]) for d in db]
    assert any(d is not None and "v" in d for d in db), "no window recorded its verso store"


# --------------------------------------------------------------------------------- the losses

def test_an_all_zero_weight_channel_is_finite_and_carries_no_gradient():
    torch.manual_seed(0)
    logit = torch.zeros(2, 2, 4, 4, 4, requires_grad=True)
    tgt = torch.zeros(2, 2, 4, 4, 4)
    tgt[:, 0, :2] = 1.0
    w = torch.ones_like(tgt)
    w[:, 1] = 0.0                                  # no verso store anywhere in this batch
    bce, dice = T.losses(logit, tgt, w=w)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    (bce + dice).backward()
    assert torch.allclose(logit.grad[:, 1], torch.zeros(1))  # the verso head row gets nothing
    assert logit.grad[:, 0].abs().sum() > 0
    # and the recto loss is exactly what a one-channel run would have seen: the dead channel does not
    # halve the dice by being averaged in
    l1 = torch.zeros(2, 1, 4, 4, 4, requires_grad=True)
    b1, d1 = T.losses(l1, tgt[:, :1], w=w[:, :1])
    assert float(bce) == pytest.approx(float(b1), rel=1e-6)
    assert float(dice) == pytest.approx(float(d1), rel=1e-6)


def test_deep_losses_stay_finite_with_a_dead_channel():
    logits = [torch.zeros(1, 2, 8, 8, 8, requires_grad=True), torch.zeros(1, 2, 4, 4, 4, requires_grad=True)]
    tgt = torch.ones(1, 2, 8, 8, 8)
    w = torch.ones_like(tgt)
    w[:, 1] = 0.0
    bce, dice = T.deep_losses(logits, tgt, w=w)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    (bce + dice).backward()
    for lg in logits:
        assert torch.allclose(lg.grad[:, 1], torch.zeros(1))
    w0 = torch.zeros_like(w)                       # nothing at all (the blank-patch aug)
    b0, d0 = T.deep_losses([l.detach() for l in logits], tgt, w=w0)
    assert torch.isfinite(b0) and torch.isfinite(d0) and float(b0) == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------------- end to end

def train_two_channel(tmp_path, root, steps=2, cascade="mix", **kw):
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    out = tmp_path / f"run_{cascade}"
    ckpt = T.train(out, size="1m", steps=steps, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
                   eval_every=steps, val_patches=2, device="cpu", ctx=(1,), rungs={3}, val_rungs=(3,),
                   stores=[f"{ct},{tg}"], val=((0, 0, 0), (32, 32, 32)), cascade=cascade,
                   teacher_regions=str(root), **kw)
    return ct, ckpt


def test_a_two_channel_training_step_runs_with_the_cascade(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    verso_store(root / "verso" / "region_0_0_0.zarr", (0, 0, 0))
    ct, ckpt = train_two_channel(tmp_path, root, verso=True, cout=2)
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st["args"]["cout"] == 2 and st["args"]["channels"][-1] == "verso"
    assert st["args"]["verso"] is True and st["args"]["cascade"] == "mix"
    assert st["ema"]["head.weight"].shape[0] == 2
    rec = json.loads((ckpt.parent / "eval.jsonl").read_text().splitlines()[-1])
    assert np.isfinite(rec["bce"]) and np.isfinite(rec["dice"])
    assert "dice_recto" in rec and "dice_verso" in rec   # the val box is covered by the store
    assert (ckpt.parent / f"val_{st['step']:06d}.png").exists()
    # the two output channels come out of predict by name
    a, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3, head="recto")
    b, _ = P.probs(ckpt, ct, 0, 0, 0, P32, P32, P32, window=P32, halo=8, device="cpu", rung=3, head="verso")
    assert a.shape == b.shape == (P32,) * 3 and np.isfinite(a).all() and np.isfinite(b).all()


def test_cout_mismatch_and_a_cout_1_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    root = tmp_path / "treg"
    with pytest.raises(AssertionError, match="--verso is what adds"):
        train_two_channel(tmp_path, root, cascade="off", cout=2)
    ct, ckpt = train_two_channel(tmp_path, root, cascade="off")  # an ordinary recto run
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st["args"]["cout"] == 1 and "verso" not in st["args"]
    ct2 = ct_pyramid(tmp_path, base=256, nlev=4)
    tg2 = pred_pyramid(tmp_path, base=256, nlev=4)
    T.train(ckpt.parent, size="1m", steps=4, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1,
            eval_every=4, val_patches=2, device="cpu", ctx=(1,), rungs={3}, val_rungs=(3,),
            stores=[f"{ct2},{tg2}"], val=((0, 0, 0), (32, 32, 32)), cascade="off",
            teacher_regions=str(root), resume=True)   # a cout=1 run resumes unchanged
    st2 = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st2["step"] == 4 and st2["args"]["cout"] == 1


def test_predict_head_verso_needs_a_two_channel_checkpoint():
    with pytest.raises(AssertionError, match="radial-sign"):
        P.resolve_head("verso", {"channels": ["recto"], "cout": 1})
    assert P.resolve_head("verso", {"channels": ["recto", "verso"], "cout": 2}) == 1
    assert P.resolve_head("recto", {"channels": ["recto", "verso"], "cout": 2}) == 0
    assert P.resolve_head("1", {}) == 1 and P.resolve_head("mean", {}) == "mean"
    assert P.resolve_head("all", {}) == "all"


# --------------------------------------------------------------------------------- the planner's fetch

@pytest.fixture
def published(tmp_path):
    """An HTTP origin serving the published verso region stores; yields (url root, its directory)."""
    import socket
    import urllib.request
    root = tmp_path / "pub"
    root.mkdir()
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                            "--directory", str(root)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5).read(1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    else:
        srv.kill()
        pytest.fail("http.server did not start")
    try:
        yield f"http://127.0.0.1:{port}", root
    finally:
        srv.kill()
        srv.wait()


def planner(tmp_path, url, local_root):
    f = tmp_path / "stores.txt"
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    f.write_text(f"{ct},{tg}\n")
    pl = S.Planner(stores_file=str(f), queue=str(tmp_path / "q"), patch=P32, rungs={2, 3}, workers=1,
                   teacher_regions=str(local_root), verso=True, verso_url=url, val=None)
    pl.ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2, 3}, sym=False,
                         verso=True, verso_regions=str(local_root))
    pl.ds._open_rungs()
    return pl


def probe(pl, k=2, lo=(0, 0, 0)):
    async def go():
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            pl.f = S.Fetcher(sess, 4)
            await pl.fetch_verso(pl.ds.srcs[0], k, lo)
    asyncio.run(go())


def test_the_planner_fetches_a_published_verso_store(tmp_path, published, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    url, pub = published
    verso_store(pub / "region_0_0_0.zarr", (0, 0, 0))
    local = tmp_path / "treg"
    pl = planner(tmp_path, url, local)
    probe(pl)
    d = local / "verso" / "region_0_0_0.zarr"
    assert (d / "zarr.json").exists() and (d / "c" / "0" / "0" / "0").exists()
    assert pl.verso_probe[str(d)][0] == "yes" and pl.verso_have == 1 and pl.verso_bytes > 0
    # the sampler now finds it, and the descriptor names it
    assert pl.ds._verso_store(pl.ds.srcs[0], 2, (0, 0, 0)) == str(d)
    probe(pl)                                   # a second window in the same region costs one dict lookup
    assert pl.f.requests == 0 and pl.verso_have == 1


def test_an_unpublished_verso_store_is_a_404_and_is_retried_later(tmp_path, published, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    url, pub = published
    local = tmp_path / "treg"
    pl = planner(tmp_path, url, local)
    probe(pl)
    d = local / "verso" / "region_0_0_0.zarr"
    assert not d.exists() and pl.verso_probe[str(d)][0] == "no" and pl.verso_have == 0
    assert pl.ds._verso_store(pl.ds.srcs[0], 2, (0, 0, 0)) is None
    assert not (local / "verso" / "region_0_0_0.zarr" / "zarr.json.absent").exists()  # no marker: it expires
    verso_store(pub / "region_0_0_0.zarr", (0, 0, 0))   # the pod publishes it
    probe(pl)
    assert pl.verso_have == 0, "the miss is remembered for VERSO_TTL"
    monkeypatch.setattr(S, "VERSO_TTL", 0.0)
    probe(pl)
    assert (d / "c" / "0" / "0" / "0").exists() and pl.verso_have == 1


def test_a_half_written_store_is_not_used(tmp_path, published, monkeypatch):
    """`done` is what says the pod finished; without it the store must not be left where a loader opens it."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    url, pub = published
    verso_store(pub / "region_0_0_0.zarr", (0, 0, 0), done=False)
    local = tmp_path / "treg"
    pl = planner(tmp_path, url, local)
    probe(pl)
    d = local / "verso" / "region_0_0_0.zarr"
    assert not (d / "zarr.json").exists() and pl.verso_probe[str(d)][0] == "no"


def test_the_planners_meta_carries_the_verso_settings(tmp_path, published, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    url, _ = published
    local = tmp_path / "treg"
    pl = planner(tmp_path, url, local)
    m = pl._meta(["a,b"])
    assert m["verso"] is True and m["verso_url"] == url and m["verso_regions"] == str(local)
    assert m["channels"][-1] == "verso"
