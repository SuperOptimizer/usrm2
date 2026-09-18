import json
import math

import numpy as np
import zarr

from usrm2 import data, predict as P, train as T


def make(tmp_path):
    rng = np.random.default_rng(0)
    ct = zarr.create_array(str(tmp_path / "ct.zarr"), shape=(256, 256, 256), chunks=(64, 64, 64),
                           dtype="uint8", fill_value=0)
    ct[:] = rng.integers(20, 200, (256, 256, 256), dtype=np.uint8)
    paths = []
    for name, org, n in [("tr", (0, 0, 0), 128), ("va", (128, 128, 128), 64)]:
        a = zarr.create_array(str(tmp_path / f"{name}.zarr"), shape=(1, n, n, n), chunks=(1, 32, 32, 32),
                              dtype="uint8", fill_value=0)
        a[:] = rng.integers(0, 255, (1, n, n, n), dtype=np.uint8)
        a.attrs.update({"channels": ["recto"], "voxel_um": 2.4, "origin_zyx": list(org), "scale": 1.0})
        paths.append(str(tmp_path / f"{name}.zarr"))
    return str(tmp_path / "ct.zarr"), paths


def test_train_and_predict(tmp_path, monkeypatch):
    ct, (tr, va) = make(tmp_path)
    umb = tmp_path / "umb.json"  # a straight scroll axis through the middle of the synthetic volume
    umb.write_text(json.dumps({"control_points": [{"z": 0, "y": 128, "x": 128}, {"z": 256, "y": 128, "x": 128}]}))
    monkeypatch.setattr(data, "UMBILICUS", str(umb))
    out = tmp_path / "run"
    ckpt = T.train(out, size="1m", steps=3, patch=32, batch=1, lr=1e-3, workers=0, warmup=2,
                   eval_every=3, val_patches=4, device="cpu", ct=ct, stores=[tr], val=va)
    rec = json.loads((out / "eval.jsonl").read_text().splitlines()[-1])
    assert math.isfinite(rec["bce"]) and math.isfinite(rec["dice"]) and 0 <= rec["mae"] <= 1
    o = P.predict(ckpt, ct, 0, 0, 0, 64, 64, 64, str(tmp_path / "pred.zarr"),
                  window=32, halo=8, device="cpu", volcomp=False)
    a = zarr.open(o, mode="r")
    assert a.shape == (1, 64, 64, 64) and a.dtype == np.uint8
    assert a.attrs["origin_zyx"] == [0, 0, 0] and a.attrs["channels"] == ["recto"]
    assert np.isfinite(a[:]).all()


def test_slide_gpu_matches_slide():
    import numpy as np, torch
    from usrm2 import predict as P
    rng = np.random.default_rng(0)
    roi = rng.integers(1, 255, (40, 56, 48), dtype=np.uint8)
    roi[:, :8] = 0  # masked strip
    fn = lambda t: torch.sigmoid(t[:, 0] * 0.1)
    a = P.slide(fn, roi, 32, 4, torch.device("cpu"))
    for batch, streams in ((1, 1), (3, 1), (1, 2), (2, 3)):
        b = P.slide_gpu(fn, roi, 32, 4, torch.device("cpu"), lambda c, o: P.zscore_t(c)[None], batch=batch, streams=streams)
        assert a.shape == b.shape and np.abs(a - b).max() < 2e-2 and np.abs(a - b).mean() < 1e-3 and (b[:, :8] == 0).all()  # fp16 accumulators


def test_slide_pads_rois_thinner_than_the_window():
    import numpy as np, torch
    from usrm2 import predict as P
    roi = np.full((8, 16, 16), 100, np.uint8)
    out = P.slide(lambda t: torch.ones((t.shape[0],) + t.shape[2:]), roi, 16, 4, torch.device("cpu"))
    assert out.shape == roi.shape and np.allclose(out, 1.0)


def test_axis_default_follows_the_global(tmp_path, monkeypatch):
    import json
    from usrm2 import data
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"control_points": [{"z": 0, "y": 5, "x": 7}, {"z": 10, "y": 5, "x": 7}]}))
    monkeypatch.setattr(data, "UMBILICUS", str(p))
    assert data.axis()[1][0] == 5


def test_put_handles_both_store_layouts(tmp_path):
    import numpy as np, zarr
    from usrm2 import predict as P
    u = np.full((4, 4, 4), 9, np.uint8)
    for shape, chunks in (((8, 8, 8), (4, 4, 4)), ((1, 8, 8, 8), (1, 4, 4, 4))):
        a = zarr.create_array(str(tmp_path / f"s{len(shape)}.zarr"), shape=shape, chunks=chunks, dtype="uint8", fill_value=0)
        P.put(a, u, 0, 4, 4)
        v = a[0] if a.ndim == 4 else a[:]
        assert v[:4, 4:, 4:].min() == 9 and v[:4, :4, :4].max() == 0


def test_teacher_boxes_shards_partition_the_sequence(tmp_path, monkeypatch):
    import numpy as np
    from usrm2 import teacher, data
    ct = np.full((512, 512, 512), 100, np.uint8)
    class A:  # a stand-in volume with a level-2 twin
        shape = ct.shape
        def __getitem__(self, s): return ct[s]
    lo = np.full((128, 128, 128), 100, np.uint8)
    class L:
        def __getitem__(self, s): return lo[s]
    monkeypatch.setattr(data, "open_zarr", lambda p: L() if p.endswith("/2") else A())
    got = {}
    def runner(out, *a, **k):
        got.setdefault(k["tag"], []).append(out)
    for i in range(3):
        teacher.boxes(str(tmp_path), n=7, size=(256, 256, 256), seed=1, volume="v/0", exclude=None, runner=runner, shard=(i, 3), tag=i)
    allb = sum(got.values(), [])
    assert len(allb) == 7 and len(set(allb)) == 7 and [len(got[i]) for i in range(3)] == [3, 2, 2]


def test_procs_argv_strip(monkeypatch):
    import sys
    from usrm2 import cli
    calls = []
    class P:
        def __init__(self, args): calls.append(args)
        def wait(self): return 0
    monkeypatch.setattr(cli, "subprocess", type("S", (), {"Popen": staticmethod(lambda args: P(args))})) if hasattr(cli, "subprocess") else None
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda args: P(args))
    monkeypatch.setattr(sys, "argv", ["usrm2", "teacher-boxes", "/tmp/o", "--n", "4", "--procs", "3", "--gpu-acc"])
    try:
        cli.main()
    except SystemExit:
        pass
    assert len(calls) == 3 and all("--procs" not in c and "3" not in c[:-2] and c[-3:-2] == ["--shard"] for c in calls)


def test_ridge_weighted_loss_and_lr_floor_and_global_norm(monkeypatch):
    import math, numpy as np, torch
    from usrm2 import train as T, data
    logit = torch.zeros(1, 1, 4, 4, 4); tgt = torch.zeros(1, 1, 4, 4, 4); tgt[..., :2] = 1.0
    b0, _ = T.losses(logit, tgt); b1, _ = T.losses(logit, tgt, ridge_w=3.0)
    assert abs(b0.item() - math.log(2)) < 1e-5 and abs(b1.item() - math.log(2)) < 1e-5  # uniform logits: same value, weights only reshuffle
    logit[..., :2] = 3.0  # right on the core
    assert T.losses(logit, tgt, ridge_w=3.0)[0] < T.losses(logit, tgt)[0]  # core weight rewards committing there
    monkeypatch.setattr(data, "NORM", (100.0, 50.0))
    x = np.full((2, 2, 2), 150, np.uint8)
    assert np.allclose(data.zscore(x), 1.0)
    monkeypatch.setattr(data, "NORM", None)
    assert np.allclose(data.zscore(x), 0.0)


def test_continuity_metric_prefers_unbroken_bands(tmp_path):
    import json, numpy as np, tifffile
    from usrm2 import evalsurf as E
    Z, X = np.meshgrid(np.arange(4, 60, 4, dtype=np.float32), np.arange(4, 60, 4, dtype=np.float32), indexing="ij")
    g = np.stack([Z, np.full_like(Z, 32.0), X], -1)
    d = tmp_path / "s" / "surf"; d.mkdir(parents=True)
    for i, c in enumerate("zyx"):
        tifffile.imwrite(d / f"{c}.tif", g[..., i])
    json.dump({"bbox": [[0, 0, 0], [64, 64, 64]]}, open(d / "meta.json", "w"))
    ax = np.array([[0.0, 100.0], [-1000.0, -1000.0], [32.0, 32.0]])
    full = np.zeros((64, 64, 64), np.uint8); full[:, 30:35, :] = 255
    broken = full.copy(); broken[:, :, ::8] = 0  # a gap every 8 voxels along x
    cf, cb = (E.continuity(v, (0, 0, 0), (64, 64, 64), tifxyz=str(tmp_path), ax=ax, min_pts=50) for v in (full, broken))
    assert cf["continuity"] > 0.95 and cb["continuity"] < cf["continuity"] - 0.3 and cb["hit_frac"] < cf["hit_frac"]


def test_context_cubes_are_centred_and_pooled(tmp_path, monkeypatch):
    import numpy as np, zarr
    from usrm2 import data
    # a tiny pyramid: level 0 64^3 with a bright block, level 1 = 2x mean pool, level 2 = 4x
    v0 = np.zeros((64, 64, 64), np.uint8); v0[16:48, 16:48, 16:48] = 200
    root = tmp_path / "vol.zarr"; root.mkdir()
    for l, v in enumerate([v0, v0.reshape(32, 2, 32, 2, 32, 2).mean((1, 3, 5)).astype(np.uint8),
                           v0.reshape(16, 4, 16, 4, 16, 4).mean((1, 3, 5)).astype(np.uint8)]):
        a = zarr.create_array(str(root / str(l)), shape=v.shape, chunks=v.shape, dtype="uint8"); a[:] = v
    monkeypatch.setattr(data, "CTX_CACHE", {})
    cubes = data.context(str(root / "0"), (16, 16, 16), (32, 32, 32), ctx=(1, 2, 3))
    assert [c.shape for c in cubes] == [(32, 32, 32)] * 3
    assert cubes[0][8:24, 8:24, 8:24].min() == 200 and cubes[0][:4].max() == 0      # 4.8um cube: the block is the middle half
    assert cubes[1][12:20, 12:20, 12:20].min() == 200 and cubes[1][:8].max() == 0  # 9.6um: quarter
    assert cubes[2].shape == (32, 32, 32) and cubes[2][14:18, 14:18, 14:18].min() == 200  # 19.2um pooled from level 2
    x = data.inputs(v0[:32, :32, :32], np.zeros((3, 32, 32, 32), np.float32), cubes)
    assert x.shape == (7, 32, 32, 32)


def test_augment_and_apply_keep_extra_channels_and_flip_only_the_vector():
    import numpy as np, torch
    from usrm2 import data, aug as A
    rng = np.random.default_rng(0)
    x = np.zeros((7, 8, 8, 8), np.float32); x[:4] = rng.random((4, 8, 8, 8)); x[6] = 1.0  # radial = +x
    t = np.zeros((1, 8, 8, 8), np.float32)
    y, _ = data.augment(rng, x, t)
    assert y.shape == x.shape and np.allclose(np.linalg.norm(y[4:], axis=0), 1.0)
    xb = torch.from_numpy(x)[None]
    yb, _ = A.apply(xb.clone(), torch.from_numpy(t)[None], A.get("all3"))
    assert yb.shape == xb.shape and torch.isfinite(yb).all()
    n = yb[0, 4:].norm(dim=0); assert ((n - 1).abs() < 1e-3).float().mean() > 0.9  # still unit vectors


def test_warm_start_widens_the_first_conv_without_changing_the_output():
    import torch
    from usrm2 import model as M
    old = M.build("1m", verbose=False, cin=4); new = M.build("1m", verbose=False, cin=7)
    src = old.state_dict(); w = src["enc.0.0.weight"]
    w2 = torch.zeros(w.shape[0], 7, *w.shape[2:]); w2[:, :1], w2[:, 4:] = w[:, :1], w[:, 1:]
    src["enc.0.0.weight"] = w2; new.load_state_dict(src)
    x = torch.randn(1, 7, 16, 16, 16); x4 = torch.cat([x[:, :1], x[:, 4:]], 1)
    old.eval(); new.eval()
    with torch.no_grad():
        assert torch.allclose(old(x4), new(x), atol=1e-5)


def test_stores_file_is_reread_when_it_grows(tmp_path, monkeypatch):
    import os, time
    ct, (tr, va) = make(tmp_path)
    umb = tmp_path / "umb.json"
    umb.write_text(json.dumps({"control_points": [{"z": 0, "y": 128, "x": 128}, {"z": 256, "y": 128, "x": 128}]}))
    monkeypatch.setattr(data, "UMBILICUS", str(umb))
    f = tmp_path / "groups.txt"
    f.write_text(tr + "\n")
    ds = data.Patches(patch=32, ct=ct, stores_file=str(f), exclude=va, air_keep=1.0, fg_keep=1.0, recheck=3)
    it = iter(ds)
    for _ in range(3):
        next(it)
    assert len(ds.paths) == 1
    time.sleep(0.05)
    f.write_text(tr + "\n" + va + "\n")  # a second group appears (the val store, just as more data here)
    os.utime(f, None)
    for _ in range(6):
        next(it)
    assert len(ds.paths) == 2 and ds.exclude == [va]
