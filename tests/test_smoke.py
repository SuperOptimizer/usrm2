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
    fn = lambda t: torch.sigmoid(t[0, 0] * 0.1)
    a = P.slide(fn, roi, 32, 4, torch.device("cpu"))
    b = P.slide_gpu(fn, roi, 32, 4, torch.device("cpu"), lambda c, o: P.zscore_t(c)[None])
    assert a.shape == b.shape and np.abs(a - b).max() < 2e-2 and np.abs(a - b).mean() < 1e-3 and (b[:, :8] == 0).all()  # fp16 accumulators


def test_slide_pads_rois_thinner_than_the_window():
    import numpy as np, torch
    from usrm2 import predict as P
    roi = np.full((8, 16, 16), 100, np.uint8)
    out = P.slide(lambda t: torch.ones(t.shape[2:]), roi, 16, 4, torch.device("cpu"))
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
