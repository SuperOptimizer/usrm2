import json
import math

import numpy as np
import zarr

from usrm2 import predict as P, train as T


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


def test_train_and_predict(tmp_path):
    ct, (tr, va) = make(tmp_path)
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
