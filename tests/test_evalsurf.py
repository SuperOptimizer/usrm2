import json

import numpy as np
import pytest

from usrm2 import evalsurf as E

tifffile = pytest.importorskip("tifffile")


def plane(tmp_path):
    """A synthetic tifxyz plane at x = 64, spanning z, y in [20, 100] with 2-voxel cells."""
    d = tmp_path / "seg" / "surf"
    d.mkdir(parents=True)
    zz, yy = np.meshgrid(np.arange(20, 102, 2, dtype=np.float32), np.arange(20, 102, 2, dtype=np.float32),
                         indexing="ij")
    for n, a in [("z", zz), ("y", yy), ("x", np.full_like(zz, 64.0))]:
        tifffile.imwrite(str(d / f"{n}.tif"), a)
    (d / "meta.json").write_text(json.dumps({"scale": [0.5, 0.5], "bbox": [[64, 20, 20], [64, 100, 100]]}))
    return str(tmp_path)


def test_plane_offset_by_two(tmp_path):
    ax = np.array([[0.0, 128.0], [0.0, 0.0], [0.0, 0.0]])  # scroll axis at y=x=0 -> normals point +x
    pts, nrm, counts = E.sites((0, 0, 0), (128, 128, 128), plane(tmp_path), ax=ax)
    assert len(pts) == 41 * 41 and list(counts) == ["surf"]
    assert np.allclose(nrm, np.array([0, 0, 1.0]), atol=1e-5)
    p = np.zeros((128, 128, 128), np.uint8)
    p[20:101, 20:101, 66] = 255  # a one-voxel band at +2 along the normal, over the meshed area
    m = E.metrics(p, (0, 0, 0), pts, nrm)
    assert m["recall@4"] == 1.0 and m["recall@2"] == 1.0 and m["recall@8"] == 1.0
    assert abs(m["offset_mean"] - 2.0) < 0.05 and m["offset_std"] < 0.05 and m["offset_le3"] == 1.0
    assert m["merge_runs"] == 1.0 and m["merge_frac"] == 0.0
    assert m["precision6"] == 1.0
