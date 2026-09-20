"""Streaming: the planner reproduces the sampler, the buffer lands where read_rung expects it, the
consumer replays the queue across W workers, waits, evicts and resumes.

The origin is a real HTTP server (`python -m http.server`) over a temp directory laid out like the public
volcomp tree, and `data.LOCAL_VOLUMES` / `data.STREAM_VOLUMES` are pointed at an empty mirror and at it.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import torch

from usrm2 import data, stream as S
from tests.test_rungs import ct_pyramid, pred_pyramid, umbilicus

P32 = 32
SCROLL = "PHercTest"


@pytest.fixture
def origin(tmp_path):
    """A served volcomp tree + an empty local mirror. Yields (mirror CT base, mirror target base)."""
    root = tmp_path / "origin"
    (root / SCROLL / "volumes").mkdir(parents=True)
    (root / SCROLL / "representations" / "predictions" / "surfaces").mkdir(parents=True)
    ct = ct_pyramid(root / SCROLL / "volumes", nlev=4)
    tg = pred_pyramid(root / SCROLL / "representations" / "predictions" / "surfaces", nlev=4)
    import socket
    import urllib.request
    with socket.socket() as sk:  # a free port, then wait for the server to answer on it
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
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    old = data.LOCAL_VOLUMES, data.STREAM_VOLUMES
    data.LOCAL_VOLUMES, data.STREAM_VOLUMES = str(mirror), f"http://127.0.0.1:{port}"
    try:
        yield (f"{mirror}/{SCROLL}/{os.path.basename(ct)}",
               f"{mirror}/{SCROLL}/representations/predictions/surfaces/{os.path.basename(tg)}",
               str(root), ct, tg)
    finally:
        data.LOCAL_VOLUMES, data.STREAM_VOLUMES = old
        srv.kill()
        srv.wait()


def stores(tmp_path, ct, tg, name="stores.txt"):
    p = tmp_path / name
    p.write_text(f"{ct},{tg}\n")
    return str(p)


@pytest.fixture
def per_window(monkeypatch):
    """Force the per-window chunk path: these pyramids are small enough that `data.full_level` would
    otherwise keep every level whole (which is right, but then nothing is fetched per window)."""
    monkeypatch.setattr(data, "CACHE_VOX", 1)
    monkeypatch.setattr(data, "CACHE_BUDGET", 1)


def run_plan(tmp_path, ct, tg, queue, limit, workers=2, ctx=(1, 2), **kw):
    S.plan(stores_file=stores(tmp_path, ct, tg), queue=str(queue), patch=P32, rungs={2, 3}, seed=0,
           workers=workers, ahead=10 ** 6, cache_gb=10 ** 6, ctx=ctx, limit=limit, jobs=8, report=10 ** 6,
           val=None, **kw)


# ------------------------------------------------------------------ URL <-> mirror

def test_remote_maps_both_subtrees(monkeypatch):
    monkeypatch.setattr(data, "LOCAL_VOLUMES", "/m")
    monkeypatch.setattr(data, "STREAM_VOLUMES", "https://h/volcomp")
    assert data.remote("/m/S/v.zarr/0/c/1/2/3") == "https://h/volcomp/S/volumes/v.zarr/0/c/1/2/3"
    assert data.remote("/m/S/representations/predictions/surfaces/p.zarr/2.4") == \
        "https://h/volcomp/S/representations/predictions/surfaces/p.zarr/2.4"
    for p in ("/m/S/v.zarr/0", "/m/S/representations/predictions/surfaces/p.zarr/2.4"):
        assert data.local(data.remote(p)) == data.remote(p)  # nothing is mirrored: the URL stays


# ------------------------------------------------------------------ the planner

def test_the_planner_draws_what_the_direct_sampler_draws(tmp_path, origin, monkeypatch):
    """Same seed, same rejection rules: the queue is the sequence the loader would have sampled."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, oct_, otg = origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=24, workers=2)
    recs = [json.loads(l) for l in open(q / S.QUEUE)]
    assert [r["i"] for r in recs] == list(range(24))

    # the same sampler, straight off the origin directory (everything local, nothing streamed)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    want = {0: [], 1: []}
    for w in range(2):
        ds = data.Patches(patch=P32, stores=[f"{oct_},{otg}"], exclude=[], rungs={2, 3}, ctx=(1, 2), seed=0)
        ds._open_rungs()
        rng = np.random.default_rng(0 + 1000 * w)
        while len(want[w]) < 12:
            d = ds._rung_draw(rng, build=False)[0]
            if d is not None:
                want[w].append(d)
    for r in recs:
        d = want[r["i"] % 2][r["i"] // 2]
        assert (r["k"], r["lo"], r["y"], r["s"]) == (d["k"], d["lo"], d["y"], d["s"])


def test_the_buffer_lands_where_read_rung_expects_it(tmp_path, origin, monkeypatch, per_window):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "NORM", (0.0, 1.0))
    mct, mtg, _, oct_, otg = origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=8, workers=1)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1, 2), stream=str(q))
    ds._open()
    rec = json.loads(open(q / S.QUEUE).readline())
    assert rec["c"], "no per-window chunks were recorded"
    for di, key in rec["c"]:
        assert S.have(f"{ds.smeta['dirs'][di]}/{key}")
    item = ds._rung_build(rec)
    k = int(item["rung"])
    assert torch.allclose(item["ct"][0].float(), torch.tensor(100.0 + 10 * (k - 2)))  # the planted CT
    assert int(item["tgt"].max()) == 220 - 30 * k                                     # the planted target


def test_replay_across_workers_has_no_gaps_or_duplicates(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=12, workers=3)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    seen = []
    for w in range(3):  # what each worker would take (the DataLoader round-robins them in this order)
        ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1, 2), stream=str(q))
        ds._open()
        recs = [json.loads(l) for l in open(q / S.QUEUE)]
        seen += [r["i"] for r in recs if r["i"] % 3 == w]
    assert sorted(seen) == list(range(12)) and len(set(seen)) == 12

    dl = torch.utils.data.DataLoader(  # and the real loader, 3 worker processes, in queue order
        data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1, 2), stream=str(q)),
        batch_size=1, num_workers=3, multiprocessing_context="forkserver")
    got = [int(b["idx"]) for b, _ in zip(dl, range(9))]
    assert got == list(range(9))


def test_the_consumer_waits_for_the_planner(tmp_path, origin, monkeypatch):
    """Start the consumer first: it blocks on the queue and its wait is reported in milliseconds."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=2, workers=1)  # 2 entries, then the planner stops
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1, 2), stream=str(q))
    out, done = [], threading.Event()

    def consume():
        for item in ds:
            out.append(item)
            if len(out) == 3:
                break
        done.set()

    th = threading.Thread(target=consume, daemon=True)
    th.start()
    for _ in range(300):  # the third window is not planned yet: the consumer must be waiting, not skipping
        if len(out) >= 2:
            break
        time.sleep(0.02)
    assert len(out) == 2 and not done.is_set()
    time.sleep(0.4)
    run_plan(tmp_path, mct, mtg, q, limit=3, workers=1)  # resumes at index 2 and appends one more
    done.wait(30)
    assert len(out) == 3 and [int(v["idx"]) for v in out] == [0, 1, 2]
    assert float(out[2]["wait"]) > 100  # it waited (ms) for the planner, and says so


def test_resume_continues_the_index_and_the_rng(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q1, q2 = tmp_path / "q1", tmp_path / "q2"
    run_plan(tmp_path, mct, mtg, q1, limit=16, workers=2)
    run_plan(tmp_path, mct, mtg, q2, limit=8, workers=2)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    run_plan(tmp_path, mct, mtg, q2, limit=16, workers=2)  # a fresh process would do the same
    a = [json.loads(l) for l in open(q1 / S.QUEUE)]
    b = [json.loads(l) for l in open(q2 / S.QUEUE)]
    assert len(b) == 16
    assert [(r["i"], r["k"], r["lo"], r["y"]) for r in a] == [(r["i"], r["k"], r["lo"], r["y"]) for r in b]


def test_eviction_respects_the_budget_and_spares_unconsumed_windows(tmp_path, origin, monkeypatch, per_window):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=30, workers=1)
    recs = [json.loads(l) for l in open(q / S.QUEUE)]
    dirs = json.load(open(q / S.META))["dirs"]
    json.dump({"i": 9, "margin": 0}, open(q / S.CONSUMED, "w"))  # the trainer has consumed 0..9

    pl = S.Planner(stores_file=stores(tmp_path, mct, mtg), queue=str(q), patch=P32, rungs={2, 3}, workers=1,
                   ctx=(1, 2), cache_gb=0)  # a zero budget: everything evictable must go
    pl.qf = str(q / S.QUEUE)
    pl.dirs, pl.dir_ix = dirs, {d: i for i, d in enumerate(dirs)}
    pl.resume()
    before = pl.cache_bytes
    assert before > 0
    pl.evict()
    assert pl.cache_bytes < before and pl.evicted > 0
    alive = {f"{dirs[di]}/{k}" for r in recs if r["i"] > 9 for di, k in r["c"]}
    for p in alive:
        assert S.have(p), f"{p}: a chunk of an unconsumed window was evicted"
    gone = {f"{dirs[di]}/{k}" for r in recs if r["i"] <= 9 for di, k in r["c"]} - alive
    assert any(not os.path.exists(p) for p in gone)


def test_a_404_is_recorded_as_absent_and_never_refetched(tmp_path, origin, monkeypatch):
    """A key the origin does not serve is air: a zero-length marker, and `have` is satisfied by it."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, root, _, otg = origin
    shutil.rmtree(pathlib.Path(otg) / "2.4" / "c" / "0")  # drop the first z row of shards on the origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=20, workers=1)
    marks = list(pathlib.Path(mtg).rglob("*.absent"))
    assert marks, "no absent marker was written"
    assert all(m.stat().st_size == 0 for m in marks)
    assert S.have(str(marks[0])[:-len(".absent")])


def test_require_targets_rejects_a_window_with_no_exported_target(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, otg = origin
    shutil.rmtree(pathlib.Path(otg) / "2.4")
    (pathlib.Path(otg) / "2.4").mkdir()
    shutil.copy(pathlib.Path(otg) / "4.8" / "zarr.json", pathlib.Path(otg) / "2.4" / "zarr.json")
    # rung 2 now has metadata but not one shard on the origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=10, workers=1, require_targets=True)
    recs = [json.loads(l) for l in open(q / S.QUEUE)]
    assert recs and all(r["k"] != 2 for r in recs)


# ------------------------------------------------------------------ chunk arithmetic

def test_rung_need_covers_exactly_what_read_rung_reads(tmp_path):
    ct = ct_pyramid(tmp_path, base=256, nlev=3)
    pyr = data.rungs(ct)
    old = data.CACHE_VOX
    try:
        data.CACHE_VOX = 1  # nothing is small enough to be kept whole: the per-window path
        arr, keys, whole = S.rung_need(pyr, 2, (32, 64, 96), P32)
        assert not whole and arr is pyr[2]
        assert keys == [(1, 2, 3)]                                  # 32^3 chunks
        arr, keys, whole = S.rung_need(pyr, 4, (0, 0, 0), P32)      # above the top: pooled from rung 4's source
        assert arr is pyr[max(r for r in pyr if r <= 4)]
        assert keys and all(len(k) == 3 for k in keys)
        data.CACHE_VOX = 1 << 30
        assert S.rung_need(pyr, 2, (0, 0, 0), P32)[2] is True       # small enough: the whole level
    finally:
        data.CACHE_VOX = old


# ------------------------------------------------------------------ end to end

def test_train_over_a_streamed_queue(tmp_path, origin, monkeypatch):
    """`usrm2 train --stream DIR`: the run consumes the queue, logs its wait and reports what it consumed."""
    from usrm2 import train as T
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    S.plan(stores_file=stores(tmp_path, mct, mtg), queue=str(q), patch=P32, rungs={2, 3}, seed=0, workers=1,
           ahead=10 ** 6, cache_gb=10 ** 6, ctx=(1,), limit=40, jobs=8, report=10 ** 6,
           val="0,0,0,64,64,64", val_rungs=(2,), val_patches=2)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    out = tmp_path / "run"
    T.train(out, size="1m", steps=20, patch=P32, batch=1, lr=1e-3, workers=0, warmup=1, eval_every=20,
            val_patches=2, device="cpu", ctx=(1,), rungs={2, 3}, val_rungs=(2,), stream=str(q),
            stores=[f"{mct},{mtg}"], val="0,0,0,64,64,64")
    logs = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    step20 = [q_ for q_ in logs if q_.get("step") == 20 and "vox_s" in q_][0]
    assert "stream_wait_ms" in step20 and step20["stream_idx"] >= 19
    assert json.load(open(q / S.CONSUMED))["i"] >= 19


# ------------------------------------------------------------------ sharded levels (the real layout)

def sharded_pyramid(root, name, nlev=3, base=128, chunks=16, shards=32, pred=False, seed=0):
    """A pyramid whose levels are SHARDED zarr v3 arrays, as the exported volcomp levels are (1024^3 shards
    of 128^3 inner chunks). The data is random, so a partially fetched shard cannot pass by accident."""
    import zarr
    root = pathlib.Path(root) / name
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lv = []
    for l in range(nlev):
        n, um = base >> l, data.rung_um(2 + l)
        nm = f"{um:g}" if pred else str(l)
        lv.append({"path": nm, "um": um})
        a = zarr.create_array(str(root / nm), shape=(n, n, n), chunks=(chunks,) * 3, shards=(shards,) * 3,
                              dtype="uint8", fill_value=0, overwrite=True)
        a[:] = rng.integers(1, 255, (n, n, n), dtype=np.uint8)
    meta = {"zarr_format": 3, "node_type": "group", "attributes": {
        "ome": {"version": "0.5", "multiscales": [{"version": "0.5", "name": name, "type": "mean",
                "axes": [{"name": q, "type": "space", "unit": "micrometer"} for q in "zyx"],
                "datasets": [{"path": q["path"], "coordinateTransformations":
                              [{"type": "scale", "scale": [q["um"]] * 3}]} for q in lv]}]},
        "volcomp": {"rung_voxel_size_um": 2.4}}}
    (root / "zarr.json").write_text(json.dumps(meta))
    return str(root)


RANGE_SERVER = """
import http.server, os, re, sys, functools
class H(http.server.SimpleHTTPRequestHandler):
    def send_head(self):                       # nginx-style single byte range, which http.server lacks
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        n = os.path.getsize(path)
        m = re.fullmatch(r"bytes=(\\d*)-(\\d*)", rng.strip())
        a, b = m.group(1), m.group(2)
        lo, hi = (n - int(b), n - 1) if a == "" else (int(a), n - 1 if b == "" else int(b))
        lo, hi = max(lo, 0), min(hi, n - 1)
        f = open(path, "rb"); f.seek(lo)
        buf = f.read(hi - lo + 1); f.close()
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{n}")
        self.send_header("Content-Length", str(len(buf)))
        self.end_headers()
        import io; return io.BytesIO(buf)
http.server.HTTPServer.allow_reuse_address = True
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])),
                       functools.partial(H, directory=sys.argv[2])).serve_forever()
"""


@pytest.fixture
def sharded_origin(tmp_path):
    import socket
    import urllib.request
    root = tmp_path / "origin"
    (root / SCROLL / "volumes").mkdir(parents=True)
    (root / SCROLL / "representations" / "predictions" / "surfaces").mkdir(parents=True)
    ct = sharded_pyramid(root / SCROLL / "volumes", "20260101000000-2.400um-0.2m-78keV-masked.zarr", seed=1)
    tg = sharded_pyramid(root / SCROLL / "representations" / "predictions" / "surfaces", "pred.zarr",
                         pred=True, seed=2)
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-c", RANGE_SERVER, str(port), str(root)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.5).read(1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.05)
    else:
        srv.kill()
        pytest.fail("the range server did not start")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    old = data.LOCAL_VOLUMES, data.STREAM_VOLUMES
    data.LOCAL_VOLUMES, data.STREAM_VOLUMES = str(mirror), f"http://127.0.0.1:{port}"
    try:
        yield (f"{mirror}/{SCROLL}/{os.path.basename(ct)}",
               f"{mirror}/{SCROLL}/representations/predictions/surfaces/pred.zarr", ct, tg)
    finally:
        data.LOCAL_VOLUMES, data.STREAM_VOLUMES = old
        srv.kill()
        srv.wait()


def test_whole_shards_are_cached_and_later_windows_hit_them(tmp_path, sharded_origin, monkeypatch,
                                                            per_window):
    """The unit of fetching and caching is the whole shard object: the buffer's copy is byte-identical to the
    origin's, and windows that land in a resident shard cost nothing (the report's hit rate)."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, oct_, otg = sharded_origin
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=12, workers=1, ctx=(1,))
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()

    ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1,), stream=str(q))
    ds._open()
    src = data.Patches(patch=P32, stores=[f"{oct_},{otg}"], exclude=[], rungs={2, 3}, ctx=(1,))
    src._open_rungs()
    for line in open(q / S.QUEUE):
        rec = json.loads(line)
        got, want = ds._rung_build(rec), src._rung_build(rec)
        assert torch.equal(got["ct"], want["ct"]), "the streamed window differs from the origin"
        assert torch.equal(got["tgt"], want["tgt"])

    # every buffered object is the origin's object, byte for byte
    n = 0
    for f in pathlib.Path(f"{mct}/0").rglob("*"):
        if f.is_file() and f.name != "zarr.json" and not f.name.endswith(".absent"):
            assert f.read_bytes() == (pathlib.Path(f"{oct_}/0") / f.relative_to(f"{mct}/0")).read_bytes()
            n += 1
    assert n, "no shard was cached"
    rec = [json.loads(l) for l in open(q / "plan.jsonl")][-1]
    assert rec["hit_rate"] > 0.3, f"shards are not being reused: {rec['hit_rate']}"
    assert rec["B_per_vox"] > 0


# ------------------------------------------------------------------ region mode

def test_region_mode_keeps_the_windows_inside_one_region(tmp_path, origin, monkeypatch):
    """`--region R --windows-per-region N`: N windows out of one snapped RxRxR region of one source at one
    rung, then the next region -- which is what makes a shard pay for itself."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, oct_, otg = origin
    ds = data.Patches(patch=P32, stores=[f"{oct_},{otg}"], exclude=[], rungs={2, 3}, sym=False,
                      air_keep=1.0, fg_keep=1.0, region=64, windows_per_region=4)
    ds._open_rungs()
    rng, st, got = np.random.default_rng(0), ds.region_state(), []
    while len(got) < 12:
        d = ds._rung_draw(rng, build=False, st=st)[0]
        if d is not None:
            got.append((d["s"], d["k"], np.array(d["lo"]), tuple(st["lo"]), tuple(st["size"])))
    for _, k, lo, rlo, rsz in got:
        assert np.all(lo >= np.array(rlo)) and np.all(lo <= np.array(rlo) + np.array(rsz))
        assert tuple(int(v) % 32 for v in rlo) == (0, 0, 0)  # snapped to the level's chunk grid
    # consecutive windows share a region: far fewer distinct regions than windows
    regions = {(g[0], g[1], g[3]) for g in got}
    assert len(regions) <= 4 < len(got)


def test_region_mode_lifts_the_cache_hit_rate(tmp_path, sharded_origin, monkeypatch, per_window):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _ = sharded_origin
    hits = {}
    for name, kw in (("plain", {}), ("region", dict(region=64, windows_per_region=8))):
        q = tmp_path / name
        S.plan(stores_file=stores(tmp_path, mct, mtg, f"s_{name}.txt"), queue=str(q), patch=P32,
               rungs={2, 3}, seed=0, workers=1, ahead=10 ** 6, cache_gb=10 ** 6, ctx=(1,), limit=24,
               jobs=8, report=10 ** 6, val=None, **kw)
        data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
        hits[name] = [json.loads(l) for l in open(q / "plan.jsonl")][-1]
    assert hits["region"]["hit_rate"] > hits["plain"]["hit_rate"]
    assert hits["region"]["B_per_vox"] < hits["plain"]["B_per_vox"]


# ------------------------------------------------------------------ per-scroll axis

def test_umbilicus_is_derived_per_scroll_and_parsed_in_both_formats(tmp_path, monkeypatch):
    from usrm2 import umbilicus as U
    monkeypatch.setattr(data, "LOCAL_VOLUMES", str(tmp_path / "m"))
    monkeypatch.setattr(data, "UMBILICUS_DIR", str(tmp_path / "umb"))
    root = pathlib.Path(data.LOCAL_VOLUMES) / "PHerc0139"
    root.mkdir(parents=True)
    ct = ct_pyramid(root, base=128, nlev=4, value=lambda l: 0)   # all air, then a blob per z slice
    a = data.rungs(ct)[2]
    v = np.zeros((128, 128, 128), np.uint8)
    v[:, 40:56, 72:88] = 200          # a "scroll" off centre: the centroid must find it
    import zarr
    zarr.open(ct + "/0", mode="r+")[:] = v
    for l in (1, 2, 3):
        n = 128 >> l
        w = np.zeros((n, n, n), np.uint8)
        w[:, 40 >> l:56 >> l, 72 >> l:88 >> l] = 200
        zarr.open(f"{ct}/{l}", mode="r+")[:] = w
    data.CTX_CACHE.clear()
    assert data.scroll_of(ct) == "PHerc0139"
    p = U.ensure(ct, rung=4)  # rung 4 = level 2 of this 2.400 um pyramid
    assert p == data.umbilicus_path("PHerc0139") and os.path.exists(p)
    ax = data.axis(p)
    assert ax.shape[0] == 3 and ax.shape[1] > 2
    assert abs(float(np.mean(ax[1])) - 47.5) < 3 and abs(float(np.mean(ax[2])) - 79.5) < 3  # rung-2 voxels

    # both published formats parse to the same points
    j = json.dumps({"control_points": [{"z": 1, "y": 2, "x": 3}, {"z": 4, "y": 5, "x": 6}]})
    assert U.parse(j) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    assert U.parse("4, 3, 2\n7, 6, 5\n") == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]  # volpkg: x, y, z, 1-based


def test_a_multi_scroll_stores_file_gives_each_source_its_own_axis(tmp_path, monkeypatch):
    from usrm2 import umbilicus as U
    monkeypatch.setattr(data, "LOCAL_VOLUMES", str(tmp_path / "m"))
    monkeypatch.setattr(data, "UMBILICUS_DIR", str(tmp_path / "umb"))
    monkeypatch.setattr(data, "UMBILICUS", str(tmp_path / "paris.json"))
    (tmp_path / "paris.json").write_text(json.dumps({"control_points": [{"z": 0, "y": 1, "x": 1}]}))
    lines = []
    for sc, cy in (("PHerc0139", 8), ("PHerc0332", 44)):
        root = pathlib.Path(data.LOCAL_VOLUMES) / sc
        root.mkdir(parents=True)
        ct = ct_pyramid(root, base=64, nlev=2, value=lambda l: 0)
        import zarr
        for l in (0, 1):
            n = 64 >> l
            w = np.zeros((n, n, n), np.uint8)
            w[:, cy >> l:(cy + 8) >> l, 8 >> l:16 >> l] = 200
            zarr.open(f"{ct}/{l}", mode="r+")[:] = w
        data.CTX_CACHE.clear()
        U.ensure(ct, rung=3)
        lines.append(f"{ct},{pred_pyramid(root, name='p.zarr', base=64, nlev=2)}")
    data.CTX_CACHE.clear()
    srcs = data.source_groups(lines)
    assert [data.scroll_of(s["ct"]) for s in srcs] == ["PHerc0139", "PHerc0332"]
    for s, sc in zip(srcs, ("PHerc0139", "PHerc0332")):
        assert s["umbilicus"] == data.umbilicus_path(sc)
    y0, y1 = float(np.mean(data.axis(srcs[0]["umbilicus"])[1])), float(np.mean(data.axis(srcs[1]["umbilicus"])[1]))
    assert y0 < y1 - 20, "the two scrolls must not share an axis"


# ------------------------------------------------------------------ the no-repeat walk

def walk_plan(tmp_path, ct, tg, queue, name="w.txt", workers=2, active_regions=3, windows_per_region=2, **kw):
    S.plan(stores_file=stores(tmp_path, ct, tg, name), queue=str(queue), patch=P32, rungs={2, 3}, seed=0,
           workers=workers, ahead=10 ** 6, cache_gb=10 ** 6, ctx=(1,), jobs=8, report=10 ** 6, val=None,
           region=64, windows_per_region=windows_per_region, walk="once",
           active_regions=active_regions, **kw)


def queue_of(q):
    return [json.loads(l) for l in open(pathlib.Path(q) / S.QUEUE)]


def test_the_region_list_tiles_the_box_and_drops_the_air(tmp_path, monkeypatch):
    """Every region is a shard-aligned tile of the target box, the tiles are disjoint and cover it, and a
    tile whose target is all air is not in the list at all."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    import zarr
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    for l, n in ((0, 256), (1, 128), (2, 64), (3, 32)):  # only the first eighth of z carries any surface
        v = np.zeros((n, n, n), np.uint8)
        v[:max(n // 8, 1)] = 200
        zarr.open(f"{tg}/{data.rung_um(2 + l):g}", mode="r+")[:] = v
    data.CTX_CACHE.clear()
    srcs = data.source_groups([f"{ct},{tg}"])
    regs = data.region_list(srcs, patch=P32, region=64, allowed={2})
    assert regs, "no region survived"
    lo = np.array([r["lo"] for r in regs])
    assert (lo % 32 == 0).all()                       # shard-aligned (32^3 chunks in the test pyramids)
    assert len({tuple(q) for q in lo}) == len(regs)   # disjoint
    assert (lo[:, 0] < 64).all(), "an all-air region was kept"
    assert set(lo[:, 1]) == set(range(0, 256, 64)) and len(regs) == 1 * 4 * 4
    assert abs(sum(r["w"] for r in regs) - 1.0) < 1e-9


def test_the_walk_weights_honour_the_source_and_rung_mix(tmp_path, monkeypatch):
    """Per (source, rung) the region weights add up to that source's volume share times the rung's
    probability -- the mix `rung_probs` defines, spread over the regions of that rung."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    lines = [f"{ct_pyramid(a, base=256, nlev=4)},{pred_pyramid(a, base=256, nlev=4)}",
             f"{ct_pyramid(b, base=128, nlev=3)},{pred_pyramid(b, base=128, nlev=3)}"]
    data.CTX_CACHE.clear()
    srcs = data.source_groups(lines)
    regs = data.region_list(srcs, patch=P32, region=64)
    sw = np.array([s["volume_um3"] for s in srcs], np.float64)
    sw /= sw.sum()
    got = {}
    for r in regs:
        got[(r["s"], r["k"])] = got.get((r["s"], r["k"]), 0.0) + r["w"]
    for i, s in enumerate(srcs):
        for k, p in data.rung_probs(s, P32).items():
            assert abs(got[(i, k)] - sw[i] * p) < 1e-9


def test_the_walk_order_is_a_weighted_shuffle_without_replacement():
    w = np.array([0.5, 0.3, 0.15, 0.05])
    n = 4000
    firsts = np.bincount([int(data.walk_order(w, s)[0]) for s in range(n)], minlength=4) / n
    assert np.abs(firsts - w).max() < 0.03, firsts   # P(first = i) = w_i exactly (Efraimidis-Spirakis)
    for s in (0, 1, 2):
        assert sorted(data.walk_order(w, s).tolist()) == [0, 1, 2, 3]  # every item, exactly once


def test_the_walk_visits_every_region_exactly_once(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q)
    recs = queue_of(q)
    regs = [json.loads(l) for l in open(q / S.REGIONS)]
    assert len(regs) > 8
    gs = [r["g"] for r in recs]
    # every region was visited (the last partial group of windows is dropped, so one region may be short)
    assert len(set(gs)) >= len(regs) - 1 and set(gs) <= set(range(len(regs)))
    assert all(gs.count(g) <= 2 for g in set(gs))  # at most --windows-per-region windows from each
    first, last = {}, {}
    for i, g in enumerate(gs):
        first.setdefault(g, i)
        last[g] = i
    for g in set(gs):  # a region's windows are contiguous in the interleave: it is never re-opened
        assert len([1 for h in gs[first[g]:last[g] + 1] if h == g]) == gs.count(g)
    done = json.load(open(q / S.EPOCH_DONE))
    assert done["regions"] == len(regs)
    assert done["windows"] == len(recs) and len(recs) % 2 == 0  # the queue ends on a stream boundary


def test_resuming_mid_walk_never_repeats_a_region(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q, limit=12)     # stop part way
    half = [r["g"] for r in queue_of(q)]
    assert half and not os.path.exists(q / S.EPOCH_DONE)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    walk_plan(tmp_path, mct, mtg, q)               # a fresh planner picks the walk up
    recs = queue_of(q)
    gs = [r["g"] for r in recs]
    assert gs[:len(half)] == half
    rest = gs[len(half):]
    assert not (set(rest) & set(half)), "a region was visited twice across the resume"
    regs = [json.loads(l) for l in open(q / S.REGIONS)]
    assert all(gs.count(g) <= 2 for g in set(gs)) and len(set(gs)) <= len(regs)


def test_the_active_regions_are_interleaved_and_released_only_when_queued(tmp_path, origin, monkeypatch):
    """K regions are open at once, their windows go out round robin, and a region is released (a new one
    opened) only after every window of it has been queued."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q, active_regions=3, windows_per_region=4)
    gs = [r["g"] for r in queue_of(q)]
    first, last = {}, {}
    for i, g in enumerate(gs):
        first.setdefault(g, i)
        last[g] = i
    open_now = [sum(1 for g in first if first[g] <= i <= last[g]) for i in range(len(gs))]
    assert max(open_now) <= 3, f"more than --active-regions regions open at once: {max(open_now)}"
    assert max(open_now) > 1, "the regions were not interleaved at all"
    pairs = [1 for a, b in zip(gs, gs[1:]) if a != b]
    assert len(pairs) > 0.5 * (len(gs) - 1), "consecutive entries mostly came from the same region"


def test_the_trainer_stops_when_the_walk_is_done(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q, workers=1)
    n = json.load(open(q / S.EPOCH_DONE))["windows"]
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1,), stream=str(q))
    ds._open()
    got = [int(item["idx"]) for item in ds]         # it ENDS instead of waiting for more
    assert got == list(range(n))


def test_a_mirrored_shard_is_a_hit_and_is_never_evicted(tmp_path, origin, monkeypatch, per_window):
    """The local mirror is not the planner's buffer: what `mirror.json` says the mirror owns is never
    fetched, never charged and never evicted."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, oct_, _ = origin
    shutil.copytree(oct_, mct)                       # the CT is mirrored here already, in full
    for l in sorted(os.listdir(mct)):
        if os.path.isdir(f"{mct}/{l}"):
            json.dump({"complete": True}, open(f"{mct}/{l}/mirror.json", "w"))
    before = {str(p) for p in pathlib.Path(mct).rglob("*") if p.is_file()}
    q = tmp_path / "q"
    run_plan(tmp_path, mct, mtg, q, limit=12, workers=1, ctx=(1,))
    rec = [json.loads(l) for l in open(q / "plan.jsonl")][-1]
    assert rec["mirror"] > 0 and rec["hit_rate"] > 0
    recs = queue_of(q)
    dirs = json.load(open(q / S.META))["dirs"]
    assert recs and not any(dirs[di].startswith(mct) for r in recs for di, _ in r["c"]), \
        "a mirrored shard was recorded as a buffered one"
    pl = S.Planner(stores_file=stores(tmp_path, mct, mtg), queue=str(q), patch=P32, rungs={2, 3}, workers=1,
                   ctx=(1,), cache_gb=0)
    pl.qf = str(q / S.QUEUE)
    pl.dirs, pl.dir_ix = dirs, {d: i for i, d in enumerate(dirs)}
    pl.resume()
    json.dump({"i": 10 ** 6, "margin": 0}, open(q / S.CONSUMED, "w"))
    pl.evict()
    assert {str(p) for p in pathlib.Path(mct).rglob("*") if p.is_file()} >= before


# ------------------------------------------------------------------ the shared walk + region teachers

def test_region_walk_is_the_same_list_the_planner_walks(tmp_path, origin, monkeypatch):
    """`stream.region_walk` is the contract the region teacher service walks: the same regions, in the
    same order, as the planner's own walk."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    mct, mtg, _, _, _ = origin
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q, workers=1)
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    regs = [json.loads(l) for l in open(q / S.REGIONS)]
    order = data.walk_order([r["w"] for r in regs], 0)
    want = [(f"{mct},{mtg}", regs[int(j)]["k"], tuple(regs[int(j)]["lo"])) for j in order]
    got = S.region_walk([f"{mct},{mtg}"], rungs={2, 3}, seed=0, patch=P32, region=64)
    assert got == want
    assert len(set(got)) == len(got)
    # the queue's regions come out in that order
    gs = list(dict.fromkeys(r["g"] for r in queue_of(q)))
    assert [want[g] for g in gs] == [want[g] for g in sorted(gs)] or len(gs) > 1


def teacher_store(path, lo2, shape=(64, 64, 64), value=90, done=True, channel="recto"):
    import zarr
    a = zarr.create_array(str(path), shape=shape, chunks=(32, 32, 32), dtype="uint8", fill_value=0,
                          overwrite=True)
    a[:] = np.full(shape, value, np.uint8)
    a.attrs["origin_zyx"] = [int(v) for v in lo2]
    a.attrs["channel"] = channel
    if done:
        a.attrs["done"] = True
    return str(path)


def test_a_finished_region_teacher_store_replaces_the_mask_target(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "REGION", 64)
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)          # the exported mask: 220 - 30 k
    root = tmp_path / "treg"
    teacher_store(root / "recto" / "region_64_64_64.zarr", (64, 64, 64), value=90)
    data.CTX_CACHE.clear()
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2, 3}, sym=False,
                      teacher_regions=str(root))
    ds._open_rungs()
    s = ds.srcs[0]
    for k, lo, want in ((2, (64, 64, 64), 90),        # inside the store: its probability
                        (3, (32, 32, 32), 90),        # rung 3 = its 2x pool (a constant pools to itself)
                        (2, (0, 0, 0), 160)):         # no store there: the exported mask
        c = data.read_rung(s["ct_pyr"], k, lo, ds.patch, dtype=np.uint8)
        t = ds._teacher_store(s, k, lo)
        tgt, w = ds._rung_target(s, k, lo, c, teacher=t)
        assert int(tgt[0].max()) == want == int(tgt[0].min()), (k, lo, t)
        assert bool((w[0] > 0).all())
    assert ds._teacher_store(s, 2, (48, 64, 64)) is None, "a window straddling two regions must fall back"
    assert ds._teacher_store(s, 4, (64, 64, 64)) is None, "only rungs 2 and 3 come from the store"


def test_an_unfinished_region_teacher_store_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "REGION", 64)
    ct = ct_pyramid(tmp_path, base=256, nlev=4)
    tg = pred_pyramid(tmp_path, base=256, nlev=4)
    root = tmp_path / "treg"
    teacher_store(root / "recto" / "region_64_64_64.zarr", (64, 64, 64), value=90, done=False)
    data.CTX_CACHE.clear()
    ds = data.Patches(patch=P32, stores=[f"{ct},{tg}"], exclude=[], rungs={2, 3}, sym=False,
                      teacher_regions=str(root))
    ds._open_rungs()
    assert ds._teacher_store(ds.srcs[0], 2, (64, 64, 64)) is None


def test_the_queue_carries_the_teacher_store_so_the_replay_reads_it(tmp_path, origin, monkeypatch):
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    monkeypatch.setattr(data, "REGION", 64)
    mct, mtg, _, _, _ = origin
    root = tmp_path / "treg"
    for z in (0, 64, 128, 192):
        for y in (0, 64, 128, 192):
            for x in (0, 64, 128, 192):
                teacher_store(root / "recto" / f"region_{z}_{y}_{x}.zarr", (z, y, x), value=90)
    q = tmp_path / "q"
    walk_plan(tmp_path, mct, mtg, q, workers=1, teacher_regions=str(root))
    recs = queue_of(q)
    assert any("t" in r for r in recs), "no window used a region teacher store"
    data.CTX_CACHE.clear(), data.CHUNK_INDEX.clear()
    ds = data.Patches(patch=P32, stores=[f"{mct},{mtg}"], exclude=[], rungs={2, 3}, ctx=(1,), stream=str(q))
    ds._open()
    assert ds.teacher_regions == str(root)          # the queue's meta carries it to the replaying worker
    rec = [r for r in recs if "t" in r and not r.get("b")][0]
    item = ds._rung_build(rec)
    assert int(item["tgt"].max()) == 90, "the replay did not read the region teacher store"


def test_walk_mix_spreads_the_coarse_rungs_over_the_whole_epoch(tmp_path, monkeypatch):
    """A rung with few regions but a big --rung-boost share is used up in the first percent of a `once`
    walk. `mix` gives it proportionally many visits instead, so its share holds throughout."""
    monkeypatch.setattr(data, "UMBILICUS", umbilicus(tmp_path))
    ct = ct_pyramid(tmp_path, base=256, nlev=5)
    tg = pred_pyramid(tmp_path, base=256, nlev=5)
    data.CTX_CACHE.clear()
    srcs = data.source_groups([f"{ct},{tg}"])
    boost = {6: 40}                                   # the coarsest rung here: one region, a big share
    regs = data.region_list(srcs, patch=P32, region=64, boost=boost)
    vis = data.region_visits(regs, cap=64)
    nreg = {k: sum(1 for r in regs if r["k"] == k) for k in {r["k"] for r in regs}}
    assert nreg[2] > 50 and nreg[6] == 1
    assert sum(1 for r in vis if r["k"] == 2) == nreg[2], "a fine rung keeps one visit per region"
    assert sum(1 for r in vis if r["k"] == 6) > 1, "the coarse rung was not given more visits"
    want = sum(r["w"] for r in regs if r["k"] == 6)   # its intended share
    for name, lst in (("once", regs), ("mix", vis)):
        order = data.walk_order([r["w"] for r in lst], 0)
        seq = [lst[int(j)]["k"] for j in order]
        got = np.mean([k == 6 for k in seq[len(seq) // 2:]])   # the SECOND half of the epoch
        if name == "once":
            assert got == 0, "the coarse rung should already be exhausted"
        else:
            assert abs(got - want) < 0.5 * want + 0.02, (got, want)
    assert abs(sum(r["w"] for r in vis) - 1.0) < 1e-9
