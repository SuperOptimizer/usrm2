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
