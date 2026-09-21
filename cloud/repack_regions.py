#!/usr/bin/env python3
"""Repack unsharded 128^3-chunk volcomp stores into the zarr v3 SHARDED layout, losslessly.

A 1024^3 region store written the old way is 512 chunk files + 2 directories: publishing one to the sftp
mirror costs ~590 operations. The sharded layout puts the same 512 encoded chunks into ONE file (c/0/0/0)
with a small index, so a store is 2 files and 2 sftp puts. The inner chunk encoding is unchanged (volcomp
q then zstd), so this converter only MOVES raw chunk bytes - nothing is decoded or re-encoded, and the
decoded array is bit-identical to the original.

    python cloud/repack_regions.py DIR [--jobs N] [--verify] [--limit N] [--dry-run]

Skips stores that are already sharded and stores whose attributes lack "done": true (a store still being
written by the teacher is only marked done at the very end, so skipping not-done never races the writer).
Each store is built in <store>.tmp and swapped in atomically; a `.published` marker is carried over.

    python cloud/repack_regions.py DIR --strip-zstd [--jobs N] [--verify] [--limit N] [--dry-run]

--strip-zstd is the second, also lossless, conversion: stores written before `predict.out_array` passed
`compressors=None` have inner codecs [volcomp q8, ZSTD] because zarr-python appends its default compressor
after the serializer. zstd on volcomp output saves 0.2% (measured on a real region store) and costs a
decode step on every chunk read, and the C-tool exports and the CT volumes are volcomp-only. This mode
zstd-DECODES every chunk payload inside each shard file, rebuilds the shard's payload region, its uint64
(offset, nbytes) index and the index crc32c, and rewrites zarr.json without the zstd codec. The volcomp
bytes themselves are never touched, so the decoded array is bit-identical; `--verify` decodes both and
compares. Attributes (including `done`), the `.published` marker and the atomic swap are as above.

*** NEVER RUN EITHER CONVERSION WHILE A READER IS RUNNING. ***

The swap is atomic, but that is not enough. An OPEN zarr array CACHES ITS METADATA: a training worker or a
stream planner that opened a store before the swap goes on decoding with the OLD codec chain / the OLD
chunk grid, and the first read after the swap raises

    numcodecs Zstd decompression error: invalid input data

That is exactly how the in-place sharding repack killed the A100 `u2_30m6_stream` run at 18:01 UTC on
2026-09-21 (a worker holding the old unsharded zarr.json read the new `c/0/0/0` shard file as chunk
(0,0,0)). An rsync that reads the store while it is swapped is the same hazard one level out: it can ship
the shard from one side of the rename and the zarr.json from the other.

So, before converting a directory:
  * A100 (`~/teacher_regions/recto`): only in a restart window, with no trainer and no planner running.
  * desk (`/vesuvius/usrm2/teacher_regions/recto`): stop the readers first --
        bash ~/sync_stop.sh && bash ~/publish_stop.sh
        python cloud/repack_regions.py /vesuvius/usrm2/teacher_regions/recto --strip-zstd --jobs 8
        bash ~/sync_start.sh && bash ~/publish_restart.sh 2
    The two teacher WRITERS may keep running: they never read a finished store, and the converter skips
    stores without `done`.
`--verify` DECODES, so it needs the volcomp shared library: on the desk run it with
`VOLCOMP_LIB=$HOME/.cache/usrm/bin/libvolcomp.so` (what `desk.sh` exports). Without it every store comes
back FAILED -- harmlessly, nothing is swapped, and the half-built `<store>.tmp` is removed.

`--check` afterwards is the cheap audit (4 bytes per store) that nothing ended up mismatched, and
`usrm2.data.read_slice` retries a decode failure once against a re-opened array so that a run survives
this happening anyway -- a backstop, not a licence.
"""
import argparse, json, os, shutil, sys, time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def shard_shape(shape, chunk=128, cap=1024):
    return tuple(min(cap, -(-int(s) // chunk) * chunk) for s in shape)


def sharded_json(meta, shard):
    """The unsharded zarr.json rewritten as the equivalent sharding_indexed one (same attributes)."""
    inner = meta["chunk_grid"]["configuration"]["chunk_shape"]
    out = dict(meta)
    out["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": [int(v) for v in shard]}}
    out["codecs"] = [{"name": "sharding_indexed",
                      "configuration": {"chunk_shape": [int(v) for v in inner],
                                        "codecs": meta["codecs"],
                                        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                                         {"name": "crc32c"}],
                                        "index_location": "end"}}]
    return out


def build_shard(key_of, cps):
    """(bytes, n_present) of one shard: the encoded inner chunks in C order, then the uint64 LE
    (offset, nbytes) index and its crc32c checksum (index_codecs = bytes + crc32c, index_location end)."""
    import numpy as np, google_crc32c
    parts, idx, off, n = [], np.full((int(np.prod(cps)), 2), 2 ** 64 - 1, np.uint64), 0, 0
    for i in range(int(np.prod(cps))):
        loc = np.unravel_index(i, tuple(int(v) for v in cps))  # C order, as the sharding spec requires
        p = key_of(loc)
        if p is None or not os.path.isfile(p):
            continue
        b = open(p, "rb").read()
        idx[i] = (off, len(b))
        parts.append(b)
        off += len(b)
        n += 1
    if not n:
        return None, 0
    ib = idx.astype("<u8").tobytes()
    ib += np.uint32(google_crc32c.value(ib)).astype("<u4").tobytes()
    return b"".join(parts) + ib, n


# ---------------------------------------------------------------- --strip-zstd

def sharding(meta):
    """(the sharding_indexed configuration, the inner codec list) of a sharded store, or (None, None)."""
    c = meta.get("codecs", [])
    if len(c) == 1 and c[0].get("name") == "sharding_indexed":
        cfg = c[0].get("configuration", {})
        return cfg, list(cfg.get("codecs", []))
    return None, None


def restrip_shard(buf, n, dec):
    """One shard file with every chunk payload run through `dec`: the payloads in the same order, then a
    fresh uint64 LE (offset, nbytes) index of `n` entries and its crc32c (index_codecs bytes + crc32c,
    index_location end). An absent entry (2**64-1, 2**64-1) stays absent."""
    import numpy as np, google_crc32c
    isz = n * 16 + 4
    assert len(buf) > isz, "shard file shorter than its index"
    old = np.frombuffer(buf[-isz:-4], dtype="<u8").reshape(n, 2)
    if google_crc32c.value(buf[-isz:-4]) != int(np.frombuffer(buf[-4:], dtype="<u4")[0]):
        raise ValueError("shard index crc32c mismatch")
    parts, idx, off, got = [], np.full((n, 2), 2 ** 64 - 1, np.uint64), 0, 0
    for i in range(n):
        o, ln = int(old[i, 0]), int(old[i, 1])
        if o == 2 ** 64 - 1:
            continue
        b = bytes(dec(buf[o:o + ln]))
        idx[i] = (off, len(b))
        parts.append(b)
        off += len(b)
        got += 1
    ib = idx.astype("<u8").tobytes()
    ib += np.uint32(google_crc32c.value(ib)).astype("<u4").tobytes()
    return b"".join(parts) + ib, got


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def check(path, verify=False, dry=False):
    """Cheap consistency check: does the store's FIRST stored chunk actually carry the codec chain its
    zarr.json declares? Reads 4 bytes per store. A `[volcomp]` store whose payload is still zstd-framed
    (or the other way round) is unreadable, which is what a copy interrupted halfway through a conversion
    looks like -- so this is what to run over a directory after a `--strip-zstd` pass that ran alongside a
    sync loop."""
    import numpy as np
    meta = json.load(open(os.path.join(path, "zarr.json")))
    cfg, inner = sharding(meta)
    if cfg is None:
        return ("unsharded", path, 0)
    want = "zstd" in [q.get("name") for q in inner]
    chunk = [int(v) for v in cfg["chunk_shape"]]
    shard = [int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]]
    n = int(np.prod([h // c for h, c in zip(shard, chunk)]))
    for root, _, files in os.walk(os.path.join(path, "c")):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as fh:
                fh.seek(-(n * 16 + 4), os.SEEK_END)
                idx = np.frombuffer(fh.read(n * 16), dtype="<u8").reshape(n, 2)
                live = [i for i in range(n) if int(idx[i, 0]) != 2 ** 64 - 1]
                if not live:
                    continue
                fh.seek(int(idx[live[0], 0]))
                got = fh.read(4) == ZSTD_MAGIC
            return ("ok" if got == want else "MISMATCH", path, int(want))
    return ("empty", path, 0)


def strip_zstd(path, verify=False, dry=False):
    """Rewrite a sharded [volcomp, zstd] store as [volcomp], losslessly. Nothing is decoded but the zstd
    layer, so the volcomp payload -- and therefore the decoded array -- is bit-identical."""
    import numpy as np
    from numcodecs import Zstd
    meta = json.load(open(os.path.join(path, "zarr.json")))
    cfg, inner = sharding(meta)
    if cfg is None:
        return ("unsharded", path, 0)  # run the plain repack first
    names = [q.get("name") for q in inner]
    if "zstd" not in names:
        return ("clean", path, 0)
    if not meta.get("attributes", {}).get("done"):
        return ("notdone", path, 0)
    if names[-1] != "zstd" or names.count("zstd") != 1 or "volcomp" not in names:
        return ("skip", path, 0)     # zstd is only strippable as the LAST inner codec
    if cfg.get("index_location", "end") != "end":
        return ("skip", path, 0)
    if dry:
        return ("would", path, 0)
    chunk = [int(v) for v in cfg["chunk_shape"]]
    shard = [int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]]
    n = int(np.prod([h // c for h, c in zip(shard, chunk)]))
    dec = Zstd().decode
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    total = 0
    for root, _, files in os.walk(os.path.join(path, "c")):
        for f in files:
            src = os.path.join(root, f)
            dstd = os.path.join(tmp, os.path.relpath(root, path))
            os.makedirs(dstd, exist_ok=True)
            buf, got = restrip_shard(open(src, "rb").read(), n, dec)
            with open(os.path.join(dstd, f), "wb") as fh:
                fh.write(buf)
            total += got
    out = dict(meta)
    out["codecs"] = [{"name": "sharding_indexed",
                      "configuration": dict(cfg, codecs=[q for q in inner if q.get("name") != "zstd"])}]
    json.dump(out, open(os.path.join(tmp, "zarr.json"), "w"), indent=2)
    if verify:
        import zarr
        try:
            import volcomp_zarr  # noqa: F401
        except Exception:
            pass
        a, b = zarr.open(path, mode="r")[:], zarr.open(tmp, mode="r")[:]
        if not np.array_equal(a, b):
            shutil.rmtree(tmp, ignore_errors=True)
            return ("MISMATCH", path, 0)
    swap(path, tmp)
    return ("ok", path, total)


def swap(path, tmp):
    """Atomically replace `path` with `tmp`, carrying the `.published` marker over."""
    pub = os.path.join(path, ".published")
    if os.path.isfile(pub):
        shutil.copy2(pub, os.path.join(tmp, ".published"))
    old = path + ".old"
    shutil.rmtree(old, ignore_errors=True)
    os.rename(path, old)
    os.rename(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def repack(path, verify=False, dry=False):
    import numpy as np
    meta = json.load(open(os.path.join(path, "zarr.json")))
    if any(c.get("name") == "sharding_indexed" for c in meta.get("codecs", [])):
        return ("sharded", path, 0)
    if not meta.get("attributes", {}).get("done"):
        return ("notdone", path, 0)
    shape = [int(v) for v in meta["shape"]]
    chunk = [int(v) for v in meta["chunk_grid"]["configuration"]["chunk_shape"]]
    shard = shard_shape(shape, chunk=chunk[-1]) if len(chunk) == 3 else None
    if shard is None or any(s % c for s, c in zip(shard, chunk)):
        return ("skip", path, 0)
    if dry:
        return ("would", path, 0)
    nsh = [-(-s // h) for s, h in zip(shape, shard)]
    ngr = [-(-s // c) for s, c in zip(shape, chunk)]
    cps = [h // c for h, c in zip(shard, chunk)]
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    total = 0
    for sz in range(nsh[0]):
        for sy in range(nsh[1]):
            for sx in range(nsh[2]):
                base = (sz * cps[0], sy * cps[1], sx * cps[2])

                def key_of(loc, base=base):
                    g = [b + int(l) for b, l in zip(base, loc)]
                    if any(v >= n for v, n in zip(g, ngr)):
                        return None  # outside the chunk grid: an absent entry
                    return os.path.join(path, "c", str(g[0]), str(g[1]), str(g[2]))
                buf, n = build_shard(key_of, cps)
                if buf is None:
                    continue
                d = os.path.join(tmp, "c", str(sz), str(sy))
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, str(sx)), "wb") as f:
                    f.write(buf)
                total += n
    json.dump(sharded_json(meta, shard), open(os.path.join(tmp, "zarr.json"), "w"), indent=2)
    if verify:
        import zarr
        try:
            import volcomp_zarr  # noqa: F401
        except Exception:
            pass
        a, b = zarr.open(path, mode="r")[:], zarr.open(tmp, mode="r")[:]
        if not np.array_equal(a, b):
            shutil.rmtree(tmp, ignore_errors=True)
            return ("MISMATCH", path, 0)
    swap(path, tmp)
    return ("ok", path, total)


def _job(a):
    fn = {"strip": strip_zstd, "check": check}.get(a[-1], repack)
    try:
        return fn(*a[:-1])
    except Exception as e:
        shutil.rmtree(a[0] + ".tmp", ignore_errors=True)  # a half-built conversion never survives a failure
        return ("FAILED", a[0], repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="decode both layouts and require identical arrays")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="only check that each store's payload matches the "
                    "codec chain its zarr.json declares (4 bytes read per store); prints MISMATCH lines")
    ap.add_argument("--strip-zstd", action="store_true",
                    help="losslessly rewrite sharded [volcomp, zstd] stores as [volcomp] (see the module "
                         "docstring): the zstd layer is decoded away, the volcomp payload is untouched")
    a = ap.parse_args()
    stores = sorted(os.path.join(a.dir, d) for d in os.listdir(a.dir)
                    if d.endswith(".zarr") and os.path.isfile(os.path.join(a.dir, d, "zarr.json")))
    if a.limit:
        stores = stores[:a.limit]
    t0, cnt = time.time(), {}
    mode = "check" if a.check else ("strip" if a.strip_zstd else "repack")
    args = [(s, a.verify, a.dry_run, mode) for s in stores]
    it = Pool(a.jobs).imap_unordered(_job, args) if a.jobs > 1 else map(_job, args)
    for i, (st, p, n) in enumerate(it):
        cnt[st] = cnt.get(st, 0) + 1
        if st in ("MISMATCH", "FAILED"):
            print(f"{st} {p} {n}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(stores)} {cnt} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(stores)} stores in {time.time() - t0:.0f}s: {cnt}", flush=True)
    return 1 if cnt.get("MISMATCH") or cnt.get("FAILED") else 0


if __name__ == "__main__":
    sys.exit(main())
