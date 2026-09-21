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
    pub = os.path.join(path, ".published")
    if os.path.isfile(pub):
        shutil.copy2(pub, os.path.join(tmp, ".published"))
    old = path + ".old"
    shutil.rmtree(old, ignore_errors=True)
    os.rename(path, old)
    os.rename(tmp, path)
    shutil.rmtree(old, ignore_errors=True)
    return ("ok", path, total)


def _job(a):
    try:
        return repack(*a)
    except Exception as e:
        return ("FAILED", a[0], repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="decode both layouts and require identical arrays")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stores = sorted(os.path.join(a.dir, d) for d in os.listdir(a.dir)
                    if d.endswith(".zarr") and os.path.isfile(os.path.join(a.dir, d, "zarr.json")))
    if a.limit:
        stores = stores[:a.limit]
    t0, cnt = time.time(), {}
    args = [(s, a.verify, a.dry_run) for s in stores]
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
