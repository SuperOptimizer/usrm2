#!/usr/bin/env python3
"""Mirror an exported prediction pyramid (zarr v3, sharded, levels named by micron) from the public volcomp tree
into the local mirror by enumerating the shard keys from the array metadata and fetching them concurrently
(404 = absent shard = fill value; skipped). Far faster than a recursive crawl of the directory listings.
    python cloud/pull_preds.py SCROLL NAME.zarr [--jobs 48]      (re-runnable; existing files are kept)
"""
import asyncio, json, math, os, sys, time
import aiohttp
scroll, name = sys.argv[1], sys.argv[2]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 48
URL = f"https://dl.ash2txt.org/community-uploads/forrest/volcomp/{scroll}/representations/predictions/surfaces/{name}"
DST = f"/vesuvius/usrm/volcomp/{scroll}/representations/predictions/surfaces/{name}"


async def get(session, rel, sem, binary=True):
    out = f"{DST}/{rel}"
    if os.path.exists(out):
        return 1
    async with sem:
        for attempt in range(4):
            try:
                async with session.get(f"{URL}/{rel}") as r:
                    if r.status == 404:
                        return 0
                    r.raise_for_status(); data = await r.read()
                os.makedirs(os.path.dirname(out), exist_ok=True)
                tmp = out + ".part"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, out)
                return len(data)
            except Exception as e:
                if attempt == 3:
                    print("FAILED", rel, e, flush=True); return -1
                await asyncio.sleep(2 * (attempt + 1))


async def main():
    sem = asyncio.Semaphore(jobs)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        await get(s, "zarr.json", sem)
        g = json.load(open(f"{DST}/zarr.json"))
        paths = [d["path"] for d in g["attributes"]["ome"]["multiscales"][0]["datasets"]]
        for p in paths:
            await get(s, f"{p}/zarr.json", sem)
            m = json.load(open(f"{DST}/{p}/zarr.json"))
            shape = m["shape"]; shard = m["chunk_grid"]["configuration"]["chunk_shape"]
            n = [math.ceil(a / b) for a, b in zip(shape, shard)]
            rels = [f"{p}/c/{z}/{y}/{x}" for z in range(n[0]) for y in range(n[1]) for x in range(n[2])]
            t0 = time.time()
            res = await asyncio.gather(*[get(s, r, sem) for r in rels])
            got = sum(1 for r in res if r and r > 1); nb = sum(r for r in res if r and r > 1)
            print(f"{p}: {len(rels)} keys, {got} shards fetched ({nb / 1e9:.2f} GB), {sum(1 for r in res if r == 1)} already local, "
                  f"{sum(1 for r in res if r == -1)} failed, {time.time() - t0:.0f} s", flush=True)
    print("PULLPREDSDONE", name, flush=True)

asyncio.run(main())
