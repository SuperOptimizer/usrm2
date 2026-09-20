#!/usr/bin/env python3
"""On the A100: build a LOCAL volcomp mirror of the CT so training never streams. Levels 1 and 2 are fetched
whole from dl.ash2txt.org (fast), level 0 only for the chunks the training boxes touch (plus a margin), levels
3-5 come from the desk. Layout mirrors the desk (/vesuvius/usrm/volcomp/<scroll>/<vol>.zarr/<level>) so
data.local() maps the streamed URL onto it and data.levels() finds the pyramid.
    python cloud/a100_mirror.py ~/groups_raw.txt [--margin 256] [--jobs 32]
    python cloud/a100_mirror.py --boxes ~/boxes_p4.txt          # boxes as "z y x Z Y X" lines instead of stores
"""
import asyncio, glob, json, os, sys, time
import aiohttp, numpy as np, zarr

URL = "https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
DST = "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr"
groups = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.expanduser("~/groups_raw.txt")
boxes_file = sys.argv[sys.argv.index("--boxes") + 1] if "--boxes" in sys.argv else None  # "z y x Z Y X" per line
margin = int(sys.argv[sys.argv.index("--margin") + 1]) if "--margin" in sys.argv else 256
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 32
os.makedirs(DST, exist_ok=True)


async def fetch(session, rel, sem):
    out = f"{DST}/{rel}"
    if os.path.exists(out):
        return 0
    async with sem:
        for attempt in range(4):
            try:
                async with session.get(f"{URL}/{rel}") as r:
                    if r.status == 404:
                        return 0
                    r.raise_for_status(); data = await r.read()
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out + ".part", "wb") as f:
                    f.write(data)
                os.replace(out + ".part", out)
                return len(data)
            except Exception as e:
                await asyncio.sleep(2 * (attempt + 1))
        print("FAILED", rel, flush=True); return 0


async def main():
    sem = asyncio.Semaphore(jobs)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        # metadata + levels 1, 2 whole
        rels = ["zarr.json"]
        for l in (0, 1, 2):
            async with s.get(f"{URL}/{l}/zarr.json") as r:
                meta = json.loads(await r.read())
            os.makedirs(f"{DST}/{l}", exist_ok=True); json.dump(meta, open(f"{DST}/{l}/zarr.json", "w"))
            shape, chunks = meta["shape"], meta["chunk_grid"]["configuration"]["chunk_shape"]
            n = [(sh + c - 1) // c for sh, c in zip(shape, chunks)]
            if l >= 1:
                rels += [f"{l}/c/{z}/{y}/{x}" for z in range(n[0]) for y in range(n[1]) for x in range(n[2])]
            else:
                n0, c0, shape0 = n, np.array(chunks), np.array(shape)
        # level 0: chunks covering every training/val box (+ margin)
        boxes = set()
        def box_list():
            if boxes_file:
                for line in open(boxes_file):
                    v = [int(q) for q in line.split()]
                    if len(v) == 6:
                        yield np.array(v[:3]), np.array(v[3:])
                return
            for line in open(groups):
                for p in line.strip().split(","):
                    if not p or "_v" in os.path.basename(p.rstrip("/")):
                        continue
                    a = zarr.open(p, mode="r"); yield np.array(a.attrs["origin_zyx"]), np.array(a.shape[-3:])
        for o, sz in box_list():
            if True:
                lo = np.maximum(o - margin, 0) // c0; hi = np.minimum(o + sz + margin, shape0) // c0
                for z in range(lo[0], hi[0] + 1):
                    for y in range(lo[1], hi[1] + 1):
                        for x in range(lo[2], hi[2] + 1):
                            boxes.add((z, y, x))
        rels += [f"0/c/{z}/{y}/{x}" for z, y, x in sorted(boxes)]
        print(f"{len(rels)} objects to check ({len(boxes)} level-0 chunks for the boxes)", flush=True)
        t0 = time.time(); done = 0; nbytes = 0
        for i in range(0, len(rels), 2000):
            res = await asyncio.gather(*[fetch(s, r, sem) for r in rels[i:i + 2000]])
            nbytes += sum(res); done += len(res)
            print(f"  {done}/{len(rels)}  {nbytes / 2**30:.1f} GiB  {time.time() - t0:.0f} s", flush=True)
    print(f"MIRRORDONE {nbytes / 2**30:.1f} GiB in {time.time() - t0:.0f} s", flush=True)

asyncio.run(main())

# what the mirror knows: levels fetched whole are complete; level 0 only inside the boxes (see data.chunk_index)
try:
    for l in (1, 2):
        json.dump({"complete": True}, open(f"{DST}/{l}/mirror.json", "w"))
    if boxes_file:
        bx = [[int(q) for q in line.split()] for line in open(boxes_file) if len(line.split()) == 6]
        json.dump({"boxes": [[max(b[0] - margin, 0), max(b[1] - margin, 0), max(b[2] - margin, 0),
                              b[3] + 2 * margin, b[4] + 2 * margin, b[5] + 2 * margin] for b in bx]},
                  open(f"{DST}/0/mirror.json", "w"))
    print("mirror.json markers written")
except Exception as e:
    print("mirror.json markers failed:", e)
