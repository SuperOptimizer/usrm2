"""Train while streaming: a planner that IS the sampler, a rolling disk buffer, a replayed queue.

`usrm2 stream-plan STORES --queue DIR ...` runs the exact rung-mode sampling of `data.Patches` -- the same
per-worker seeds, the same draw order, the same air / low-foreground / dense rejection rules -- but instead
of returning tensors it

  1. FETCHES, from the public volcomp tree, every chunk the reads of a candidate window will touch (the CT
     at the window's rung, the target levels, then the nine context rungs once the window is accepted),
     into the LOCAL mirror layout (`data.local` maps both ways, `data.remote` is the inverse), `.part` +
     rename, and marks a 404 with a zero-length `<chunk>.absent` file so nothing refetches it (an absent key
     on the origin is air, i.e. the array's fill value);
  2. applies the rejection rules to the fetched data;
  3. appends the accepted window's descriptor (source, rung, corner, symmetry, the raw-aug draws and the
     chunk keys it needs) to `queue.jsonl` with a monotonic index.

Entry i belongs to loader worker i % W, and the entries of one worker come from ONE rng stream seeded
`seed + 1000 * w`, exactly as `data.Patches.__iter__` seeds it -- so replaying the queue with W workers
reproduces, window for window, what the direct sampler would have drawn had the whole tree been local.

`Patches(stream=DIR)` (i.e. `usrm2 train ... --stream DIR`) replays: worker w takes entries w, w+W, ...,
waits (with backoff, and logs the wait as `stream_wait_ms`) when the planner has not got there yet, and
reads each window from the buffer exactly as today.

The planner keeps `ahead` windows in front of the consumer and deletes, once the buffer passes `--cache-gb`,
the chunks no queued-but-unconsumed window still references (LRU by last reference). Whole levels small
enough that `data.full_level` decodes them (<= `data.CACHE_VOX`) are fetched once and never evicted; nor are
the `.absent` markers or any `zarr.json`.
"""
import asyncio
import json
import os
import time

import numpy as np

from usrm2 import data, umbilicus as U

META, QUEUE, STATE, PROGRESS, CONSUMED = "meta.json", "queue.jsonl", "state.json", "progress", "consumed"
REGIONS, WALK, EPOCH_DONE = "regions.jsonl", "walk.json", "epoch_done"
VERSO_TTL = 1800.0  # seconds an UNPUBLISHED verso region store stays unpublished in the planner's memory


# ------------------------------------------------------------------ which chunks a read touches

def chunk_grid(arr):
    """The write-chunk (shard) size of an array, as `data.chunk_index` sees it."""
    return np.array(getattr(arr, "shards", None) or arr.chunks, np.int64)[-3:]


def chunk_key(arr, ix):
    """The key of chunk (z, y, x), relative to the array directory."""
    z, y, x = (int(v) for v in ix)
    v2 = int(getattr(getattr(arr, "metadata", None), "zarr_format", 3)) == 2
    return f"{z}.{y}.{x}" if v2 else f"c/{z}/{y}/{x}"


def keys_in(arr, a, b):
    """The chunk indices of `arr` covering the voxel range [a, b)."""
    g, S = chunk_grid(arr), np.array(arr.shape[-3:], np.int64)
    a, b = np.maximum(np.asarray(a, np.int64), 0), np.minimum(np.asarray(b, np.int64), S)
    if (b <= a).any():
        return []
    lo, hi = a // g, -(-b // g)
    return [(z, y, x) for z in range(lo[0], hi[0]) for y in range(lo[1], hi[1]) for x in range(lo[2], hi[2])]


def all_keys(arr):
    n = -(-np.array(arr.shape[-3:], np.int64) // chunk_grid(arr))
    return [(z, y, x) for z in range(n[0]) for y in range(n[1]) for x in range(n[2])]


# A volcomp level is a SHARDED zarr v3 array: the object is a 1024^3 shard holding 512 inner 128^3 chunks
# (a CT level-0 shard is ~17 MB, a level-2 shard ~35 MB). The SHARD is the unit of fetching and of caching:
# it is written into the mirror at its normal path and stays there until eviction, so every later window
# whose reads land in it costs nothing. One shard covers 64 windows' worth of volume at 256^3, so the
# residency of the coarse levels (whose whole level is a handful of shards) is what the hit rate lives on.

def rung_range(pyr, k, lo, p):
    """(array, a, b, whole) that `data.read_rung(pyr, k, lo, p)` will read: the voxel range [a, b) of the
    source level, or whole=True when `data.full_level` keeps that level decoded and reads all of it.

    It mirrors `read_rung` / `read_block` / `full_level`: the source level is the highest rung at or below k,
    and when either that level or the rung-k view of it is small enough to be kept whole, the WHOLE level is
    read (once) instead of a window of it."""
    src = max(r for r in pyr if r <= k)
    arr = pyr[src]
    S = np.array(arr.shape[-3:], np.int64)
    if (int(np.prod(data.rung_shape(pyr, k))) <= data.CACHE_VOX
            or int(np.prod(S)) <= data.CACHE_VOX):
        return arr, np.zeros(3, np.int64), S, True
    p, lo = data.shape3(p), np.asarray(lo, np.int64)
    e = 1 << (k - src)
    jlo, jhi = np.maximum(-lo, 0), np.minimum(-(-S // e) - lo, p)
    if (jhi <= jlo).any():  # the cube does not touch the array at all
        return arr, np.zeros(3, np.int64), np.zeros(3, np.int64), False
    return arr, (lo + jlo) * e, np.minimum((lo + jhi) * e, S), False


def rung_need(pyr, k, lo, p):
    """(array, chunk indices, whole) of the same read: the outer (shard) keys it touches."""
    arr, a, b, whole = rung_range(pyr, k, lo, p)
    return arr, (all_keys(arr) if whole else keys_in(arr, a, b)), whole


def ctx_need(pyr, k, lo, p, ctx):
    """`data.context`'s reads: one (array, keys, whole) per context offset."""
    c0 = np.asarray(lo, np.int64) + data.shape3(p) // 2
    out = []
    for d in ctx:
        lo_d = c0 // (1 << int(d)) - data.shape3(p) // 2
        out.append(rung_need(pyr, k + int(d), lo_d, p))
    return out


# ------------------------------------------------------------------ fetching

class Fetcher:
    """Concurrent GETs of mirror objects from their origin (`data.remote`). 404 = absent = a marker file."""

    def __init__(self, session, jobs=48, retries=4):
        self.session, self.sem, self.retries = session, asyncio.Semaphore(jobs), retries
        self.lock = {}  # one download per shard: concurrent windows wanting the same object wait for it
        self.bytes = self.fetched = self.absent = self.have = self.failed = self.requests = 0
        self.mirror = 0  # shards the pre-existing local mirror already owns: never fetched, never evicted

    async def get(self, path):
        """(status, bytes) with status in have / new / absent / fail. `path` is the LOCAL mirror path.
        `have` counts the buffer hits -- a shard some earlier window already pulled -- and `fetched` the
        misses, which is what the hit rate in the report is made of."""
        if os.path.exists(path):
            self.have += 1
            return "have", 0
        if os.path.exists(path + ".absent"):
            self.have += 1
            return "absent", 0
        lk = self.lock.setdefault(path, asyncio.Lock())
        async with lk:
            try:
                return await self._get(path)
            finally:
                if not lk.locked():
                    self.lock.pop(path, None)

    async def _get(self, path):
        if os.path.exists(path):  # another window pulled this shard while we waited for the lock
            self.have += 1
            return "have", 0
        if os.path.exists(path + ".absent"):
            self.have += 1
            return "absent", 0
        url = data.remote(path)
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    async with self.session.get(url) as r:
                        if r.status == 404:
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            open(path + ".absent", "wb").close()
                            self.absent += 1
                            return "absent", 0
                        r.raise_for_status()
                        buf = await r.read()
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path + ".part", "wb") as f:
                        f.write(buf)
                    os.replace(path + ".part", path)
                    self.bytes += len(buf)
                    self.fetched += 1
                    self.requests += 1
                    return "new", len(buf)
                except Exception as e:  # noqa: BLE001
                    if attempt == self.retries - 1:
                        print(f"stream-plan: FAILED {url}: {e!r}", flush=True)
                        self.failed += 1
                        return "fail", 0
                    await asyncio.sleep(2 * (attempt + 1))


    async def fetch_url(self, url, path, retries=2):
        """GET an explicit URL into `path`. Unlike `get` this leaves NO `.absent` marker on a 404: it is
        used for objects that do not exist YET (the verso region stores, published continuously while the
        run trains), where "absent" has to be allowed to expire. Returns have / new / absent / fail."""
        if os.path.exists(path):
            self.have += 1
            return "have"
        lk = self.lock.setdefault(path, asyncio.Lock())
        async with lk:
            try:
                if os.path.exists(path):
                    self.have += 1
                    return "have"
                async with self.sem:            # the same keep-alive session and the same job budget as
                    for attempt in range(max(int(retries), 1)):   # everything else: no burst of its own
                        try:
                            async with self.session.get(url) as r:
                                if r.status == 404:
                                    return "absent"
                                r.raise_for_status()
                                buf = await r.read()
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            with open(path + ".part", "wb") as fh:
                                fh.write(buf)
                            os.replace(path + ".part", path)
                            self.bytes += len(buf)
                            self.fetched += 1
                            self.requests += 1
                            return "new"
                        except Exception as e:  # noqa: BLE001
                            if attempt == max(int(retries), 1) - 1:
                                print(f"stream-plan: FAILED {url}: {e!r}", flush=True)
                                self.failed += 1
                                return "fail"
                            await asyncio.sleep(2 * (attempt + 1))
            finally:
                if not lk.locked():
                    self.lock.pop(path, None)


async def fetch_group_meta(f, base):
    """Fetch a pyramid group's `zarr.json` and every level's `zarr.json` into the mirror, so that
    `data.rungs(base)` opens the pyramid locally. Level names come from the group's OME multiscales; the
    integer names of a CT mirror are probed as well (a 404 probe leaves nothing behind)."""
    st, _ = await f.get(f"{base}/zarr.json")
    names = []
    if st in ("have", "new"):
        j = json.load(open(f"{base}/zarr.json"))
        at = j.get("attributes", j) if isinstance(j, dict) else {}
        ms = (at.get("ome") or at).get("multiscales") if isinstance(at, dict) else None
        names = [str(d["path"]) for d in ms[0]["datasets"]] if ms else []
    else:  # a CT mirror without a group object: the integer level names are probed below
        os.makedirs(base, exist_ok=True)
    cand = list(dict.fromkeys(names + [str(i) for i in range(data.NRUNGS)]))
    res = await asyncio.gather(*[f.get(f"{base}/{n}/zarr.json") for n in cand])
    keep = []
    for n, (st, _) in zip(cand, res):
        if st in ("have", "new"):
            keep.append(n)
            continue
        for q in (f"{base}/{n}/zarr.json.absent", f"{base}/{n}"):  # a probe must not leave a level directory
            try:
                os.remove(q) if os.path.isfile(q) else os.rmdir(q)
            except OSError:
                pass
    assert keep, f"{data.remote(base)}: no pyramid levels on the origin"
    return keep


# ------------------------------------------------------------------ the no-repeat walk

def region_walk(stores, rungs=True, seed=0, patch=256, region=1024, boost=None, exclude=(), records=False,
                walk="once", visits_max=64, epoch=0):
    """THE region walk, as a plain list -- the contract between the stream planner and anything else that
    has to visit the same regions in the same order (cloud/teacher_regions.py runs the upstream teacher
    over them and writes `data.teacher_region_path` stores, which the loader then prefers as the rung-2/3
    target; see `data.read_teacher`).

        [(source line, rung, (z, y, x) origin in rung-k voxels), ...]   in VISIT order

    with `records=True` the full dicts instead (adding "size", the tile edge, and "w", the draw weight).
    Both sides MUST pass the same stores, rungs, boost, patch, region, exclude and seed: the list is
    enumerated by `data.region_list` (shard-aligned tiles of each target box, all-air ones dropped) and
    ordered by `data.walk_order`, and every one of those is an input to both. At rung 2 an origin is a
    multiple of 1024 (the CT and the export are both 1024^3-sharded there), which is the region name.

    `stores` is a stores file path or the lines themselves; `rungs` True or the allowed set. `walk`/`visits_max`/
    `epoch` must match the planner's (`--walk mix` = region_visits then the shuffle; a region is listed at its first visit)."""
    if isinstance(stores, str) and os.path.exists(stores):
        stores = [l.strip() for l in open(stores) if l.strip() and not l.startswith("#")]
    lines = [stores] if isinstance(stores, str) else list(stores)
    ex = [e if isinstance(e, (tuple, list)) and len(e) == 2 and not isinstance(e[0], str) else data.val_box(e)
          for e in ([exclude] if isinstance(exclude, str) else list(exclude or []))]
    regs = data.region_list(data.source_groups(lines), patch=patch, region=region,
                            allowed=None if rungs is True else set(rungs), boost=boost, exclude=ex)
    if walk == "mix":  # exactly what the planner does in --walk mix: visits, then the seeded shuffle
        regs = data.region_visits(regs, visits_max)
    order = data.walk_order([r["w"] for r in regs], seed + 7919 * epoch)
    out, seen = [], set()
    for j in order:  # a region visited several times is listed once, at its first visit
        r = regs[int(j)]
        key = (r["s"], r["k"], tuple(r["lo"]))
        if key not in seen:
            seen.add(key); out.append(r)
    return out if records else [(lines[r["s"]], r["k"], tuple(r["lo"])) for r in out]


class Walk:
    """The region list and a cursor over it: every region visited ONCE, in the weighted-shuffled order
    `data.walk_order` gives (see the walk section of usrm2/data.py).

    `regions.jsonl` is the list in enumeration order and `walk.json` the cursor ({"i", "epoch"}) plus the
    fingerprint of the configuration it was built for. The visit ORDER is not stored: it is a pure function
    of (seed, epoch), so a resumed planner recomputes it and continues at the cursor -- nothing before the
    cursor is ever handed out again. With `epochs > 1` the list is re-permuted with the next epoch's seed
    when it runs out; after the last epoch the planner writes `epoch_done` and stops."""

    def __init__(self, d, seed=0, epochs=1):
        self.dir, self.seed, self.epochs = str(d), int(seed), max(int(epochs), 1)
        self.regions, self.order, self.i, self.epoch = [], None, 0, 0

    @property
    def path(self):
        return os.path.join(self.dir, REGIONS)

    def load(self, fp):
        """Pick an existing list up when it was built for the same configuration."""
        try:
            st = json.load(open(os.path.join(self.dir, WALK)))
        except Exception:  # noqa: BLE001
            return False
        if st.get("fp") != fp or not os.path.exists(self.path):
            return False
        self.regions = [json.loads(l) for l in open(self.path) if l.strip()]
        self.i, self.epoch = int(st.get("i", 0)), int(st.get("epoch", 0))
        self._permute()
        return True

    def build(self, regions, fp):
        with open(self.path + ".tmp", "w") as f:
            for r in regions:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        os.replace(self.path + ".tmp", self.path)
        self.regions, self.i, self.epoch, self.fp = regions, 0, 0, fp
        self._permute()
        self.save(fp)

    def _permute(self):
        self.order = data.walk_order([r["w"] for r in self.regions], self.seed + 7919 * self.epoch)

    def save(self, fp):
        tmp = os.path.join(self.dir, WALK + ".tmp")
        json.dump({"i": self.i, "epoch": self.epoch, "n": len(self.regions), "fp": fp}, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, WALK))

    @property
    def done(self):
        return self.epoch >= self.epochs or (self.epoch == self.epochs - 1 and self.i >= len(self.regions))

    def take(self, fp):
        """(ordinal, region) -- the ordinal is unique across epochs and seeds the region's rng -- or None
        when the walk is over."""
        if self.i >= len(self.regions):
            if self.epoch + 1 >= self.epochs:
                return None
            self.epoch, self.i = self.epoch + 1, 0
            self._permute()
        j = int(self.order[self.i])
        ordinal = self.epoch * len(self.regions) + self.i
        self.i += 1
        self.save(fp)
        return ordinal, self.regions[j]


# ------------------------------------------------------------------ the planner

class Hook:
    """What `data.Patches._rung_draw` calls before it reads. The sampler runs in a thread (one per worker
    stream, so the streams overlap); every fetch is handed back to the planner's event loop."""

    def __init__(self, pl, loop):
        self.pl, self.loop, self.keys = pl, loop, []

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def data(self, s, k, lo):
        return self._run(self.pl.fetch_data(self, s, k, lo))

    def ctx(self, s, k, lo):
        return self._run(self.pl.fetch_ctx(self, s, k, lo))


class Planner:
    """The sampler, fetching instead of returning. `run()` is the whole process."""

    def __init__(self, stores_file, queue, patch=256, rungs=True, rung_boost=None, seed=0, workers=4,
                 ahead=400, cache_gb=20.0, ctx=(), aug="geo", dense_pow=0.0, require_targets=False,
                 val=None, jobs=48, report=30.0, limit=0, val_rungs=data.VAL_RUNGS, val_patches=32,
                 region=0, windows_per_region=64, walk=None, active_regions=4, epochs=1, region_fails=0,
                 teacher_regions=None, visits_max=64, cascade="off", verso=False, verso_regions=None,
                verso_url=None, planes=(), scan_meta=None):
        from usrm2 import aug as A
        self.dir = str(queue)
        self.stores_file, self.seed, self.W, self.ahead = str(stores_file), int(seed), int(workers), int(ahead)
        self.cache_max, self.jobs, self.report, self.limit = float(cache_gb) * 2 ** 30, int(jobs), float(report), int(limit)
        self.patch, self.ctx, self.require_targets = data.shape3(patch), tuple(ctx), bool(require_targets)
        self.cfg = A.get(aug)
        self.kw = dict(patch=patch, rungs=rungs, rung_boost=dict(rung_boost or {}), ctx=tuple(ctx),
                       aug=self.cfg, sym=self.cfg.get("sym", True), dense_pow=dense_pow,
                       require_targets=bool(require_targets), seed=int(seed),
                       region=int(region or 0), windows_per_region=int(windows_per_region),
                       region_fails=int(region_fails or 0), teacher_regions=teacher_regions,
                       cascade=str(cascade or "off"), verso=bool(verso),
                       verso_regions=str(verso_regions) if verso_regions else teacher_regions,
                       planes=data.parse_planes(planes), scan_meta=scan_meta)
        # The METADATA / RADIUS planes (section 29) change the STEM, so a queue planned with one plane
        # set may only be replayed by a run that builds the same one; meta.json records it and
        # `data.Patches._open_stream` asserts it, exactly as it does for --ctx and --cascade.
        self.planes = data.parse_planes(planes)
        # THE VERSO OUTPUT (docs/unified_design.md section 23). The verso target has no pyramid: it is
        # published, region by region, as the pod finishes it. With `--verso-regions-url` the planner
        # fetches a region's store the first time it plans that region -- two objects, `zarr.json` and the
        # single shard `c/0/0/0` -- into `<verso_regions>/verso/`, where the training workers read it from
        # like any local region store. A 404 is "not published yet" and is remembered for VERSO_TTL only.
        self.verso, self.verso_url = bool(verso), (str(verso_url) if verso_url else None)
        assert not (self.verso and self.verso_url) or self.kw["verso_regions"], \
            "--verso-regions-url needs --teacher-regions (or --verso-regions): a directory to download into"
        self.verso_probe = {}          # local store path -> (state, when); "no" expires, "yes" does not
        self.verso_bytes = self.verso_have = 0
        self.cascade = str(cascade or "off")
        self.walk_mode = None if not walk else str(walk)
        assert self.walk_mode in (None, "once", "mix"), f"--walk {walk}: 'once' or 'mix'"
        assert not self.walk_mode or region, "--walk needs --region (the walk is over regions)"
        self.K = max(int(active_regions), 1)
        self.visits_max = max(int(visits_max), 1)
        self.epochs = max(int(epochs), 1)
        self.walk = None
        self.active = [None] * self.K   # the open regions, emitted round robin
        self.rr = 0                     # whose turn it is
        self.regions_done = self.region_bytes = 0
        self.group = []                 # emitted windows not yet written: the queue grows a stream's worth at a time
        self.mirror = {}                # level dir -> which shards the LOCAL mirror already knows (startup)
        self.val, self.val_rungs, self.val_patches = val, tuple(val_rungs), int(val_patches)
        self.pin = set()   # the validation grid's chunks: fetched once, never evicted
        self.axis_rung = 9  # the rung a missing scroll axis is derived from (307 um: a few MB per scroll)
        self.dirs, self.dir_ix = [], {}          # the level directories entries refer to, by index
        self.whole, self.whole_any, self.whole_lock = {}, {}, {}  # level dir -> bytes kept forever / served / guard
        self.ref, self.size = {}, {}             # chunk path -> last referencing queue index / its size
        self.cache_bytes = self.evicted = self.evicted_bytes = 0
        self.index = 0                           # the next queue index to emit
        self.pending = [[] for _ in range(self.W)]
        self.states = [None] * self.W
        self.lines, self.dirty, self.full = [], False, False
        self.verso_lock = {}          # one probe/download per region, however many windows want it

    # ---- bookkeeping -------------------------------------------------------

    def path_of(self, arr, ix):
        d = data.array_dir(arr)
        di = self.dir_ix.get(d)
        if di is None:
            di = self.dir_ix[d] = len(self.dirs)
            self.dirs.append(d)
            self.dirty = True
        return di, chunk_key(arr, ix), f"{d}/{chunk_key(arr, ix)}"

    def charge(self, path, sz=None):
        """Book a buffered shard against the cache budget. Every fetched shard is booked, including the ones
        a REJECTED candidate pulled: those are referenced by no queue entry (ref -1) and are the first thing
        eviction takes, but they are on the disk and the budget has to see them."""
        if sz is None:
            sz = os.path.getsize(path) if os.path.exists(path) else 0
        self.cache_bytes += sz - self.size.get(path, 0)
        self.size[path] = sz
        if path not in self.pin:
            self.ref.setdefault(path, -1)
        return sz

    def mirrored(self, d, ix):
        """Does the pre-existing local mirror already own this shard? `data.chunk_index`, snapshotted before
        the planner fetched anything, is the authority: `mirror.json` says which shards the mirror KNOWS
        (a complete level, or the boxes of a partially pulled one -- there an absent file is air), plus
        whatever was on the disk at startup. Such a shard is a hit that is never fetched and never evicted;
        only what the planner pulled itself is the rolling buffer."""
        pres = self.mirror.get(d, False)
        if pres is None:  # the whole level was already known at startup
            return True
        if pres is False:
            return False
        z, y, x = (int(v) for v in ix)
        return bool(z < pres.shape[0] and y < pres.shape[1] and x < pres.shape[2] and pres[z, y, x])

    async def _fetch(self, arr, keys, record=None, pin=False):
        """Fetch a level's shards, skipping the ones the local mirror owns. Returns True when anything is
        readable there (a mirrored shard that is not on the disk is air, exactly as the mirror means it)."""
        want, got = [], False
        for ix in keys:
            di, key, q = self.path_of(arr, ix)
            if self.mirrored(self.dirs[di], ix):
                self.f.mirror += 1
                got = got or os.path.exists(q)
                continue
            want.append((di, key, q))
        res = await asyncio.gather(*[self.f.get(q) for _, _, q in want])
        nb = 0
        for (di, key, q), (st, v) in zip(want, res):
            if st not in ("have", "new"):
                continue
            got, nb = True, nb + v
            self.charge(q)
            if pin:
                self.pin.add(q)
                self.ref.pop(q, None)
            elif record is not None:
                record.append((di, key))
        return got, nb

    async def _need(self, hook, pyr, k, lo, record=True, pin=False):
        """Fetch the shards one `read_rung` will touch, whole. Returns False when the origin served none."""
        arr, a, b, whole = rung_range(pyr, k, lo, self.patch)
        d = data.array_dir(arr)
        if whole:  # a level small enough that every worker keeps it decoded: fetched once, kept for the run
            if d in self.whole:
                return self.whole_any[d]
            async with self.whole_lock.setdefault(d, asyncio.Lock()):
                if d not in self.whole:
                    self.whole_any[d], self.whole[d] = await self._fetch(arr, all_keys(arr), record=None)
                    self.cache_bytes += self.whole[d]
            return self.whole_any[d]
        return (await self._fetch(arr, keys_in(arr, a, b),
                                  record=None if pin or not record else hook.keys, pin=pin))[0]

    async def fetch_data(self, hook, s, k, lo):
        """The CT cube and the targets of a candidate window. False = reject (`--require-targets` and the
        origin serves no target chunk at all for this window: nothing was exported there).

        Under `--verso-regions-url` this is also where the window's VERSO region store is made local, before
        the sampler asks `data.Patches._verso_store` whether there is one: one probe per region (cached), so
        the cost per window is a dict lookup."""
        await self.fetch_verso(s, k, lo)
        res = await asyncio.gather(self._need(hook, s["ct_pyr"], k, lo),
                                   *[self._need(hook, t["pyr"], k, lo) for t in s["targets"].values()])
        return bool(any(res[1:])) if self.require_targets and s["targets"] else True

    async def fetch_verso(self, s, k, lo):
        """Make the verso region store covering this window local, if it is published. One HEAD-equivalent
        (a GET of `zarr.json`, a few hundred bytes) decides: 404 = the pod has not got there yet, remembered
        for `VERSO_TTL` seconds so a long run picks the region up later; `done` false = still being written,
        the same; otherwise the shard is pulled too and the store is complete on disk. Nothing here is
        charged to the rolling chunk buffer -- a region store is read by every window of the region and is
        not evicted (see the docs: `<DIR>/verso` grows with the walk)."""
        if not (self.verso and self.verso_url) or int(k) not in (2, 3) or s is not self.ds.srcs[0]:
            return
        d = int(k) - 2
        lo2 = np.asarray(lo, np.int64) << d
        hi2 = ((np.asarray(lo, np.int64) + self.patch) << d) - 1
        a, b = lo2 // data.REGION, hi2 // data.REGION
        if not np.array_equal(a, b) or (lo2 < 0).any():
            return                                      # the window straddles two regions: no store anyway
        org = a * data.REGION
        path = data.teacher_region_path(org, data.VERSO, self.kw["verso_regions"])
        st, when = self.verso_probe.get(path, (None, 0.0))
        if st == "yes" or (st == "no" and time.time() - when < VERSO_TTL):
            return
        lk = self.verso_lock.setdefault(path, asyncio.Lock())
        async with lk:
            st, when = self.verso_probe.get(path, (None, 0.0))
            if st == "yes" or (st == "no" and time.time() - when < VERSO_TTL):
                return
            url = data.verso_region_url(org, self.verso_url)
            b0 = self.f.bytes
            got = await self.f.fetch_url(f"{url}/zarr.json", f"{path}/zarr.json")
            ok = got in ("have", "new")
            if ok:
                try:
                    j = json.load(open(f"{path}/zarr.json"))
                    at = j.get("attributes", j) if isinstance(j, dict) else {}
                    ok = bool(at.get("done"))
                except Exception:  # noqa: BLE001  (half-written or not JSON)
                    ok = False
                if not ok:
                    os.remove(f"{path}/zarr.json")      # not finished: do not leave a store a loader opens
            if ok:
                for q in data.REGION_FILES[1:]:
                    ok = ok and await self.f.fetch_url(f"{url}/{q}", f"{path}/{q}") in ("have", "new")
            self.verso_probe[path] = ("yes" if ok else "no", time.time())
            if ok:
                self.verso_bytes += self.f.bytes - b0
                self.verso_have += 1
                print(f"stream-plan: verso region {org.tolist()} -> {path} "
                      f"({(self.f.bytes - b0) / 2 ** 20:.1f} MiB, {self.verso_have} so far)", flush=True)

    async def fetch_ctx(self, hook, s, k, lo):
        """The nine context cubes, fetched only once the window has been accepted -- plus, under
        `--cascade`, what the cascade channel reads: the rung-(k+1) TARGET block over the patch footprint
        (`mask`/`mix`) and, for `self`/`mix`, the TENTH context cube (rung k + ctx[-1] + 1)."""
        c0 = np.asarray(lo, np.int64) + self.patch // 2
        jobs = [self._need(hook, s["ct_pyr"], k + int(d), c0 // (1 << int(d)) - self.patch // 2)
                for d in self.ctx]
        if self.cascade != "off":
            d = (int(self.ctx[-1]) + 1) if self.ctx else 1
            if self.cascade in ("self", "mix"):
                jobs.append(self._need(hook, s["ct_pyr"], k + d, c0 // (1 << d) - self.patch // 2))
            if k + 1 < data.NRUNGS:  # the coarse target block is half a patch, but shards are shards
                jobs += [self._need(hook, t["pyr"], k + 1, np.asarray(lo, np.int64) // 2)
                         for t in s["targets"].values()]
        await asyncio.gather(*jobs)
        return True

    # ---- the queue ---------------------------------------------------------

    # ---- the walk ----------------------------------------------------------

    def fingerprint(self, lines):
        """What the region list was built for: a different one means a different list."""
        return json.dumps({"stores": lines, "patch": [int(v) for v in self.patch],
                           "region": self.kw["region"], "seed": self.seed, "epochs": self.epochs,
                           "rungs": self.kw["rungs"] if self.kw["rungs"] is True else sorted(self.kw["rungs"]),
                           "boost": {str(k): v for k, v in self.kw["rung_boost"].items()},
                           "walk": self.walk_mode, "visits_max": self.visits_max,
                           "val": _val_meta(self.val)}, sort_keys=True)

    async def build_walk(self, lines):
        """Enumerate every region once (or pick up the list a previous planner left here)."""
        self.walk_fp = self.fingerprint(lines)
        self.walk = Walk(self.dir, seed=self.seed, epochs=self.epochs)
        loaded = self.walk.load(self.walk_fp)
        if not loaded or not self.walk.done:  # a marker from an earlier walk must not stop the new trainer
            try:
                os.remove(os.path.join(self.dir, EPOCH_DONE))
            except OSError:
                pass
        if loaded:
            print(f"stream-plan: walk resumed at region {self.walk.i}/{len(self.walk.regions)} "
                  f"of epoch {self.walk.epoch + 1}/{self.epochs}", flush=True)
            return
        t0 = time.time()
        for s in self.ds.srcs:  # the occupancy check reads one coarse level of each target: pull it first
            for t in s["targets"].values():
                arr = t["pyr"][data.occupancy_rung(t["pyr"])]
                await self._fetch(arr, all_keys(arr), record=None)
        regions = data.region_list(self.ds.srcs, patch=self.patch, region=self.kw["region"],
                                   allowed=None if self.kw["rungs"] is True else set(self.kw["rungs"]),
                                   boost=self.kw["rung_boost"], exclude=self.ds.ex,
                                   log=lambda q: print("stream-plan " + q, flush=True))
        assert regions, "the walk is empty: no region of any source has a target at any rung"
        nreg = len(regions)
        if self.walk_mode == "mix":  # visits proportional to the weight: the rung mix holds all the way
            regions = data.region_visits(regions, self.visits_max)
        self.walk.build(regions, self.walk_fp)
        n = self.kw["windows_per_region"]
        print(f"stream-plan: walk over {nreg} regions / {len(regions)} visits ({len(regions) * n} windows "
              f"per epoch, {self.epochs} epoch(s)) enumerated in {time.time() - t0:.1f} s", flush=True)

    def _meta(self, lines):
        return {"stores": lines, "stores_file": self.stores_file, "patch": [int(v) for v in self.patch],
                "workers": self.W, "seed": self.seed, "ctx": list(self.ctx),
                "rungs": self.kw["rungs"] if self.kw["rungs"] is True else sorted(self.kw["rungs"]),
                "rung_boost": {str(k): v for k, v in self.kw["rung_boost"].items()},
                "aug": self.cfg, "dense_pow": self.kw["dense_pow"], "cascade": self.cascade,
                "planes": list(self.planes),
                "require_targets": self.require_targets, "channels": self.ds.channels,
                "region": self.kw["region"], "windows_per_region": self.kw["windows_per_region"],
                "walk": self.walk_mode, "active_regions": self.K, "epochs": self.epochs,
                "teacher_regions": self.kw["teacher_regions"],
                "verso": self.verso, "verso_regions": self.kw["verso_regions"], "verso_url": self.verso_url,
                "regions": len(self.walk.regions) if self.walk else 0,
                "val": _val_meta(self.val),
                "dirs": self.dirs, "whole": sorted(self.whole)}

    def write_meta(self, lines):
        tmp = os.path.join(self.dir, META + ".tmp")
        json.dump(self._meta(lines), open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, META))

    def consumed(self):
        """The queue index the trainer has certainly consumed (its own report, minus the loader's prefetch)."""
        try:
            j = json.load(open(os.path.join(self.dir, CONSUMED)))
            return int(j["i"]) - int(j.get("margin", 0))
        except Exception:  # noqa: BLE001
            return -1

    def emit(self, desc, keys):
        if self.dirty:  # a level the consumer's meta.json does not know yet: republish the directory table
            self.write_meta(self.lines)
            self.dirty = False
        rec = dict(desc)
        rec["i"] = self.index
        rec["c"] = [[di, key] for di, key in keys]
        with open(self.qf, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
        for di, key in keys:
            p = f"{self.dirs[di]}/{key}"
            if p in self.pin:  # a validation chunk: kept for the whole run
                continue
            self.charge(p)  # a partial shard grows as later windows add inner chunks to it
            self.ref[p] = self.index
        self.index += 1

    def save_state(self):
        tmp = os.path.join(self.dir, STATE + ".tmp")
        json.dump({"i": self.index, "rng": self.states}, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, STATE))

    def evict(self):
        """Delete the chunks of consumed windows, oldest reference first, down to 90 % of the budget."""
        if self.cache_bytes <= self.cache_max:
            return
        bound = self.consumed()
        target = 0.9 * self.cache_max
        for p, i in sorted(self.ref.items(), key=lambda q: q[1]):
            if i > bound or self.cache_bytes <= target:
                break
            try:
                os.remove(p)
            except OSError:
                pass
            self.cache_bytes -= self.size.pop(p, 0)
            self.ref.pop(p, None)
            self.evicted += 1

    async def ensure_axis(self, ct):
        """Every scroll needs its OWN axis for the radial channel. A published umbilicus is used when the
        origin serves one; otherwise it is derived from the scroll's own CT, which means pulling one coarse
        level whole first (rung 9 of Paris 4 is 2.3 MB)."""
        sc = data.scroll_of(ct)
        if not sc or os.path.exists(data.umbilicus_path(sc)):
            return
        t1 = time.time()
        try:
            pyr = data.rungs(ct)
            k = max(r for r in pyr if r <= self.axis_rung)
            arr = pyr[k]
            await asyncio.gather(*[self.f.get(f"{data.array_dir(arr)}/{chunk_key(arr, ix)}")
                                   for ix in all_keys(arr)])
            data.CTX_CACHE.clear()
            u = U.ensure(ct, scroll=sc, urls=U.published_urls(ct, sc), rung=self.axis_rung)
            print(f"stream-plan: axis for {sc} -> {u} ({time.time() - t1:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001  (fall back to the configured default and say so)
            print(f"stream-plan: no axis for {sc} ({e!r}); falling back to {data.UMBILICUS}", flush=True)

    async def prefetch_val(self):
        """The held-out box, at every rung `evaluate` scores it, read from the same pyramids as training
        (data.val_grid_rungs). Its chunks are fetched once at startup and pinned: validation must not read
        zeros out of an unfetched mirror, and the buffer must not evict them 500 steps later."""
        if self.val is None or not self.ds.srcs:
            return 0
        o2, s2 = data.val_box(self.val)
        p3, s, hook, n = self.patch, self.ds.srcs[0], Hook(self, None), 0
        for k in self.val_rungs:
            d = k - 2
            org, sz = (o2 >> d, np.maximum(s2 >> d, 1)) if d >= 0 else (o2 << -d, s2 << -d)
            corners = [(z, y, x) for z in range(0, max(int(sz[0]) - int(p3[0]), 0) + 1, int(p3[0]))
                       for y in range(0, max(int(sz[1]) - int(p3[1]), 0) + 1, int(p3[1]))
                       for x in range(0, max(int(sz[2]) - int(p3[2]), 0) + 1, int(p3[2]))]
            if self.val_patches and len(corners) > self.val_patches:
                corners = [corners[i] for i in np.linspace(0, len(corners) - 1, self.val_patches).astype(int)]
            for i in range(0, len(corners), 4):  # four windows at a time, all their levels in parallel
                jobs = []
                for c in corners[i:i + 4]:
                    lo = org + np.array(c, np.int64)
                    c0 = lo + p3 // 2
                    jobs.append(self._need(hook, s["ct_pyr"], k, lo, pin=True))
                    jobs += [self._need(hook, t["pyr"], k, lo, pin=True) for t in s["targets"].values()]
                    jobs += [self._need(hook, s["ct_pyr"], k + int(dd), c0 // (1 << int(dd)) - p3 // 2,
                                        pin=True) for dd in self.ctx]
                    n += 1
                await asyncio.gather(*jobs)
        return n

    # ---- resume ------------------------------------------------------------

    def resume(self):
        """Pick the queue up where it stopped: the index and per-chunk references from queue.jsonl, the rng
        state of each worker stream from state.json (saved with every emitted entry)."""
        st = {}
        if os.path.exists(os.path.join(self.dir, STATE)):
            st = json.load(open(os.path.join(self.dir, STATE)))
        self.states = list((st.get("rng") or []))[:self.W]
        self.states += [None] * (self.W - len(self.states))
        if not os.path.exists(self.qf):
            return
        keep = [l for l in open(self.qf) if l.endswith("\n")]  # a torn last line is dropped
        n = int(st.get("i", len(keep)))
        keep = keep[:n] if 0 <= n <= len(keep) else keep
        with open(self.qf, "w") as f:
            f.writelines(keep)
        self.index = len(keep)
        for line in keep:
            rec = json.loads(line)
            for di, key in rec["c"]:
                if di < len(self.dirs):
                    self.ref[f"{self.dirs[di]}/{key}"] = rec["i"]
        for q in list(self.ref):
            self.size[q] = os.path.getsize(q) if os.path.exists(q) else 0
        self.cache_bytes = sum(self.size.values())
        print(f"stream-plan: {len(self.ref)} buffered objects known from the queue", flush=True)
        print(f"stream-plan: resuming at index {self.index} ({self.cache_bytes / 2 ** 30:.2f} GiB buffered)",
              flush=True)

    # ---- the run -----------------------------------------------------------

    async def run(self):
        import aiohttp
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, PROGRESS), exist_ok=True)
        self.qf = os.path.join(self.dir, QUEUE)
        lines = [l.strip() for l in open(self.stores_file) if l.strip() and not l.startswith("#")]
        assert lines, f"{self.stores_file} lists no store groups"
        timeout = aiohttp.ClientTimeout(total=600, sock_connect=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.f = Fetcher(session, self.jobs)
            t0 = time.time()
            bases = list(dict.fromkeys(q.strip() for l in lines for q in l.split(",") if q.strip()))
            for b in bases:
                await fetch_group_meta(self.f, data.pyramid_base(b))
            print(f"stream-plan: {len(bases)} pyramids, metadata in {time.time() - t0:.1f} s", flush=True)
            for ct in dict.fromkeys(l.split(",")[0].strip() for l in lines):
                await self.ensure_axis(ct)
            self.ds = data.Patches(stores=lines, exclude=[] if self.val is None else self.val, **self.kw)
            self.ds._open_rungs()
            # the directory table has to exist before resume() can map the recorded keys back to paths;
            # and `data.chunk_index`, read here before anything is fetched, is the record of what the
            # pre-existing local mirror owns (never fetched, never evicted -- see `mirrored`)
            for s in self.ds.srcs:
                for pyr in [s["ct_pyr"]] + [t["pyr"] for t in s["targets"].values()]:
                    for a in pyr.values():
                        self.path_of(a, (0, 0, 0))
                        self.mirror[data.array_dir(a)] = data.chunk_index(a)[1]
            nm = sum(1 for v in self.mirror.values() if v is None)
            print(f"stream-plan: {nm}/{len(self.mirror)} levels already complete in the local mirror",
                  flush=True)
            self.lines = lines
            self.resume()
            if self.walk_mode:
                await self.build_walk(lines)
            self.write_meta(lines)  # before the val prefetch: the trainer may already be opening the pyramids
            self.dirty = False
            t1 = time.time()
            nv = await self.prefetch_val()
            open(os.path.join(self.dir, "val_ready"), "w").write(str(nv))  # the trainer waits for this
            if nv:
                print(f"stream-plan: {nv} validation windows prefetched in {time.time() - t1:.1f} s "
                      f"({self.f.bytes / 1e6:.1f} MB, {self.f.fetched} chunks so far)", flush=True)
            loop = asyncio.get_running_loop()
            self.stop = False
            self.drained = [False] * self.K
            tasks = ([asyncio.create_task(self.visit(j, loop)) for j in range(self.K)] if self.walk_mode
                     else [asyncio.create_task(self.stream(w, loop)) for w in range(self.W)])
            tasks.append(asyncio.create_task(self.emitter(lines)))
            tasks.append(asyncio.create_task(self.keeper()))
            try:
                await asyncio.gather(*tasks)
            finally:
                self.stop = True
                for t in tasks:
                    t.cancel()
                self.report_line(t0, 0, 0)

    async def keeper(self):
        """Evict, and decide whether the buffer is full. A shard is ~17-56 MB, so `--cache-gb` -- not
        `--ahead` -- is usually what stops the planner: the streams and the emitter both park while the
        buffer is over budget and nothing more is evictable (nothing else would keep the disk bounded
        before the trainer has consumed its first window)."""
        while not self.stop:
            self.evict()
            self.full = self.cache_bytes > self.cache_max or self.index - self.consumed() > self.ahead
            await asyncio.sleep(0.5)

    async def stream(self, w, loop):
        """One loader worker's rng stream: draw, fetch, reject, park the accepted windows for the emitter."""
        rng = np.random.default_rng(self.seed + 1000 * w)
        if self.states[w]:
            rng.bit_generator.state = _unjson_state(self.states[w])
        hi = max(4, self.ahead // self.W)
        st = self.ds.region_state()
        while not self.stop:
            if self.full or len(self.pending[w]) >= hi:
                await asyncio.sleep(0.05)
                continue
            hook = Hook(self, loop)
            desc = await asyncio.to_thread(_draw, self.ds, rng, hook, st)
            if desc is not None:
                self.pending[w].append((desc, hook.keys, _json_state(rng.bit_generator.state)))

    async def visit(self, j, loop):
        """One of the K ACTIVE REGIONS. It takes the next region of the walk, draws that region's windows
        into its own buffer, and is released -- free to take another region -- only once the emitter has
        queued every one of them. All K regions stay resident while they are open, which is what keeps the
        cache hit rate up while consecutive queue entries come from different regions."""
        cap = max(4, self.ahead // self.K)
        while not self.stop:
            slot = self.active[j]
            if slot is not None and slot["done"] and not slot["out"]:  # every window queued: let it go
                self.active[j], slot = None, None
                self.regions_done += 1
            if slot is None:
                got = self.walk.take(self.walk_fp)
                if got is None:  # the walk is over
                    self.drained[j] = True
                    return
                ordinal, reg = got
                self.active[j] = slot = {"g": int(ordinal), "out": [], "done": False,
                                         "rng": np.random.default_rng([self.seed, int(ordinal)]),
                                         "st": self.ds.region_state(reg)}
            if slot["done"] or self.full or len(slot["out"]) >= cap:
                await asyncio.sleep(0.02)
                continue
            hook = Hook(self, loop)
            desc = await asyncio.to_thread(_draw, self.ds, slot["rng"], hook, slot["st"])
            if desc is not None:
                desc["g"] = slot["g"]
                slot["out"].append((desc, hook.keys))
            st = slot["st"]
            if st["left"] <= 0 or st["fails"] >= self.ds.region_fails:
                slot["done"] = True

    def next_slot(self):
        """Round robin over the active regions: consecutive entries come from different regions."""
        for d in range(self.K):
            j = (self.rr + d) % self.K
            s = self.active[j]
            if s is not None and s["out"]:
                self.rr = (j + 1) % self.K
                return s
        return None

    def finish(self, lines):
        """The walk is done. The last partial group of windows (fewer than one per replay stream) is never
        written, so the queue ends on a stream boundary and a DDP run's ranks stop together."""
        n, rec = self.index, {"windows": self.index, "dropped": len(self.group),
                              "regions": self.regions_done, "epochs": self.epochs}
        self.group = []
        tmp = os.path.join(self.dir, EPOCH_DONE + ".tmp")
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, os.path.join(self.dir, EPOCH_DONE))
        print(f"stream-plan: walk complete -- {self.regions_done} regions, {n} windows " + json.dumps(rec),
              flush=True)
        self.stop = True

    async def emitter(self, lines):
        """Entry i goes to stream i % W, so the queue is the round robin the DataLoader will replay."""
        t0, last, b0, n0 = time.time(), time.time(), self.f.bytes, self.f.fetched
        while not self.stop:
            if time.time() - last > self.report:
                last, b0, n0, t0 = self.report_line(t0, b0, n0), self.f.bytes, self.f.fetched, time.time()
            if self.limit and self.index >= self.limit:
                self.stop = True
                return
            if self.full:  # over --cache-gb, or --ahead windows in front of the trainer: wait for it
                await asyncio.sleep(0.2)
                continue
            if self.walk_mode:
                slot = self.next_slot()
                if slot is None:
                    if all(self.drained):
                        return self.finish(lines)
                    await asyncio.sleep(0.02)
                    continue
                self.group.append(slot["out"].pop(0))
                if len(self.group) >= self.W:  # the queue only ever ends on a stream boundary
                    for d, ks in self.group:
                        self.emit(d, ks)
                    self.group = []
                    self.save_state()
                continue
            w = self.index % self.W
            if not self.pending[w]:
                await asyncio.sleep(0.02)
                continue
            desc, keys, state = self.pending[w].pop(0)
            self.emit(desc, keys)
            self.states[w] = state
            self.save_state()

    def report_line(self, t0, b0, n0):
        dt = max(time.time() - t0, 1e-6)
        rec = {"t": round(time.time()), "index": self.index, "consumed": self.consumed(),
               "MB_s": round((self.f.bytes - b0) / 1e6 / dt, 1), "chunks_s": round((self.f.fetched - n0) / dt, 1),
               "cache_GiB": round(self.cache_bytes / 2 ** 30, 3), "chunks": self.f.fetched, "full": self.full,
               "GB_total": round(self.f.bytes / 1e9, 3), "absent": self.f.absent, "failed": self.f.failed,
               "evicted": self.evicted, "whole_MiB": round(sum(self.whole.values()) / 2 ** 20, 1),
               "requests": self.f.requests, "pinned": len(self.pin),
               "hit_rate": round((self.f.have + self.f.mirror) /
                                 max(self.f.have + self.f.mirror + self.f.fetched + self.f.absent, 1), 4),
               "mirror": self.f.mirror,
               "B_per_vox": round(self.f.bytes / max(self.index * int(np.prod(self.patch)), 1), 4)}
        if self.walk:
            rec.update(regions=self.regions_done, regions_left=len(self.walk.regions) - self.walk.i,
                       regions_total=len(self.walk.regions), epoch=self.walk.epoch, active=self.K,
                       region_MiB=round(self.f.bytes / max(self.regions_done, 1) / 2 ** 20, 1),
                       **({"verso_stores": self.verso_have,
                           "verso_MiB": round(self.verso_bytes / 2 ** 20, 1)} if self.verso else {}))
            if self.regions_done >= self.K and self.K * self.f.bytes / self.regions_done > 0.5 * self.cache_max:
                print(f"stream-plan: WARNING --cache-gb {self.cache_max / 2 ** 30:.0f} is small for "
                      f"--active-regions {self.K} ({rec['region_MiB']} MiB fetched per region)", flush=True)
        print("stream-plan " + json.dumps(rec), flush=True)
        with open(os.path.join(self.dir, "plan.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return time.time()


def _val_meta(val):
    try:
        return None if val is None else [[int(v) for v in q] for q in data.val_box(val)]
    except Exception:  # noqa: BLE001  (a held-out box named by a store this machine does not mirror)
        return str(val)


def _draw(ds, rng, hook, st):
    """The sampler itself, run in a thread (its fetches go back to the event loop through the hook)."""
    return ds._rung_draw(rng, hook=hook, build=False, st=st)[0]


def _json_state(st):
    """A numpy bit-generator state -> JSON (its integers are 128-bit, which json handles, but numpy ints
    are not JSON-able)."""
    if isinstance(st, dict):
        return {k: _json_state(v) for k, v in st.items()}
    if isinstance(st, (np.integer,)):
        return int(st)
    if isinstance(st, np.ndarray):
        return [int(v) for v in st.tolist()]
    return st


def _unjson_state(st):
    return st


def plan(**kw):
    asyncio.run(Planner(**kw).run())


# ------------------------------------------------------------------ the consumer side

def read_meta(d):
    return json.load(open(os.path.join(str(d), META)))


def have(path):
    """Is a chunk in the buffer, or known to be absent on the origin?"""
    return os.path.exists(path) or os.path.exists(path + ".absent")


def tail(path, stop=None, poll=0.05, poll_max=1.0):
    """Yield (line, seconds waited) from a file that is still being appended to; a torn last line is
    re-read rather than parsed."""
    wait, back = 0.0, poll
    while not os.path.exists(path):
        time.sleep(back)
        wait, back = wait + back, min(back * 1.5, poll_max)
    f = open(path)
    while True:
        pos = f.tell()
        line = f.readline()
        if line and line.endswith("\n"):
            yield line, wait
            wait, back = 0.0, poll
            continue
        f.seek(pos)
        if stop is not None and stop():
            return
        time.sleep(back)
        wait, back = wait + back, min(back * 1.5, poll_max)
