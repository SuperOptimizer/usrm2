#!/usr/bin/env python3
"""Pod: VERSO labels for every rung-2 region of the Paris 4 walk, published to dl.ash2txt.org -- v2.

v2 = v1 plus the EXTRA FIELD STORES. `verso_core_v2` runs one sliding-window pass per region and reads
every head the checkpoint has off that one output (docs/unified_design.md section 30). When the
checkpoint has a DISTANCE channel (cout_t > 2, i.e. a Phase-B run, section 29) the region writes, beside
the usual `region_<z>_<y>_<x>.zarr`:

    region_<z>_<y>_<x>_recto.zarr       uint8  probability * 255                        volcomp q8
    region_<z>_<y>_<x>_surf_sdist.zarr  uint8  d = (v - 128) * 0.25 voxels, 0 = NO DATA  q0 LOSSLESS
    region_<z>_<y>_<x>_nz|ny|nx.zarr    uint8  component = (v - 128) / 127, ZYX          q0
    region_<z>_<y>_<x>_gmag.zarr        uint8  |grad d| * 127                            q0
    region_<z>_<y>_<x>_thickness.zarr   uint8  t = v * 0.25 voxels                       q0
    region_<z>_<y>_<x>_conf.zarr        uint8  confidence * 255                          q0

exactly the encodings, the attrs (`axis_order`, `encoding`, `no_data`, `sign_convention` in words) and
the q of `usrm2 export-tracer`, so a tracer cannot tell a region store from an exported box. They are
uploaded in the same sftp batch as the verso store and are published or not published together.

**For the current production checkpoint (u2, cout 2, no distance channel) v2 is a NO-OP**: the field list
is empty, `verso_core_v2` takes v1's code path kernel for kernel, and the one store written is
byte-identical to v1's. `--compare-v1` runs both cores on one region and asserts exactly that.

**The sign.** v1 runs the flip trick (`--sign -1`: a recto-trained student pointed at the other face). A
distance field from a MIRRORED world has the opposite sign convention to the stores', so v2 refuses to
write field stores with a negative sign -- a Phase-B checkpoint has a real verso channel and is run at
+1.

One region at a time on the GPU (verso_core: the unified 14-channel student at rung 2 with the radial
vector NEGATED, torch.compile'd, bf16, window 256 halo 32, fp16 GPU accumulators), while two helper
threads keep the card fed and the disk clear:

  prefetch : the next `--ahead` regions' level-0 CT shards (levels 1-9 are mirrored whole), fetched with
             ONE keep-alive session and few concurrent requests (verso_mirror.py --boxfile).
  publish  : finished stores, `--batch-up` at a time in ONE sftp session (.part then rename), then the
             local store and the region's level-0 shard are deleted and a `.published` marker written.

Resumable: a region with a marker, or already present in the remote listing taken at start, is skipped.
"""
import argparse, json, os, queue, shutil, subprocess, threading, time
import numpy as np
import torch

from usrm2 import data, predict
import verso_core_v2 as VC

REM = "/volcomp/PHercParis4/representations/predictions/teacher_regions/verso-2.4um"
ROOT = "/workspace"
STORES = f"{ROOT}/verso_regions"
MARKS = f"{ROOT}/verso_published"
CTDIR = "/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr"


def sh(script, *args, timeout=3600):
    return subprocess.run(["bash", script, *[str(a) for a in args]], capture_output=True, text=True,
                          timeout=timeout)


def name(o):
    return f"region_{o[0]}_{o[1]}_{o[2]}"


# The tracer contract's encodings (usrm2/predict.py section 29.6), reused verbatim so a region store and
# an `export-tracer` box are the same bytes. q = 8 for a probability, 0 (LOSSLESS) for every field: their
# code 0 means NO DATA and their other codes are a distance or a normal component, not a probability a
# codec may round (docs/unified_design.md 29.1).
SIGN_WORDS = ("d > 0 and n pointing from the VERSO face towards the RECTO face, i.e. radially OUTWARD "
              "from the scroll axis (dot(n, radial) > 0)")


def field_stores(fields, ckpt_name, step, region, window, halo):
    """{suffix: (uint8 array, encoding, q, extra attrs)} for one region's extra stores.

    `fields` is `verso_core_v2.run_region`'s plane dict. The distance is converted to the RECTO-FACE
    convention here (`d = m - t/2` when the checkpoint is a midline one), and the normal and the gradient
    magnitude are the Scharr gradient of THAT field -- never the net's own normal head and never its
    autograd gradient -- so what a tracer reads is exactly the gradient of what it reads."""
    import numpy as _np
    dch = "midline" if "midline" in fields else ("sdist" if "sdist" in fields else None)
    out = {}
    base = dict(region=region, window=window, halo=halo, ckpt=ckpt_name, step=step, done=True)
    for nm in ("recto", "verso"):
        if nm in fields:
            out[nm] = (VC.u8(fields[nm]), "prob_u8", 8, {"channels": [nm]})
    if dch is None:
        return out
    th = fields.get("thickness")
    d, n, mag, valid = VC.tracer_fields(fields[dch], th)
    if "recto" in fields:      # CT == 0 is masked by both sides; `run_region` already zeroes it
        valid = valid & (fields["recto"] > 0)
    out["surf_sdist"] = (VC.enc_signed(d, valid), "signed_u8_off128_q0.25", 0, {})
    for j, nm in enumerate(("nz", "ny", "nx")):
        out[nm] = (VC.enc_normal(n[j], valid & (mag > 1e-3)), "normal_u8_off128_div127", 0, {})
    out["gmag"] = (_np.clip(_np.rint(mag * VC.NORMAL_SCALE), 0, 255).astype(_np.uint8),
                   "gradmag_u8_x127", 0, {})
    if th is not None:
        out["thickness"] = (_np.clip(_np.rint(th / VC.TRACER_UNIT), 0, 255).astype(_np.uint8),
                            "unsigned_u8_q0.25", 0, {})
    if "conf" in fields:
        out["conf"] = (VC.u8(fields["conf"]), "conf_u8", 0, {})
    for k in out:
        out[k][3].update(base)
    return out


def l0_key(o):
    return f"{CTDIR}/0/c/{o[0] // 1024}/{o[1] // 1024}/{o[2] // 1024}"


def fetch_boxes(regions, sizes, jobs=16):
    """Every CT shard a BATCH of regions reads: level 0 (each region itself, rung 2) and the context boxes at
    rungs 3-11 (verso_boxes). One region needs only ~6-10 shards, too few to fill 16 connections, so the
    batch is fetched in ONE session: 6 regions give ~40-60 objects and the link actually saturates. Shards
    already on disk are skipped, and levels 1-2 stay cached for the neighbouring regions of the walk."""
    f = f"{ROOT}/.boxes_batch.txt"
    with open(f, "w") as fh:
        for o in regions:
            subprocess.run(["python3", f"{ROOT}/verso_boxes.py", *[str(v) for v in o],
                            *[str(v) for v in sizes[o]], "5"], stdout=fh, check=True)
    r = subprocess.run(["python3", f"{ROOT}/verso_mirror.py", "--boxfile", f, "--jobs", str(jobs)],
                       capture_output=True, text=True, timeout=7200)
    os.remove(f)
    return r.returncode == 0


def remote_listing():
    r = sh(f"{ROOT}/sftp_list.sh")
    return names_in(r.stdout)


def names_in(text):
    """The store names an `ls -1` prints: it prints FULL PATHS, and a half-published store as <name>.part."""
    out = set()
    for l in text.splitlines():
        for tok in l.split():
            b = os.path.basename(tok.rstrip("/"))
            if b.startswith("region_") and b.endswith(".zarr"):
                out.add(b)
    return out


def upload(stores):
    """One sftp session for a batch of finished stores: .part tree, puts, rename."""
    lines, names = [], [os.path.basename(s) for s in stores]
    for s in stores:
        n = os.path.basename(s)
        lines.append(f"-mkdir {REM}/{n}.part")
        for root, dirs, _ in os.walk(s):
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), s)
                lines.append(f"-mkdir {REM}/{n}.part/{rel}")
        for root, _, files in os.walk(s):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, s)
                lines.append(f"put {p} {REM}/{n}.part/{rel}")
        lines.append(f"-rename {REM}/{n}.part {REM}/{n}")
    lines.append(f"ls -1 {REM}")   # the session verifies itself: every name must be listed
    lines.append("quit")
    b = f"{ROOT}/.sftp/up.batch"
    open(b, "w").write("\n".join(lines) + "\n")
    sh(f"{ROOT}/sftp_put.sh", b)
    try:
        out = open(f"{ROOT}/.sftp/up.out").read()
    except OSError:
        return False
    listed = names_in(out)
    return all(n in listed for n in names)


def one_region(a, dev):
    """Do ONE region, locally, and (with --compare-v1) prove v2 == v1 on this checkpoint.

    The comparison is BIT-IDENTICAL, not tolerant: with no field heads `verso_core_v2` executes v1's
    statements in v1's order on the same inputs, so the only way the uint8 stores can differ is a bug.
    (`np.array_equal`, not `allclose` -- the memory note's bf16+cuDNN run-to-run floor is dice 0.994, so
    a tolerant check on this path would prove nothing.)"""
    import numpy as _np
    o = tuple(int(v) for v in a.one_region)
    os.makedirs(STORES, exist_ok=True)
    ct = data.open_zarr(VC.VOL)
    size = tuple(int(min(a.region, s - v)) for s, v in zip(ct.shape[-3:], o))
    pyr, ax = data.rungs(VC.VOL), data.axis()
    net = VC.Net(a.ckpt, dev, compile_mode=(a.compile or None), window=a.window)
    assert not (net.nplanes and a.sign < 0), "field heads with a negative --sign: see the module docstring"
    print(f"one_region {name(o)} size {size}: v2 planes {net.plane_names or '(v1: verso only)'}", flush=True)
    R = VC.RegionInputs(o, size, a.window, a.halo, dev, sign=a.sign, pyr=pyr, ax=ax)
    out, nw = VC.run_region(net, R, batch=1, acc_dtype=torch.float16)
    if net.nplanes:
        fields = {nm: out[i].float().cpu().numpy() for i, nm in enumerate(net.plane_names)}
        main = fields["verso"] if "verso" in fields else fields[net.plane_names[0]]
        u8 = _np.clip(_np.rint(main * 255), 0, 255).astype(_np.uint8)
    else:
        fields, u8 = None, torch.clamp(torch.round(out * 255), 0, 255).to(torch.uint8).cpu().numpy()
    del out
    paths = []
    p0 = f"{STORES}/{name(o)}.zarr"
    shutil.rmtree(p0, ignore_errors=True)
    arr = predict.out_array(p0, u8.shape, o, volcomp=True, volume=VC.VOL, rung=2)
    arr[:] = u8
    arr.attrs.update({"channels": ["verso"], "mode": "verso", "region": a.region,
                      "radial_sign": a.sign, "ckpt": os.path.basename(a.ckpt), "step": net.step,
                      "window": a.window, "halo": a.halo, "done": True})
    paths.append(p0)
    if fields:
        for sfx, (v, enc, q, extra) in field_stores(fields, os.path.basename(a.ckpt), net.step,
                                                    a.region, a.window, a.halo).items():
            fp = f"{STORES}/{name(o)}_{sfx}.zarr"
            shutil.rmtree(fp, ignore_errors=True)
            fa = predict.out_array(fp, v.shape, o, volcomp=True, volume=VC.VOL, rung=2,
                                   channels=(sfx,), q=q)
            fa[:] = v[None] if fa.ndim == 4 else v
            fa.attrs.update({"encoding": enc, "unit": "voxels_of_this_rung", "no_data": 0,
                             "axis_order": "ZYX", "sign_convention": SIGN_WORDS, "radial_sign": a.sign,
                             **extra})
            paths.append(fp)
    print(json.dumps({"region": name(o), "windows": nw, "planes": net.plane_names,
                      "stores": [os.path.basename(q) for q in paths],
                      "vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}), flush=True)
    if a.compare_v1:
        import verso_core as VC1
        n1 = VC1.Net(a.ckpt, dev, compile_mode=(a.compile or None), window=a.window)
        R1 = VC1.RegionInputs(o, size, a.window, a.halo, dev, sign=a.sign, pyr=pyr, ax=ax)
        o1, nw1 = VC1.run_region(n1, R1, batch=1, acc_dtype=torch.float16)
        u1 = torch.clamp(torch.round(o1 * 255), 0, 255).to(torch.uint8).cpu().numpy()
        same = bool(_np.array_equal(u1, u8))
        d = _np.abs(u1.astype(_np.int16) - u8.astype(_np.int16))
        print(json.dumps({"compare": "v1_vs_v2", "windows_v1": nw1, "windows_v2": nw,
                          "bit_identical": same, "max_abs_diff": int(d.max()),
                          "n_differing": int((d > 0).sum()), "voxels": int(d.size)}), flush=True)
        assert same, "v2 is NOT bit-identical to v1 on this checkpoint -- do not switch"
        print("V2COMPAREOK", flush=True)
    if not a.no_upload:
        assert upload(paths), "upload failed"
        print("uploaded", [os.path.basename(q) for q in paths], flush=True)
    return 0


def main():
    global STORES
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/ckpt.pt")
    ap.add_argument("--regions", default=f"{ROOT}/verso_regions.txt")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--halo", type=int, default=32)
    ap.add_argument("--region", type=int, default=1024)
    ap.add_argument("--compile", default="max-autotune-no-cudagraphs")
    ap.add_argument("--ahead", type=int, default=3,
                    help="regions kept fetched in front of the GPU (the prefetcher is serial, so this only "
                         "buys a buffer: throughput is set by the fetch rate itself)")
    ap.add_argument("--fetch-jobs", type=int, default=16,
                    help="concurrent keep-alive GETs of the per-region prefetch; ONE session, see verso_mirror")
    ap.add_argument("--batch-up", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1))
    ap.add_argument("--sign", type=float, default=-1.0, help="radial sign; v1's flip trick is -1. A "
                    "checkpoint with FIELD heads may only run at +1: a distance from a mirrored world "
                    "has the opposite sign convention to the stores'")
    ap.add_argument("--one-region", nargs=3, type=int, default=None, metavar=("Z", "Y", "X"),
                    help="do exactly this region and stop (the v1/v2 comparison)")
    ap.add_argument("--no-upload", action="store_true", help="write the stores locally and stop")
    ap.add_argument("--stores-dir", default=STORES, help="where the stores are written")
    ap.add_argument("--compare-v1", action="store_true", help="also run the v1 core on --one-region and "
                    "assert the verso store is BIT-IDENTICAL; prints the comparison and exits")
    a = ap.parse_args()
    STORES = a.stores_dir
    os.makedirs(STORES, exist_ok=True)
    os.makedirs(MARKS, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    if a.one_region:
        return one_region(a, dev)
    regs = [tuple(int(q) for q in l.split()) for l in open(a.regions) if l.strip()]
    regs = [o for i, o in enumerate(regs) if i % a.shard[1] == a.shard[0]]
    done = {n[:-5] for n in remote_listing()}
    todo = [o for o in regs if name(o) not in done and not os.path.exists(f"{MARKS}/{name(o)}")]
    print(f"{len(regs)} regions for this worker, {len(regs) - len(todo)} already published, {len(todo)} to do",
          flush=True)

    ct = data.open_zarr(VC.VOL)
    sizes = {o: tuple(int(min(a.region, s - v)) for s, v in zip(ct.shape[-3:], o)) for o in todo}

    # ---- prefetch thread: the next `ahead` regions' level-0 shards
    pf_done, _consumed = {}, [0]

    def prefetcher():
        for i in range(0, len(todo), a.ahead):
            batch = todo[i:i + a.ahead]
            while len(pf_done) - _consumed[0] > 2 * a.ahead:  # at most two batches in front
                time.sleep(1)
            t0 = time.time()
            ok = fetch_boxes(batch, sizes, jobs=a.fetch_jobs)
            dt = (time.time() - t0) / max(len(batch), 1)      # per-region share of the batch fetch
            for o in batch:
                pf_done[o] = (ok, dt)

    # ---- publish thread
    up_q = queue.Queue()
    stats = {"uploaded": 0, "up_bytes": 0, "up_s": 0.0}
    def publisher():
        pend, stop = [], False
        while not (stop and not pend):
            try:
                item = up_q.get(timeout=30)
            except queue.Empty:
                item = "__tick__"
            if item is None:
                stop = True
            elif item != "__tick__":
                pend.append(item)           # a LIST of a region's stores, verso first
            if pend and (stop or len(pend) >= a.batch_up or item == "__tick__"):
                t0 = time.time()
                flat = [q for grp in pend for q in grp]
                nb = sum(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(s) for f in fs)
                         for s in flat)
                if upload(flat):
                    stats["uploaded"] += len(pend); stats["up_bytes"] += nb
                    stats["up_s"] += time.time() - t0
                    for grp in pend:
                        # ONE marker per region, named after its verso store, exactly as v1: the field
                        # stores are published in the same batch and never on their own
                        open(f"{MARKS}/{os.path.basename(grp[0])}", "w").close()
                        for s in grp:
                            shutil.rmtree(s, ignore_errors=True)
                    print(f"published {len(pend)} stores, {nb/2**20:.0f} MiB in {time.time()-t0:.1f}s "
                          f"({nb/2**20/max(time.time()-t0,1e-3):.1f} MiB/s); total {stats['uploaded']}", flush=True)
                    pend = []
                else:
                    print(f"UPLOAD FAILED for {[os.path.basename(g[0]) for g in pend]}; retrying in 60 s",
                          flush=True)
                    time.sleep(60)
    tp = threading.Thread(target=prefetcher, daemon=True); tp.start()
    tu = threading.Thread(target=publisher, daemon=True); tu.start()

    net = VC.Net(a.ckpt, dev, compile_mode=(a.compile or None), window=a.window)
    assert not (net.nplanes and a.sign < 0), (
        "this checkpoint has FIELD heads and --sign is negative: a distance field measured in a mirrored "
        "world has the opposite sign convention to the stores'. Run a Phase-B checkpoint at --sign 1 and "
        "take the verso band from its own verso channel.")
    print(f"verso_run_v2: {len(net.plane_names)} planes {net.plane_names or '(v1: verso only)'}, "
          f"sign {a.sign}, step {net.step}", flush=True)
    pyr, ax = data.rungs(VC.VOL), data.axis()

    # ---- read / compute / write are pipelined: the CT + context super-cube reads of region N+1 and the
    # volcomp encode of region N-1 both run while the GPU is inside region N. Without this the GPU idles
    # for io_s + write_s (~5.4 s of a 17.8 s region, a ~70% duty cycle).
    rd_q, wr_q = queue.Queue(maxsize=1), queue.Queue(maxsize=2)
    todo_n = todo[:a.limit] if a.limit else todo

    def reader():
        s = torch.cuda.Stream(dev)          # H2D of the next region on a side stream
        for o in todo_n:
            while o not in pf_done:
                time.sleep(0.5)
            _consumed[0] += 1
            t0 = time.time()
            try:
                with torch.cuda.stream(s):
                    R = VC.RegionInputs(o, sizes[o], a.window, a.halo, dev, sign=a.sign, pyr=pyr, ax=ax)
                s.synchronize()             # the compute stream must see the copies
            except Exception as e:          # noqa: BLE001
                print(f"READ FAILED {name(o)}: {e!r}", flush=True)
                continue
            rd_q.put((o, R, time.time() - t0))
        rd_q.put(None)

    def write_region(o, u8, fields):
        """The verso store -- byte for byte what v1 writes -- plus, when the checkpoint has field heads,
        one extra store per field beside it. Returns the list of store paths, verso first."""
        paths = []
        path = f"{STORES}/{name(o)}.zarr"
        shutil.rmtree(path, ignore_errors=True)
        arr = predict.out_array(path, u8.shape, o, volcomp=True, volume=VC.VOL, rung=2)
        arr[:] = u8
        arr.attrs.update({"channels": ["verso"], "mode": "verso", "region": a.region,
                          "radial_sign": a.sign, "ckpt": os.path.basename(a.ckpt), "step": net.step,
                          "window": a.window, "halo": a.halo, "done": True})
        paths.append(path)
        for sfx, (v, enc, q, extra) in (field_stores(fields, os.path.basename(a.ckpt), net.step,
                                                     a.region, a.window, a.halo).items()
                                        if fields else ()):
            fp = f"{STORES}/{name(o)}_{sfx}.zarr"
            shutil.rmtree(fp, ignore_errors=True)
            fa = predict.out_array(fp, v.shape, o, volcomp=True, volume=VC.VOL, rung=2,
                                   channels=(sfx,), q=q)
            fa[:] = v[None] if fa.ndim == 4 else v
            fa.attrs.update({"encoding": enc, "unit": "voxels_of_this_rung", "no_data": 0,
                             "axis_order": "ZYX", "sign_convention": SIGN_WORDS, "radial_sign": a.sign,
                             **extra})
            paths.append(fp)
        return paths

    def writer():
        while True:
            item = wr_q.get()
            if item is None:
                break
            o, u8, nw, fields = item
            t0 = time.time()
            paths = write_region(o, u8, fields)
            del u8
            wr_t.append(time.time() - t0)
            up_q.put(paths)
            try:  # the region's level-0 shard is a single zarr v3 shard FILE and is never read again
                os.remove(l0_key(o))
            except OSError:
                pass

    wr_t = []
    tr = threading.Thread(target=reader, daemon=True); tr.start()
    tw = threading.Thread(target=writer, daemon=True); tw.start()

    warm = False
    n, t_run = 0, time.time()
    while True:
        t_wait = time.time()
        item = rd_q.get()
        if item is None:
            break
        o, R, t_io = item
        t_wait = time.time() - t_wait      # >0 only when the reader could not keep up
        if not warm:
            wu = [q for q in R.offs if R.window_ct(q).any()][:1]
            if wu:
                net(torch.stack([R.prep(wu[0])]))
                torch.cuda.synchronize()
            warm = True
            t_run = time.time()            # do not charge the compile warm-up to the rate
        t0 = time.time()
        out, nw = VC.run_region(net, R, batch=1, acc_dtype=torch.float16)
        if net.nplanes:
            # the MAIN store stays the verso probability; the rest go to the field stores
            fields = {nm: out[i].float().cpu().numpy() for i, nm in enumerate(net.plane_names)}
            main = fields["verso"] if "verso" in fields else fields[net.plane_names[0]]
            u8 = np.clip(np.rint(main * 255), 0, 255).astype(np.uint8)
        else:
            fields = None
            u8 = torch.clamp(torch.round(out * 255), 0, 255).to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize()
        t_gpu = time.time() - t0
        del out, R
        wr_q.put((o, u8, nw, fields))
        del u8
        n += 1
        if n % 20 == 0:
            torch.cuda.empty_cache()
        print(json.dumps({"i": n, "region": name(o), "windows": nw, "io_s": round(t_io, 1),
                          "gpu_s": round(t_gpu, 1), "wait_s": round(t_wait, 1),
                          "write_s": round(wr_t[-1], 1) if wr_t else 0.0,
                          "l0_fetch_s": round(pf_done[o][1], 1),
                          "vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                          "rate_s_per_region": round((time.time() - t_run) / n, 1)}), flush=True)
    wr_q.put(None)
    tw.join(timeout=1800)
    up_q.put(None)
    tu.join(timeout=1800)
    print(f"VERSORUNDONE {n} regions, {stats['uploaded']} published, "
          f"{(time.time()-t_run)/max(n,1):.1f} s/region", flush=True)


if __name__ == "__main__":
    main()
