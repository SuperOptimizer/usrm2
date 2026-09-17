import argparse


def main(argv=None):
    ap = argparse.ArgumentParser("usrm2")
    ap.add_argument("--umbilicus", default=None, help="scroll axis json (default: PHerc Paris 4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("out_dir")
    t.add_argument("--size", default="1m", choices=["1m", "3m", "5m"])
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--patch", type=int, default=128)
    t.add_argument("--batch", type=int, default=1)  # 128^3 batch 2 needs >3.5 GiB
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--val-patches", type=int, default=32)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stores", nargs="+", default=None, help="teacher stores to train on (default: data.TRAIN); "
                   "'a.zarr,a_m7.zarr' = several teachers over one box, one head each")
    t.add_argument("--val", default=None, help="teacher store for validation (default: data.VAL); same comma form")
    t.add_argument("--aug", default="geo", help="augmentation preset (see aug.PRESETS)")
    t.add_argument("--no-radial", action="store_true", help="zero the radial channels 1..3")
    b = sub.add_parser("ablate", help="train one run per augmentation preset, sequentially")
    b.add_argument("out_dir")
    b.add_argument("--presets", default="geo,all")
    b.add_argument("--size", default="1m", choices=["1m", "3m", "5m"])
    b.add_argument("--steps", type=int, default=3000)
    b.add_argument("--patch", type=int, default=96)
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--lr", type=float, default=3e-4)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--eval-every", type=int, default=500)
    b.add_argument("--val-patches", type=int, default=32)
    b.add_argument("--stores", nargs="+", default=None)
    b.add_argument("--val", default=None)
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--patch", type=int, default=128)
    e.add_argument("--val-patches", type=int, default=32)
    e.add_argument("--val", default=None, help="teacher store to score (default: the checkpoint's own)")
    e.add_argument("--ct", default=None)
    p = sub.add_parser("predict")
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--volume", default=None)
    p.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    p.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--plain", action="store_true", help="plain zarr (1,Z,Y,X) instead of volcomp")
    p.add_argument("--ome", action="store_true", help="zarr v2 OME group at full volume shape (tracer drop-in)")
    p.add_argument("--tta", type=int, default=0, help="average over this many axis flips (8 = all)")
    p.add_argument("--head", default="0", help="head index of a multi-teacher student, or mean / prod / max")
    p.add_argument("--lut-to", nargs="*", default=(), metavar="REF", help="also average with the input histogram-matched to REF volumes")
    s = sub.add_parser("evalsurf", help="score a checkpoint or store against the published surfaces")
    s.add_argument("--box", type=int, nargs=6, default=None, metavar=("Z0", "Y0", "X0", "Z", "Y", "X"))
    s.add_argument("--ckpt")
    s.add_argument("--store")
    s.add_argument("--teacher", help="a second store, scored on the same points")
    s.add_argument("--tifxyz", default=None)
    s.add_argument("--volume", default=None)
    s.add_argument("--png", default=None)
    s.add_argument("--window", type=int, default=128)
    s.add_argument("--tta", type=int, default=0)
    s.add_argument("--head", default="0", help="head index of a multi-teacher student, or mean / prod / max")
    s.add_argument("--lut-to", nargs="*", default=(), metavar="REF")
    s.add_argument("--halo", type=int, default=16)
    s.add_argument("--device", default=None)
    t = sub.add_parser("teacher", help="run the upstream teacher over a box")
    t.add_argument("out")
    t.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    t.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    t.add_argument("--volume", default=None)
    t.add_argument("--tta", type=int, default=0, help="average over this many axis flips (8 = all)")
    t.add_argument("--lut-to", nargs="*", default=(), metavar="REF", help="also average with the input histogram-matched to each REF volume")
    t.add_argument("--model", default="recto", choices=["recto", "m7"], help="m7 = the 8um nnU-Net on level 2, upsampled")
    t.add_argument("--backend", default="torch", choices=["torch", "trt"], help="trt = tsm's fp16 TensorRT engine")
    t.add_argument("--gpu-acc", action="store_true", help="box, normalization and accumulators on the GPU (needs ~8 GB spare)")
    t.add_argument("--window", type=int, default=None, help="recto 256 / m7 192 by default; bigger = less halo overlap")
    t.add_argument("--batch", type=int, default=1, help="windows per forward (with --gpu-acc)")
    b = sub.add_parser("teacher-boxes", help="run the teacher over many random non-air boxes")
    b.add_argument("out_dir")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--size", type=int, nargs=3, default=(384, 2048, 2048), metavar=("Z", "Y", "X"))
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--volume", default=None, help="CT zarr array (default Paris 4); pair with --umbilicus")
    b.add_argument("--exclude", default="default", help="val store to avoid (default: data.VAL; 'none' for other scrolls)")
    b.add_argument("--tta", type=int, default=0, help="axis-flip TTA (4 = identity + 3 single flips)")
    b.add_argument("--model", default="recto", choices=["recto", "m7"], help="m7 = the 8um nnU-Net on level 2, upsampled")
    b.add_argument("--backend", default="torch", choices=["torch", "trt"], help="trt = tsm's fp16 TensorRT engine")
    b.add_argument("--gpu-acc", action="store_true", help="box, normalization and accumulators on the GPU (needs ~8 GB spare)")
    b.add_argument("--window", type=int, default=None, help="recto 256 / m7 192 by default; bigger = less halo overlap")
    b.add_argument("--batch", type=int, default=1, help="windows per forward (with --gpu-acc)")
    a = ap.parse_args(argv)
    from usrm2 import data, model, predict as P, train as T
    if a.umbilicus:
        data.UMBILICUS = a.umbilicus
    if a.cmd == "train":
        T.train(a.out_dir, size=a.size, steps=a.steps, patch=a.patch, batch=a.batch, lr=a.lr,
                workers=a.workers, eval_every=a.eval_every, val_patches=a.val_patches, resume=a.resume,
                aug=a.aug, no_radial=a.no_radial,
                **{k: v for k, v in dict(stores=a.stores, val=a.val).items() if v})
    elif a.cmd == "ablate":
        from usrm2 import ablate
        ablate.sweep(a.out_dir, a.presets.split(","), size=a.size, steps=a.steps, patch=a.patch,
                     batch=a.batch, lr=a.lr, workers=a.workers, eval_every=a.eval_every,
                     val_patches=a.val_patches,
                     **{k: v for k, v in dict(stores=a.stores, val=a.val).items() if v})
    elif a.cmd == "eval":
        import torch
        st = torch.load(a.ckpt, map_location="cpu")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model.build(st["args"]["size"], cout=st["args"].get("cout", 1)).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in st["ema"].items()})
        sa = st["args"]  # the run's own validation set unless overridden
        grid = data.val_grid(patch=a.patch, limit=a.val_patches, ct=a.ct or sa.get("ct", data.CT), store=a.val or sa.get("val", data.VAL))
        print("val", a.val or sa.get("val", data.VAL))
        if st["args"].get("no_radial"):
            for x, _ in grid:
                x[1:] = 0
        print(st["step"], T.evaluate(net, grid, dev))
    elif a.cmd == "evalsurf":
        from usrm2 import evalsurf as E
        assert a.ckpt or a.store, "need --ckpt or --store"
        b = a.box or (*E.VAL_BOX[0], *E.VAL_BOX[1])
        from usrm2 import teacher
        luts = [teacher.lut_to(a.volume or data.CT, r) for r in a.lut_to]
        E.run(b[:3], b[3:], ckpt=a.ckpt, store=a.store, teacher=a.teacher, tifxyz=a.tifxyz or E.TIFXYZ,
              volume=a.volume, window=a.window, halo=a.halo, device=a.device, png_path=a.png, tta=a.tta, luts=luts,
              head=a.head if a.head in P.HEADS else int(a.head))
    elif a.cmd == "teacher-boxes":
        from usrm2 import teacher
        ex = data.VAL if a.exclude == "default" else (None if a.exclude == "none" else a.exclude)
        runner = __import__("usrm2.m7", fromlist=["run"]).run if a.model == "m7" else None
        teacher.boxes(a.out_dir, n=a.n, size=tuple(a.size), seed=a.seed, volume=a.volume or data.CT, exclude=ex, tta=a.tta, runner=runner, backend=a.backend, **({"window": a.window} if a.window else {}),
                      **({"gpu_acc": True, "batch": a.batch} if a.gpu_acc and a.model == "recto" else {}))
    elif a.cmd == "teacher":
        from usrm2 import teacher
        vol = a.volume or data.CT
        if a.model == "m7":
            from usrm2 import m7
            m7.run(a.out, *a.origin, *a.size, volume=vol, tta=a.tta, backend=a.backend, **({"window": a.window} if a.window else {}))
        else:
            teacher.run(a.out, *a.origin, *a.size, volume=vol, tta=a.tta, luts=[teacher.lut_to(vol, r) for r in a.lut_to], backend=a.backend,
                        gpu_acc=a.gpu_acc, batch=a.batch, **({"window": a.window} if a.window else {}))
    else:
        from usrm2 import teacher
        vol = a.volume or data.CT
        P.predict(a.ckpt, vol, *a.origin, *a.size, a.out, window=a.window, halo=a.halo, volcomp=not a.plain, ome=a.ome,
                  tta=a.tta, luts=[teacher.lut_to(vol, r) for r in a.lut_to], head=a.head if a.head in P.HEADS else int(a.head))


if __name__ == "__main__":
    main()
