import argparse


def main(argv=None):
    ap = argparse.ArgumentParser("usrm2")
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
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--patch", type=int, default=128)
    e.add_argument("--val-patches", type=int, default=32)
    p = sub.add_parser("predict")
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--volume", default=None)
    p.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    p.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--halo", type=int, default=16)
    p.add_argument("--plain", action="store_true", help="plain zarr (1,Z,Y,X) instead of volcomp")
    t = sub.add_parser("teacher", help="run the upstream teacher over a box")
    t.add_argument("out")
    t.add_argument("--origin", type=int, nargs=3, required=True, metavar=("Z0", "Y0", "X0"))
    t.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    t.add_argument("--volume", default=None)
    a = ap.parse_args(argv)
    from usrm2 import data, model, predict as P, train as T
    if a.cmd == "train":
        T.train(a.out_dir, size=a.size, steps=a.steps, patch=a.patch, batch=a.batch, lr=a.lr,
                workers=a.workers, eval_every=a.eval_every, val_patches=a.val_patches, resume=a.resume)
    elif a.cmd == "eval":
        import torch
        st = torch.load(a.ckpt, map_location="cpu")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = model.build(st["args"]["size"]).to(dev)
        net.load_state_dict({k: v.to(dev) for k, v in st["ema"].items()})
        print(st["step"], T.evaluate(net, data.val_grid(patch=a.patch, limit=a.val_patches), dev))
    elif a.cmd == "teacher":
        from usrm2 import teacher
        teacher.run(a.out, *a.origin, *a.size, volume=a.volume or data.CT)
    else:
        P.predict(a.ckpt, a.volume or data.CT, *a.origin, *a.size, a.out,
                  window=a.window, halo=a.halo, volcomp=not a.plain)


if __name__ == "__main__":
    main()
