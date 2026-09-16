# usrm2 — Ultra Small Recto Model 2

A ~1.5M-parameter 3D U-Net distilled from the upstream `surface_recto_3dunet` teacher:
input is a z-scored CT patch, output is one recto-probability logit per voxel.

## Conventions
- All arrays are indexed `(z, y, x)` in level-0 (2.4 um) voxels.
- Probability maps are `uint8 = p * 255`; `attrs = {channels: ["recto"], voxel_um: 2.4,
  origin_zyx: [z, y, x], scale: 1.0}` where `origin_zyx` is the level-0 index of element 0.
- CT is the volcomp zarr `.../PHercParis4/...-masked.zarr/0` (uint8, 0 = air/masked);
  reading needs `volcomp_zarr` (set `VOLCOMP_LIB`, see `desk.sh`).
- Teacher stores are `(1, Z, Y, X)`; predictions written with the volcomp codec are `(Z, Y, X)`
  (the codec only accepts 128^3 chunks); `--plain` writes `(1, Z, Y, X)` plain zarr instead.

## Commands
    usrm2 train /vesuvius/usrm2/runs/p4_1m --size 1m --steps 20000 --patch 128 --batch 1
    usrm2 eval  /vesuvius/usrm2/runs/p4_1m/ckpt.pt
    usrm2 predict RUN/ckpt.pt out.zarr --origin 34432 15104 18432 --size 256 256 256
