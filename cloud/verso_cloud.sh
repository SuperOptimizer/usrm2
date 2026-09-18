#!/bin/bash
# On the instance: verso targets for every paired group this instance produced itself (~/out/boxesN + boxesN_m7),
# two worker processes on the one GPU (separate processes are what scales on Thunder). Outputs land next to the
# stores (~/out/boxesN_v, boxesN_vraw, ...) and the desk pulls them down. usage: bash cloud/verso_cloud.sh
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1
export USRM2_CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
export USRM2_UMBILICUS=$HOME/umbilicus.json USRM2_VAL=$HOME/eval.zarr
O=$HOME/out; G=""
for dd in $O/boxes*/; do d=$(basename ${dd%/}); case $d in *_m7|*_v|*_vraw) continue;; esac
  for b in $O/$d/box_*.zarr; do
    [ -d "$b" ] || continue; n=$(basename $b)
    [ -d $O/${d}_m7/$n/c ] || continue   # both teachers present (m7 written)
    G="$G $b,$O/${d}_m7/$n"
  done
done
echo "verso (cloud) for $(echo $G | wc -w) groups, 2 workers: $(date)"
usrm2 verso ~/student.pt --shard 0 2 --stores $G > ~/verso_gen_0.out 2>&1 &
usrm2 verso ~/student.pt --shard 1 2 --stores $G > ~/verso_gen_1.out 2>&1 &
wait
echo "CLOUDVERSODONE $(date)"
