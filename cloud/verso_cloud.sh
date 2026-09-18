#!/bin/bash
# On the instance: verso targets for every paired group this instance produced itself (~/out/boxesN + boxesN_m7),
# WORKERS (default 5) worker processes on the one GPU: each is CPU-bound on one core (the sliding window's python
# side), the instance has 6 vCPUs and ~7 GB VRAM per worker. Outputs land next to the stores (~/out/boxesN_v,
# boxesN_vraw, ...) and the desk pulls them down. usage: WORKERS=5 bash cloud/verso_cloud.sh
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1
export USRM2_CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
export USRM2_UMBILICUS=$HOME/umbilicus.json USRM2_VAL=$HOME/eval.zarr
O=$HOME/out; G=""
for z in $O/*.zarr; do  # desk-made single stores shipped up (a.zarr + a_m7.zarr)
  n=$(basename $z .zarr); case $n in *_m7|*_v|*_vraw|eval*|p4val*) continue;; esac
  [ -d $O/${n}_m7.zarr/c ] && G="$G $z,$O/${n}_m7.zarr"
done
for dd in $O/boxes*/; do d=$(basename ${dd%/}); case $d in *_m7|*_v|*_vraw) continue;; esac
  for b in $O/$d/box_*.zarr; do
    [ -d "$b" ] || continue; n=$(basename $b)
    [ -d $O/${d}_m7/$n/c ] || continue   # both teachers present (m7 written)
    G="$G $b,$O/${d}_m7/$n"
  done
done
W=${WORKERS:-5}; TL=${TILE:-384}  # 5 workers at tile 512 occasionally spike past 48 GB (OOM); 384 leaves headroom
echo "verso (cloud) for $(echo $G | wc -w) groups, $W workers, tile $TL: $(date)"
for i in $(seq 0 $((W - 1))); do usrm2 verso ~/student.pt --shard $i $W --tile $TL --stores $G > ~/verso_gen_$i.out 2>&1 & done
wait
echo "CLOUDVERSODONE $(date)"
