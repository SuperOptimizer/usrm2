#!/bin/bash
# On the instance: verso targets for every paired group under ~/teacher (mirrored from the desk), taken from the END
# of the same list the desk works from the front, so the two meet in the middle. usage: bash cloud/verso_cloud.sh I K
I=${1:-0}; K=${2:-1}
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1
export USRM2_CT=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
export USRM2_UMBILICUS=$HOME/umbilicus.json USRM2_VAL=$HOME/teacher/eval.zarr
T=$HOME/teacher
VG="$T/eval.zarr,$T/p4val256_m7_tta0.zarr $(sed "s#/vesuvius/usrm2/teacher#$T#g" ~/val_extra.txt | tr '\n' ' ')"
G="$T/a.zarr,$T/a_m7.zarr $T/b.zarr,$T/b_m7.zarr"
for dd in $T/boxes*/; do d=$(basename ${dd%/}); case $d in *_m7|*_tta4|*_v|*_vraw) continue;; esac
  for b in $T/$d/box_*.zarr; do
    [ -d "$b" ] || continue; n=$(basename $b)
    [ -d $T/${d}_m7/$n ] || continue
    grep -q "^/vesuvius/usrm2/teacher/$d/$n," ~/val_extra.txt && continue
    G="$G $b,$T/${d}_m7/$n"
  done
done
echo "verso (cloud, reverse, shard $I/$K) for $(echo $VG $G | wc -w) groups: $(date)"
usrm2 verso ~/student.pt --reverse --shard $I $K --stores $VG $G
echo CLOUDVERSODONE
