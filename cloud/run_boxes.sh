#!/bin/bash
# On the instance: recto teacher over N random boxes (seed S) of the streamed volume, then m7 at the same origins.
# usage: cloud/run_boxes.sh SEED N        -> ~/out/boxes<SEED>/ and ~/out/boxes<SEED>_m7/
S=$1; N=$2
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1
V=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
O=~/out; mkdir -p $O
usrm2 teacher-boxes $O/boxes$S --n $N --seed $S --volume $V --exclude ~/eval.zarr > $O/boxes$S.log 2>&1
for b in $O/boxes$S/box_*.zarr; do
  n=$(basename $b); o=${n#box_}; o=${o%.zarr}; o=${o//_/ }
  [ -d $O/boxes${S}_m7/$n ] || usrm2 teacher $O/boxes${S}_m7/$n --origin $o --size 384 2048 2048 --volume $V --model m7 2>&1 | grep -i "traceback\|error"
  echo "m7 $n"
done >> $O/boxes$S.log 2>&1
echo CLOUDBOXESDONE >> $O/boxes$S.log
