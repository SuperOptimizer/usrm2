#!/bin/bash
# On the instance: recto teacher over N random boxes (seed S) of the streamed volume, then m7 at the same origins.
# usage: cloud/run_boxes.sh SEED N        -> ~/out/boxes<SEED>/ and ~/out/boxes<SEED>_m7/
S=$1; N=$2
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1 USRM2_TRT=$HOME/trt
B=${BACKEND:-torch}  # BACKEND=trt once ~/trt holds the ONNX graphs (engines build on first use)
V=https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
O=~/out; mkdir -p $O
for attempt in 1 2 3 4 5 6; do  # a streaming error kills the process; finished boxes are skipped on the retry
  for d in $O/boxes$S/box_*.zarr; do [ -d "$d" ] && ! grep -q "$(basename $d)" $O/boxes$S.log 2>/dev/null && rm -rf "$d"; done
  usrm2 teacher-boxes $O/boxes$S --n $N --seed $S --volume $V --exclude ~/eval.zarr --backend $B >> $O/boxes$S.log 2>&1
  grep -q "^box $N/$N" $O/boxes$S.log && break
  echo "retry $attempt $(date)" >> $O/boxes$S.log; sleep 30
done
for b in $O/boxes$S/box_*.zarr; do
  [ -d "$b" ] || continue
  n=$(basename $b); o=${n#box_}; o=${o%.zarr}; o=${o//_/ }
  [ -d $O/boxes${S}_m7/$n ] || usrm2 teacher $O/boxes${S}_m7/$n --origin $o --size 384 2048 2048 --volume $V --model m7 --backend $B 2>&1 | grep -i "traceback\|error"
  echo "m7 $n"
done >> $O/boxes$S.log 2>&1
echo CLOUDBOXESDONE >> $O/boxes$S.log
