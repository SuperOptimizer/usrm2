#!/bin/bash
# On the A100: the big-patch multi-resolution 4-head run. usage: SIZE=26m CKPT=-1 PATCH="384 384 384" bash cloud/a100_train.sh RUN
# Reads only local data (~/teacher stores, the local CT mirror at /vesuvius/usrm/volcomp incl. pyramid levels).
R=${1:-r4_a100}; SIZE=${SIZE:-26m}; CKPT=${CKPT:--1}; PATCH=${PATCH:-384 384 384}; STEPS=${STEPS:-40000}
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USRM2_UMBILICUS=$HOME/umbilicus.json USRM2_CT=/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
T=$HOME/teacher
VAL="$T/eval.zarr,$T/p4val256_m7_tta0.zarr,$T/eval_vraw.zarr,$T/p4val256_m7_tta0_vraw.zarr"
for v in $(sed "s#/vesuvius/usrm2/teacher#$T#g" ~/val_extra.txt); do a=${v%%,*}; b=${v##*,}; VAL="$VAL $a,$b,${a%/*}_vraw/${a##*/},${b%/*}_vraw/${b##*/}"; done
mkdir -p ~/runs/$R
usrm2 train ~/runs/$R --size $SIZE --steps $STEPS --patch $PATCH --batch 1 --accum ${ACCUM:-2} --workers ${WORKERS:-7} \
  --eval-every 500 --val-patches 8 --aug all3 --ctx 1 2 3 --dense-pow 1.5 --ridge-w 2.0 --norm global --ema 0.9995 --lr-floor 0.02 \
  --compile --ckpt-act $CKPT ${INIT:+--init-from $INIT} --stores-file ~/groups_raw.txt --val $VAL ${EXTRA:-} > ~/runs/$R/nohup.log 2>&1
echo "A100RUNDONE $R $(date)"
