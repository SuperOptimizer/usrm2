#!/bin/bash
# Masked-cube pretraining on one A6000 (docs/unified_design.md section 28, experiment 11 / R10).
#
#   usage: SIZE=30m6 PATCH=256 STEPS=20000 bash cloud/pretrain_a6000.sh RUN
#
# PLANNER-FREE by design. `stream-plan` exists because the TARGET stores are published region by region and
# have to be fetched as they appear; pretraining reads no target at all, so there is nothing to wait for and
# the run samples the local CT mirror directly (the ordinary `data.Patches` path, --workers processes).
#
# DATA REQUIREMENT. One line per source in $GROUPS, exactly the format `train --stores-file` takes:
#
#     <ct pyramid group>,<target pyramid group>
#
# Only the CT pyramid is READ. The target group is still named because it is what bounds the sampled box
# (`data.target_box`) and what fixes each source's native rung; its voxels are never looked at, so a
# freshly exported, still-empty target pyramid with the right `box` attribute is enough. Every level the
# chosen rungs touch must be on disk locally: at --rungs 2-4 with --ctx 1..9 that is CT levels 0..11 of each
# pyramid (the coarse ones are a few MB; `usrm2 rebuild-pyramid` / cloud/make_levels.py build them).
# Rungs 0 and 1 exist only for a source whose native rung is 0 or 1 (a 0.55/1.129 um scan); on a corpus of
# 2.4 um mirrors `--rungs 0-4` prints "rungs [0, 1] are not on disk anywhere" and trains at 2-4.
#
# AUDIT FIRST (synthesis_v2 experiment 11): count DISTINCT SCANS in $GROUPS, not voxels. The R10 evidence
# comes from ~39k volumes; two scrolls is a different regime and the ablation is what decides adoption.
#
# The checkpoint warm-starts the fine-tuning arm:
#     usrm2 train ~/runs/ft_pre ... --init-from ~/runs/$R/ckpt.pt
# and the from-scratch arm is the same command without --init-from, at the same --steps.
set -euo pipefail
R=${1:-pre_a6000}; SIZE=${SIZE:-30m6}; PATCH=${PATCH:-256}; STEPS=${STEPS:-20000}
RUNGS=${RUNGS:-0-4}; CTX=${CTX:-1..9}; BATCH=${BATCH:-1}; ACCUM=${ACCUM:-2}; WORKERS=${WORKERS:-6}
GROUPS=${GROUPS:-$HOME/groups_raw.txt}
cd ~; . venv/bin/activate
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USRM2_UMBILICUS=$HOME/umbilicus.json USRM2_CT=/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr/0
T=$HOME/teacher
VAL=${VAL:-"$T/eval.zarr"}   # held out from sampling AND used for the reconstruction metric
mkdir -p ~/runs/$R
# --ckpt-act 2 --add-skip 1 are the 256^3 memory settings of the fine-tuning run; the trunk must be built
# the same way for the state dict to line up key for key.
usrm2 pretrain ~/runs/$R --size "$SIZE" --steps "$STEPS" --patch $PATCH --batch "$BATCH" --accum "$ACCUM" \
  --workers "$WORKERS" --rungs "$RUNGS" --ctx $CTX --aug ${AUG:-all3} --norm global \
  --ema ${EMA:-0.9995} --lr ${LR:-3e-4} --lr-floor 0.02 --ckpt-act ${CKPT:-2} --add-skip ${ADDSKIP:-1} \
  --compile --eval-every 500 --val-patches 4 \
  --mask-block ${MASK_BLOCK:-32} --mask-lo ${MASK_LO:-0.5} --mask-hi ${MASK_HI:-0.75} \
  --sheet-p ${SHEET_P:-0.5} --sheet-pct ${SHEET_PCT:-0.7} \
  --stores-file "$GROUPS" --val $VAL ${EXTRA:-} > ~/runs/$R/nohup.log 2>&1
echo "PRETRAINDONE $R $(date)"
