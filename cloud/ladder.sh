#!/bin/bash
# EXPERIMENT 12, the size ladder (docs/unified_design.md section 30):
#   --size 15m / 30m6 / 60m at MATCHED STEPS, on the SAME windows in the SAME order, differing ONLY in
#   --size. Then `usrm2 ladder-report` fits 1 - dice against log(params) PER RUNG.
#
# usage:
#   MODE=parallel GPUS="0 1 2" STEPS=20000 bash cloud/ladder.sh ~/runs/ladder ~/stores.txt ~/queue
#   MODE=serial              STEPS=20000 bash cloud/ladder.sh ~/runs/ladder ~/stores.txt ~/queue
#   DRY=1 bash cloud/ladder.sh ...        # print the commands and stop
#
# MODE=parallel (the default and the honest one): ONE stream plan, every rung replaying it under its own
#   --stream-tag, all running at once, one GPU each. Nothing about a replay is destructive -- queue.jsonl
#   is append-only and read by index -- and the tag is what keeps the per-worker replay cursor
#   (<queue>/progress.<tag>/) and the planner's eviction bound (<queue>/consumed.<tag>) separate. The
#   planner evicts only below the MINIMUM over the tags, so the buffer advances at the slowest rung's
#   pace; give it CACHE_GB >= 2x what one run would need, because the 60m is ~2x the 15m's step time.
#   Every rung must use the same --workers: the queue's meta.json fixes the stream count.
# MODE=serial: one queue PER RUNG, planned with the same --seed, run one after another. The planner is
#   deterministic in (stores, seed, patch, rungs, boost, region, walk, epochs), so the three queues hold
#   the same windows in the same order -- at 3x the origin bandwidth and 3x the disk, which is why this
#   is the fallback. A shared queue cannot serve SEQUENTIAL runs: the rolling buffer is long gone by the
#   time the second rung starts at index 0.
set -u
OUT=${1:-$HOME/runs/ladder}; STORES=${2:-$HOME/stores.txt}; QUEUE=${3:-$HOME/queue}
MODE=${MODE:-parallel}; GPUS=${GPUS:-0 1 2}; SIZES=${SIZES:-15m 30m6 60m}
STEPS=${STEPS:-20000}; SEED=${SEED:-0}; WORKERS=${WORKERS:-8}; CACHE_GB=${CACHE_GB:-80}
PATCH=${PATCH:-256}; BATCH=${BATCH:-2}; ACCUM=${ACCUM:-1}; LR=${LR:-3e-4}; LRSCALE=${LRSCALE:-same}
CKPT_ACT=${CKPT_ACT:-1}; DEEP=${DEEP:-2}; ADDSKIP=${ADDSKIP:-1}
# The flag line every rung shares, verbatim. Anything but --size, --lr, --stream and --stream-tag belongs
# here; if a flag differs between rungs the experiment measures that flag, not the size.
BASE=${BASE:-"--steps $STEPS --patch $PATCH --batch $BATCH --accum $ACCUM --ckpt-act $CKPT_ACT \
--deep $DEEP --add-skip $ADDSKIP --workers $WORKERS --rungs 2-11 --ctx 1..9 --rung-boost 2=2 \
--val-rungs 2,3,4,6 --stores-file $STORES --cascade mix --cascade-self-p-anneal 0.1 0.7 --cascade-drop 0.1 \
--verso --teacher-regions $HOME/teacher_regions --aug full2 --sched wsd --ema auto --eval-every 500 \
--val-patches 8 --norm global"}

mkdir -p "$OUT"
QUEUES=shared; [ "$MODE" = serial ] && QUEUES=per-run

echo "=== ladder: $MODE, sizes [$SIZES], $STEPS steps, queue $QUEUE ($QUEUES)"
usrm2 ladder "$OUT" --sizes $SIZES --queue "$QUEUE" --queues $QUEUES --stores-file "$STORES" \
     --lr "$LR" --lr-scale "$LRSCALE" --base "$BASE" | tee "$OUT/commands.txt"
[ "${DRY:-0}" = 1 ] && { echo "(DRY=1: nothing launched)"; exit 0; }

plan_one() {  # plan_one QUEUEDIR LOGSUFFIX
  mkdir -p "$1"
  setsid nohup usrm2 stream-plan "$STORES" --queue "$1" --patch $PATCH --ctx 1..9 --rungs 2-11 \
      --rung-boost 2=2 --workers $WORKERS --seed $SEED --cache-gb $CACHE_GB --cascade mix --verso \
      --teacher-regions "$HOME/teacher_regions" --walk mix --region 1024 --windows-per-region 64 \
      --val-rungs 2,3,4,6 --aug full2 \
      > "$OUT/plan$2.log" 2>&1 < /dev/null &
  echo "  planner for $1 -> $OUT/plan$2.log (pid $!)"
}

if [ "$MODE" = parallel ]; then
  plan_one "$QUEUE" ""
  sleep 30                      # let the planner get ahead before the trainers start waiting on it
  i=0
  for sz in $SIZES; do
    g=$(echo $GPUS | cut -d' ' -f$((i + 1))); g=${g:-0}
    lr=$(python3 -c "from usrm2 import ladder as L; print(L.lr_for('$sz', '30m6', $LR, '$LRSCALE'))")
    CUDA_VISIBLE_DEVICES=$g setsid nohup usrm2 train "$OUT/$sz" --size $sz --lr $lr \
        --stream "$QUEUE" --stream-tag $sz $BASE > "$OUT/$sz.log" 2>&1 < /dev/null &
    echo "  $sz on GPU $g -> $OUT/$sz.log (pid $!)"
    i=$((i + 1))
  done
  wait
else
  for sz in $SIZES; do
    plan_one "${QUEUE}_$sz" "_$sz"
    sleep 30
    lr=$(python3 -c "from usrm2 import ladder as L; print(L.lr_for('$sz', '30m6', $LR, '$LRSCALE'))")
    usrm2 train "$OUT/$sz" --size $sz --lr $lr --stream "${QUEUE}_$sz" $BASE > "$OUT/$sz.log" 2>&1
    echo "  $sz done"
  done
fi

echo "=== ladder runs finished; the report:"
usrm2 ladder-report $(for sz in $SIZES; do echo "$OUT/$sz"; done) --json "$OUT/ladder.json"
echo "LADDERDONE $(date -u)"
