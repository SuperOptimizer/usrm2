#!/bin/bash
# Swap the pod's verso production loop between v1 (`verso_run.py`) and v2 (`verso_run_v2.py`), cleanly and
# resumably. Lives ON THE POD as /workspace/switch_v2.sh.
#
#   bash /workspace/switch_v2.sh v2      # stop the loop, point it at verso_run_v2.py, restart
#   bash /workspace/switch_v2.sh v1      # back to v1
#   bash /workspace/switch_v2.sh status  # which one the loop is pointed at, and whether it is running
#
# Why this is safe to run mid-walk: the loop is resumable by construction -- a region is skipped when it
# is in the remote listing taken at start or has a local `.published` marker -- so stopping it between
# regions loses at most the region in flight, which the next start redoes. The one thing that must not
# happen is stopping it MID-UPLOAD, which is why the stop goes through `stop_everything.sh` (the pod's
# own script; kill patterns live in script files on the host, never on an ssh command line) and this
# script then waits for the sftp session to be gone before it touches anything.
#
# v2 is a NO-OP for a checkpoint without field heads (cout_t <= 2): it runs v1's code path kernel for
# kernel and writes the same bytes. Prove it on ONE region before switching:
#
#   python3 /workspace/verso_run_v2.py --one-region Z Y X --no-upload --compare-v1 \
#           --stores-dir /workspace/v2_test
#
# which prints {"compare": "v1_vs_v2", "bit_identical": true, ...} and "V2COMPAREOK", or asserts.
set -u
W=/workspace
LOOP=$W/verso_loop.sh
MODE=${1:-status}

running() { pgrep -f "verso_run(_v2)?\.py" > /dev/null && echo yes || echo no; }
target()  { grep -oE "verso_run(_v2)?\.py" "$LOOP" | head -1; }

if [ "$MODE" = status ]; then
  echo "loop points at: $(target)"
  echo "running: $(running)"
  pgrep -af "verso_run(_v2)?\.py" | grep -v "bash -c" | head -3
  exit 0
fi

case "$MODE" in
  v1) WANT=verso_run.py ;;
  v2) WANT=verso_run_v2.py ;;
  *)  echo "usage: switch_v2.sh v1|v2|status"; exit 2 ;;
esac

[ -f "$W/$WANT" ] || { echo "MISSING $W/$WANT"; exit 3; }
if [ "$(target)" = "$WANT" ] && [ "$(running)" = yes ]; then
  echo "already on $WANT and running; nothing to do"; exit 0
fi

echo "=== stopping the loop ($(date -u))"
bash $W/stop_everything.sh
for i in $(seq 1 60); do                 # wait for an sftp session in flight to finish or die
  pgrep -f "sftp .*dl.ash2txt.org" > /dev/null || break
  sleep 2
done
sleep 2
echo "still running: $(running)"

cp -a "$LOOP" "$LOOP.bak.$(date -u +%Y%m%dT%H%M%SZ)"
sed -i -E "s#verso_run(_v2)?\.py#$WANT#" "$LOOP"
echo "=== loop now points at: $(target)"
grep -n "python3" "$LOOP"

echo "=== restarting ($(date -u))"
bash $W/start_loop.sh
sleep 5
echo "running: $(running)"
