#!/bin/bash
# ON the instance: stop any box run, drop unfinished box dirs, start a fresh run. usage: bash inst_relaunch.sh SEED N
S=$1; N=$2
for p in $(pgrep -f "cloud/run_boxes.s[h]") $(pgrep -f "usrm2 teacher-boxe[s]") $(pgrep -f "usrm2.cli teacher-boxe[s]") $(pgrep -f "usrm2 teache[r] "); do kill $p; done
sleep 3
for d in ~/out/boxes$S/box_*.zarr; do [ -d "$d" ] && ! grep -q "$(basename $d)" ~/out/boxes$S.log 2>/dev/null && rm -rf "$d"; done
echo "kept $(ls ~/out/boxes$S 2>/dev/null | wc -l) boxes"
nohup env BACKEND=${BACKEND:-torch} PROCS=${PROCS:-2} SIZE="${SIZE:-384 2048 2048}" USRM2_CUDNN_BENCH=${USRM2_CUDNN_BENCH:-1} bash ~/usrm2/cloud/run_boxes.sh $S $N > ~/run_boxes.log 2>&1 &
sleep 2; pgrep -af "cloud/run_boxes.s[h]" | cut -c1-80
