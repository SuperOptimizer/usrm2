#!/bin/bash
# ON the instance: restart just one seed's run on the current code. usage: bash inst_restart_one.sh SEED N
S=$1; N=$2
for p in $(pgrep -f "run_boxes.sh $S [0-9]") $(pgrep -f "teacher-boxes /home/ubuntu/out/boxes$S "); do kill $p; done
sleep 3
for d in ~/out/boxes$S/box_*.zarr; do [ -d "$d" ] && ! grep -q "$(basename $d)" ~/out/boxes$S.log 2>/dev/null && rm -rf "$d"; done
nohup env BACKEND=torch bash ~/usrm2/cloud/run_boxes.sh $S $N > ~/run_boxes_$S.log 2>&1 &
sleep 2; pgrep -af "run_boxes.sh $S [0-9]" | cut -c1-80
