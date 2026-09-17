#!/bin/bash
# desk daemon: an A6000 with no box run gets the next seed immediately (3 lean workers, 1024-boxes) + a pull loop.
# Seeds come from ~/cloud_seed_next (one number). Log: ~/cloud_keepbusy.out
cd ~/usrm2
[ -f ~/cloud_seed_next ] || echo 9 > ~/cloud_seed_next
while true; do
  for h in tnr-0 tnr-1; do
    if ! ssh -o ConnectTimeout=20 $h "pgrep -f 'cloud/run_boxes.s[h]' > /dev/null" 2>/dev/null; then
      st=$(ssh -o ConnectTimeout=20 $h "echo up" 2>/dev/null); [ "$st" = "up" ] || { echo "$(date) $h unreachable"; continue; }
      S=$(cat ~/cloud_seed_next); echo $((S + 1)) > ~/cloud_seed_next
      ssh $h "PROCS=3 SIZE=\"384 1024 1024\" USRM2_CUDNN_BENCH=0 bash inst_relaunch.sh $S 200" | tail -1
      nohup bash cloud/pull.sh $h $S > ~/cloud_pull_$S.log 2>&1 &
      echo "$(date) $h idle -> seed $S started"
    fi
  done
  sleep 120
done
