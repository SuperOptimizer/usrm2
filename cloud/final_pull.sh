#!/bin/bash
# From forlindesk2: stop the instance verso workers, pull everything finished from tnr-0/tnr-1 ~/out into
# /vesuvius/usrm2/teacher (all teacher stores; verso stores only when done) and the A100's logs/results into
# ~/a100_final. Run BEFORE deleting the instances. Log: ~/final_pull.out
T=/vesuvius/usrm2/teacher
for h in tnr-0 tnr-1; do
  ssh $h 'pkill -f "usrm2.cli verso" ; pkill -f verso_cloud.sh; sleep 2; pgrep -af "cli verso" | wc -l' 2>/dev/null
done
pkill -f pull_verso.sh
for h in tnr-0 tnr-1; do
  echo "== $h teacher stores $(date)"
  rsync -a --info=stats1 --exclude "*_v" --exclude "*_vraw" --exclude "*_v.zarr" --exclude "*_vraw.zarr" --exclude "*.partial" $h:out/ $T/ 2>&1 | grep -E "Number of|Total transferred"
  echo "== $h verso stores $(date)"
  for d in $(ssh $h 'cd ~/out && grep -l "\"done\": true" boxes*_v/*/zarr.json boxes*_vraw/*/zarr.json *_v.zarr/zarr.json *_vraw.zarr/zarr.json 2>/dev/null | xargs -n1 dirname'); do
    [ -f $T/$d/zarr.json ] && grep -q '"done": true' $T/$d/zarr.json && continue
    mkdir -p $T/$d && rsync -a $h:out/$d/ $T/$d/
  done
  echo "== $h partial (not pulled): $(ssh $h 'cd ~/out && grep -L "\"done\": true" boxes*_v/*/zarr.json boxes*_vraw/*/zarr.json 2>/dev/null | wc -l')"
done
mkdir -p ~/a100_final
rsync -a tnr-2:'*.out' tnr-2:'*.log' tnr-2:'*.sh' tnr-2:'*.txt' tnr-2:runs ~/a100_final/ 2>&1 | tail -n 2
echo "FINALPULLDONE $(date)"
