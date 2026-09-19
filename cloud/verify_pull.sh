#!/bin/bash
# From forlindesk2: list every done verso store and every teacher store on tnr-0/tnr-1 and report those missing
# (or not done) on the desk. Prints MISSING lines and a summary.
T=/vesuvius/usrm2/teacher
for h in tnr-0 tnr-1; do
  n=0; miss=0
  for d in $(ssh $h 'cd ~/out && grep -l "\"done\": true" boxes*_v/*/zarr.json boxes*_vraw/*/zarr.json 2>/dev/null | xargs -n1 dirname; ls -d boxes*/*.zarr | grep -v "_v/\|_vraw/"'); do
    n=$((n+1))
    if [ ! -f $T/$d/zarr.json ]; then echo "MISSING $h $d"; miss=$((miss+1)); continue; fi
    case $d in *_v/*|*_vraw/*) grep -q '"done": true' $T/$d/zarr.json || { echo "NOTDONE $h $d"; miss=$((miss+1)); };; esac
  done
  echo "$h: $n stores checked, $miss missing"
done
