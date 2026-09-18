#!/bin/bash
# From forlindesk2: keep pulling finished verso stores from an instance into /vesuvius/usrm2/teacher/ (only stores
# whose zarr.json says done, so the desk workers skip exactly the finished groups). usage: cloud/pull_verso.sh SSH_TARGET
H=$1; T=/vesuvius/usrm2/teacher
sync_done() {
  for d in $(ssh $H 'cd ~/teacher && grep -l "\"done\": true" */*/zarr.json *_v*.zarr/zarr.json 2>/dev/null | xargs -n1 dirname' 2>/dev/null); do
    case $d in *_v|*_vraw|*_v/*|*_vraw/*|*_v.zarr|*_vraw.zarr) ;; *) continue;; esac
    [ -f $T/$d/zarr.json ] && grep -q '"done": true' $T/$d/zarr.json && continue
    mkdir -p $T/$d && rsync -a $H:teacher/$d/ $T/$d/ 2>/dev/null
  done
}
until ssh $H "grep -q CLOUDVERSODONE verso_gen.out" 2>/dev/null; do sync_done; sleep 600; done
sync_done; echo "PULLED verso $H $(date)"
