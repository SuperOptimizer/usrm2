#!/bin/bash
# From forlindesk2: keep pulling finished verso stores (zarr.json says done) from an instance's ~/out into
# /vesuvius/usrm2/teacher/ until the instance reports CLOUDVERSODONE. usage: cloud/pull_verso.sh SSH_TARGET
H=$1; T=/vesuvius/usrm2/teacher
sync_done() {
  for d in $(ssh $H 'cd ~/out && grep -l "\"done\": true" boxes*_v/*/zarr.json boxes*_vraw/*/zarr.json *_v.zarr/zarr.json *_vraw.zarr/zarr.json 2>/dev/null | xargs -n1 dirname' 2>/dev/null); do
    [ -f $T/$d/zarr.json ] && grep -q '"done": true' $T/$d/zarr.json && continue
    mkdir -p $T/$d && rsync -a $H:out/$d/ $T/$d/ 2>/dev/null
  done
}
until ssh $H "grep -q CLOUDVERSODONE verso_gen.out" 2>/dev/null; do sync_done; sleep 600; done
sync_done; echo "PULLED verso $H $(date)"
