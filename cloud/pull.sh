#!/bin/bash
# From forlindesk2: keep pulling finished boxes from an instance into /vesuvius/usrm2/teacher/ until it reports done.
# usage: cloud/pull.sh SSH_TARGET SEED
H=$1; S=$2; T=/vesuvius/usrm2/teacher
until ssh $H "grep -q CLOUDBOXESDONE out/boxes$S.log" 2>/dev/null; do
  rsync -a --exclude "*.partial" $H:out/boxes$S/ $T/boxes$S/ 2>/dev/null; rsync -a $H:out/boxes${S}_m7/ $T/boxes${S}_m7/ 2>/dev/null
  sleep 600
done
rsync -a $H:out/boxes$S/ $T/boxes$S/; rsync -a $H:out/boxes${S}_m7/ $T/boxes${S}_m7/; rsync -a $H:out/boxes$S.log $T/
echo PULLED $S
