#!/bin/bash
# From forlindesk2: make sure no worker is left on tnr-0/tnr-1, pull whatever finished meanwhile, and delete
# the three instances when nothing is missing.
for h in tnr-0 tnr-1; do
  ssh $h 'pkill -9 -f "usrm2" ; pkill -9 -f verso_cloud; sleep 1; echo "$(hostname) python left: $(pgrep -c python)"'
done
bash ~/usrm2/cloud/pull_missing.sh | tail -n 3
if [ "$(bash ~/usrm2/cloud/verify_pull.sh | grep -c '^MISSING\|^NOTDONE')" = 0 ]; then
  for i in 2 0 1; do ~/tnr2/tnr delete $i -y </dev/null 2>&1 | tail -n 1; done
  ~/tnr2/tnr status 2>&1 | tail -n 4
else
  echo "STILL MISSING, not deleting"
fi
