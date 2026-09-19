#!/bin/bash
# From forlindesk2: pull the stores verify_pull.sh reports MISSING/NOTDONE, then verify again.
T=/vesuvius/usrm2/teacher
bash ~/usrm2/cloud/verify_pull.sh | grep "^MISSING\|^NOTDONE" | while read _ h d; do
  mkdir -p $T/$d && rsync -a $h:out/$d/ $T/$d/ && echo "pulled $h $d"
done
bash ~/usrm2/cloud/verify_pull.sh | grep -v "^MISSING\|^NOTDONE" ; bash ~/usrm2/cloud/verify_pull.sh | grep -c "^MISSING\|^NOTDONE"
