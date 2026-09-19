#!/bin/bash
# forlindesk2: once tnr-2 is provisioned, sync the latest code, start the store pull + group refresh loops, and
# run the memory/throughput benchmark. usage: bash cloud/a100_after_provision.sh
set -u
rsync -a --exclude .venv --exclude .git --exclude __pycache__ ~/usrm2/ tnr-2:usrm2/
ssh tnr-2 'cd ~ && . venv/bin/activate && pip install -q -e usrm2 --no-deps 2>&1 | tail -n 1; python -c "import usrm2, torch; print(\"code ok\", torch.__version__)"'
ssh tnr-2 'nohup bash usrm2/cloud/a100_pull.sh > ~/a100_pull.out 2>&1 < /dev/null &'
ssh tnr-2 'nohup bash usrm2/cloud/a100_groups.sh > ~/a100_groups.out 2>&1 < /dev/null &'
ssh tnr-2 'cd ~ && . venv/bin/activate && export VOLCOMP_LIB=$HOME/lib/libvolcomp.so && cd usrm2 && python cloud/a100_bench.py 5m,26m,45m,30m6 256,384,512' > ~/a100_bench.out 2>&1
echo "bench done $(date)"; cat ~/a100_bench.out
