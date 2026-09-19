#!/bin/bash
# forlindesk2: (1) quick levels 3-5 at q=1 from the existing level 2 so training can use them today;
# (2) the full offline rebuild 0 -> 1 (q=4) -> 2 (q=2) -> 3,4,5 (q=1), swapped in level by level as each finishes.
cd ~/usrm2; export VOLCOMP_LIB=$HOME/.cache/usrm/bin/libvolcomp.so
B=/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr
for l in 3 4 5 6 7 8 9; do nice .venv/bin/python cloud/make_levels.py $B $((l - 1)) $l 1 --threads 8; done
echo "QUICKLEVELSDONE $(date)"
nice -n 15 .venv/bin/python cloud/make_levels.py $B 0 1 4 --threads 12
nice -n 15 .venv/bin/python cloud/make_levels.py $B 1 2 2 --threads 12
for l in 3 4 5 6 7 8 9; do nice -n 15 .venv/bin/python cloud/make_levels.py $B $((l - 1)) $l 1 --threads 8; done
echo "PYRAMIDDONE $(date)"
