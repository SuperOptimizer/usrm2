#!/bin/bash
# From forlindesk2: ship code, models, the val box (excluded from sampling) and the volcomp library to an instance.
# usage: cloud/push.sh SSH_TARGET   (e.g. tnr-0 once `tnr connect 0` has written the ssh config)
H=$1
ssh $H mkdir -p usrm2 models lib
rsync -a --exclude .venv --exclude .git --exclude .claude ~/usrm2/ $H:usrm2/
rsync -a ~/.cache/usrm/bin/libvolcomp.so $H:lib/
rsync -a /vesuvius/usrm2/teacher/eval.zarr/ $H:eval.zarr/
rsync -a --progress /vesuvius/tsm/models/surface_recto_3dunet.pth /vesuvius/tsm/models/surface_m7_nnunet.pth $H:models/
ssh $H "sudo mkdir -p /vesuvius/tsm && sudo ln -sfn ~/models /vesuvius/tsm/models"  # the checkpoint paths in usrm2 are absolute
echo PUSHED $H
