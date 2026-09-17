#!/bin/bash
# On the instance: install whatever the teacher imports still miss (vesuvius is installed --no-deps), then check m7 too.
cd ~; . venv/bin/activate
declare -A MAP=([nrrd]=pynrrd [yaml]=pyyaml [cv2]=opencv-python-headless [skimage]=scikit-image [sklearn]=scikit-learn [PIL]=pillow [rich]=rich)
for i in $(seq 1 12); do
  m=$(python -c "from vesuvius.models.build.build_network_from_config import NetworkFromConfig; from usrm2 import m7, teacher; print('IMPORTS_OK')" 2>&1 | grep -o "No module named '[^']*'" | head -1 | sed "s/No module named '\([^'.]*\).*/\1/")
  [ -z "$m" ] && break
  pkg=${MAP[$m]:-$m}; echo "installing $pkg for $m"; pip install -q "$pkg" || pip install -q --no-deps "$pkg"
done
python -c "from vesuvius.models.build.build_network_from_config import NetworkFromConfig; from usrm2 import m7, teacher; print('IMPORTS_OK')" 2>&1 | grep -v Warn | tail -1
