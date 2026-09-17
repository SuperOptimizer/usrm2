#!/bin/bash
# Provision a Thunder Compute A6000 (Ubuntu) for teacher inference. Run ON the instance after cloud/push.sh.
set -e
cd ~
sudo apt-get install -y -q python3-venv rsync > /dev/null 2>&1 || true
python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q "zarr>=3" numpy scipy fsspec aiohttp requests pillow tifffile pynrrd pyyaml
pip install -q --no-deps einops timm  # timm would pull a PyPI torchvision that mismatches the cu128 torch
pip install -q --no-deps vesuvius dynamic-network-architectures
pip install -q "volcomp-zarr @ git+https://github.com/SuperOptimizer/volume-compressor#subdirectory=python"
pip install -q -e usrm2 --no-deps
bash usrm2/cloud/fixdeps.sh
export VOLCOMP_LIB=$HOME/lib/libvolcomp.so
python -c "import torch, zarr, volcomp_zarr; print(torch.cuda.get_device_name(0), torch.__version__, zarr.__version__)"
echo PROVISIONED
