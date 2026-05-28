Installation for ReCogDrive
# Installation for ReCogDrive

After successfully downloading the NAVSIM dataset:

## 1. Pretraining (SFT on VLM)
If you want to perform **ReCogDrive pretraining** and do **SFT training on VLM**, install InternVL dependencies:
```bash
pip install -r internvl_chat/internvl_chat.txt
```

##  2. Training & Evaluation on NAVSIM
If you want to train and evaluate ReCogDrive on NAVSIM, install the requirements:

```bash
cd /path/to/ReCogDrive
pip install -e .
```

## 3. Optional VGGT Teacher Installation
VGGT is only needed when generating real VGGT expert-token caches. Keep the
project environment on Python 3.9; do not create a separate Python 3.10 VGGT
environment for this project.

Preferred install path:

```bash
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
bash scripts/install_vggt_py39.sh /path/to/vggt
```

The helper installs VGGT into the active Python 3.9 environment with:

```bash
python -m pip install --no-deps --ignore-requires-python -e /path/to/vggt
```

This installs the VGGT source package only. It intentionally avoids dependency
resolution so the existing torch, torchvision, CUDA, and Python 3.9 setup is
preserved.

If you do not want to install the package, making the source tree importable is
also sufficient:

```bash
export PYTHONPATH=/path/to/vggt:${PYTHONPATH}
python -c "import sys, vggt; print(sys.version); print(vggt)"
```

