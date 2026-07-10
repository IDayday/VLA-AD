#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}
DATASET_REPO=${DATASET_REPO:-owl10/ReCogDrive_Pretraining}
LOCAL_DIR=${LOCAL_DIR:-/mnt/project/recogdrive_pretraining}

cd "${VLA_AD_ROOT}"
mkdir -p "${LOCAL_DIR}"

# Keep download traffic direct from the container unless the caller sets proxy
# variables again explicitly for this process.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --include "Bench2drive_Traj/**" "Bench2drive_QA/**" \
  --local-dir "${LOCAL_DIR}"
