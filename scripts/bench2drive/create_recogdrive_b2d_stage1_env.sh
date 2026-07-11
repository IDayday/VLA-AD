#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BASE_PYTHON=${BASE_PYTHON:-/root/miniconda3/envs/navsim/bin/python}
ENV_DIR=${ENV_DIR:-/mnt/project/recogdrive_stage1_env}
REQUIREMENTS=${REQUIREMENTS:-${VLA_AD_ROOT}/configs/envs/recogdrive_b2d_stage1_requirements.txt}
FLASH_ATTN_WHEEL_NAME=flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8%2Bcu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
FLASH_ATTN_WHEEL_SHA256=debb44efae117dd9124c40931ccfb8065175aef7facea964819749c95e87018a

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Base Python is not executable: ${BASE_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${REQUIREMENTS}" ]]; then
  echo "Requirements lock not found: ${REQUIREMENTS}" >&2
  exit 2
fi
if ! command -v curl >/dev/null; then
  echo "curl is required to fetch the pinned FlashAttention wheel" >&2
  exit 2
fi
if [[ -e "${ENV_DIR}" && "${REBUILD_STAGE1_ENV:-0}" != "1" ]]; then
  echo "Environment already exists: ${ENV_DIR}" >&2
  echo "Set REBUILD_STAGE1_ENV=1 to rebuild this dedicated environment." >&2
  exit 2
fi

"${BASE_PYTHON}" -m venv --clear --copies "${ENV_DIR}"
PYTHON=${ENV_DIR}/bin/python

"${PYTHON}" -m pip install \
  pip==24.0 setuptools==69.5.1 wheel==0.43.0 packaging==24.0 ninja==1.11.1.1
"${PYTHON}" -m pip install torch==2.2.2 torchvision==0.17.2
"${PYTHON}" -m pip install -r "${REQUIREMENTS}"
ARTIFACT_DIR=${ENV_DIR}/artifacts
mkdir -p "${ARTIFACT_DIR}"
FLASH_ATTN_WHEEL=${ARTIFACT_DIR}/${FLASH_ATTN_WHEEL_NAME}
curl --fail --location --retry 20 --retry-delay 2 \
  --output "${FLASH_ATTN_WHEEL}" "${FLASH_ATTN_WHEEL_URL}"
printf '%s  %s\n' "${FLASH_ATTN_WHEEL_SHA256}" "${FLASH_ATTN_WHEEL}" | sha256sum --check
"${PYTHON}" -m pip install --no-deps "${FLASH_ATTN_WHEEL}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "${PYTHON}" "${VLA_AD_ROOT}/scripts/bench2drive/check_recogdrive_b2d_stage1_environment.py" \
  --min-gpus 1 \
  --report "${ENV_DIR}/environment_preflight.json"
