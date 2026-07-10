#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8765}
CONFIG=${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_il.yaml}
PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_vlm_il_train_full_epoch1/best.ckpt}
VLM_PATH=${VLM_PATH:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
PRECISION=${PRECISION:-bf16}
INIT_ACTION_MODE=${INIT_ACTION_MODE:-token_noise}
INIT_ACTION_SEED=${INIT_ACTION_SEED:-20260709}

cd "${VLA_AD_ROOT}"
exec "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" python scripts/bench2drive/serve_recogdrive_b2d.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --config "${CONFIG}" \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --vlm-path "${VLM_PATH}" \
  --precision "${PRECISION}" \
  --init-action-mode "${INIT_ACTION_MODE}" \
  --init-action-seed "${INIT_ACTION_SEED}"
