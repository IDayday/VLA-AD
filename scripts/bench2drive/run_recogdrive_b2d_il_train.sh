#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}

CONFIG=${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_il.yaml}
CHUNK_CACHE_ROOT=${CHUNK_CACHE_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_vlm_cache_full_v1}
CHUNK_NAME_PATTERN=${CHUNK_NAME_PATTERN:-shard_*}
INIT_POLICY_CHECKPOINT=${INIT_POLICY_CHECKPOINT:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}
OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_vlm_il_train_full_stage}

GLOBAL_EPOCHS=${GLOBAL_EPOCHS:-4}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
PRECISION=${PRECISION:-bf16}
LR_ACTION_HEAD=${LR_ACTION_HEAD:-2e-5}
LR_SCHEDULER=${LR_SCHEDULER:-official-cosine}
LR_SCHEDULER_EPOCHS=${LR_SCHEDULER_EPOCHS:-200}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-3}
SAVE_EVERY=${SAVE_EVERY:-1000}
SEED=${SEED:-20260709}

cd "${VLA_AD_ROOT}"
exec "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" python scripts/train_recogdrive_expert_chunked.py \
  --config "${CONFIG}" \
  --init-policy-checkpoint "${INIT_POLICY_CHECKPOINT}" \
  --chunk-cache-root "${CHUNK_CACHE_ROOT}" \
  --chunk-name-pattern "${CHUNK_NAME_PATTERN}" \
  --output-dir "${OUTPUT_DIR}" \
  --global-epochs "${GLOBAL_EPOCHS}" \
  --flat-global-dataset \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --precision "${PRECISION}" \
  --lr-action-head "${LR_ACTION_HEAD}" \
  --lr-scheduler "${LR_SCHEDULER}" \
  --lr-scheduler-epochs "${LR_SCHEDULER_EPOCHS}" \
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --save-every-epoch \
  --seed "${SEED}"
