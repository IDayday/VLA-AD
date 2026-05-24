#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the NAVSIM dataset root.}"
: "${NAVSIM_EXP_ROOT:?Set NAVSIM_EXP_ROOT to the NAVSIM experiment/cache root.}"
: "${RECOGDRIVE_EXPERT_CACHE_DIR:=${NAVSIM_EXP_ROOT}/recogdrive_expert_cache}"
: "${RECOGDRIVE_JEPA_MODEL_PATH:?Set RECOGDRIVE_JEPA_MODEL_PATH. Use any stable string for --teacher-backend dummy.}"
: "${RECOGDRIVE_VGGT_MODEL_PATH:?Set RECOGDRIVE_VGGT_MODEL_PATH. Use any stable string for --teacher-backend dummy.}"

SPLIT=${SPLIT:-trainval}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
TEACHER_BACKEND=${TEACHER_BACKEND:-dummy}
DEVICE=${DEVICE:-cuda}
PRECISION=${PRECISION:-fp16}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-8}
RECOGDRIVE_NUM_JEPA_TOKENS=${RECOGDRIVE_NUM_JEPA_TOKENS:-4}
RECOGDRIVE_NUM_VGGT_TOKENS=${RECOGDRIVE_NUM_VGGT_TOKENS:-4}
RECOGDRIVE_JEPA_DIM=${RECOGDRIVE_JEPA_DIM:-768}
RECOGDRIVE_VGGT_DIM=${RECOGDRIVE_VGGT_DIM:-2048}

TARGET_ARGS=()
if [[ "${COMPUTE_FUTURE_TARGETS:-1}" == "1" ]]; then
  TARGET_ARGS+=(--compute-future-targets)
fi

MAX_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

python "${REPO_ROOT}/navsim/planning/script/run_recogdrive_expert_feature_caching.py" \
  --split "${SPLIT}" \
  --scene-filter "${TRAIN_TEST_SPLIT}" \
  --output-cache-dir "${RECOGDRIVE_EXPERT_CACHE_DIR}" \
  --jepa-checkpoint-or-model-path "${RECOGDRIVE_JEPA_MODEL_PATH}" \
  --vggt-checkpoint-or-model-path "${RECOGDRIVE_VGGT_MODEL_PATH}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-jepa-tokens "${RECOGDRIVE_NUM_JEPA_TOKENS}" \
  --num-vggt-tokens "${RECOGDRIVE_NUM_VGGT_TOKENS}" \
  --dummy-jepa-dim "${RECOGDRIVE_JEPA_DIM}" \
  --dummy-vggt-dim "${RECOGDRIVE_VGGT_DIM}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --teacher-backend "${TEACHER_BACKEND}" \
  --navsim-log-path "${OPENSCENE_DATA_ROOT}/navsim_logs/${SPLIT}" \
  --sensor-blobs-path "${OPENSCENE_DATA_ROOT}/sensor_blobs/${SPLIT}" \
  "${TARGET_ARGS[@]}" \
  "${MAX_SAMPLE_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"
