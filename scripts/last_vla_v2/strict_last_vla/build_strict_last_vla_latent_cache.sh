#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/project/VLA-AD/cache/strict_last_vla_latent}"
SPLIT="${SPLIT:-train}"

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/strict_last_vla/build_strict_last_vla_latent_cache.py
  --output-root "${OUTPUT_ROOT}"
  --split "${SPLIT}"
)

[[ -n "${NAVSIM_LOG_PATH:-}" ]] && cmd+=(--navsim-log-path "${NAVSIM_LOG_PATH}")
[[ -n "${SENSOR_BLOBS_PATH:-}" ]] && cmd+=(--sensor-blobs-path "${SENSOR_BLOBS_PATH}")
[[ -n "${RECOGDRIVE_VLM_PATH:-}" ]] && cmd+=(--vlm-path "${RECOGDRIVE_VLM_PATH}")
[[ -n "${STAGE1_CHECKPOINT:-}" ]] && cmd+=(--stage1-checkpoint "${STAGE1_CHECKPOINT}")
[[ -n "${BASE_CHUNK_ROOT:-}" ]] && cmd+=(--base-chunk-root "${BASE_CHUNK_ROOT}")
[[ -n "${TEACHER_CACHE_ROOT:-}" ]] && cmd+=(--teacher-cache-root "${TEACHER_CACHE_ROOT}")
if [[ "${INCLUDE_TEACHER_TARGETS:-0}" == "1" ]]; then
  cmd+=(--include-teacher-targets)
fi
if [[ "${ALLOW_EVAL_TEACHER_TARGETS:-0}" == "1" ]]; then
  cmd+=(--allow-eval-teacher-targets)
fi

if [[ "${RUN_CACHE:-0}" != "1" ]]; then
  printf 'RUN_CACHE is not 1; strict latent cache generation will run gated dry mode.\n'
fi
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
