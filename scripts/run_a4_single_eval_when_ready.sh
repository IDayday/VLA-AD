#!/usr/bin/env bash
set -Eeuo pipefail

: "${OUT_ROOT:?OUT_ROOT is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${CONT_CONFIG:?CONT_CONFIG is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${EVAL_CHUNK_CACHE_ROOT:?EVAL_CHUNK_CACHE_ROOT is required}"
: "${EVAL_CHUNK_NAME_PATTERN:?EVAL_CHUNK_NAME_PATTERN is required}"
: "${METRIC_CACHE_DIR:?METRIC_CACHE_DIR is required}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.eval.log"
METRICS_FILE="${EVAL_DIR}/metrics.json"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"

{
  echo "[$(date -Is)] eval watcher started: ${RUN_NAME}"
  echo "checkpoint: ${CHECKPOINT}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"
} >>"${LOG_FILE}"

if [[ -f "${METRICS_FILE}" ]]; then
  echo "[$(date -Is)] metrics already exists, skipping: ${METRICS_FILE}" >>"${LOG_FILE}"
  exit 0
fi

while [[ ! -f "${CHECKPOINT}" ]]; do
  if [[ "${WAIT_FOR_CHECKPOINT}" != "1" ]]; then
    echo "[$(date -Is)] checkpoint missing and WAIT_FOR_CHECKPOINT!=1: ${CHECKPOINT}" >>"${LOG_FILE}"
    exit 2
  fi
  echo "[$(date -Is)] waiting for checkpoint: ${CHECKPOINT}" >>"${LOG_FILE}"
  sleep "${POLL_SECONDS}"
done

cmd=(
  "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
  --config "${CONT_CONFIG}"
  --checkpoint "${CHECKPOINT}"
  --chunk-cache-root "${EVAL_CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${EVAL_CHUNK_NAME_PATTERN}"
  --metric-cache-dir "${METRIC_CACHE_DIR}"
  --precision fp32
  --output-dir "${EVAL_DIR}"
)

if [[ -n "${EVAL_MAX_SAMPLES:-}" ]]; then
  cmd+=(--max-samples "${EVAL_MAX_SAMPLES}")
fi

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "[$(date -Is)] starting eval: ${RUN_NAME}" >>"${LOG_FILE}"
"${cmd[@]}" >>"${LOG_FILE}" 2>&1
echo "[$(date -Is)] eval finished: ${RUN_NAME}" >>"${LOG_FILE}"
