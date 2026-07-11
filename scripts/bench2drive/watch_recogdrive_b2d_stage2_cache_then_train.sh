#!/usr/bin/env bash
set -Eeuo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}

CACHE_RUN_ROOT=${CACHE_RUN_ROOT:?Set CACHE_RUN_ROOT to the full Stage2 cache run directory}
CHUNK_CACHE_ROOT=${CHUNK_CACHE_ROOT:-${CACHE_RUN_ROOT}/train}
VLM_PATH=${VLM_PATH:?Set VLM_PATH to the accepted Stage1 checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR for scratch Stage2 training}
EXPECTED_SHARDS=${EXPECTED_SHARDS:-8}
EXPECTED_CLIPS=${EXPECTED_CLIPS:-1000}
EXPECTED_RECORDS=${EXPECTED_RECORDS:-202656}
POLL_SECONDS=${POLL_SECONDS:-30}
CACHE_TMUX_SESSION=${CACHE_TMUX_SESSION:-}

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29551}
GLOBAL_EPOCHS=${GLOBAL_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PRECISION=${PRECISION:-bf16}
SAVE_EVERY=${SAVE_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-20}
SEED=${SEED:-20260709}

WATCH_DIR=${WATCH_DIR:-${CACHE_RUN_ROOT}/watcher}
WATCH_LOG=${WATCH_LOG:-${WATCH_DIR}/watch.log}
TRAIN_LOG=${TRAIN_LOG:-${WATCH_DIR}/stage2_train.log}
WATCH_STATUS=${WATCH_STATUS:-${WATCH_DIR}/status}
VALIDATOR=${VLA_AD_ROOT}/scripts/bench2drive/validate_recogdrive_b2d_stage2_cache.py
TRAIN_LAUNCHER=${VLA_AD_ROOT}/scripts/bench2drive/run_recogdrive_b2d_stage2_closest_public.sh

mkdir -p "${WATCH_DIR}"
GIT_COMMIT=$(cd "${VLA_AD_ROOT}" && git rev-parse HEAD)

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCH_LOG}"
}

write_status() {
  printf '%s\n' "$1" > "${WATCH_STATUS}"
}

validate_cache() {
  local -a args=(
    "${PYTHON_BIN}"
    "${VALIDATOR}"
    --cache-root "${CHUNK_CACHE_ROOT}"
    --expected-shards "${EXPECTED_SHARDS}"
    --expected-clips "${EXPECTED_CLIPS}"
    --expected-vlm-path "${VLM_PATH}"
  )
  if [[ -n "${EXPECTED_RECORDS}" ]]; then
    args+=(--expected-records "${EXPECTED_RECORDS}")
  fi
  "${args[@]}"
}

log "watcher started; cache=${CHUNK_CACHE_ROOT}, Stage2 output=${OUTPUT_DIR}"
log "watcher git commit=${GIT_COMMIT}; expected shards/clips/records=${EXPECTED_SHARDS}/${EXPECTED_CLIPS}/${EXPECTED_RECORDS}"
write_status "waiting_for_cache"

while true; do
  if [[ -f "${OUTPUT_DIR}/launch_env.txt" || -f "${OUTPUT_DIR}/checkpoint_load.json" ]]; then
    log "Stage2 already has launch artifacts; refusing a duplicate launch."
    write_status "stage2_already_launched"
    exit 0
  fi

  validation_output=""
  if validation_output=$(validate_cache 2>&1); then
    printf '%s\n' "${validation_output}" > "${WATCH_DIR}/cache_validation.json"
    log "cache validation passed; launching random-init Stage2 now."
    write_status "launching_stage2"
    break
  fi

  log "cache not ready: ${validation_output}"
  if [[ -n "${CACHE_TMUX_SESSION}" ]] && ! tmux has-session -t "${CACHE_TMUX_SESSION}" 2>/dev/null; then
    log "cache tmux session ${CACHE_TMUX_SESSION} disappeared before validation passed."
    write_status "cache_failed"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

set +e
env \
  CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT}" \
  EXPECTED_VLM_PATH="${VLM_PATH}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  RANDOM_INIT_POLICY=1 \
  GLOBAL_EPOCHS="${GLOBAL_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR}" \
  PRECISION="${PRECISION}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  LOG_EVERY="${LOG_EVERY}" \
  SEED="${SEED}" \
  GPU_LIST="${GPU_LIST}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_PORT="${MASTER_PORT}" \
  bash "${TRAIN_LAUNCHER}" >> "${TRAIN_LOG}" 2>&1
train_status=$?
set -e

if [[ "${train_status}" -eq 75 ]]; then
  log "another Stage2 launcher owns the output lock; duplicate launch suppressed."
  write_status "stage2_already_launched"
  exit 0
fi
if [[ "${train_status}" -eq 0 ]]; then
  log "Stage2 training completed successfully."
  write_status "stage2_complete"
else
  log "Stage2 training failed with status ${train_status}; inspect ${TRAIN_LOG}."
  write_status "stage2_failed"
fi
exit "${train_status}"
