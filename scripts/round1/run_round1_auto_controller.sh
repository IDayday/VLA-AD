#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-training-vla-zt-peer}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:?Set A0_INIT_CHECKPOINT.}"
A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT:?Set A0_REFERENCE_CHECKPOINT.}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:?Set NAVTEST_CHUNK_CACHE_ROOT.}"
NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:?Set NAVTEST_CHUNK_NAME_PATTERN.}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:?Set METRIC_CACHE_DIR.}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
LOCAL_PROGRESSIVE_MASTER_PORT="${LOCAL_PROGRESSIVE_MASTER_PORT:-29532}"
REMOTE_PROGRESSIVE_MASTER_PORT="${REMOTE_PROGRESSIVE_MASTER_PORT:-29542}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
CORRUPTION_MAX_SAMPLES="${CORRUPTION_MAX_SAMPLES:-1000}"
FULL_CORRUPTION="${FULL_CORRUPTION:-0}"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
CONTROLLER_LOG="${LOG_DIR}/round1_auto_controller.log"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${CONTROLLER_LOG}"
}

wait_local_pid_file() {
  local pid_file="$1"
  local label="$2"
  local pid
  while [[ ! -f "${pid_file}" ]]; do
    log "Waiting for ${label} pid file: ${pid_file}"
    sleep 60
  done
  pid="$(cat "${pid_file}")"
  log "Watching ${label} local pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 300
  done
  log "${label} local pid=${pid} exited"
}

wait_remote_pid_file() {
  local pid_file="$1"
  local label="$2"
  local pid
  while [[ ! -f "${pid_file}" ]]; do
    log "Waiting for ${label} remote pid file: ${pid_file}"
    sleep 60
  done
  pid="$(cat "${pid_file}")"
  log "Watching ${label} remote pid=${pid} on ${REMOTE_HOST}"
  while ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "kill -0 ${pid} 2>/dev/null"; do
    sleep 300
  done
  log "${label} remote pid=${pid} exited"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    log "Missing required ${label}: ${path}"
    return 1
  fi
}

launch_progressive_lastrd_only() {
  local pid_file="${LOG_DIR}/wave2_progressive_lastrd_only_server1.pid"
  local out_dir="${OUT_ROOT}/wave2_progressive/lastrd_only"
  mkdir -p "${LOG_DIR}"
  log "Launching Wave2 Progressive SFT LastRD-only on local container"
  setsid env PATH="/root/miniconda3/envs/navsim/bin:${PATH}" \
    OUT_ROOT="${OUT_ROOT}" \
    TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT}" \
    TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT}" \
    A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT}" \
    A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT}" \
    STAGE1_5_CHECKPOINT="${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt" \
    MASTER_PORT="${LOCAL_PROGRESSIVE_MASTER_PORT}" \
    bash "${REPO_ROOT}/scripts/round1/run_progressive_lastrd_only_server1.sh" \
    > "${LOG_DIR}/wave2_progressive_lastrd_only_server1.nohup.log" 2>&1 < /dev/null &
  echo $! > "${pid_file}"
  log "Launched local Progressive LastRD-only pid=$(cat "${pid_file}") output=${out_dir}"
}

launch_progressive_hybrid() {
  local pid_file="${LOG_DIR}/wave2_progressive_hybrid_server2.remote_pid"
  log "Launching Wave2 Progressive SFT hybrid on ${REMOTE_HOST}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "
    cd '${REPO_ROOT}' &&
    OUT_ROOT='${OUT_ROOT}';
    mkdir -p \"\$OUT_ROOT/logs\";
    nohup env PATH=/root/miniconda3/envs/navsim/bin:\$PATH \
      OUT_ROOT=\"\$OUT_ROOT\" \
      TRAIN_CHUNK_CACHE_ROOT='${TRAIN_CHUNK_CACHE_ROOT}' \
      TRAIN_TEST_SPLIT='${TRAIN_TEST_SPLIT}' \
      A0_INIT_CHECKPOINT='${A0_INIT_CHECKPOINT}' \
      A0_REFERENCE_CHECKPOINT='${A0_REFERENCE_CHECKPOINT}' \
      STAGE1_5_CHECKPOINT='${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt' \
      MASTER_PORT='${REMOTE_PROGRESSIVE_MASTER_PORT}' \
      LAST_RD_POLICY_KD_WEIGHT=0.05 \
      LAST_RD_POLICY_KD_MODE=noise \
      LAST_RD_RISK_LOSS_WEIGHT=0.0 \
      bash scripts/round1/run_progressive_hybrid_server2.sh \
      > \"\$OUT_ROOT/logs/wave2_progressive_hybrid_server2.nohup.log\" 2>&1 < /dev/null &
    echo \$! > \"\$OUT_ROOT/logs/wave2_progressive_hybrid_server2.remote_pid\"
  "
  require_file "${pid_file}" "remote progressive pid file"
  log "Launched remote Progressive hybrid pid=$(cat "${pid_file}")"
}

run_eval_and_summary() {
  log "Starting Wave3 checkpoint sweep eval"
  OUT_ROOT="${OUT_ROOT}" \
  NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
  NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN}" \
  METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
  RUN_NAME=all \
  GPU_ID="${EVAL_GPU_ID}" \
  bash "${REPO_ROOT}/scripts/round1/eval_progressive_checkpoint_sweep.sh"

  log "Summarizing Wave3 results"
  python "${REPO_ROOT}/scripts/round1/summarize_round1_results.py" \
    --root "${OUT_ROOT}" \
    --baseline-pdms 0.864891

  log "Starting Wave4 corruption eval"
  OUT_ROOT="${OUT_ROOT}" \
  NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
  NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN}" \
  METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
  EVAL_MAX_SAMPLES="${CORRUPTION_MAX_SAMPLES}" \
  FULL_CORRUPTION="${FULL_CORRUPTION}" \
  GPU_ID="${EVAL_GPU_ID}" \
  bash "${REPO_ROOT}/scripts/round1/eval_best_corruption.sh"

  log "Summarizing Wave4 results"
  python "${REPO_ROOT}/scripts/round1/summarize_round1_results.py" \
    --root "${OUT_ROOT}" \
    --baseline-pdms 0.864891
}

main() {
  log "Round1 auto-controller started"
  log "OUT_ROOT=${OUT_ROOT}"
  log "A0 official checkpoint=${A0_INIT_CHECKPOINT}"

  wait_local_pid_file "${LOG_DIR}/wave1_stage1_5_full_server1.pid" "Wave1 Stage1.5 full"
  wait_remote_pid_file "${LOG_DIR}/wave1_stage1_5_jepa_only_server2.remote_pid" "Wave1 Stage1.5 JEPA-only"

  require_file "${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt" "Stage1.5 full adapter"
  python "${REPO_ROOT}/scripts/audit_last_rd_adapter_checkpoint.py" \
    --checkpoint "${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt" \
    --output "${OUT_ROOT}/wave1_stage1_5/full/adapter_audit_controller.json"
  log "Stage1.5 full adapter audit passed"

  launch_progressive_lastrd_only
  launch_progressive_hybrid
  wait_local_pid_file "${LOG_DIR}/wave2_progressive_lastrd_only_server1.pid" "Wave2 Progressive LastRD-only"
  wait_remote_pid_file "${LOG_DIR}/wave2_progressive_hybrid_server2.remote_pid" "Wave2 Progressive hybrid"

  run_eval_and_summary
  log "Round1 auto-controller completed"
}

main "$@"
