#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
SUPPORT_ROOT=${SUPPORT_ROOT:-$(cat "${REPO_ROOT}/outputs/latest_sg_fps_support_navtrain_v3.txt")}
RUN_ID=${RUN_ID:-sg_fps_external_navtrain_ddv2_drivor_$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_ID}}
DDV2_OUT_ROOT=${DDV2_OUT_ROOT:-${OUT_ROOT}/ddv2_candidates}
DRIVOR_OUT_ROOT=${DRIVOR_OUT_ROOT:-${OUT_ROOT}/drivor_candidates}
NUM_SHARDS=${NUM_SHARDS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PARTIAL_EVERY=${PARTIAL_EVERY:-512}
DDV2_NUM_WORKERS=${DDV2_NUM_WORKERS:-4}
DDV2_PREFETCH_FACTOR=${DDV2_PREFETCH_FACTOR:-2}
DDV2_PIN_MEMORY=${DDV2_PIN_MEMORY:-true}
DRIVOR_NUM_WORKERS=${DRIVOR_NUM_WORKERS:-4}
DRIVOR_PREFETCH_FACTOR=${DRIVOR_PREFETCH_FACTOR:-2}
DRIVOR_PIN_MEMORY=${DRIVOR_PIN_MEMORY:-false}
POLL_SECONDS=${POLL_SECONDS:-300}
WAIT_FOR_RESELECT_AUDIT=${WAIT_FOR_RESELECT_AUDIT:-true}
RUN_DDV2=${RUN_DDV2:-true}
RUN_DRIVOR=${RUN_DRIVOR:-true}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"
echo "${OUT_ROOT}" > "${REPO_ROOT}/outputs/latest_sg_fps_external_navtrain_candidates.txt"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/watcher.log"
}

live_pid_count() {
  local pid_dir="$1"
  local count=0
  local pid_file pid
  shopt -s nullglob
  for pid_file in "${pid_dir}"/*.pid; do
    pid=$(cat "${pid_file}" 2>/dev/null || true)
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      count=$((count + 1))
    fi
  done
  shopt -u nullglob
  echo "${count}"
}

wait_support_done() {
  log "waiting for support build: ${SUPPORT_ROOT}"
  while true; do
    local alive=0
    local count=0
    local i pid_file pid
    count=$(find "${SUPPORT_ROOT}/support_v3" -maxdepth 1 -type f -name '*.pkl.xz' 2>/dev/null | wc -l)
    for i in 0 1 2 3 4 5 6 7; do
      pid_file="${SUPPORT_ROOT}/shard_${i}/launcher.pid"
      if [[ -f "${pid_file}" ]]; then
        pid=$(cat "${pid_file}" 2>/dev/null || true)
        if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
          alive=$((alive + 1))
        fi
      fi
    done
    log "support_status alive_shards=${alive} record_count=${count}"
    if [[ "${alive}" -eq 0 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
  if [[ "${WAIT_FOR_RESELECT_AUDIT}" == "true" ]]; then
    log "waiting for support reselect audit"
    while [[ ! -f "${SUPPORT_ROOT}/sg_fps_support_audit_after_reselect.json" ]]; do
      sleep "${POLL_SECONDS}"
    done
    log "support reselect audit ready: ${SUPPORT_ROOT}/sg_fps_support_audit_after_reselect.json"
  fi
}

launch_ddv2() {
  log "launching DDV2 full navtrain candidates: ${DDV2_OUT_ROOT}"
  REPO_ROOT="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${DDV2_OUT_ROOT}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PARTIAL_EVERY="${PARTIAL_EVERY}" \
  NUM_WORKERS="${DDV2_NUM_WORKERS}" \
  PREFETCH_FACTOR="${DDV2_PREFETCH_FACTOR}" \
  PIN_MEMORY="${DDV2_PIN_MEMORY}" \
  RESUME=true \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_generate_ddv2_navtrain_candidates.sh" \
    > "${OUT_ROOT}/logs/ddv2_launcher.outer.log" 2>&1
}

launch_drivor() {
  log "launching DriveOR full navtrain candidates: ${DRIVOR_OUT_ROOT}"
  REPO_ROOT="${REPO_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUT_ROOT="${DRIVOR_OUT_ROOT}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  PARTIAL_EVERY="${PARTIAL_EVERY}" \
  NUM_WORKERS="${DRIVOR_NUM_WORKERS}" \
  PREFETCH_FACTOR="${DRIVOR_PREFETCH_FACTOR}" \
  PIN_MEMORY="${DRIVOR_PIN_MEMORY}" \
  RESUME=true \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_generate_drivor_navtrain_candidates.sh" \
    > "${OUT_ROOT}/logs/drivor_launcher.outer.log" 2>&1
}

wait_candidate_job() {
  local name="$1"
  local root="$2"
  log "waiting for ${name}: ${root}"
  while true; do
    local live summaries partials submissions
    live=$(live_pid_count "${root}/pids")
    summaries=$(find "${root}/submissions" -name summary.json 2>/dev/null | wc -l)
    partials=$(find "${root}/submissions" -name summary.partial.json 2>/dev/null | wc -l)
    submissions=$(find "${root}/submissions" -name submission.pkl 2>/dev/null | wc -l)
    log "${name}_status live=${live} summaries=${summaries}/${NUM_SHARDS} submissions=${submissions}/${NUM_SHARDS} partial_summaries=${partials}"
    if [[ "${live}" -eq 0 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
  local final_count
  final_count=$(find "${root}/submissions" -name summary.json 2>/dev/null | wc -l)
  if [[ "${final_count}" -ne "${NUM_SHARDS}" ]]; then
    log "${name} incomplete: final_summary_count=${final_count}/${NUM_SHARDS}"
    return 1
  fi
  log "${name} complete"
}

{
  echo "run_id=${RUN_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "support_root=${SUPPORT_ROOT}"
  echo "ddv2_out_root=${DDV2_OUT_ROOT}"
  echo "drivor_out_root=${DRIVOR_OUT_ROOT}"
  echo "num_shards=${NUM_SHARDS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "partial_every=${PARTIAL_EVERY}"
  echo "run_ddv2=${RUN_DDV2}"
  echo "run_drivor=${RUN_DRIVOR}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT_ROOT}/commands.log"

wait_support_done
if [[ "${RUN_DDV2}" == "true" ]]; then
  launch_ddv2
  wait_candidate_job "ddv2" "${DDV2_OUT_ROOT}"
elif [[ "${RUN_DDV2}" == "wait_existing" ]]; then
  wait_candidate_job "ddv2" "${DDV2_OUT_ROOT}"
fi
if [[ "${RUN_DRIVOR}" == "true" ]]; then
  launch_drivor
  wait_candidate_job "drivor" "${DRIVOR_OUT_ROOT}"
fi
log "external candidate generation pipeline complete"
