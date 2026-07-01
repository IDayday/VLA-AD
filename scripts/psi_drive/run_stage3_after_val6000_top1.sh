#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
STAGE2_RUN_ROOT="${STAGE2_RUN_ROOT:-/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z}"
STAGE2_CHECKPOINT_ROOT="${STAGE2_CHECKPOINT_ROOT:-${STAGE2_RUN_ROOT}/checkpoints/raw}"
VAL6000_STATUS_TSV="${VAL6000_STATUS_TSV:-${STAGE2_RUN_ROOT}/eval/val6000/checkpoint_eval_status.tsv}"
VAL6000_TOP5_TSV="${VAL6000_TOP5_TSV:-${STAGE2_RUN_ROOT}/rankings/val6000/current_top5.tsv}"
EXPECTED_CHECKPOINTS="${EXPECTED_CHECKPOINTS:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WAIT_FOR_VAL6000="${WAIT_FOR_VAL6000:-1}"
FAIL_ON_VAL6000_FAILED="${FAIL_ON_VAL6000_FAILED:-1}"
STATUS_ONLY="${STATUS_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
LAUNCH_WATCHERS="${LAUNCH_WATCHERS:-1}"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-psi_drive_stage3_sr_pgrpo_from_stage2_val6000_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/${EXPERIMENT_NAME}}"
STAGE3_LOG_DIR="${OUTPUT_DIR}/logs"
STAGE3_STATE_DIR="${OUTPUT_DIR}/state"
ORCHESTRATOR_LOG="${ORCHESTRATOR_LOG:-${STAGE3_LOG_DIR}/stage3_after_val6000_orchestrator.log}"

STAGE3_TRAIN_SCRIPT="${STAGE3_TRAIN_SCRIPT:-${REPO_ROOT}/scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh}"
STAGE3_WATCHER_SCRIPT="${STAGE3_WATCHER_SCRIPT:-${REPO_ROOT}/scripts/evaluation/watch_psi_stage3_checkpoints.sh}"

mkdir -p "${STAGE3_LOG_DIR}" "${STAGE3_STATE_DIR}"

exec 9>"${STAGE3_STATE_DIR}/stage3_after_val6000.lock"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another orchestrator already holds ${STAGE3_STATE_DIR}/stage3_after_val6000.lock"
  exit 0
fi

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${ORCHESTRATOR_LOG}"
}

checkpoint_target_count() {
  if [[ -n "${EXPECTED_CHECKPOINTS}" ]]; then
    printf '%s\n' "${EXPECTED_CHECKPOINTS}"
    return
  fi
  find "${STAGE2_CHECKPOINT_ROOT}" -maxdepth 1 -type f -name '*.ckpt' | wc -l
}

val6000_counts() {
  if [[ ! -f "${VAL6000_STATUS_TSV}" ]]; then
    printf '0\t0\t0\n'
    return
  fi
  awk -F '\t' '
    NR > 1 && $4 == "done" { done[$3] = 1 }
    NR > 1 && $4 == "failed" { failed[$3] = 1 }
    NR > 1 && $4 == "started" { started[$3] = 1 }
    END { print length(done) "\t" length(failed) "\t" length(started) }
  ' "${VAL6000_STATUS_TSV}"
}

print_status() {
  local target done failed started
  target="$(checkpoint_target_count)"
  IFS=$'\t' read -r done failed started < <(val6000_counts)
  log "val6000 status done=${done}/${target} started=${started} failed=${failed}"
  if [[ -f "${VAL6000_TOP5_TSV}" ]]; then
    awk -F '\t' 'NR == 2 {printf "current_top1 checkpoint=%s sha=%s pdms=%s object=%s\n", $2, $3, $6, $4}' "${VAL6000_TOP5_TSV}" | tee -a "${ORCHESTRATOR_LOG}"
  fi
}

val6000_complete() {
  local target done failed started
  target="$(checkpoint_target_count)"
  IFS=$'\t' read -r done failed started < <(val6000_counts)
  if [[ "${FAIL_ON_VAL6000_FAILED}" == "1" && "${failed}" -gt 0 ]]; then
    log "val6000 has failed checkpoint evaluations failed=${failed}; refusing to launch Stage3"
    exit 3
  fi
  [[ "${done}" -ge "${target}" && "${target}" -gt 0 ]]
}

select_stage2_checkpoint() {
  if [[ ! -f "${VAL6000_TOP5_TSV}" ]]; then
    log "missing val6000 ranking file: ${VAL6000_TOP5_TSV}"
    exit 4
  fi
  local row checkpoint_id sha object_path source_path pdms
  row="$(awk -F '\t' 'NR == 2 {print $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6}' "${VAL6000_TOP5_TSV}")"
  if [[ -z "${row}" ]]; then
    log "val6000 ranking file has no Top-1 row: ${VAL6000_TOP5_TSV}"
    exit 4
  fi
  IFS=$'\t' read -r checkpoint_id sha object_path source_path pdms <<< "${row}"
  if [[ ! -f "${object_path}" ]]; then
    log "selected Stage2 object checkpoint does not exist: ${object_path}"
    exit 4
  fi
  {
    printf 'checkpoint_id\tsha256\tobject_path\tsource_path\tpdms\tselected_at\n'
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${checkpoint_id}" \
      "${sha}" \
      "${object_path}" \
      "${source_path}" \
      "${pdms}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${OUTPUT_DIR}/stage2_val6000_selected_top1.tsv"
  printf '%s\n' "${object_path}"
}

start_background() {
  local name="$1"
  shift
  local pid_file="${STAGE3_STATE_DIR}/${name}.pid"
  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}")"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      log "${name} already running pid=${old_pid}"
      return
    fi
  fi
  log "starting ${name}: $*"
  nohup "$@" >> "${STAGE3_LOG_DIR}/${name}.nohup.log" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" > "${pid_file}"
  log "started ${name} pid=${pid}"
}

launch_stage3() {
  local stage2_ckpt="$1"
  log "selected Stage2 checkpoint for Stage3: ${stage2_ckpt}"
  log "Stage3 output_dir=${OUTPUT_DIR}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry-run: would launch Stage3 training and watchers"
    PSI_STAGE2_INIT_CKPT="${stage2_ckpt}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
      DRY_RUN=1 \
      bash "${STAGE3_TRAIN_SCRIPT}" | tee -a "${ORCHESTRATOR_LOG}"
    return 0
  fi

  start_background \
    stage3_train \
    env \
      PSI_STAGE2_INIT_CKPT="${stage2_ckpt}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
      bash "${STAGE3_TRAIN_SCRIPT}"

  if [[ "${LAUNCH_WATCHERS}" == "1" ]]; then
    start_background \
      stage3_watch_val6000 \
      env \
        STAGE=stage3 \
        RUN_ROOT="${OUTPUT_DIR}" \
        CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints/raw" \
        EVAL_SPLIT=val6000 \
        bash "${STAGE3_WATCHER_SCRIPT}"
    start_background \
      stage3_watch_navtest \
      env \
        STAGE=stage3 \
        RUN_ROOT="${OUTPUT_DIR}" \
        CHECKPOINT_ROOT="${OUTPUT_DIR}/checkpoints/raw" \
        EVAL_SPLIT=navtest \
        bash "${STAGE3_WATCHER_SCRIPT}"
  fi
}

print_status
if [[ "${STATUS_ONLY}" == "1" ]]; then
  exit 0
fi

while ! val6000_complete; do
  if [[ "${WAIT_FOR_VAL6000}" != "1" ]]; then
    log "val6000 is not complete and WAIT_FOR_VAL6000=0; exiting without launching Stage3"
    exit 5
  fi
  sleep "${POLL_SECONDS}"
  print_status
done

stage2_ckpt="$(select_stage2_checkpoint)"
launch_stage3 "${stage2_ckpt}"
