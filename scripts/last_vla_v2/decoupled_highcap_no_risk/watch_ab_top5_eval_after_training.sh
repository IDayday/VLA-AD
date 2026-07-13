#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUT_ROOT NAVTEST_CHUNK_CACHE_ROOT METRIC_CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
RUN_EVAL_ONCE="${RUN_EVAL_ONCE:-1}"
LOG_DIR="${OUT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/ab_top5_eval_watcher.log"

A_RUN_ROOT="${A_RUN_ROOT:-${OUT_ROOT}/A_frozen_vlm_stage2_progressive}"
B_RUN_ROOT="${B_RUN_ROOT:-${OUT_ROOT}/B_lora_stage2_progressive}"
B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT:-}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

local_training_running_for() {
  local output_dir="$1"
  ps -eo cmd | grep 'run_training_recogdrive.py' | grep -F "output_dir=${output_dir}" | grep -v grep >/dev/null 2>&1
}

remote_training_running_for() {
  local output_dir="$1"
  if [[ -z "${REMOTE_HOST}" ]]; then
    return 1
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "ps -eo cmd | grep 'run_training_recogdrive.py' | grep -F 'output_dir=${output_dir}' | grep -v grep >/dev/null" \
    >/dev/null 2>&1
}

training_running() {
  local_training_running_for "${A_RUN_ROOT}" && return 0
  local_training_running_for "${B_RUN_ROOT}" && return 0
  remote_training_running_for "${B_RUN_ROOT}" && return 0
  return 1
}

stable_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local size_a size_b
  size_a="$(stat -c '%s' "${path}" 2>/dev/null || echo 0)"
  [[ "${size_a}" -gt 0 ]] || return 1
  sleep "${STABLE_SECONDS}"
  [[ -f "${path}" ]] || return 1
  size_b="$(stat -c '%s' "${path}" 2>/dev/null || echo 0)"
  [[ "${size_a}" == "${size_b}" && "${size_b}" -gt 0 ]]
}

topk_count() {
  local run_root="$1"
  find "${run_root}/lightning_logs" -path '*/checkpoints/epoch=*-step=*.ckpt' -type f 2>/dev/null | wc -l
}

ready_for_eval() {
  [[ -d "${A_RUN_ROOT}" && -d "${B_RUN_ROOT}" ]] || return 1
  stable_file "${A_RUN_ROOT}/latest.ckpt" || return 1
  stable_file "${B_RUN_ROOT}/latest.ckpt" || return 1
  [[ "$(topk_count "${A_RUN_ROOT}")" -gt 0 ]] || return 1
  [[ "$(topk_count "${B_RUN_ROOT}")" -gt 0 ]] || return 1
}

run_eval() {
  cd "${PROJECT_ROOT}"
  log "starting A/B top-5 NAVTEST eval sweep"
  OUT_ROOT="${OUT_ROOT}" \
  NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
  METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
  CONFIG="${CONFIG}" \
  VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT}" \
  B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH}" \
  B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  INCLUDE_TOPK_CHECKPOINTS=1 \
  INCLUDE_CHECKPOINTS_TO_EVAL="${INCLUDE_CHECKPOINTS_TO_EVAL:-0}" \
  PARALLEL_EVAL="${PARALLEL_EVAL:-1}" \
  EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}" \
  MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-8}" \
  bash scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh \
    >>"${LOG_FILE}" 2>&1
  log "A/B top-5 NAVTEST eval sweep finished"
}

log "watching A=${A_RUN_ROOT}"
log "watching B=${B_RUN_ROOT}"
log "B eval uses online VLM=${B_ONLINE_VLM_PATH} with LoRA adapter=${B_ONLINE_VLM_LORA_ADAPTER_DIR}"

while true; do
  if training_running; then
    log "training still running; waiting"
    sleep "${POLL_SECONDS}"
    continue
  fi
  if ready_for_eval; then
    run_eval
    if [[ "${RUN_EVAL_ONCE}" == "1" || "${RUN_EVAL_ONCE}" == "true" ]]; then
      exit 0
    fi
  else
    log "training not running, but latest/top-k checkpoints are not ready yet"
  fi
  sleep "${POLL_SECONDS}"
done
