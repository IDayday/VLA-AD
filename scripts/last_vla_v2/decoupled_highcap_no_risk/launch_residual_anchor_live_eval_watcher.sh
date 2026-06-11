#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/residual_anchor_live_ckpt_eval_${RUN_ID}}"

PURE_RUN_ROOT="${PURE_RUN_ROOT:-/mnt/project/VLA-AD/outputs/recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot}"
LASTVLA_RUN_ROOT="${LASTVLA_RUN_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_A_remote_20260610T030302Z/A_frozen_vlm_stage2_progressive}"
VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_navtest_20260610T051017Z}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
PREVIOUS_EVAL_ROOT="${PREVIOUS_EVAL_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_ckpt_eval_20260610T051017Z}"
EXTRA_SEED_EVAL_ROOT="${EXTRA_SEED_EVAL_ROOT:-}"

POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
MAX_NEW_PER_CYCLE="${MAX_NEW_PER_CYCLE:-0}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"

mkdir -p "${OUT_ROOT}/logs"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${COMMANDS_LOG}"
}

if [[ -n "${WAIT_FOR_PID}" ]]; then
  log "waiting for existing first-pass evaluator pid=${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" >/dev/null 2>&1; do
    sleep "${WAIT_POLL_SECONDS}"
  done
  log "first-pass evaluator pid=${WAIT_FOR_PID} has exited; starting live watcher"
fi

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/decoupled_highcap_no_risk/watch_residual_anchor_live_eval.py
  --out-root "${OUT_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --pure-run-root "${PURE_RUN_ROOT}"
  --lastvla-run-root "${LASTVLA_RUN_ROOT}"
  --vlm-text-anchor-cache-root "${VLM_TEXT_ANCHOR_CACHE_ROOT}"
  --navtest-chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
  --metric-cache-dir "${METRIC_CACHE_DIR}"
  --seed-eval-root "${PREVIOUS_EVAL_ROOT}"
  --poll-seconds "${POLL_SECONDS}"
  --stable-seconds "${STABLE_SECONDS}"
  --num-shards "${NUM_SHARDS}"
  --gpus-csv "${GPUS_CSV}"
  --max-new-per-cycle "${MAX_NEW_PER_CYCLE}"
)

if [[ -n "${EXTRA_SEED_EVAL_ROOT}" ]]; then
  cmd+=(--seed-eval-root "${EXTRA_SEED_EVAL_ROOT}")
fi

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

cd "${PROJECT_ROOT}"
exec "${cmd[@]}"
