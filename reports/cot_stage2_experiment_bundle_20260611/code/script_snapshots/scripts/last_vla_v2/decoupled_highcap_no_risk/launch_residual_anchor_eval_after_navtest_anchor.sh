#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_BASE_ROOT="${EVAL_BASE_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_ckpt_eval_${RUN_ID}}"

NAVTEST_ANCHOR_CACHE_ROOT="${NAVTEST_ANCHOR_CACHE_ROOT:?set NAVTEST_ANCHOR_CACHE_ROOT}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
EXPECTED_NAVTEST_ANCHORS="${EXPECTED_NAVTEST_ANCHORS:-12146}"

PURE_RUN_ROOT="${PURE_RUN_ROOT:-/mnt/project/VLA-AD/outputs/recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot}"
LASTVLA_A_RUN_ROOT="${LASTVLA_A_RUN_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_A_remote_20260610T030302Z/A_frozen_vlm_stage2_progressive}"
PURE_CONFIG="${PURE_CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/original_recogdrive_residual_anchor_eval_flat.yaml}"
LASTVLA_CONFIG="${LASTVLA_CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml}"

POLL_SECONDS="${POLL_SECONDS:-120}"
ANCHOR_WAIT_TIMEOUT_SECONDS="${ANCHOR_WAIT_TIMEOUT_SECONDS:-0}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
A_NUM_SHARDS="${A_NUM_SHARDS:-8}"
STAGGER_SECONDS="${STAGGER_SECONDS:-8}"
LAUNCHER="${LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"

mkdir -p "${EVAL_BASE_ROOT}/logs"
COMMANDS_LOG="${EVAL_BASE_ROOT}/commands.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${COMMANDS_LOG}"
}

validate_anchor_cache() {
  "${PYTHON_BIN}" - "${NAVTEST_ANCHOR_CACHE_ROOT}" "${EXPECTED_NAVTEST_ANCHORS}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
index_path = root / "index.jsonl"
if not index_path.is_file():
    raise SystemExit(f"missing anchor index: {index_path}")

total = 0
parse_fail = 0
missing_file = 0
with index_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        if rec.get("parse_ok") is False:
            parse_fail += 1
        path_value = rec.get("path") or ""
        path = Path(path_value)
        if not path_value or not (path if path.is_absolute() else root / path).is_file():
            missing_file += 1

payload = {
    "anchor_cache_root": str(root),
    "expected": expected,
    "total": total,
    "parse_fail": parse_fail,
    "missing_file": missing_file,
}
print(json.dumps(payload, sort_keys=True))
if total != expected or parse_fail or missing_file:
    raise SystemExit(3)
PY
}

wait_for_anchor_cache() {
  local start
  start="$(date +%s)"
  while true; do
    if validate_anchor_cache >>"${EVAL_BASE_ROOT}/logs/anchor_validation.log" 2>&1; then
      log "navtest anchor cache validated: ${NAVTEST_ANCHOR_CACHE_ROOT}"
      return 0
    fi
    if [[ "${ANCHOR_WAIT_TIMEOUT_SECONDS}" -gt 0 ]]; then
      local now
      now="$(date +%s)"
      if (( now - start >= ANCHOR_WAIT_TIMEOUT_SECONDS )); then
        log "timeout waiting for navtest anchor cache: ${NAVTEST_ANCHOR_CACHE_ROOT}"
        return 4
      fi
    fi
    log "waiting for navtest anchor cache: ${NAVTEST_ANCHOR_CACHE_ROOT}"
    sleep "${POLL_SECONDS}"
  done
}

run_eval_for_line() {
  local name="$1"
  local run_root="$2"
  local config="$3"
  local out_root="${EVAL_BASE_ROOT}/${name}"

  [[ -d "${run_root}" ]] || { log "missing run root for ${name}: ${run_root}"; return 2; }
  [[ -f "${config}" ]] || { log "missing eval config for ${name}: ${config}"; return 2; }

  log "starting ${name} residual-anchor ckpt eval"
  (
    cd "${PROJECT_ROOT}"
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      SOURCE_OUT_ROOT="${run_root}" \
      A_RUN_ROOT="${run_root}" \
      OUT_ROOT="${out_root}" \
      LINE=A \
      CONFIG="${config}" \
      NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
      METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      VLM_TEXT_ANCHOR_CACHE_ROOT="${NAVTEST_ANCHOR_CACHE_ROOT}" \
      TRAJECTORY_OUTPUT_KEY=pred_traj \
      PRECISION=fp32 \
      A_NUM_SHARDS="${A_NUM_SHARDS}" \
      A_GPUS_CSV="${EVAL_GPUS}" \
      STAGGER_SECONDS="${STAGGER_SECONDS}" \
      LAUNCHER="${LAUNCHER}" \
      INCLUDE_TOPK_CKPTS=1 \
      INCLUDE_FIXED_STEP_CKPTS=0 \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/run_current_ab_ckpt_navtest_parallel_eval.sh
  ) >"${EVAL_BASE_ROOT}/logs/${name}.outer.log" 2>&1
  log "finished ${name} residual-anchor ckpt eval"
}

{
  echo "run_id=${RUN_ID}"
  echo "eval_base_root=${EVAL_BASE_ROOT}"
  echo "navtest_anchor_cache_root=${NAVTEST_ANCHOR_CACHE_ROOT}"
  echo "navtest_chunk_cache_root=${NAVTEST_CHUNK_CACHE_ROOT}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "pure_run_root=${PURE_RUN_ROOT}"
  echo "lastvla_a_run_root=${LASTVLA_A_RUN_ROOT}"
  echo "pure_config=${PURE_CONFIG}"
  echo "lastvla_config=${LASTVLA_CONFIG}"
} >>"${COMMANDS_LOG}"

wait_for_anchor_cache
run_eval_for_line pure_recogdrive "${PURE_RUN_ROOT}" "${PURE_CONFIG}"
run_eval_for_line lastvla_A "${LASTVLA_A_RUN_ROOT}" "${LASTVLA_CONFIG}"
log "all residual-anchor eval launch jobs finished"
