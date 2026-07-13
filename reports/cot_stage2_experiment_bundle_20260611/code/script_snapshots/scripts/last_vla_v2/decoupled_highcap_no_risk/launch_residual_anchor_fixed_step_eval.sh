#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/residual_anchor_fixed_step_eval_${RUN_ID}}"

PURE_RUN_ROOT="${PURE_RUN_ROOT:-/mnt/project/VLA-AD/outputs/recogdrive_stage2_residual_anchor_base2b_20260610T034055Z_setsid_freezecot}"
LASTVLA_RUN_ROOT="${LASTVLA_RUN_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_A_remote_20260610T030302Z/A_frozen_vlm_stage2_progressive}"
PURE_CONFIG="${PURE_CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/original_recogdrive_residual_anchor_eval_flat.yaml}"
LASTVLA_CONFIG="${LASTVLA_CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml}"

NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/vlm_text_anchor_2bbase_navtest_20260610T051017Z}"
STEPS_CSV="${STEPS_CSV:-00050000,00060000,00080000,00100000,00120000}"
EXPERIMENTS_CSV="${EXPERIMENTS_CSV:-pure_recogdrive,lastvla_A}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7}"
PRECISION="${PRECISION:-fp32}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
STAGGER_SECONDS="${STAGGER_SECONDS:-8}"
POLL_SECONDS="${POLL_SECONDS:-20}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/summary"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/driver.log"
}

run_one() {
  local experiment="$1"
  local run_root="$2"
  local config="$3"
  local step="$4"
  local ckpt="${run_root}/step_${step}.ckpt"
  local eval_root="${OUT_ROOT}/eval/${experiment}/step_${step}"

  if [[ ! -f "${ckpt}" ]]; then
    log "skip missing ${experiment} step_${step}: ${ckpt}"
    return 0
  fi

  log "start ${experiment} step_${step}: ${ckpt}"
  OUT_ROOT="${eval_root}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SOURCE_OUT_ROOT="${OUT_ROOT}" \
  LINE=A \
  CONFIG="${config}" \
  A_RUN_ROOT="${run_root}" \
  NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
  METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
  VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT}" \
  TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" \
  PRECISION="${PRECISION}" \
  A_NUM_SHARDS="${NUM_SHARDS}" \
  A_GPUS_CSV="${GPUS_CSV}" \
  STAGGER_SECONDS="${STAGGER_SECONDS}" \
  POLL_SECONDS="${POLL_SECONDS}" \
  INCLUDE_TOPK_CKPTS=0 \
  INCLUDE_FIXED_STEP_CKPTS=1 \
  FIXED_STEP_CKPTS_CSV="${step}" \
  scripts/last_vla_v2/decoupled_highcap_no_risk/run_current_ab_ckpt_navtest_parallel_eval.sh \
    >"${OUT_ROOT}/logs/${experiment}_step_${step}.outer.log" 2>&1
  log "done ${experiment} step_${step}"
}

IFS=',' read -r -a steps <<<"${STEPS_CSV}"
IFS=',' read -r -a experiments <<<"${EXPERIMENTS_CSV}"
for step in "${steps[@]}"; do
  [[ -n "${step}" ]] || continue
  for experiment in "${experiments[@]}"; do
    case "${experiment}" in
      pure_recogdrive)
        run_one "pure_recogdrive" "${PURE_RUN_ROOT}" "${PURE_CONFIG}" "${step}"
        ;;
      lastvla_A)
        run_one "lastvla_A" "${LASTVLA_RUN_ROOT}" "${LASTVLA_CONFIG}" "${step}"
        ;;
      "")
        ;;
      *)
        echo "unknown experiment in EXPERIMENTS_CSV: ${experiment}" >&2
        exit 2
        ;;
    esac
  done
done

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
rows = []
for summary in sorted((out_root / "eval").glob("*/*/summary/A_summary.csv")):
    experiment = summary.parts[-4]
    step = summary.parts[-3].removeprefix("step_")
    with summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = {"experiment": experiment, "step": step, **row}
            rows.append(row)

summary_dir = out_root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
fields = [
    "experiment", "step", "line", "run", "state", "checkpoint",
    "trajectory_output_key", "num_shards", "num_samples", "num_pdm_valid",
    "num_pdm_missing_metric_cache", "num_pdm_failed", "PDMS", "NC", "DAC",
    "TTC", "comfort", "EP", "DDC", "trajectory_l1", "metrics_path",
]
with (summary_dir / "combined_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {len(rows)} rows to {summary_dir / 'combined_summary.csv'}")
PY

log "finished all fixed-step evals"
