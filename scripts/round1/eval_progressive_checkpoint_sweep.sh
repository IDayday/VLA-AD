#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:?Set NAVTEST_CHUNK_CACHE_ROOT.}"
NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:?Set NAVTEST_CHUNK_NAME_PATTERN.}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:?Set METRIC_CACHE_DIR.}"
RUN_NAME="${RUN_NAME:-all}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
GPU_ID="${GPU_ID:-0}"

EVAL_ROOT="${OUT_ROOT}/wave3_eval"
mkdir -p "${EVAL_ROOT}"
COMMANDS_LOG="${EVAL_ROOT}/commands.log"
: > "${COMMANDS_LOG}"

if [[ "${RUN_NAME}" == "all" ]]; then
  RUNS=(hybrid lastrd_only)
else
  RUNS=("${RUN_NAME}")
fi

config_for_run() {
  case "$1" in
    hybrid) echo "${REPO_ROOT}/configs/last_rd/last_rd_progressive_sft_hybrid_eval.yaml" ;;
    lastrd_only) echo "${REPO_ROOT}/configs/last_rd/last_rd_progressive_sft_lastrd_only_eval.yaml" ;;
    *) echo "Unknown RUN_NAME=$1" >&2; return 1 ;;
  esac
}

dir_for_run() {
  case "$1" in
    hybrid) echo "${OUT_ROOT}/wave2_progressive/hybrid" ;;
    lastrd_only) echo "${OUT_ROOT}/wave2_progressive/lastrd_only" ;;
    *) echo "Unknown RUN_NAME=$1" >&2; return 1 ;;
  esac
}

collect_checkpoints() {
  local run_dir="$1"
  local step
  for step in 00050000 00060000 00080000 00100000 00120000; do
    find "${run_dir}" -type f \( -name "*step_${step}.ckpt" -o -name "*step=${step}.ckpt" -o -name "*${step}*.ckpt" \) -print
  done
  find "${run_dir}" -type f \( -name "latest.ckpt" -o -name "last.ckpt" -o -name "*epoch=*.ckpt" -o -name "*.ckpt" \) -print
}

for run in "${RUNS[@]}"; do
  run_dir="$(dir_for_run "${run}")"
  config="$(config_for_run "${run}")"
  if [[ ! -d "${run_dir}" ]]; then
    echo "Skipping ${run}: run directory not found: ${run_dir}" | tee -a "${COMMANDS_LOG}"
    continue
  fi
  declare -A seen=()
  while IFS= read -r ckpt; do
    [[ -f "${ckpt}" ]] || continue
    if [[ -n "${seen[${ckpt}]:-}" ]]; then
      continue
    fi
    seen["${ckpt}"]=1
    stem="$(basename "${ckpt}")"
    stem="${stem%.ckpt}"
    stem="${stem//[^A-Za-z0-9_.=-]/_}"
    out_dir="${EVAL_ROOT}/${run}/${stem}"
    cmd=(
      python "${REPO_ROOT}/scripts/eval_recogdrive_expert_pdm.py"
      --config "${config}"
      --checkpoint "${ckpt}"
      --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
      --metric-cache-dir "${METRIC_CACHE_DIR}"
      --precision fp32
      --output-dir "${out_dir}"
    )
    if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
      cmd+=(--max-samples "${EVAL_MAX_SAMPLES}")
    fi
    printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPU_ID}" | tee -a "${COMMANDS_LOG}"
    printf '%q ' "${cmd[@]}" | tee -a "${COMMANDS_LOG}"
    printf '\n' | tee -a "${COMMANDS_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
  done < <(collect_checkpoints "${run_dir}" | sort)
done
