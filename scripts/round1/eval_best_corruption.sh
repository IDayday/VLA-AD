#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:?Set NAVTEST_CHUNK_CACHE_ROOT.}"
NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:?Set NAVTEST_CHUNK_NAME_PATTERN.}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:?Set METRIC_CACHE_DIR.}"
BEST_HYBRID_CKPT="${BEST_HYBRID_CKPT:-}"
BEST_LASTRD_ONLY_CKPT="${BEST_LASTRD_ONLY_CKPT:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
FULL_CORRUPTION="${FULL_CORRUPTION:-0}"
GPU_ID="${GPU_ID:-0}"

SUMMARY_CSV="${OUT_ROOT}/round1_summary.csv"
OUT_DIR="${OUT_ROOT}/wave4_corruption"
mkdir -p "${OUT_DIR}"
COMMANDS_LOG="${OUT_DIR}/commands.log"
: > "${COMMANDS_LOG}"

lookup_best() {
  local run="$1"
  python - "${SUMMARY_CSV}" "${run}" <<'PY'
import csv
import sys
path, run = sys.argv[1:3]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for row in rows:
    if row.get("run_name") == run and str(row.get("is_best_for_run")).lower() in {"1", "true", "yes"}:
        print(row.get("checkpoint_path", ""))
        raise SystemExit(0)
raise SystemExit(f"No best checkpoint found for {run} in {path}")
PY
}

if [[ -z "${BEST_HYBRID_CKPT}" && -f "${SUMMARY_CSV}" ]]; then
  BEST_HYBRID_CKPT="$(lookup_best hybrid)"
fi
if [[ -z "${BEST_LASTRD_ONLY_CKPT}" && -f "${SUMMARY_CSV}" ]]; then
  BEST_LASTRD_ONLY_CKPT="$(lookup_best lastrd_only)"
fi

run_corruption() {
  local run="$1"
  local ckpt="$2"
  local config="$3"
  [[ -n "${ckpt}" ]] || { echo "Skipping ${run}: no best checkpoint provided or found." | tee -a "${COMMANDS_LOG}"; return 0; }
  [[ -f "${ckpt}" ]] || { echo "Missing checkpoint for ${run}: ${ckpt}" >&2; exit 1; }

  local modes=(none zero-all-last-rd zero-jepa-dynamic shuffle-jepa-dynamic zero-vggt-geometry shuffle-vggt-geometry zero-ego-tokens)
  local mode flag out mode_dir
  for mode in "${modes[@]}"; do
    mode_dir="${OUT_DIR}/${run}/${mode}"
    cmd=(
      python "${REPO_ROOT}/scripts/eval_recogdrive_last_rd_corruption_pdm.py"
      --config "${config}"
      --checkpoint "${ckpt}"
      --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
      --metric-cache-dir "${METRIC_CACHE_DIR}"
      --precision fp32
      --output-dir "${mode_dir}"
    )
    if [[ "${FULL_CORRUPTION}" != "1" ]]; then
      cmd+=(--max-samples "${EVAL_MAX_SAMPLES}")
    fi
    case "${mode}" in
      none) ;;
      zero-all-last-rd) cmd+=(--zero-all-last-rd) ;;
      zero-jepa-dynamic) cmd+=(--zero-jepa-dynamic) ;;
      shuffle-jepa-dynamic) cmd+=(--shuffle-jepa-dynamic) ;;
      zero-vggt-geometry) cmd+=(--zero-vggt-geometry) ;;
      shuffle-vggt-geometry) cmd+=(--shuffle-vggt-geometry) ;;
      zero-ego-tokens) cmd+=(--zero-ego-tokens) ;;
      *) echo "Unknown corruption mode ${mode}" >&2; exit 1 ;;
    esac
    printf 'CUDA_VISIBLE_DEVICES=%s ' "${GPU_ID}" | tee -a "${COMMANDS_LOG}"
    printf '%q ' "${cmd[@]}" | tee -a "${COMMANDS_LOG}"
    printf '\n' | tee -a "${COMMANDS_LOG}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
  done
}

run_corruption hybrid "${BEST_HYBRID_CKPT}" "${REPO_ROOT}/configs/last_rd/last_rd_progressive_sft_hybrid_eval.yaml"
run_corruption lastrd_only "${BEST_LASTRD_ONLY_CKPT}" "${REPO_ROOT}/configs/last_rd/last_rd_progressive_sft_lastrd_only_eval.yaml"
