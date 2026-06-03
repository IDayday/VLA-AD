#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
DRY_RUN="${DRY_RUN:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project/VLA-AD/checkpoints}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
OUT_DIR="${OUT_DIR:-${ROUND_DIR}/stage6_input_closure}"
MANIFEST_YAML="${MANIFEST_YAML:-configs/risk_vla/round1_input_manifest.yaml}"

mkdir -p "${OUT_DIR}/dryrun_logs"

echo "[run_15] EXECUTE=${EXECUTE}; this wrapper never starts training/eval."
echo "[run_15] output dir: ${OUT_DIR}"

python scripts/risk_vla/find_pdm_csv_candidates.py \
  --search-root "${BIT_EXP_ROOT}" \
  --search-root "${BIT_WORK_ROOT}" \
  --search-root "${CHECKPOINT_ROOT}" \
  --output-csv "${OUT_DIR}/pdm_csv_candidates.csv" \
  --output-md "${OUT_DIR}/pdm_csv_candidates.md" \
  --max-depth "${MAX_DEPTH:-5}" \
  >"${OUT_DIR}/dryrun_logs/find_pdm_csv_candidates.log" 2>&1 || true

MANIFEST_ARGS=()
if [[ -f "${MANIFEST_YAML}" ]]; then
  echo "[run_15] validating manifest: ${MANIFEST_YAML}"
  set +e
  python scripts/risk_vla/validate_round1_input_manifest.py \
    --manifest-yaml "${MANIFEST_YAML}" \
    --output-json "${OUT_DIR}/input_manifest_resolved.json" \
    --output-md "${OUT_DIR}/input_manifest_resolved.md" \
    >"${OUT_DIR}/dryrun_logs/validate_manifest.log" 2>&1
  manifest_status=$?
  set -e
  echo "[run_15] manifest validation exit: ${manifest_status}"
  MANIFEST_ARGS=(--input-manifest-yaml "${MANIFEST_YAML}")
else
  echo "[run_15] no manifest found at ${MANIFEST_YAML}; using discovery report."
fi

python scripts/risk_vla/discover_round1_inputs.py \
  --output-json "${OUT_DIR}/input_discovery.json" \
  --output-md "${OUT_DIR}/input_discovery.md" \
  >"${OUT_DIR}/dryrun_logs/discover_inputs.log" 2>&1 || true

if [[ ${#MANIFEST_ARGS[@]} -gt 0 ]]; then
  PLAN_INPUT_ARGS=("${MANIFEST_ARGS[@]}")
else
  PLAN_INPUT_ARGS=(--input-report-json "${OUT_DIR}/input_discovery.json")
fi

python scripts/risk_vla/build_round1_execution_plan.py \
  "${PLAN_INPUT_ARGS[@]}" \
  --output-dir "${OUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --limit-train-batches "${LIMIT_TRAIN_BATCHES}" \
  --limit-val-batches "${LIMIT_VAL_BATCHES}" \
  --write \
  >"${OUT_DIR}/dryrun_logs/build_execution_plan.log" 2>&1 || true

STEPS=(
  run_00_collect_matched_pdm.sh
  run_01_build_risk_labels.sh
  run_02_merge_labels_to_chunk_cache.sh
  run_03_train_risk_head_diagnostic.sh
  run_04_export_risk_predictions.sh
  run_05_aggregate_risk_diagnostics.sh
  run_06_oracle_router_pilot_eval.sh
  run_07_predicted_router_pilot_eval.sh
  run_08_build_round1_report.sh
)

{
  echo "# RISK-VLA Stage 6 Input Closure Summary"
  echo
  echo "This is a dry-run/input-closure report for low-score risk scenarios / critical-risk subsets."
  echo
  echo "- output dir: \`${OUT_DIR}\`"
  echo "- execute: \`${EXECUTE}\`"
  echo "- dry_run: \`${DRY_RUN}\`"
  echo
  echo "## Candidate Search"
  echo
  if [[ -f "${OUT_DIR}/pdm_csv_candidates.md" ]]; then
    cat "${OUT_DIR}/pdm_csv_candidates.md"
  else
    echo "_Candidate search report missing._"
  fi
  echo
  echo "## Manifest Status"
  echo
  if [[ -f "${OUT_DIR}/input_manifest_resolved.md" ]]; then
    cat "${OUT_DIR}/input_manifest_resolved.md"
  else
    echo "_No explicit manifest was validated._"
  fi
  echo
  echo "## Input Discovery"
  echo
  if [[ -f "${OUT_DIR}/input_discovery.md" ]]; then
    cat "${OUT_DIR}/input_discovery.md"
  else
    echo "_Input discovery report missing._"
  fi
  echo
  echo "## Dry-Run Steps"
} >"${OUT_DIR}/stage6_input_closure_summary.md"

for step in "${STEPS[@]}"; do
  log="${OUT_DIR}/dryrun_logs/${step%.sh}.log"
  set +e
  DRY_RUN=1 EXECUTE=0 ROUND_DIR="${OUT_DIR}/round1_outputs" bash "scripts/risk_vla/round1/${step}" >"${log}" 2>&1
  status=$?
  set -e
  echo "- \`${step}\`: exit ${status}, log \`${log}\`" >>"${OUT_DIR}/stage6_input_closure_summary.md"
done

{
  echo
  echo "## Required Next Inputs"
  echo
  echo "- If A0/B3 CSVs are still missing, provide them in \`configs/risk_vla/round1_input_manifest.yaml\` under \`analysis_pdm\`."
  echo "- If train/val PDM labels are missing, provide navtrain/navval PDM CSVs under \`trainval_pdm\`, or use an existing labeled overlay only if it has already passed leakage checks."
  echo "- Do not use navtest/test labels for R0 training."
} >>"${OUT_DIR}/stage6_input_closure_summary.md"

echo "[run_15] summary: ${OUT_DIR}/stage6_input_closure_summary.md"
