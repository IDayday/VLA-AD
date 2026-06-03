#!/usr/bin/env bash
set -euo pipefail

BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
LOG_DIR="${ROUND_DIR}/logs/dryrun"
mkdir -p "${LOG_DIR}"

DISCOVERY_JSON="${LOG_DIR}/input_discovery.json"
DISCOVERY_MD="${LOG_DIR}/input_discovery.md"
SUMMARY_MD="${LOG_DIR}/round1_dryrun_summary.md"

echo "[dryrun] writing logs to ${LOG_DIR}"

DISCOVERY_CMD=(
  python scripts/risk_vla/discover_round1_inputs.py
  --output-json "${DISCOVERY_JSON}"
  --output-md "${DISCOVERY_MD}"
)
[[ -n "${A0_PDM_CSV:-}" ]] && DISCOVERY_CMD+=(--a0-pdm-csv "${A0_PDM_CSV}")
[[ -n "${BIT_PDM_CSV:-}" ]] && DISCOVERY_CMD+=(--bit-pdm-csv "${BIT_PDM_CSV}")
[[ -n "${CHUNK_CACHE_DIR:-}" ]] && DISCOVERY_CMD+=(--chunk-cache-dir "${CHUNK_CACHE_DIR}")
[[ -n "${CHECKPOINT_DIR:-}" ]] && DISCOVERY_CMD+=(--checkpoint-dir "${CHECKPOINT_DIR}")

"${DISCOVERY_CMD[@]}" >"${LOG_DIR}/input_discovery.log" 2>&1 || true

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
  echo "# RISK-VLA Round 1 Dry-Run Summary"
  echo
  echo "Logs: \`${LOG_DIR}\`"
  echo
  echo "## Commands Checked"
  echo
} >"${SUMMARY_MD}"

for step in "${STEPS[@]}"; do
  log="${LOG_DIR}/${step%.sh}.log"
  echo "[dryrun] ${step}"
  set +e
  DRY_RUN=1 EXECUTE=0 ROUND_DIR="${ROUND_DIR}" bash "scripts/risk_vla/round1/${step}" >"${log}" 2>&1
  status=$?
  set -e
  echo "- \`${step}\`: exit ${status}, log \`${log}\`" >>"${SUMMARY_MD}"
done

{
  echo
  echo "## Inputs Resolved / Missing"
  echo
  if [[ -f "${DISCOVERY_MD}" ]]; then
    cat "${DISCOVERY_MD}"
  else
    echo "_Input discovery report missing._"
  fi
  echo
  echo "## Intended Outputs"
  echo
  echo "- matched PDM table under \`${ROUND_DIR}/matched_pdm\`"
  echo "- risk labels under \`${ROUND_DIR}/risk_labels\`"
  echo "- labeled overlay under \`${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}/risk_vla_round1_labeled_overlay\`"
  echo "- R0 diagnostic artifacts under \`${ROUND_DIR}/R0_risk_head_diagnostic\`"
  echo "- risk diagnostics under \`${ROUND_DIR}/risk_diagnostics\`"
  echo
  echo "## Next Executable Commands"
  echo
  echo "- R0 only: \`EXECUTE=1 bash scripts/risk_vla/round1/run_10_execute_r0_diagnostic_small.sh\`"
  echo "- Prediction export: \`EXECUTE=1 bash scripts/risk_vla/round1/run_11_export_r0_predictions_small.sh\`"
  echo "- Aggregation: \`EXECUTE=1 bash scripts/risk_vla/round1/run_12_aggregate_r0_diagnostics.sh\`"
} >>"${SUMMARY_MD}"

echo "[dryrun] summary: ${SUMMARY_MD}"
