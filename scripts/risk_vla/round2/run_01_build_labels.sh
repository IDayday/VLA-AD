#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"
SPLIT="${SPLIT:-train}"
PURPOSE="${PURPOSE:-training}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
DRY_RUN="${DRY_RUN:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${EXP_ROOT}/utility_labels" "${EXP_ROOT}/finegrained_labels"
if [[ -z "${A0_PDM_TABLE:-}" || -z "${CANDIDATE_PDM_TABLES:-}" ]]; then
  echo "Missing A0_PDM_TABLE or CANDIDATE_PDM_TABLES. Provide explicit paths; no private paths are guessed." | tee "${EXP_ROOT}/utility_labels/missing_inputs.txt"
  exit 0
fi
candidate_args=()
IFS=';' read -r -a candidate_specs <<< "${CANDIDATE_PDM_TABLES}"
for spec in "${candidate_specs[@]}"; do candidate_args+=(--candidate "${spec}"); done
args=(--baseline "name=A0,path=${A0_PDM_TABLE},strategy=base" "${candidate_args[@]}" --split "${SPLIT}" --purpose "${PURPOSE}" --output-dir "${EXP_ROOT}/utility_labels")
if [[ -n "${MAX_SAMPLES}" ]]; then args+=(--max-tokens "${MAX_SAMPLES}"); fi
if [[ "${DRY_RUN}" == "1" ]]; then args+=(--dry-run); fi
python scripts/risk_vla/build_strategy_utility_labels.py "${args[@]}"
if [[ -n "${FINEGRAINED_PDM_TABLE:-}" ]]; then
  fg_args=(--pdm-table "${FINEGRAINED_PDM_TABLE}" --split "${SPLIT}" --purpose "${PURPOSE}" --output-dir "${EXP_ROOT}/finegrained_labels")
  if [[ -n "${MAX_SAMPLES}" ]]; then fg_args+=(--max-rows "${MAX_SAMPLES}"); fi
  if [[ "${DRY_RUN}" == "1" ]]; then fg_args+=(--dry-run); fi
  python scripts/risk_vla/build_finegrained_risk_labels.py "${fg_args[@]}"
fi
