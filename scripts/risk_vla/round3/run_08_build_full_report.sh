#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
REPORT_DIR="${REPORT_DIR:-${BIT_EXP_ROOT}/reports}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${REPORT_DIR}"
dry_args=()
[[ "${DRY_RUN}" == "1" ]] && dry_args+=(--dry-run)
python scripts/risk_vla/aggregate_risk_vla_v3_full_matrix.py --exp-root "${BIT_EXP_ROOT}" --output-dir "${REPORT_DIR}" "${dry_args[@]}"
python scripts/risk_vla/build_risk_vla_v3_leaderboard.py --exp-root "${BIT_EXP_ROOT}" --output-dir "${REPORT_DIR}" "${dry_args[@]}"
python scripts/risk_vla/run_risk_vla_v3_full_matrix.py \
  --manifest-yaml "${BIT_EXP_ROOT}/full_experiment_manifest.yaml" \
  --output-dir "${REPORT_DIR}" \
  "${dry_args[@]}"
echo "Wrote ${REPORT_DIR}"
