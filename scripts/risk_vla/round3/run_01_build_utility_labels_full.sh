#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
SPLIT="${SPLIT:-train}"
PURPOSE="${PURPOSE:-training}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_EXP_ROOT}/utility_labels/${SPLIT}_full}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

write_blocker() {
  cat > "${OUTPUT_DIR}/UTILITY_LABEL_BLOCKER.md" <<EOF
# Utility Label Blocker

Missing explicit PDM inputs. Set:

- BASELINE_PDM_SPEC='name=A0,path=/abs/base.csv,strategy=base,source_method=A0'
- CANDIDATE_PDM_SPECS='name=BiT,path=/abs/bit.csv,strategy=path_intent;name=D5,path=/abs/d5.csv,strategy=d5_conservative'

No guessed paths are used. Minimum formal scale: ${MIN_REAL_SAMPLES} matched tokens.
EOF
}

if [[ -z "${BASELINE_PDM_SPEC:-}" || -z "${CANDIDATE_PDM_SPECS:-}" ]]; then
  write_blocker
  echo "Blocked: missing BASELINE_PDM_SPEC or CANDIDATE_PDM_SPECS"
  [[ "${DRY_RUN}" == "1" ]] && exit 0 || exit 1
fi

args=(
  --baseline "${BASELINE_PDM_SPEC}"
  --split "${SPLIT}"
  --purpose "${PURPOSE}"
  --output-dir "${OUTPUT_DIR}"
  --min-real-samples "${MIN_REAL_SAMPLES}"
)
IFS=';' read -r -a candidate_specs <<< "${CANDIDATE_PDM_SPECS}"
for spec in "${candidate_specs[@]}"; do
  [[ -n "${spec}" ]] && args+=(--candidate "${spec}")
done
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-tokens "${MAX_TOKENS}")
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)

python scripts/risk_vla/build_strategy_utility_labels.py "${args[@]}"
if [[ "${DRY_RUN}" != "1" ]]; then
  python scripts/risk_vla/check_strategy_utility_labels.py \
    --labels-jsonl "${OUTPUT_DIR}/strategy_utility_labels.jsonl" \
    --purpose "${PURPOSE}" \
    --output-json "${OUTPUT_DIR}/label_check.json"
  python scripts/risk_vla/build_safe_alignment_pairs.py \
    --labels-jsonl "${OUTPUT_DIR}/strategy_utility_labels.jsonl" \
    --output-dir "${OUTPUT_DIR}/safe_alignment_pairs"
  python scripts/risk_vla/aggregate_strategy_utility_report.py \
    --labels-jsonl "${OUTPUT_DIR}/strategy_utility_labels.jsonl" \
    --output-dir "${OUTPUT_DIR}/aggregate"
fi
