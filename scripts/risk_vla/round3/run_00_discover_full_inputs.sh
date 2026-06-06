#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-${BIT_WORK_ROOT}/cache/risk_vla_v3}"
MIN_REAL_SAMPLES="${MIN_REAL_SAMPLES:-10000}"
MAX_DEPTH="${MAX_DEPTH:-4}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${BIT_EXP_ROOT}" "${BIT_CACHE_ROOT}"

DISCOVERY_JSON="${BIT_EXP_ROOT}/full_input_discovery.json"
DISCOVERY_MD="${BIT_EXP_ROOT}/FULL_INPUT_DISCOVERY_REPORT.md"
MANIFEST_YAML="${BIT_EXP_ROOT}/full_experiment_manifest.yaml"
MANIFEST_MD="${BIT_EXP_ROOT}/FULL_EXPERIMENT_MANIFEST_REPORT.md"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat <<EOF
{"project_root":"${PROJECT_ROOT}","work_root":"${BIT_WORK_ROOT}","exp_root":"${BIT_EXP_ROOT}","cache_root":"${BIT_CACHE_ROOT}","min_real_samples":${MIN_REAL_SAMPLES},"max_depth":${MAX_DEPTH},"would_write_discovery":"${DISCOVERY_JSON}","would_write_manifest":"${MANIFEST_YAML}"}
EOF
  exit 0
fi

python scripts/risk_vla/discover_full_training_inputs.py \
  --project-root "${PROJECT_ROOT}" \
  --work-root "${BIT_WORK_ROOT}" \
  --exp-root "${BIT_EXP_ROOT}" \
  --cache-root "${BIT_CACHE_ROOT}" \
  --output-json "${DISCOVERY_JSON}" \
  --output-md "${DISCOVERY_MD}" \
  --min-real-samples "${MIN_REAL_SAMPLES}" \
  --max-depth "${MAX_DEPTH}"

python scripts/risk_vla/build_full_experiment_manifest.py \
  --discovery-json "${DISCOVERY_JSON}" \
  --output-yaml "${MANIFEST_YAML}" \
  --output-md "${MANIFEST_MD}" \
  --min-real-samples "${MIN_REAL_SAMPLES}"

echo "Wrote ${DISCOVERY_MD}"
echo "Wrote ${MANIFEST_YAML}"
