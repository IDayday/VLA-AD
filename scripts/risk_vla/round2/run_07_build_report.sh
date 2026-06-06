#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${PROJECT_ROOT}/experiments/risk_vla}"
EXP_ROOT="${EXP_ROOT:-${BIT_EXP_ROOT}/v2}"

cd "${PROJECT_ROOT}"
python scripts/risk_vla/aggregate_risk_vla_v2_results.py --exp-root "${EXP_ROOT}" --output-dir "${EXP_ROOT}"
