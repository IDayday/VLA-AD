#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"

cd "${PROJECT_ROOT}"
exec scripts/risk_vla/round2/run_02_build_candidate_bank.sh "$@"
