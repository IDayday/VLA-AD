#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PSI_STAGE2_INIT_MODE=random
export MAX_EPOCHS="${MAX_EPOCHS:-200}"
export LR="${LR:-1e-4}"
export MASTER_PORT="${MASTER_PORT:-63699}"
export CACHE_TRAIN_ALL_RECORDS="${CACHE_TRAIN_ALL_RECORDS:-true}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-psi_drive_stage2_apsd_random_init_clean_$(date -u +%Y%m%dT%H%M%SZ)}"

exec "${REPO_ROOT}/scripts/psi_drive/run_stage2_apsd_8gpu.sh" "$@"
