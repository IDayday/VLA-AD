#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Expose more geometry-balanced modes than the conservative beta=0.25 profile,
# while bounding their detached epsilon-residual mass relative to the GT anchor.
# All settings remain overridable for matched ablations.
export DPSI_TARGET_DISTRIBUTION="${DPSI_TARGET_DISTRIBUTION:-mode_balanced}"
export DPSI_MODE_DENSITY_BANDWIDTH="${DPSI_MODE_DENSITY_BANDWIDTH:-0.40}"
export DPSI_USE_SOURCE_WEIGHT="${DPSI_USE_SOURCE_WEIGHT:-true}"
export DPSI_BETA_MAX="${DPSI_BETA_MAX:-0.35}"
export DPSI_PAIR_TARGET_RANDOMNESS="${DPSI_PAIR_TARGET_RANDOMNESS:-true}"
export DPSI_RESIDUAL_BUDGET_ENABLED="${DPSI_RESIDUAL_BUDGET_ENABLED:-true}"
export DPSI_NON_GT_RESIDUAL_MASS_CAP="${DPSI_NON_GT_RESIDUAL_MASS_CAP:-0.38}"
export DPSI_TARGET_SAMPLE_M="${DPSI_TARGET_SAMPLE_M:-4}"
export DPSI_TARGET_SAMPLE_M_AFTER_WARMUP="${DPSI_TARGET_SAMPLE_M_AFTER_WARMUP:-4}"

exec "${REPO_ROOT}/scripts/training/sg_fps/run_train_pta_fs_dit.sh"
