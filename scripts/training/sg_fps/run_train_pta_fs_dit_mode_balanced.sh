#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# This profile changes only the within-scene Stage2 target distribution. The
# support archive, model architecture, and diffusion objective remain fixed.
# The 0.25 cap is the largest tested dose with a full-navtest PDMS delta whose
# paired confidence interval still overlaps the legacy control.
export DPSI_TARGET_DISTRIBUTION="${DPSI_TARGET_DISTRIBUTION:-mode_balanced}"
export DPSI_MODE_DENSITY_BANDWIDTH="${DPSI_MODE_DENSITY_BANDWIDTH:-0.40}"
export DPSI_USE_SOURCE_WEIGHT="${DPSI_USE_SOURCE_WEIGHT:-true}"
export DPSI_BETA_MAX="${DPSI_BETA_MAX:-0.25}"

exec "${REPO_ROOT}/scripts/training/sg_fps/run_train_pta_fs_dit.sh"
