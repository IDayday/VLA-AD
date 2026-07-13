#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH to a validated SG-FPS v6 archive}"
: "${FS_NORM_STATS_PATH:?set FS_NORM_STATS_PATH to statistics built from that v6 archive}"

# v6 is the clean Stage2 training distribution. A previous Stage2 checkpoint is
# neither loaded nor used by the archive admission/ranking contract.
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  echo "full_training_v6 must start Stage2 from random initialization; unset CHECKPOINT_PATH." >&2
  exit 2
fi
export ALLOW_RANDOM_INIT=true
export FS_NORM_REQUIRE_ARCHIVE_MATCH="${FS_NORM_REQUIRE_ARCHIVE_MATCH:-true}"

export DPSI_TARGET_DISTRIBUTION=full_training_v6
export DPSI_BETA_MAX="${DPSI_BETA_MAX:-0.50}"
export DPSI_BETA_WARMUP_EPOCHS="${DPSI_BETA_WARMUP_EPOCHS:-40}"
export DPSI_TARGET_SAMPLE_M="${DPSI_TARGET_SAMPLE_M:-2}"
export DPSI_TARGET_SAMPLE_M_AFTER_WARMUP="${DPSI_TARGET_SAMPLE_M_AFTER_WARMUP:-2}"
export DPSI_PAIR_TARGET_RANDOMNESS=true
export DPSI_RESIDUAL_BUDGET_ENABLED=true
export DPSI_NON_GT_RESIDUAL_MASS_CAP="${DPSI_NON_GT_RESIDUAL_MASS_CAP:-0.35}"
export DPSI_FRONTIER_CURRICULUM_ENABLED=true
export DPSI_FRONTIER_DIFFICULTY_START="${DPSI_FRONTIER_DIFFICULTY_START:-0.30}"
export DPSI_FRONTIER_DIFFICULTY_END="${DPSI_FRONTIER_DIFFICULTY_END:-1.0}"
export DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS="${DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS:-40}"
export DPSI_FRONTIER_GT_ONLY_SCENE_WEIGHT=1.0
export DPSI_FRONTIER_SCENE_SAMPLING_ENABLED=false
export DPSI_USE_REWARD_MARGIN_WEIGHT=false
export DPSI_USE_SOURCE_WEIGHT=false
export DPSI_SCENE_NORMALIZE_WEIGHTS=true

# Match the proven A5 from-random optimization horizon. The first five epochs
# are an in-run dynamic gate; a passing job continues without restarting.
export LR="${LR:-1e-4}"
export MAX_EPOCHS="${MAX_EPOCHS:-200}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-200}"
export SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-3}"

export TRAJECTORY_AUX_WEIGHT="${TRAJECTORY_AUX_WEIGHT:-0.05}"
export FEASIBILITY_AUX_WEIGHT="${FEASIBILITY_AUX_WEIGHT:-0.01}"

exec "${REPO_ROOT}/scripts/training/sg_fps/run_train_pta_fs_dit.sh"
