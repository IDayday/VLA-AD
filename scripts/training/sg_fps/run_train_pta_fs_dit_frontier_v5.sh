#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH to a validated SG-FPS v5 archive}"
: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH to the frozen A5 Stage2 initialization}"

export FS_NORM_REQUIRE_ARCHIVE_MATCH="${FS_NORM_REQUIRE_ARCHIVE_MATCH:-false}"
export ALLOW_RANDOM_INIT=false

# One GT anchor plus at most one independently evidenced mode. Scenes without
# evidence remain GT-only; beta is conditional mode mass, not a scene quota.
export DPSI_TARGET_DISTRIBUTION=learning_frontier_v5
export DPSI_BETA_MAX="${DPSI_BETA_MAX:-0.45}"
export DPSI_BETA_WARMUP_EPOCHS="${DPSI_BETA_WARMUP_EPOCHS:-0}"
export DPSI_TARGET_SAMPLE_M=2
export DPSI_TARGET_SAMPLE_M_AFTER_WARMUP=2
export DPSI_PAIR_TARGET_RANDOMNESS=true
export DPSI_RESIDUAL_BUDGET_ENABLED=true
export DPSI_NON_GT_RESIDUAL_MASS_CAP="${DPSI_NON_GT_RESIDUAL_MASS_CAP:-0.45}"
export DPSI_FRONTIER_GT_ONLY_SCENE_WEIGHT="${DPSI_FRONTIER_GT_ONLY_SCENE_WEIGHT:-0.25}"
export DPSI_FRONTIER_CURRICULUM_ENABLED="${DPSI_FRONTIER_CURRICULUM_ENABLED:-false}"
export DPSI_FRONTIER_DIFFICULTY_START="${DPSI_FRONTIER_DIFFICULTY_START:-0.45}"
export DPSI_FRONTIER_DIFFICULTY_END="${DPSI_FRONTIER_DIFFICULTY_END:-1.0}"
export DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS="${DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS:-25}"
export DPSI_FRONTIER_SCENE_SAMPLING_ENABLED="${DPSI_FRONTIER_SCENE_SAMPLING_ENABLED:-true}"
export DPSI_FRONTIER_SCENE_INDEX_PATH="${DPSI_FRONTIER_SCENE_INDEX_PATH:-}"
export DPSI_FRONTIER_SCENE_UNIFORM_RATIO="${DPSI_FRONTIER_SCENE_UNIFORM_RATIO:-0.50}"
export DPSI_FRONTIER_SCENE_PRIORITY_EXPONENT="${DPSI_FRONTIER_SCENE_PRIORITY_EXPONENT:-0.50}"
export DPSI_FRONTIER_SCENE_WARMUP_EPOCHS="${DPSI_FRONTIER_SCENE_WARMUP_EPOCHS:-0}"
if [[ "${DPSI_FRONTIER_SCENE_SAMPLING_ENABLED,,}" == "true" && -z "${DPSI_FRONTIER_SCENE_INDEX_PATH}" ]]; then
  echo "DPSI_FRONTIER_SCENE_INDEX_PATH is required when v5 frontier scene sampling is enabled." >&2
  exit 2
fi
export DPSI_USE_REWARD_MARGIN_WEIGHT=false
export DPSI_USE_SOURCE_WEIGHT=false
export DPSI_SCENE_NORMALIZE_WEIGHTS=true

# This is a calibrated continuation from A5, not another 200-epoch fit. The
# 128-scene causal probe contracted after the learning rate stayed high beyond
# 25 data epochs, while the 25-epoch cosine point preserved the mode stratum.
export LR="${LR:-1e-5}"
export MAX_EPOCHS="${MAX_EPOCHS:-25}"
export SCHEDULER_EPOCHS="${SCHEDULER_EPOCHS:-25}"
export SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-0}"

export TRAJECTORY_AUX_WEIGHT="${TRAJECTORY_AUX_WEIGHT:-0.0}"
export FEASIBILITY_AUX_WEIGHT="${FEASIBILITY_AUX_WEIGHT:-0.0}"

exec "${REPO_ROOT}/scripts/training/sg_fps/run_train_pta_fs_dit.sh"
