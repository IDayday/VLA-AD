#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH to the SG-FPS v3 support archive}"

export ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${SUPPORT_ARCHIVE_PATH}}"
export SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}"
export CACHE_MODE="${CACHE_MODE:-offline}"
export HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
export OFFLINE_RL_ENABLED="${OFFLINE_RL_ENABLED:-true}"
export GRPO_BUFFER_GUIDANCE_ENABLED="${GRPO_BUFFER_GUIDANCE_ENABLED:-true}"
export SG_FPS_USE_DPSI="${SG_FPS_USE_DPSI:-true}"
export GRPO_REWARD_MODE="${GRPO_REWARD_MODE:-feasible_pareto}"
export GRPO_USE_FEASIBLE_PARETO="${GRPO_USE_FEASIBLE_PARETO:-true}"
export GRPO_FP_USE_PDAS="${GRPO_FP_USE_PDAS:-true}"
export RUN_STAGE3="${RUN_STAGE3:-1}"
export DRY_RUN="${DRY_RUN:-0}"

exec "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_rl_2b_local.sh" "$@"
