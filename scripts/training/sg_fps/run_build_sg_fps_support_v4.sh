#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${POLICY_CHECKPOINT:?set POLICY_CHECKPOINT to the Stage2 initialization checkpoint}"
: "${FS_NORM_STATS_PATH:?set FS_NORM_STATS_PATH to the checkpoint FS-Norm statistics}"

# The production Stage2 domain contains both NAVTRAIN and NAVVAL logs. Keeping
# this explicit prevents a valid-looking train-only archive from being promoted.
export BUILD_LOG_SPLIT="${BUILD_LOG_SPLIT:-train_val}"

# v4 removes the hidden AWAC pre-selector and external mechanical expansion.
# All raw internal and external proposals enter one evaluator/quality/mode gate.
export SG_FPS_ARCHIVE_VERSION=4
export SG_FPS_SELECTION_STRATEGY=mode_pareto_v4
export SG_FPS_USE_RAW_INTERNAL_CANDIDATES=true
export SG_FPS_EXPAND_EXTERNAL_CANDIDATES=false
export SG_FPS_SUPPORT_TOP_M="${SG_FPS_SUPPORT_TOP_M:-4}"
export SG_FPS_V4_MAX_GT_ADE_M="${SG_FPS_V4_MAX_GT_ADE_M:-1.5}"
export SG_FPS_V4_MAX_GT_FDE_M="${SG_FPS_V4_MAX_GT_FDE_M:-4.0}"
export SG_FPS_V4_MODE_DISTANCE_THRESHOLD="${SG_FPS_V4_MODE_DISTANCE_THRESHOLD:-0.25}"
export SG_FPS_V4_REWARD_GAIN_CAP="${SG_FPS_V4_REWARD_GAIN_CAP:-0.05}"
export SG_FPS_V4_MAX_GT_REWARD_DROP="${SG_FPS_V4_MAX_GT_REWARD_DROP:-0.05}"
export SG_FPS_V4_PARETO_EPS="${SG_FPS_V4_PARETO_EPS:-0.01}"
export SG_FPS_V4_EXCLUDE_DERIVED_EXTERNAL=true
export SG_FPS_V4_REQUIRE_POLICY_REACHABILITY=true
export SG_FPS_V4_MAX_POLICY_SNSAD="${SG_FPS_V4_MAX_POLICY_SNSAD:-0.50}"
export SG_FPS_V4_MIN_POLICY_NEIGHBORS="${SG_FPS_V4_MIN_POLICY_NEIGHBORS:-2}"
export SG_FPS_V4_STAGE3_EP_TOLERANCE="${SG_FPS_V4_STAGE3_EP_TOLERANCE:-0.02}"
export SG_FPS_V4_STAGE3_REFERENCE_MARGIN_WEIGHT="${SG_FPS_V4_STAGE3_REFERENCE_MARGIN_WEIGHT:-0.20}"
export SG_FPS_V4_STAGE3_REFERENCE_MARGIN_SCALE="${SG_FPS_V4_STAGE3_REFERENCE_MARGIN_SCALE:-0.05}"
export SG_FPS_V4_STAGE3_MIN_FEASIBLE_ROLLOUTS="${SG_FPS_V4_STAGE3_MIN_FEASIBLE_ROLLOUTS:-2}"
export SG_FPS_V4_STAGE3_MIN_SCORE_SPAN="${SG_FPS_V4_STAGE3_MIN_SCORE_SPAN:-0.01}"
export SG_FPS_KEEP_GT_BY_DEFAULT=true

# Current-policy rollouts define a local learning frontier. They are candidates,
# not privileged targets, and pass through the same evaluator/Pareto gates.
export ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES:-8}"
export ONLINE_USE_CURRENT_POLICY=true
export ONLINE_USE_OLD_POLICY=false
export USE_FS_NORM=true
export USE_PLANNING_TOKEN_ADAPTER=true
export PLANNING_NUM_TOKENS="${PLANNING_NUM_TOKENS:-16}"
export PLANNING_NUM_HEADS="${PLANNING_NUM_HEADS:-8}"
export PLANNING_CONDITION_LAYERS="${PLANNING_CONDITION_LAYERS:-cross_attention}"
export PLANNING_GATE_INIT="${PLANNING_GATE_INIT:-0.05}"
export PLANNING_CONTEXT_GATE_INIT="${PLANNING_CONTEXT_GATE_INIT:-0.05}"
export PLANNING_CONDITION_DROPOUT="${PLANNING_CONDITION_DROPOUT:-0.10}"
export USE_JEPA=false
export USE_VGGT=false

exec "${REPO_ROOT}/scripts/training/sg_fps/run_build_sg_fps_support.sh"
