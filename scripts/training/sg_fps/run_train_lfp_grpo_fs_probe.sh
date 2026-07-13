#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${LFP_FS_PROFILE:-cov_c2}"
export LFP_FS_TRANSITION_STD_ENABLED=false
export LFP_FS_ENDPOINT_STD_X_M=0.0
export LFP_FS_ENDPOINT_STD_Y_M=0.0
export LFP_FS_ENDPOINT_STD_HEADING_RAD=0.0
export LFP_MIN_GROUP_PAIRWISE_ADE_M=0.0
export LFP_MIN_GROUP_SCALAR_SPAN=0.0
export LFP_ADVANTAGE_DEADBAND=0.0
export LFP_QUALITY_POSITIVE_CREDIT_GUARD=false
export LFP_QUALITY_REFERENCE_TOLERANCE=0.02
export LFP_REFERENCE_PARETO_GATE_ENABLED=false
export LFP_FRONTIER_SAFETY_FIRST=false
export LFP_ADVANTAGE_NORMALIZATION=global_std
export LFP_FRONTIER_TRADEOFF_WEIGHT=0.0
export LFP_FRONTIER_TRADEOFF_EPS=0.01
export LFP_FRONTIER_DIVERSITY_CAPACITY_ENABLED=false
export LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH="${LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH:-}"
export LFP_FRONTIER_DIVERSITY_GAP_WEIGHT="${LFP_FRONTIER_DIVERSITY_GAP_WEIGHT:-0.05}"
export LFP_FRONTIER_DIVERSITY_DISPERSION_FLOOR="${LFP_FRONTIER_DIVERSITY_DISPERSION_FLOOR:-0.05}"

case "${PROFILE}" in
  baseline)
    ;;
  cov_c2)
    export LFP_FS_TRANSITION_STD_ENABLED=true
    export LFP_FS_ENDPOINT_STD_X_M=0.40
    export LFP_FS_ENDPOINT_STD_Y_M=0.16
    export LFP_FS_ENDPOINT_STD_HEADING_RAD=0.014
    ;;
  cov_c4)
    export LFP_FS_TRANSITION_STD_ENABLED=true
    export LFP_FS_ENDPOINT_STD_X_M=0.80
    export LFP_FS_ENDPOINT_STD_Y_M=0.32
    export LFP_FS_ENDPOINT_STD_HEADING_RAD=0.028
    ;;
  gate_only)
    export LFP_MIN_GROUP_PAIRWISE_ADE_M=0.03
    export LFP_MIN_GROUP_SCALAR_SPAN=0.005
    export LFP_ADVANTAGE_DEADBAND=0.05
    ;;
  reference_pareto)
    export LFP_REFERENCE_PARETO_GATE_ENABLED=true
    ;;
  quality_guard)
    export LFP_QUALITY_POSITIVE_CREDIT_GUARD=true
    ;;
  reference_pareto_quality)
    export LFP_REFERENCE_PARETO_GATE_ENABLED=true
    export LFP_QUALITY_POSITIVE_CREDIT_GUARD=true
    ;;
  tradeoff_curriculum)
    export LFP_FRONTIER_TRADEOFF_WEIGHT=1.0
    ;;
  safety_first)
    export LFP_FRONTIER_SAFETY_FIRST=true
    ;;
  diversity_capacity)
    : "${LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH:?Set LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH}"
    export LFP_FRONTIER_SAFETY_FIRST=true
    export LFP_FRONTIER_DIVERSITY_CAPACITY_ENABLED=true
    export LFP_FRONTIER_DIVERSITY_PRIORITY_MODE=credit_multiplicative
    ;;
  diversity_capacity_additive)
    : "${LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH:?Set LFP_FRONTIER_DIVERSITY_CAPACITY_CACHE_PATH}"
    export LFP_FRONTIER_SAFETY_FIRST=true
    export LFP_FRONTIER_DIVERSITY_CAPACITY_ENABLED=true
    export LFP_FRONTIER_DIVERSITY_PRIORITY_MODE=additive_preservation
    ;;
  scene_group_std)
    export LFP_ADVANTAGE_NORMALIZATION=scene_group_std
    ;;
  reference_quality_tradeoff)
    export LFP_REFERENCE_PARETO_GATE_ENABLED=true
    export LFP_QUALITY_POSITIVE_CREDIT_GUARD=true
    export LFP_FRONTIER_TRADEOFF_WEIGHT=1.0
    ;;
  cov_c2_gate)
    export LFP_FS_TRANSITION_STD_ENABLED=true
    export LFP_FS_ENDPOINT_STD_X_M=0.40
    export LFP_FS_ENDPOINT_STD_Y_M=0.16
    export LFP_FS_ENDPOINT_STD_HEADING_RAD=0.014
    export LFP_MIN_GROUP_PAIRWISE_ADE_M=0.03
    export LFP_MIN_GROUP_SCALAR_SPAN=0.005
    export LFP_ADVANTAGE_DEADBAND=0.05
    ;;
  *)
    echo "Unknown LFP_FS_PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

export RUN_NAME="${RUN_NAME:-stage3_lfp_grpo_${PROFILE}_$(date -u +%Y%m%dT%H%M%SZ)}"

args=("$@")
if [[ -n "${MAX_STEPS:-}" ]]; then
  args+=("+trainer.params.max_steps=${MAX_STEPS}")
fi
exec "${SCRIPT_DIR}/run_train_lfp_grpo_v1.sh" "${args[@]}"
