#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DRY_RUN="${DRY_RUN:-1}"
export RUN_STAGE3="${RUN_STAGE3:-0}"
export RUN_TRAIN="${RUN_TRAIN:-0}"
export LAUNCH_EVAL_WATCHERS="${LAUNCH_EVAL_WATCHERS:-1}"

exec "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_grpo_core_pareto_2b_local.sh"
