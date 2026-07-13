#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LFP_BENCHMARK=navsim_v2
export LFP_V2_REQUIRE_TLC="${LFP_V2_REQUIRE_TLC:-true}"
: "${LFP_V2_NAVSIM_ROOT:?Set LFP_V2_NAVSIM_ROOT to the official autonomousvision/navsim v2 checkout}"
export RUN_NAME="${RUN_NAME:-stage3_lfp_grpo_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
exec "${SCRIPT_DIR}/run_train_lfp_grpo_v1.sh" "$@"
