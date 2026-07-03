#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH}"

export ELITE_BUFFER_DIR="${SUPPORT_ARCHIVE_PATH}"
export SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}"
export HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
export ONLINE_AWAC_CANDIDATES=false
export STAGE3_OBJECTIVE="${STAGE3_OBJECTIVE:-dpsi}"
export VALIDATE_ELITE_BUFFER="${VALIDATE_ELITE_BUFFER:-false}"
export SG_FPS_USE_DPSI=true
export SG_FPS_DPSI_TOP_M="${SG_FPS_DPSI_TOP_M:-12}"
export USE_FS_NORM="${USE_FS_NORM:-false}"
export FS_NORM_STATS_PATH="${FS_NORM_STATS_PATH:-}"
export X0_AUX_WEIGHT="${X0_AUX_WEIGHT:-0.0}"
export GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.0}"
export RUN_STAGE3="${RUN_STAGE3:-1}"
export DRY_RUN="${DRY_RUN:-0}"

exec "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh"
