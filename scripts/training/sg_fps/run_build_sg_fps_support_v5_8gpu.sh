#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export SG_FPS_BUILD_VARIANT=v5
exec "${REPO_ROOT}/scripts/training/sg_fps/run_build_sg_fps_support_v4_8gpu.sh"
