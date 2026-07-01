#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STAGE="${STAGE:-stage3}"
exec "${SCRIPT_DIR}/watch_psi_stage2_checkpoint_eval_8gpu.sh" "$@"
