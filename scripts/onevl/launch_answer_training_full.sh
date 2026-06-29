#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
timestamp=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_${timestamp}}"
LOG_DIR="${OUT_ROOT}/logs"
CMD_LOG="${OUT_ROOT}/commands.log"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_answer_training_full.sh}"

mkdir -p "${LOG_DIR}"

cmd=(
  setsid
  env
  OUT_ROOT="${OUT_ROOT}"
  bash "${RUNNER}"
)

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >> "${CMD_LOG}"

"${cmd[@]}" > "${LOG_DIR}/launcher.outer.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${OUT_ROOT}/launcher.pid"
echo "launched answer full: pid=${pid} out_root=${OUT_ROOT}"
