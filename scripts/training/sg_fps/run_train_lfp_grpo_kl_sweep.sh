#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
  echo "usage: $0 navsim_v1|navsim_v2 0.002|0.005|0.010 [Hydra overrides...]" >&2
  exit 2
fi
BENCHMARK="$1"
KL_COEFF="$2"
shift 2
case "${KL_COEFF}" in
  0.002|0.005|0.010) ;;
  *) echo "KL coefficient must be 0.002, 0.005, or 0.010" >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REFERENCE_KL_COEFF="${KL_COEFF}"
export RUN_NAME="${RUN_NAME:-${BENCHMARK}_LFP_KL_${KL_COEFF}_$(date -u +%Y%m%dT%H%M%SZ)}"
exec "${SCRIPT_DIR}/run_train_lfp_grpo_ablation.sh" "${BENCHMARK}" S4_full_lfp_grpo "$@"
