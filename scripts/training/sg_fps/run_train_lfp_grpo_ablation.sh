#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
  echo "usage: $0 navsim_v1|navsim_v2 S0_old_core_pareto_v2|S1_global_norm|S2_global_norm_pareto_gate|S3_lfp_energy_no_progress|S4_full_lfp_grpo [Hydra overrides...]" >&2
  exit 2
fi

BENCHMARK="$1"
ABLATION="$2"
shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BENCHMARK}" != "navsim_v1" && "${BENCHMARK}" != "navsim_v2" ]]; then
  echo "benchmark must be navsim_v1 or navsim_v2" >&2
  exit 2
fi

case "${ABLATION}" in
  S0_old_core_pareto_v2)
    if [[ "${BENCHMARK}" != "navsim_v1" ]]; then
      echo "S0_old_core_pareto_v2 is the legacy NAVSIM v1 PDMS baseline and has no official NAVSIM v2 reward adapter; refusing a mislabeled v2 run" >&2
      exit 2
    fi
    export STAGE3_ALGORITHM=legacy
    export STAGE3_OBJECTIVE=grpo
    export GRPO_USE_CORE_PARETO=true
    export GRPO_USE_FEASIBLE_PARETO=false
    export GRPO_USE_GSPO_RATIO=false
    export BC_COEFF_START=0
    export BC_COEFF_END=0
    export RUN_NAME="${RUN_NAME:-${BENCHMARK}_${ABLATION}_$(date -u +%Y%m%dT%H%M%SZ)}"
    exec "${SCRIPT_DIR}/../run_recogdrive_stage3_rl_2b_local.sh" "$@"
    ;;
  S1_global_norm)
    export LFP_PARETO_GATE_ENABLED=false
    export LFP_PROGRESS_GATE_ENABLED=false
    export LFP_CURRICULUM_ENABLED=false
    export LFP_FRONTIER_PROGRESS_WEIGHT=0.0
    ;;
  S2_global_norm_pareto_gate)
    export LFP_PARETO_GATE_ENABLED=true
    export LFP_PROGRESS_GATE_ENABLED=true
    export LFP_CURRICULUM_ENABLED=false
    export LFP_FRONTIER_PROGRESS_WEIGHT=0.0
    ;;
  S3_lfp_energy_no_progress)
    export LFP_PARETO_GATE_ENABLED=true
    export LFP_PROGRESS_GATE_ENABLED=true
    export LFP_CURRICULUM_ENABLED=true
    export LFP_FRONTIER_PROGRESS_WEIGHT=0.0
    ;;
  S4_full_lfp_grpo)
    export LFP_PARETO_GATE_ENABLED=true
    export LFP_PROGRESS_GATE_ENABLED=true
    export LFP_CURRICULUM_ENABLED=true
    export LFP_FRONTIER_PROGRESS_WEIGHT=0.50
    ;;
  *)
    echo "unknown ablation: ${ABLATION}" >&2
    exit 2
    ;;
esac

export RUN_NAME="${RUN_NAME:-${BENCHMARK}_${ABLATION}_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "${BENCHMARK}" == "navsim_v1" ]]; then
  exec "${SCRIPT_DIR}/run_train_lfp_grpo_v1.sh" "$@"
else
  exec "${SCRIPT_DIR}/run_train_lfp_grpo_v2.sh" "$@"
fi
