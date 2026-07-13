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

    # Reproduce the Stage3 optimization recipe recorded by the successful
    # Core-Pareto v2 run. The caller still owns the Stage2 checkpoint and its
    # representation settings, which keeps S0 usable as a matched control.
    export LR=1e-4
    export MAX_EPOCHS=20
    export BATCH_SIZE=2
    export ACCUMULATE_GRAD_BATCHES=4
    export GRPO_SAMPLE_TIME=16
    export BC_ANNEAL=true
    export BC_COEFF_START=0.10
    export BC_COEFF_END=0.05
    export BC_ANNEAL_EPOCHS=5
    export REFERENCE_KL_COEFF=0.02
    export REFERENCE_KL_CHUNK_SIZE=16
    export GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL=4
    export GRPO_BEHAVIOR_POLICY_SAMPLE=true
    export GRPO_NORMALIZE_ADVANTAGE_BATCH=true
    export GRPO_ADVANTAGE_CLIP_ABS=5.0
    export GRPO_MIN_SAMPLING_DENOISING_STD=0.04
    export GRPO_MIN_LOGPROB_DENOISING_STD=0.10
    export GRPO_SCHEDULER_EPOCHS=20
    export GRPO_SCHEDULER_WARMUP_EPOCHS=0
    export GRPO_SCHEDULER_MIN_LR=1e-5
    export TRAINER_GRADIENT_CLIP_VAL=1.0
    export CHECKPOINT_EVERY_N_TRAIN_STEPS=300
    export GRPO_USE_CORE_PARETO=true
    export GRPO_USE_FEASIBLE_PARETO=false
    export GRPO_USE_GSPO_RATIO=false
    export GRPO_USE_DYNAMIC_GROUP_WEIGHT=true
    export GRPO_USE_DIVERSITY_REWARD=false
    export GRPO_CORE_PARETO_USE_ADAPTIVE_DUAL=false
    export GRPO_CORE_PARETO_DUAL_LR=0.02
    export GRPO_CORE_PARETO_USE_PHENOTYPE_BUCKET_GRPO=false
    export GRPO_CORE_PARETO_BUFFER_BONUS_ENABLED=false
    export GRPO_FP_USE_PDAS=false
    export OFFLINE_RL_ENABLED=false
    export GRPO_BUFFER_GUIDANCE_ENABLED=false
    export GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0
    export GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0
    export GRPO_SELF_IMITATION_LOSS_WEIGHT=0
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
