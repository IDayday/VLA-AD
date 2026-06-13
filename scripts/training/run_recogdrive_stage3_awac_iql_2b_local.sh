#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/navsim/bin/torchrun}"
TORCHRUN_EXTRA_ARGS="${TORCHRUN_EXTRA_ARGS:-}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_awac_iql_2b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

IL_CHECKPOINT="${IL_CHECKPOINT:-${RECOGDRIVE_IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}}"
VLM_PATH="${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${RECOGDRIVE_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}}"
HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-${RECOGDRIVE_HIDDEN_CACHE_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_agent_cache_dir_train_2b}}"
ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${NAVSIM_DATA_ROOT}/trainval_sensor_blobs/trainval}"

CACHE_MODE="${CACHE_MODE:-offline}"  # online or offline hidden-state features
ONLINE_AWAC_CANDIDATES="${ONLINE_AWAC_CANDIDATES:-false}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
DRY_RUN="${DRY_RUN:-1}"

STAGE3_LR="${LR:-1e-4}"
STAGE3_MAX_EPOCHS="${MAX_EPOCHS:-20}"
STAGE3_BATCH_SIZE="${BATCH_SIZE:-8}"
STAGE3_ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
STAGE3_SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-0}"
STAGE3_SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1e-5}"
STAGE3_LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
STAGE3_CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
STAGE3_VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1.0}"
STAGE3_NUM_SANITY_VAL_STEPS="${NUM_SANITY_VAL_STEPS:-0}"
STAGE3_TRAIN_TEST_SPLIT="navtrain"
STAGE3_EXPERIMENT_NAME="training_recogdrive_agent"
TRAINER_STRATEGY="${TRAINER_STRATEGY:-ddp}"
STAGE3_OBJECTIVE="${STAGE3_OBJECTIVE:-awac_iql}"
STAGE3_AGENT_GRPO="${AGENT_GRPO:-False}"
STAGE3_GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-8}"

AWAC_ELITE_TOP_M="${AWAC_ELITE_TOP_M:-8}"
AWAC_BC_LOSS_WEIGHT="${AWAC_BC_LOSS_WEIGHT:-0.05}"
AWAC_BC_LOSS_SCHEDULE="${AWAC_BC_LOSS_SCHEDULE:-constant}"
AWAC_BC_LOSS_WEIGHT_START="${AWAC_BC_LOSS_WEIGHT_START:-${AWAC_BC_LOSS_WEIGHT}}"
AWAC_BC_LOSS_WEIGHT_END="${AWAC_BC_LOSS_WEIGHT_END:-${AWAC_BC_LOSS_WEIGHT}}"
AWAC_BC_LOSS_SCHEDULE_EPOCHS="${AWAC_BC_LOSS_SCHEDULE_EPOCHS:-1}"
AWAC_GRPO_LOSS_WEIGHT="${AWAC_GRPO_LOSS_WEIGHT:-0.0}"
AWAC_GRPO_LOSS_SCHEDULE="${AWAC_GRPO_LOSS_SCHEDULE:-constant}"
AWAC_GRPO_LOSS_WEIGHT_START="${AWAC_GRPO_LOSS_WEIGHT_START:-0.0}"
AWAC_GRPO_LOSS_WARMUP_START_EPOCH="${AWAC_GRPO_LOSS_WARMUP_START_EPOCH:-0}"
AWAC_GRPO_LOSS_WARMUP_EPOCHS="${AWAC_GRPO_LOSS_WARMUP_EPOCHS:-1}"
AWAC_BASELINE_MODE="${AWAC_BASELINE_MODE:-max_gt_il}"
AWAC_ADVANTAGE_TEMPERATURE="${AWAC_ADVANTAGE_TEMPERATURE:-0.03}"
AWAC_WEIGHT_MAX="${AWAC_WEIGHT_MAX:-20.0}"
AWAC_REQUIRE_NC="${AWAC_REQUIRE_NC:-true}"
AWAC_REQUIRE_DAC="${AWAC_REQUIRE_DAC:-true}"
AWAC_REQUIRE_DDC_GUARD="${AWAC_REQUIRE_DDC_GUARD:-true}"
AWAC_DDC_MIN_ABSOLUTE="${AWAC_DDC_MIN_ABSOLUTE:-0.99}"
AWAC_DDC_MAX_RELATIVE_DROP="${AWAC_DDC_MAX_RELATIVE_DROP:-0.01}"
AWAC_REQUIRE_TTC_GUARD="${AWAC_REQUIRE_TTC_GUARD:-false}"
AWAC_TTC_MIN_ABSOLUTE="${AWAC_TTC_MIN_ABSOLUTE:-0.95}"
AWAC_TTC_MAX_RELATIVE_DROP="${AWAC_TTC_MAX_RELATIVE_DROP:-0.02}"
AWAC_PRIOR_DISTANCE_WEIGHT="${AWAC_PRIOR_DISTANCE_WEIGHT:-0.02}"
AWAC_JERK_PENALTY_WEIGHT="${AWAC_JERK_PENALTY_WEIGHT:-0.005}"
AWAC_SELECT_BY="${AWAC_SELECT_BY:-pdms_minus_prior}"
AWAC_ONLINE_POLICY_SAMPLES="${AWAC_ONLINE_POLICY_SAMPLES:-8}"
AWAC_STRICT_REWARD_SUBMETRICS="${AWAC_STRICT_REWARD_SUBMETRICS:-true}"
AWAC_MISSING_SUBMETRIC_POLICY="${AWAC_MISSING_SUBMETRIC_POLICY:-error}"
AWAC_USE_BATCHED_PDM_SCORING="${AWAC_USE_BATCHED_PDM_SCORING:-true}"
AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION="${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION:-true}"
AWAC_USE_FAST_PDM_SCORER="${AWAC_USE_FAST_PDM_SCORER:-true}"
AWAC_PDM_BATCH_CHUNK_SIZE="${AWAC_PDM_BATCH_CHUNK_SIZE:-0}"
AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-true}"
AWAC_PDM_SHADOW_MAX_SAMPLES="${AWAC_PDM_SHADOW_MAX_SAMPLES:-4}"
AWAC_PDM_SHADOW_MAX_ABS_DIFF="${AWAC_PDM_SHADOW_MAX_ABS_DIFF:-0.0}"
AWAC_REQUIRE_BUFFER_VALID_MASK="${AWAC_REQUIRE_BUFFER_VALID_MASK:-true}"
AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK="${AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK:-true}"
AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD="${AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD:-false}"
AWAC_CACHE_ELITE_RECORDS_IN_MEMORY="${AWAC_CACHE_ELITE_RECORDS_IN_MEMORY:-false}"
AWAC_PRELOAD_ELITE_BUFFER="${AWAC_PRELOAD_ELITE_BUFFER:-false}"
AWAC_ELITE_BUFFER_PRELOAD_MAX_RECORDS="${AWAC_ELITE_BUFFER_PRELOAD_MAX_RECORDS:-0}"
AWAC_SELECT_VALID_TOPK_ONLY="${AWAC_SELECT_VALID_TOPK_ONLY:-true}"
AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES="${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES:-false}"
AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT="${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT:-0.0}"
AWAC_USE_FINAL_HEADING_GUARD="${AWAC_USE_FINAL_HEADING_GUARD:-true}"
AWAC_TARGET_FILTER_MODE="${AWAC_TARGET_FILTER_MODE:-none}"
AWAC_TARGET_TOP_K="${AWAC_TARGET_TOP_K:-1}"
AWAC_TARGET_MIN_ADVANTAGE="${AWAC_TARGET_MIN_ADVANTAGE:-0.0}"
AWAC_COMPONENT_ADVANTAGE_ENABLED="${AWAC_COMPONENT_ADVANTAGE_ENABLED:-false}"
AWAC_COMPONENT_PROGRESS_WEIGHT="${AWAC_COMPONENT_PROGRESS_WEIGHT:-0.05}"
AWAC_COMPONENT_SAFETY_PENALTY_WEIGHT="${AWAC_COMPONENT_SAFETY_PENALTY_WEIGHT:-0.20}"
AWAC_COMPONENT_ADVANTAGE_CLIP_MIN="${AWAC_COMPONENT_ADVANTAGE_CLIP_MIN:--0.10}"
AWAC_COMPONENT_ADVANTAGE_CLIP_MAX="${AWAC_COMPONENT_ADVANTAGE_CLIP_MAX:-0.05}"
AWAC_SOURCE_BALANCE_ENABLED="${AWAC_SOURCE_BALANCE_ENABLED:-false}"
AWAC_SOURCE_BALANCE_MIN_FACTOR="${AWAC_SOURCE_BALANCE_MIN_FACTOR:-0.50}"
AWAC_SOURCE_BALANCE_MAX_FACTOR="${AWAC_SOURCE_BALANCE_MAX_FACTOR:-2.00}"
AWAC_PAIRWISE_RANK_LOSS_WEIGHT="${AWAC_PAIRWISE_RANK_LOSS_WEIGHT:-0.0}"
AWAC_PAIRWISE_RANK_MARGIN="${AWAC_PAIRWISE_RANK_MARGIN:-0.02}"
AWAC_PAIRWISE_RANK_MIN_REWARD_GAP="${AWAC_PAIRWISE_RANK_MIN_REWARD_GAP:-0.02}"
AWAC_PAIRWISE_RANK_MAX_PAIRS_PER_SCENE="${AWAC_PAIRWISE_RANK_MAX_PAIRS_PER_SCENE:-2}"
AWAC_INVALID_REPULSION_LOSS_WEIGHT="${AWAC_INVALID_REPULSION_LOSS_WEIGHT:-0.0}"
AWAC_INVALID_REPULSION_MARGIN="${AWAC_INVALID_REPULSION_MARGIN:-0.05}"
AWAC_INVALID_REPULSION_MAX_PAIRS_PER_SCENE="${AWAC_INVALID_REPULSION_MAX_PAIRS_PER_SCENE:-2}"
AWAC_PREFERENCE_DPO_LOSS_WEIGHT="${AWAC_PREFERENCE_DPO_LOSS_WEIGHT:-0.0}"
AWAC_PREFERENCE_DPO_BETA="${AWAC_PREFERENCE_DPO_BETA:-8.0}"
AWAC_PREFERENCE_DPO_LABEL_SMOOTHING="${AWAC_PREFERENCE_DPO_LABEL_SMOOTHING:-0.0}"
AWAC_PREFERENCE_DPO_REFERENCE_FREE="${AWAC_PREFERENCE_DPO_REFERENCE_FREE:-false}"
AWAC_PREFERENCE_DPO_PAIR_MODE="${AWAC_PREFERENCE_DPO_PAIR_MODE:-best_vs_gt_il}"
AWAC_PREFERENCE_DPO_MIN_REWARD_GAP="${AWAC_PREFERENCE_DPO_MIN_REWARD_GAP:-0.02}"
AWAC_PREFERENCE_DPO_MAX_PAIRS_PER_SCENE="${AWAC_PREFERENCE_DPO_MAX_PAIRS_PER_SCENE:-2}"
AWAC_PREFERENCE_DPO_GAP_WEIGHT_MODE="${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MODE:-none}"
AWAC_PREFERENCE_DPO_GAP_WEIGHT_SCALE="${AWAC_PREFERENCE_DPO_GAP_WEIGHT_SCALE:-0.05}"
AWAC_PREFERENCE_DPO_GAP_WEIGHT_MIN="${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MIN:-0.0}"
AWAC_PREFERENCE_DPO_GAP_WEIGHT_MAX="${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MAX:-3.0}"
VALIDATE_ELITE_BUFFER="${VALIDATE_ELITE_BUFFER:-true}"

if awk "BEGIN { exit !(${AWAC_GRPO_LOSS_WEIGHT} > 0.0) }"; then
  STAGE3_AGENT_GRPO=True
  if [[ "${STAGE3_OBJECTIVE}" == "awac_iql" ]]; then
    STAGE3_OBJECTIVE="hybrid"
  fi
fi

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUT_ROOT}/hydra}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "${CACHE_MODE}" != "online" && "${CACHE_MODE}" != "offline" ]]; then
  echo "CACHE_MODE must be online or offline, got: ${CACHE_MODE}" >&2
  exit 2
fi
if [[ "${ONLINE_AWAC_CANDIDATES}" != "true" && "${ONLINE_AWAC_CANDIDATES}" != "false" ]]; then
  echo "ONLINE_AWAC_CANDIDATES must be true or false, got: ${ONLINE_AWAC_CANDIDATES}" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "1" && "${ONLINE_AWAC_CANDIDATES}" == "false" ]]; then
  if [[ ! -d "${ELITE_BUFFER_DIR}" ]] || ! find "${ELITE_BUFFER_DIR}" -name '*.pkl.xz' -print -quit | grep -q .; then
    echo "ELITE_BUFFER_DIR must contain *.pkl.xz records unless ONLINE_AWAC_CANDIDATES=true: ${ELITE_BUFFER_DIR}" >&2
    exit 2
  fi
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "Python or torchrun is not executable: ${PYTHON_BIN}, ${TORCHRUN_BIN}" >&2
    exit 2
  fi
  if [[ ! -f "${IL_CHECKPOINT}" ]]; then
    echo "Stage2 IL checkpoint does not exist: ${IL_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ ! -d "${METRIC_CACHE_DIR}" ]]; then
    echo "Training metric cache directory does not exist: ${METRIC_CACHE_DIR}" >&2
    exit 2
  fi
fi

mkdir -p "${OUT_ROOT}"

HYDRA_ARGS=(
  "agent=recogdrive_agent"
  "agent.stage3_objective=${STAGE3_OBJECTIVE}"
  "agent.grpo=${STAGE3_AGENT_GRPO}"
  "agent.grpo_sample_time=${STAGE3_GRPO_SAMPLE_TIME}"
  "agent.lr=${STAGE3_LR}"
  "agent.scheduler_epochs=${STAGE3_MAX_EPOCHS}"
  "agent.scheduler_warmup_epochs=${STAGE3_SCHEDULER_WARMUP_EPOCHS}"
  "agent.scheduler_min_lr=${STAGE3_SCHEDULER_MIN_LR}"
  "agent.vlm_path=${VLM_PATH}"
  "agent.cam_type=single"
  "agent.cache_mode=False"
  "agent.vlm_type=internvl"
  "agent.checkpoint_path=${IL_CHECKPOINT}"
  "agent.dit_type=small"
  "agent.vlm_size=small"
  "agent.sampling_method=ddim"
  "agent.metric_cache_path=${METRIC_CACHE_DIR}"
  "agent.reference_policy_checkpoint=${IL_CHECKPOINT}"
  "agent.offline_rl_enabled=True"
  "agent.offline_rl_elite_buffer_path=${ELITE_BUFFER_DIR}"
  "agent.offline_rl_build_candidates_online=${ONLINE_AWAC_CANDIDATES}"
  "agent.offline_rl_online_policy_samples=${AWAC_ONLINE_POLICY_SAMPLES}"
  "agent.offline_rl_strict_reward_submetrics=${AWAC_STRICT_REWARD_SUBMETRICS}"
  "agent.offline_rl_missing_submetric_policy=${AWAC_MISSING_SUBMETRIC_POLICY}"
  "agent.offline_rl_use_batched_pdm_scoring=${AWAC_USE_BATCHED_PDM_SCORING}"
  "agent.offline_rl_use_exact_array_pdm_state_conversion=${AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION}"
  "agent.offline_rl_use_fast_pdm_scorer=${AWAC_USE_FAST_PDM_SCORER}"
  "agent.offline_rl_pdm_batch_chunk_size=${AWAC_PDM_BATCH_CHUNK_SIZE}"
  "agent.offline_rl_pdm_shadow_check=${AWAC_PDM_SHADOW_CHECK}"
  "agent.offline_rl_pdm_shadow_max_samples=${AWAC_PDM_SHADOW_MAX_SAMPLES}"
  "agent.offline_rl_pdm_shadow_max_abs_diff=${AWAC_PDM_SHADOW_MAX_ABS_DIFF}"
  "agent.offline_rl_require_buffer_valid_mask=${AWAC_REQUIRE_BUFFER_VALID_MASK}"
  "agent.offline_rl_allow_v1_buffer_recompute_valid_mask=${AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK}"
  "agent.offline_rl_recompute_buffer_valid_mask_on_load=${AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD}"
  "agent.offline_rl_cache_elite_records_in_memory=${AWAC_CACHE_ELITE_RECORDS_IN_MEMORY}"
  "agent.offline_rl_preload_elite_buffer=${AWAC_PRELOAD_ELITE_BUFFER}"
  "agent.offline_rl_elite_buffer_preload_max_records=${AWAC_ELITE_BUFFER_PRELOAD_MAX_RECORDS}"
  "agent.offline_rl_select_valid_topk_only=${AWAC_SELECT_VALID_TOPK_ONLY}"
  "agent.offline_rl_train_invalid_fallback_candidates=${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES}"
  "agent.offline_rl_fallback_invalid_candidate_weight=${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT}"
  "agent.offline_rl_use_final_heading_guard=${AWAC_USE_FINAL_HEADING_GUARD}"
  "agent.offline_rl_target_filter_mode=${AWAC_TARGET_FILTER_MODE}"
  "agent.offline_rl_target_top_k=${AWAC_TARGET_TOP_K}"
  "agent.offline_rl_target_min_advantage=${AWAC_TARGET_MIN_ADVANTAGE}"
  "agent.offline_rl_component_advantage_enabled=${AWAC_COMPONENT_ADVANTAGE_ENABLED}"
  "agent.offline_rl_component_progress_weight=${AWAC_COMPONENT_PROGRESS_WEIGHT}"
  "agent.offline_rl_component_safety_penalty_weight=${AWAC_COMPONENT_SAFETY_PENALTY_WEIGHT}"
  "agent.offline_rl_component_advantage_clip_min=${AWAC_COMPONENT_ADVANTAGE_CLIP_MIN}"
  "agent.offline_rl_component_advantage_clip_max=${AWAC_COMPONENT_ADVANTAGE_CLIP_MAX}"
  "agent.offline_rl_source_balance_enabled=${AWAC_SOURCE_BALANCE_ENABLED}"
  "agent.offline_rl_source_balance_min_factor=${AWAC_SOURCE_BALANCE_MIN_FACTOR}"
  "agent.offline_rl_source_balance_max_factor=${AWAC_SOURCE_BALANCE_MAX_FACTOR}"
  "agent.offline_rl_pairwise_rank_loss_weight=${AWAC_PAIRWISE_RANK_LOSS_WEIGHT}"
  "agent.offline_rl_pairwise_rank_margin=${AWAC_PAIRWISE_RANK_MARGIN}"
  "agent.offline_rl_pairwise_rank_min_reward_gap=${AWAC_PAIRWISE_RANK_MIN_REWARD_GAP}"
  "agent.offline_rl_pairwise_rank_max_pairs_per_scene=${AWAC_PAIRWISE_RANK_MAX_PAIRS_PER_SCENE}"
  "agent.offline_rl_invalid_repulsion_loss_weight=${AWAC_INVALID_REPULSION_LOSS_WEIGHT}"
  "agent.offline_rl_invalid_repulsion_margin=${AWAC_INVALID_REPULSION_MARGIN}"
  "agent.offline_rl_invalid_repulsion_max_pairs_per_scene=${AWAC_INVALID_REPULSION_MAX_PAIRS_PER_SCENE}"
  "agent.offline_rl_preference_dpo_loss_weight=${AWAC_PREFERENCE_DPO_LOSS_WEIGHT}"
  "agent.offline_rl_preference_dpo_beta=${AWAC_PREFERENCE_DPO_BETA}"
  "agent.offline_rl_preference_dpo_label_smoothing=${AWAC_PREFERENCE_DPO_LABEL_SMOOTHING}"
  "agent.offline_rl_preference_dpo_reference_free=${AWAC_PREFERENCE_DPO_REFERENCE_FREE}"
  "agent.offline_rl_preference_dpo_pair_mode=${AWAC_PREFERENCE_DPO_PAIR_MODE}"
  "agent.offline_rl_preference_dpo_min_reward_gap=${AWAC_PREFERENCE_DPO_MIN_REWARD_GAP}"
  "agent.offline_rl_preference_dpo_max_pairs_per_scene=${AWAC_PREFERENCE_DPO_MAX_PAIRS_PER_SCENE}"
  "agent.offline_rl_preference_dpo_gap_weight_mode=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MODE}"
  "agent.offline_rl_preference_dpo_gap_weight_scale=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_SCALE}"
  "agent.offline_rl_preference_dpo_gap_weight_min=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MIN}"
  "agent.offline_rl_preference_dpo_gap_weight_max=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MAX}"
  "agent.offline_rl_awac_loss_weight=1.0"
  "agent.offline_rl_bc_loss_weight=${AWAC_BC_LOSS_WEIGHT}"
  "agent.offline_rl_bc_loss_schedule=${AWAC_BC_LOSS_SCHEDULE}"
  "agent.offline_rl_bc_loss_weight_start=${AWAC_BC_LOSS_WEIGHT_START}"
  "agent.offline_rl_bc_loss_weight_end=${AWAC_BC_LOSS_WEIGHT_END}"
  "agent.offline_rl_bc_loss_schedule_epochs=${AWAC_BC_LOSS_SCHEDULE_EPOCHS}"
  "agent.offline_rl_grpo_loss_weight=${AWAC_GRPO_LOSS_WEIGHT}"
  "agent.offline_rl_grpo_loss_schedule=${AWAC_GRPO_LOSS_SCHEDULE}"
  "agent.offline_rl_grpo_loss_weight_start=${AWAC_GRPO_LOSS_WEIGHT_START}"
  "agent.offline_rl_grpo_loss_warmup_start_epoch=${AWAC_GRPO_LOSS_WARMUP_START_EPOCH}"
  "agent.offline_rl_grpo_loss_warmup_epochs=${AWAC_GRPO_LOSS_WARMUP_EPOCHS}"
  "agent.offline_rl_baseline_mode=${AWAC_BASELINE_MODE}"
  "agent.offline_rl_advantage_temperature=${AWAC_ADVANTAGE_TEMPERATURE}"
  "agent.offline_rl_weight_max=${AWAC_WEIGHT_MAX}"
  "agent.offline_rl_elite_top_m=${AWAC_ELITE_TOP_M}"
  "agent.offline_rl_require_nc=${AWAC_REQUIRE_NC}"
  "agent.offline_rl_require_dac=${AWAC_REQUIRE_DAC}"
  "agent.offline_rl_require_ddc_guard=${AWAC_REQUIRE_DDC_GUARD}"
  "agent.offline_rl_ddc_min_absolute=${AWAC_DDC_MIN_ABSOLUTE}"
  "agent.offline_rl_ddc_max_relative_drop=${AWAC_DDC_MAX_RELATIVE_DROP}"
  "agent.offline_rl_require_ttc_guard=${AWAC_REQUIRE_TTC_GUARD}"
  "agent.offline_rl_ttc_min_absolute=${AWAC_TTC_MIN_ABSOLUTE}"
  "agent.offline_rl_ttc_max_relative_drop=${AWAC_TTC_MAX_RELATIVE_DROP}"
  "agent.offline_rl_prior_distance_weight=${AWAC_PRIOR_DISTANCE_WEIGHT}"
  "agent.offline_rl_jerk_penalty_weight=${AWAC_JERK_PENALTY_WEIGHT}"
  "agent.offline_rl_select_by=${AWAC_SELECT_BY}"
  "trainer.params.max_epochs=${STAGE3_MAX_EPOCHS}"
  "trainer.params.num_nodes=${NODES}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "trainer.params.strategy=${TRAINER_STRATEGY}"
  "trainer.params.accumulate_grad_batches=${STAGE3_ACCUMULATE_GRAD_BATCHES}"
  "trainer.params.limit_val_batches=${STAGE3_LIMIT_VAL_BATCHES}"
  "trainer.params.check_val_every_n_epoch=${STAGE3_CHECK_VAL_EVERY_N_EPOCH}"
  "trainer.params.val_check_interval=${STAGE3_VAL_CHECK_INTERVAL}"
  "trainer.params.num_sanity_val_steps=${STAGE3_NUM_SANITY_VAL_STEPS}"
  "dataloader.params.batch_size=${STAGE3_BATCH_SIZE}"
  "experiment_name=${STAGE3_EXPERIMENT_NAME}"
  "train_test_split=${STAGE3_TRAIN_TEST_SPLIT}"
  "force_cache_computation=False"
)

if [[ "${CACHE_MODE}" == "offline" ]]; then
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=True"
    "cache_path=${HIDDEN_CACHE_DIR}"
    "use_cache_without_dataset=True"
  )
else
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=False"
    "cache_path=null"
    "use_cache_without_dataset=False"
    "navsim_log_path=${NAVSIM_LOG_PATH}"
    "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  )
fi

if [[ -n "${MAX_SCENES:-}" ]]; then
  HYDRA_ARGS+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

TORCHRUN_EXTRA_ARGS_ARRAY=()
if [[ -n "${TORCHRUN_EXTRA_ARGS}" ]]; then
  # Space-separated torchrun flags for diagnostics, e.g. "--tee=3 --redirects=3 --log_dir=/tmp/torchrun".
  read -r -a TORCHRUN_EXTRA_ARGS_ARRAY <<< "${TORCHRUN_EXTRA_ARGS}"
fi

CMD=(
  "${TORCHRUN_BIN}"
  "${TORCHRUN_EXTRA_ARGS_ARRAY[@]}"
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py"
  "${HYDRA_ARGS[@]}"
)

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "stage3_objective=${STAGE3_OBJECTIVE}"
  echo "agent_grpo=${STAGE3_AGENT_GRPO}"
  echo "grpo_sample_time=${STAGE3_GRPO_SAMPLE_TIME}"
  echo "lr=${STAGE3_LR}"
  echo "max_epochs=${STAGE3_MAX_EPOCHS}"
  echo "scheduler_min_lr=${STAGE3_SCHEDULER_MIN_LR}"
  echo "batch_size=${STAGE3_BATCH_SIZE}"
  echo "trainer_strategy=${TRAINER_STRATEGY}"
  echo "limit_val_batches=${STAGE3_LIMIT_VAL_BATCHES}"
  echo "check_val_every_n_epoch=${STAGE3_CHECK_VAL_EVERY_N_EPOCH}"
  echo "val_check_interval=${STAGE3_VAL_CHECK_INTERVAL}"
  echo "num_sanity_val_steps=${STAGE3_NUM_SANITY_VAL_STEPS}"
  echo "torchrun_extra_args=${TORCHRUN_EXTRA_ARGS}"
  echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
  echo "online_awac_candidates=${ONLINE_AWAC_CANDIDATES}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  echo "awac_strict_reward_submetrics=${AWAC_STRICT_REWARD_SUBMETRICS}"
  echo "awac_missing_submetric_policy=${AWAC_MISSING_SUBMETRIC_POLICY}"
  echo "awac_require_buffer_valid_mask=${AWAC_REQUIRE_BUFFER_VALID_MASK}"
  echo "awac_allow_v1_buffer_recompute_valid_mask=${AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK}"
  echo "awac_recompute_buffer_valid_mask_on_load=${AWAC_RECOMPUTE_BUFFER_VALID_MASK_ON_LOAD}"
  echo "awac_cache_elite_records_in_memory=${AWAC_CACHE_ELITE_RECORDS_IN_MEMORY}"
  echo "awac_preload_elite_buffer=${AWAC_PRELOAD_ELITE_BUFFER}"
  echo "awac_elite_buffer_preload_max_records=${AWAC_ELITE_BUFFER_PRELOAD_MAX_RECORDS}"
  echo "awac_select_valid_topk_only=${AWAC_SELECT_VALID_TOPK_ONLY}"
  echo "awac_train_invalid_fallback_candidates=${AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES}"
  echo "awac_fallback_invalid_candidate_weight=${AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT}"
  echo "awac_use_final_heading_guard=${AWAC_USE_FINAL_HEADING_GUARD}"
  echo "awac_target_filter_mode=${AWAC_TARGET_FILTER_MODE}"
  echo "awac_target_top_k=${AWAC_TARGET_TOP_K}"
  echo "awac_target_min_advantage=${AWAC_TARGET_MIN_ADVANTAGE}"
  echo "awac_component_advantage_enabled=${AWAC_COMPONENT_ADVANTAGE_ENABLED}"
  echo "awac_component_progress_weight=${AWAC_COMPONENT_PROGRESS_WEIGHT}"
  echo "awac_component_safety_penalty_weight=${AWAC_COMPONENT_SAFETY_PENALTY_WEIGHT}"
  echo "awac_component_advantage_clip_min=${AWAC_COMPONENT_ADVANTAGE_CLIP_MIN}"
  echo "awac_component_advantage_clip_max=${AWAC_COMPONENT_ADVANTAGE_CLIP_MAX}"
  echo "awac_source_balance_enabled=${AWAC_SOURCE_BALANCE_ENABLED}"
  echo "awac_source_balance_min_factor=${AWAC_SOURCE_BALANCE_MIN_FACTOR}"
  echo "awac_source_balance_max_factor=${AWAC_SOURCE_BALANCE_MAX_FACTOR}"
  echo "awac_pairwise_rank_loss_weight=${AWAC_PAIRWISE_RANK_LOSS_WEIGHT}"
  echo "awac_pairwise_rank_margin=${AWAC_PAIRWISE_RANK_MARGIN}"
  echo "awac_pairwise_rank_min_reward_gap=${AWAC_PAIRWISE_RANK_MIN_REWARD_GAP}"
  echo "awac_pairwise_rank_max_pairs_per_scene=${AWAC_PAIRWISE_RANK_MAX_PAIRS_PER_SCENE}"
  echo "awac_invalid_repulsion_loss_weight=${AWAC_INVALID_REPULSION_LOSS_WEIGHT}"
  echo "awac_invalid_repulsion_margin=${AWAC_INVALID_REPULSION_MARGIN}"
  echo "awac_invalid_repulsion_max_pairs_per_scene=${AWAC_INVALID_REPULSION_MAX_PAIRS_PER_SCENE}"
  echo "awac_preference_dpo_loss_weight=${AWAC_PREFERENCE_DPO_LOSS_WEIGHT}"
  echo "awac_preference_dpo_beta=${AWAC_PREFERENCE_DPO_BETA}"
  echo "awac_preference_dpo_label_smoothing=${AWAC_PREFERENCE_DPO_LABEL_SMOOTHING}"
  echo "awac_preference_dpo_reference_free=${AWAC_PREFERENCE_DPO_REFERENCE_FREE}"
  echo "awac_preference_dpo_pair_mode=${AWAC_PREFERENCE_DPO_PAIR_MODE}"
  echo "awac_preference_dpo_min_reward_gap=${AWAC_PREFERENCE_DPO_MIN_REWARD_GAP}"
  echo "awac_preference_dpo_max_pairs_per_scene=${AWAC_PREFERENCE_DPO_MAX_PAIRS_PER_SCENE}"
  echo "awac_preference_dpo_gap_weight_mode=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MODE}"
  echo "awac_preference_dpo_gap_weight_scale=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_SCALE}"
  echo "awac_preference_dpo_gap_weight_min=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MIN}"
  echo "awac_preference_dpo_gap_weight_max=${AWAC_PREFERENCE_DPO_GAP_WEIGHT_MAX}"
  echo "awac_bc_loss_schedule=${AWAC_BC_LOSS_SCHEDULE}"
  echo "awac_bc_loss_weight=${AWAC_BC_LOSS_WEIGHT}"
  echo "awac_bc_loss_weight_start=${AWAC_BC_LOSS_WEIGHT_START}"
  echo "awac_bc_loss_weight_end=${AWAC_BC_LOSS_WEIGHT_END}"
  echo "awac_bc_loss_schedule_epochs=${AWAC_BC_LOSS_SCHEDULE_EPOCHS}"
  echo "awac_grpo_loss_weight=${AWAC_GRPO_LOSS_WEIGHT}"
  echo "awac_grpo_loss_schedule=${AWAC_GRPO_LOSS_SCHEDULE}"
  echo "awac_grpo_loss_weight_start=${AWAC_GRPO_LOSS_WEIGHT_START}"
  echo "awac_grpo_loss_warmup_start_epoch=${AWAC_GRPO_LOSS_WARMUP_START_EPOCH}"
  echo "awac_grpo_loss_warmup_epochs=${AWAC_GRPO_LOSS_WARMUP_EPOCHS}"
  echo "awac_advantage_temperature=${AWAC_ADVANTAGE_TEMPERATURE}"
  echo "awac_weight_max=${AWAC_WEIGHT_MAX}"
  echo "awac_require_ttc_guard=${AWAC_REQUIRE_TTC_GUARD}"
  echo "awac_ttc_min_absolute=${AWAC_TTC_MIN_ABSOLUTE}"
  echo "awac_ttc_max_relative_drop=${AWAC_TTC_MAX_RELATIVE_DROP}"
  echo "awac_prior_distance_weight=${AWAC_PRIOR_DISTANCE_WEIGHT}"
  echo "awac_jerk_penalty_weight=${AWAC_JERK_PENALTY_WEIGHT}"
  echo "awac_select_by=${AWAC_SELECT_BY}"
  echo "validate_elite_buffer=${VALIDATE_ELITE_BUFFER}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${OUT_ROOT}/resolved_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_command.txt"
  exit 0
fi

if [[ "${RUN_STAGE3:-0}" != "1" && "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "Stage3 AWAC/IQL training is disabled by default. Set RUN_STAGE3=1 or RUN_TRAIN=1 to execute." >&2
  exit 2
fi

if [[ "${ONLINE_AWAC_CANDIDATES}" == "false" && "${VALIDATE_ELITE_BUFFER}" == "true" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/training/validate_recogdrive_stage3_awac_elite_buffer.py" \
    --buffer-dir "${ELITE_BUFFER_DIR}" \
    --summary-json "${OUT_ROOT}/elite_buffer_validation_summary.json" \
    --summary-csv "${OUT_ROOT}/elite_buffer_validation_summary.csv" \
    --strict-v2
fi

if [[ "${KILL_GPU_STRESS}" == "1" ]]; then
  pkill -f '^python /mnt/project/gpu_stress.py$' || true
  sleep 5
fi

exec "${CMD[@]}"
