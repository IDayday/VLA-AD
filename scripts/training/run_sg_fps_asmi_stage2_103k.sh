#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  RECOGDRIVE_VLM_PATH
  CACHE_PATH
  TRAIN_TEST_SPLIT
  SUPPORT_ARCHIVE_PATH
  OUTPUT_DIR
  MASTER_PORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_DIR}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "OUTPUT_DIR already exists: ${OUTPUT_DIR}. Set ALLOW_OVERWRITE=1 to reuse it." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TRAINER_DEVICES="${TRAINER_DEVICES:-${NPROC_PER_NODE}}"
SEED="${SEED:-0}"
KEY_STEPS="${KEY_STEPS:-10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000}"
HYDRA_EXPERIMENT="${HYDRA_EXPERIMENT:-sg_fps_asmi_stage2_103k_fs_norm}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
if [[ -z "${ACCUMULATE_GRAD_BATCHES:-}" ]]; then
  denom=$((NPROC_PER_NODE * MICRO_BATCH_SIZE))
  if (( denom <= 0 || TARGET_EFFECTIVE_BATCH_SIZE % denom != 0 )); then
    echo "Cannot derive ACCUMULATE_GRAD_BATCHES: target_effective_batch=${TARGET_EFFECTIVE_BATCH_SIZE}, nproc=${NPROC_PER_NODE}, micro_batch=${MICRO_BATCH_SIZE}" >&2
    exit 2
  fi
  ACCUMULATE_GRAD_BATCHES=$((TARGET_EFFECTIVE_BATCH_SIZE / denom))
fi
KEY_EPOCH_INTERVAL="${KEY_EPOCH_INTERVAL:-1}"
EFFECTIVE_BATCH_SIZE=$((NPROC_PER_NODE * MICRO_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))
if (( EFFECTIVE_BATCH_SIZE != TARGET_EFFECTIVE_BATCH_SIZE )); then
  echo "Effective batch mismatch: got ${EFFECTIVE_BATCH_SIZE}, expected ${TARGET_EFFECTIVE_BATCH_SIZE}. Adjust MICRO_BATCH_SIZE/ACCUMULATE_GRAD_BATCHES." >&2
  exit 2
fi

if [[ ("${HYDRA_EXPERIMENT}" == *"fs_norm"* || "${HYDRA_EXPERIMENT}" == "pta_fs_dit_stage2_103k") && -z "${FS_NORM_STATS_PATH:-}" ]]; then
  echo "HYDRA_EXPERIMENT=${HYDRA_EXPERIMENT} requires FS_NORM_STATS_PATH." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/sg_fps_asmi_stage2_103k.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  "+experiment=${HYDRA_EXPERIMENT}"
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH}"
  "cache_path=${CACHE_PATH}"
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "agent.offline_rl_support_archive_path=${SUPPORT_ARCHIVE_PATH}"
  "trainer.params.devices=${TRAINER_DEVICES}"
  "trainer.params.default_root_dir=${OUTPUT_DIR}"
  "output_dir=${OUTPUT_DIR}"
  "seed=${SEED}"
  "cache_train_all_records=true"
  "dataloader.params.batch_size=${MICRO_BATCH_SIZE}"
  "dataloader.params.pin_memory=false"
  "trainer.params.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
)

if [[ -n "${FS_NORM_STATS_PATH:-}" ]]; then
  cmd+=("agent.fs_norm_stats_path=${FS_NORM_STATS_PATH}")
fi
if [[ -n "${FS_NORM_REQUIRE_ARCHIVE_MATCH:-}" ]]; then
  cmd+=("agent.fs_norm_require_archive_match=${FS_NORM_REQUIRE_ARCHIVE_MATCH}")
fi
if [[ "${HYDRA_EXPERIMENT}" == *"fs_norm"* || "${HYDRA_EXPERIMENT}" == "pta_fs_dit_stage2_103k" ]]; then
  cmd+=("agent.fs_norm_target_clip=${FS_NORM_TARGET_CLIP:-0.0}")
  cmd+=("agent.fs_norm_output_clip=${FS_NORM_OUTPUT_CLIP:-12.0}")
  cmd+=("agent.fs_norm_output_clip_mode=${FS_NORM_OUTPUT_CLIP_MODE:-stats_bounds}")
fi
optional_overrides=(
  "CHECKPOINT_PATH:agent.checkpoint_path"
  "ALLOW_RANDOM_INIT:agent.allow_random_init"
  "LR:agent.lr"
  "SCHEDULER_EPOCHS:agent.scheduler_epochs"
  "SCHEDULER_WARMUP_EPOCHS:agent.scheduler_warmup_epochs"
  "MAX_EPOCHS:trainer.params.max_epochs"
  "USE_PLANNING_TOKEN_ADAPTER:agent.use_planning_token_adapter"
  "PLANNING_NUM_TOKENS:agent.planning_num_tokens"
  "PLANNING_NUM_HEADS:agent.planning_num_heads"
  "PLANNING_CONDITION_LAYERS:agent.planning_condition_layers"
  "PLANNING_GATE_INIT:agent.planning_gate_init"
  "PLANNING_CONTEXT_GATE_INIT:agent.planning_context_gate_init"
  "PLANNING_CONDITION_DROPOUT:agent.planning_condition_dropout"
  "TRAJECTORY_AUX_WEIGHT:agent.trajectory_aux_weight"
  "FEASIBILITY_AUX_WEIGHT:agent.feasibility_aux_weight"
  "AUX_ALPHA_POWER:agent.aux_alpha_power"
  "AUX_WARMUP_EPOCHS:agent.aux_warmup_epochs"
  "TANGENT_MARGIN_RAD:agent.tangent_margin_rad"
  "CURVATURE_MARGIN:agent.curvature_margin"
  "MIN_SEGMENT_LENGTH:agent.min_segment_length"
  "X0_AUX_WEIGHT:agent.x0_aux_weight"
  "DELTA_AUX_WEIGHT:agent.delta_aux_weight"
  "GEO_AUX_WEIGHT:agent.geo_aux_weight"
  "DPSI_TARGET_DISTRIBUTION:agent.offline_rl_dpsi_target_distribution"
  "DPSI_MODE_DENSITY_BANDWIDTH:agent.offline_rl_dpsi_mode_density_bandwidth"
  "DPSI_USE_SOURCE_WEIGHT:agent.offline_rl_dpsi_use_source_weight"
  "DPSI_USE_REWARD_MARGIN_WEIGHT:agent.offline_rl_dpsi_use_reward_margin_weight"
  "DPSI_SCENE_NORMALIZE_WEIGHTS:agent.offline_rl_dpsi_scene_normalize_weights"
  "DPSI_BETA_MAX:agent.offline_rl_dpsi_beta_max"
  "DPSI_BETA_WARMUP_EPOCHS:agent.offline_rl_dpsi_beta_warmup_epochs"
  "DPSI_PAIR_TARGET_RANDOMNESS:agent.offline_rl_dpsi_pair_target_randomness"
  "DPSI_RESIDUAL_BUDGET_ENABLED:agent.offline_rl_dpsi_residual_budget_enabled"
  "DPSI_NON_GT_RESIDUAL_MASS_CAP:agent.offline_rl_dpsi_non_gt_residual_mass_cap"
  "DPSI_FRONTIER_GT_ONLY_SCENE_WEIGHT:agent.offline_rl_dpsi_frontier_gt_only_scene_weight"
  "DPSI_FRONTIER_CURRICULUM_ENABLED:agent.offline_rl_dpsi_frontier_curriculum_enabled"
  "DPSI_FRONTIER_DIFFICULTY_START:agent.offline_rl_dpsi_frontier_difficulty_start"
  "DPSI_FRONTIER_DIFFICULTY_END:agent.offline_rl_dpsi_frontier_difficulty_end"
  "DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS:agent.offline_rl_dpsi_frontier_difficulty_warmup_epochs"
  "DPSI_TARGET_SAMPLE_M:agent.offline_rl_dpsi_target_sample_m"
  "DPSI_TARGET_SAMPLE_M_AFTER_WARMUP:agent.offline_rl_dpsi_target_sample_m_after_warmup"
  "DPSI_FRONTIER_SCENE_SAMPLING_ENABLED:stage2_frontier_sampling.enabled"
  "DPSI_FRONTIER_SCENE_INDEX_PATH:stage2_frontier_sampling.scene_index_path"
  "DPSI_FRONTIER_SCENE_UNIFORM_RATIO:stage2_frontier_sampling.uniform_ratio"
  "DPSI_FRONTIER_SCENE_PRIORITY_EXPONENT:stage2_frontier_sampling.priority_exponent"
  "DPSI_FRONTIER_SCENE_WARMUP_EPOCHS:stage2_frontier_sampling.warmup_epochs"
)
for mapping in "${optional_overrides[@]}"; do
  env_name="${mapping%%:*}"
  hydra_key="${mapping#*:}"
  if [[ -n "${!env_name:-}" ]]; then
    cmd+=("${hydra_key}=${!env_name}")
  fi
done
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  read -r -a extra_overrides <<<"${EXTRA_HYDRA_OVERRIDES}"
  cmd+=("${extra_overrides[@]}")
fi

{
  printf '[%s] SG_FPS_ASMI_DPSI=1 lr=%q max_epochs=%q scheduler_epochs=%q scheduler_warmup_epochs=%q micro_bs=%q nproc=%q accum=%q effective_bs=%q target_effective_bs=%q val_loss_disabled ' \
    "$(date -Is)" "${LR:-config}" "${MAX_EPOCHS:-config}" "${SCHEDULER_EPOCHS:-config}" \
    "${SCHEDULER_WARMUP_EPOCHS:-config}" "${MICRO_BATCH_SIZE}" "${NPROC_PER_NODE}" \
    "${ACCUMULATE_GRAD_BATCHES}" "${EFFECTIVE_BATCH_SIZE}" "${TARGET_EFFECTIVE_BATCH_SIZE}"
  printf 'A0_STAGE2_KEY_STEPS=%q ' "${KEY_STEPS}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "Starting SG-FPS ASMI Stage2. Log: ${TRAIN_LOG}"
RECOGDRIVE_KEY_EPOCH_INTERVAL="${KEY_EPOCH_INTERVAL}" A0_STAGE2_KEY_STEPS="${KEY_STEPS}" "${cmd[@]}" >"${TRAIN_LOG}" 2>&1
