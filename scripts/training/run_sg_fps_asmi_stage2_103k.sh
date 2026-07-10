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
if [[ "${HYDRA_EXPERIMENT}" == *"fs_norm"* || "${HYDRA_EXPERIMENT}" == "pta_fs_dit_stage2_103k" ]]; then
  cmd+=("agent.fs_norm_target_clip=${FS_NORM_TARGET_CLIP:-0.0}")
  cmd+=("agent.fs_norm_output_clip=${FS_NORM_OUTPUT_CLIP:-12.0}")
  cmd+=("agent.fs_norm_output_clip_mode=${FS_NORM_OUTPUT_CLIP_MODE:-stats_bounds}")
fi
optional_overrides=(
  "USE_PLANNING_TOKEN_ADAPTER:agent.use_planning_token_adapter"
  "PLANNING_TOKEN_SOURCE:agent.planning_token_source"
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
  printf '[%s] RANDOM_INIT_DIT=1 SG_FPS_ASMI_DPSI_FS_X0_DELTA_GEO=1 lr=1e-4 micro_bs=%q nproc=%q accum=%q effective_bs=%q target_effective_bs=%q 200ep warmup3 minlr1e-6 val_loss_disabled ' "$(date -Is)" "${MICRO_BATCH_SIZE}" "${NPROC_PER_NODE}" "${ACCUMULATE_GRAD_BATCHES}" "${EFFECTIVE_BATCH_SIZE}" "${TARGET_EFFECTIVE_BATCH_SIZE}"
  printf 'A0_STAGE2_KEY_STEPS=%q ' "${KEY_STEPS}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "Starting SG-FPS ASMI Stage2. Log: ${TRAIN_LOG}"
RECOGDRIVE_KEY_EPOCH_INTERVAL="${KEY_EPOCH_INTERVAL}" A0_STAGE2_KEY_STEPS="${KEY_STEPS}" "${cmd[@]}" >"${TRAIN_LOG}" 2>&1
