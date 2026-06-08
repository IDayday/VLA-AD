#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_HIGHCAP_TRAIN_CHUNK_ROOT OUT_ROOT MASTER_PORT VLM_PATH NAVSIM_LOG_PATH SENSOR_BLOBS_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
export NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
ROOT="${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_regenerated_train_hidden_cache"
B4="${ROOT}/progressive_sft_decoupled"
mkdir -p "${ROOT}/logs" "${B1}" "${B2}" "${B4}"
COMMANDS_LOG="${ROOT}/commands.log"

LORA_PRESET="${LORA_PRESET:-attention_mlp}"
LORA_SCOPE="${LORA_SCOPE:-llm}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_BIAS="${LORA_BIAS:-none}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-true}"
LORA_USE_DORA="${LORA_USE_DORA:-false}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_HIDDEN_ANCHOR_EVERY_N_STEPS="${LORA_HIDDEN_ANCHOR_EVERY_N_STEPS:-4}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
B1_BATCH_SIZE="${B1_BATCH_SIZE:-4}"
B1_ACCUMULATE_GRAD_BATCHES="${B1_ACCUMULATE_GRAD_BATCHES:-}"
B1_EPOCHS="${B1_EPOCHS:-5}"
B1_KEY_EPOCHS="${B1_KEY_EPOCHS:-2,3,5}"

common_overrides=(
  agent.last_vla_condition_mode=decoupled_cot_residual
  agent.last_vla_cot_bottleneck_mode=false
  agent.last_vla_raw_vlm_context_to_dit=true
  agent.last_vla_use_risk_head=false
  agent.num_jepa_tokens=128
  agent.use_vggt=false
  agent.num_dynamic_tokens=128
  agent.num_vggt_tokens=128
  agent.num_geometry_tokens=192
  agent.num_risk_tokens=0
  agent.last_vla_cot_num_tokens=192
  agent.last_vla_cot_num_steps=5
  agent.last_vla_require_full_geometry=true
  agent.last_vla_allow_patch_geometry_fallback=false
  agent.last_vla_geometry_teacher_dim=512
  agent.last_vla_geometry_grid_rows=12
  agent.last_vla_geometry_grid_cols=16
)
validation_overrides=()
if [[ "${SKIP_VALIDATION}" == "1" || "${SKIP_VALIDATION}" == "true" ]]; then
  validation_overrides+=(trainer.params.limit_val_batches=0 trainer.params.check_val_every_n_epoch=999999)
fi
b1_overrides=()
if [[ -n "${B1_EPOCHS:-}" ]]; then b1_overrides+=(trainer.params.max_epochs="${B1_EPOCHS}"); fi
if [[ -n "${B1_ACCUMULATE_GRAD_BATCHES}" ]]; then
  b1_overrides+=(trainer.params.accumulate_grad_batches="${B1_ACCUMULATE_GRAD_BATCHES}")
fi

cmd_b1=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-8}" --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_decoupled_vlm_lora_cot_alignment_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  cache_path=null
  use_cache_without_dataset=false
  force_cache_computation=false
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "output_dir=${B1}"
  agent.cache_hidden_state=false
  agent.train_backbone=true
  agent.vlm_path="${VLM_PATH}"
  agent.vlm_type="${VLM_TYPE:-internvl}"
  agent.use_expert_features=true
  agent.expert_feature_source=chunk
  agent.expert_cache_dir="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
  agent.allow_expert_target_features=true
  agent.last_vla_train_vlm_lora=true
  agent.last_vla_vlm_lora_preset="${LORA_PRESET}"
  agent.last_vla_vlm_lora_scope="${LORA_SCOPE}"
  agent.last_vla_vlm_lora_r="${LORA_R}"
  agent.last_vla_vlm_lora_alpha="${LORA_ALPHA}"
  agent.last_vla_vlm_lora_dropout="${LORA_DROPOUT}"
  agent.last_vla_vlm_lora_bias="${LORA_BIAS}"
  agent.last_vla_vlm_lora_use_rslora="${LORA_USE_RSLORA}"
  agent.last_vla_vlm_lora_use_dora="${LORA_USE_DORA}"
  agent.last_vla_hidden_anchor_every_n_steps="${LORA_HIDDEN_ANCHOR_EVERY_N_STEPS}"
  "${common_overrides[@]}"
  "${validation_overrides[@]}"
  "${b1_overrides[@]}"
  dataloader.params.batch_size="${B1_BATCH_SIZE}"
  trainer.params.devices="${NPROC_PER_NODE:-8}"
  trainer.params.strategy=ddp_find_unused_parameters_true
)
if [[ -n "${LORA_TARGET_MODULES}" ]]; then
  cmd_b1+=(agent.last_vla_vlm_lora_target_modules="${LORA_TARGET_MODULES}")
fi
cmd_extract=(
  "${PYTHON_BIN}" scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py
  --checkpoint "${B1}/latest.ckpt"
  --output-dir "${B1}/adapters"
  --base-vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --preset "${LORA_PRESET}"
  --scope "${LORA_SCOPE}"
  --r "${LORA_R}"
  --alpha "${LORA_ALPHA}"
  --dropout "${LORA_DROPOUT}"
  --bias "${LORA_BIAS}"
  --target-modules "${LORA_TARGET_MODULES}"
  --lora-target-report "${B1}/lora_target_report.json"
  --lora-training-config "${B1}/lora_training_config.json"
)
if [[ "${LORA_USE_RSLORA}" == "true" || "${LORA_USE_RSLORA}" == "1" ]]; then
  cmd_extract+=(--use-rslora)
else
  cmd_extract+=(--no-rslora)
fi
if [[ "${LORA_USE_DORA}" == "true" || "${LORA_USE_DORA}" == "1" ]]; then cmd_extract+=(--use-dora); fi

cmd_cache=(
  "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py
  --base-chunk-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
  --output-chunk-root "${B2}"
  --vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --vlm-lora-adapter-dir "${B1}/adapters/vlm_lora"
  --precision "${PRECISION:-bf16}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --batch-size "${LORA_CACHE_BATCH_SIZE:-1}"
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
  --cache-variant decoupled_highcap_no_risk
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_cache+=(--max-samples "${MAX_SAMPLES}"); fi

cmd_b4=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE:-8}" --master_port "$((MASTER_PORT + 1))"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_decoupled_progressive_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache_path=${B2}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "output_dir=${B4}"
  agent.last_vla_adapter_checkpoint="${B1}/adapters/last_vla_cot_adapter.pt"
  "${common_overrides[@]}"
  "${validation_overrides[@]}"
  trainer.params.devices="${NPROC_PER_NODE:-8}"
  trainer.params.strategy=ddp_find_unused_parameters_true
)

{
  date -Is
  printf 'RECOGDRIVE_KEY_EPOCHS=%q ' "${B1_KEY_EPOCHS}"
  printf '%q ' "${cmd_b1[@]}"; printf '\n'
  printf '%q ' "${cmd_extract[@]}"; printf '\n'
  printf '%q ' "${cmd_cache[@]}"; printf '\n'
  printf '%q ' "${cmd_b4[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

cat >"${ROOT}/checkpoints_to_eval.txt" <<EOF
${B4}/step_00050000.ckpt
${B4}/step_00060000.ckpt
${B4}/step_00080000.ckpt
${B4}/step_00100000.ckpt
${B4}/step_00120000.ckpt
${B4}/latest.ckpt
EOF

cat >"${ROOT}/eval_warning.txt" <<EOF
Line B navtest must use a LoRA-regenerated hidden cache, not the original frozen-VLM navtest cache.
EOF

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
[[ -d "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]] || { echo "FULL_HIGHCAP_TRAIN_CHUNK_ROOT missing: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" >&2; exit 2; }
RECOGDRIVE_KEY_EPOCHS="${B1_KEY_EPOCHS}" "${cmd_b1[@]}" >"${ROOT}/logs/vlm_lora_cot_alignment.log" 2>&1
"${cmd_extract[@]}" >"${ROOT}/logs/extract_adapters.log" 2>&1
if [[ "${STOP_AFTER_EXTRACT:-0}" == "1" ]]; then
  echo "STOP_AFTER_EXTRACT=1; completed LoRA alignment and adapter extraction." >>"${COMMANDS_LOG}"
  exit 0
fi
"${cmd_cache[@]}" >"${ROOT}/logs/regenerate_hidden_cache.log" 2>&1
"${cmd_b4[@]}" >"${ROOT}/logs/progressive_sft_decoupled.log" 2>&1
