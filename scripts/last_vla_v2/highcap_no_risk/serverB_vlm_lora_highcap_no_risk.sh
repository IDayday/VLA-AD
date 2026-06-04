#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_GEOMETRY_CHUNK_ROOT A0_INIT_CHECKPOINT OUT_ROOT MASTER_PORT VLM_PATH NAVSIM_LOG_PATH SENSOR_BLOBS_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
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
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
ROOT="${OUT_ROOT}/serverB_lora_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_hidden_cache_highcap_no_risk"
B3="${ROOT}/progressive_bottleneck"
mkdir -p "${ROOT}/logs" "${B1}" "${B2}" "${B3}"
COMMANDS_LOG="${ROOT}/commands.log"

common_overrides=(
  agent.num_jepa_tokens=128
  agent.num_dynamic_tokens=128
  agent.num_vggt_tokens=128
  agent.num_geometry_tokens=192
  agent.num_risk_tokens=0
  agent.last_vla_cot_num_tokens=192
  agent.last_vla_vlm_summary_tokens=64
  agent.last_vla_use_risk_head=false
  agent.last_vla_risk_loss_weight=0.0
  agent.last_vla_require_full_geometry=true
  agent.last_vla_allow_patch_geometry_fallback=false
  agent.last_vla_geometry_teacher_dim=512
  agent.last_vla_geometry_grid_rows=12
  agent.last_vla_geometry_grid_cols=16
  agent.last_vla_raw_vlm_context_to_dit=false
)

cmd_b1=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_vlm_lora_cot_alignment_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  cache_path=null
  use_cache_without_dataset=false
  force_cache_computation=false
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "output_dir=${B1}"
  agent.cache_hidden_state=false
  agent.train_backbone=true
  agent.last_vla_train_vlm_lora=true
  agent.last_vla_vlm_lora_preset="${LORA_PRESET}"
  agent.last_vla_vlm_lora_scope="${LORA_SCOPE}"
  agent.last_vla_vlm_lora_r="${LORA_R}"
  agent.last_vla_vlm_lora_alpha="${LORA_ALPHA}"
  agent.last_vla_vlm_lora_dropout="${LORA_DROPOUT}"
  agent.last_vla_vlm_lora_bias="${LORA_BIAS}"
  agent.last_vla_vlm_lora_use_rslora="${LORA_USE_RSLORA}"
  agent.last_vla_vlm_lora_use_dora="${LORA_USE_DORA}"
  "agent.last_vla_vlm_lora_target_modules='${LORA_TARGET_MODULES}'"
  agent.last_vla_hidden_anchor_every_n_steps="${LORA_HIDDEN_ANCHOR_EVERY_N_STEPS}"
  agent.vlm_path="${VLM_PATH}"
  agent.vlm_type="${VLM_TYPE:-internvl}"
  agent.use_expert_features=true
  agent.expert_feature_source=chunk
  agent.expert_cache_dir="${FULL_GEOMETRY_CHUNK_ROOT}"
  agent.allow_expert_target_features=true
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  "${common_overrides[@]}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
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
if [[ "${LORA_USE_DORA}" == "true" || "${LORA_USE_DORA}" == "1" ]]; then
  cmd_extract+=(--use-dora)
fi
cmd_cache=(
  "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py
  --base-chunk-root "${FULL_GEOMETRY_CHUNK_ROOT}"
  --output-chunk-root "${B2}"
  --vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --vlm-lora-adapter-dir "${B1}/adapters/vlm_lora"
  --precision "${PRECISION:-bf16}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --batch-size "${LORA_CACHE_BATCH_SIZE:-1}"
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
  --cache-variant highcap_no_risk
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_cache+=(--max-samples "${MAX_SAMPLES}"); fi
cmd_b3=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "$((MASTER_PORT + 1))"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_progressive_bottleneck_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache_path=${B2}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "output_dir=${B3}"
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  agent.last_vla_adapter_checkpoint="${B1}/adapters/last_vla_cot_adapter.pt"
  "${common_overrides[@]}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)

{
  date -Is
  printf '%q ' "${cmd_b1[@]}"; printf '\n'
  printf '%q ' "${cmd_extract[@]}"; printf '\n'
  printf '%q ' "${cmd_cache[@]}"; printf '\n'
  printf '%q ' "${cmd_b3[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

cat >"${ROOT}/checkpoints_to_eval.txt" <<EOF
${B3}/step_00050000.ckpt
${B3}/step_00060000.ckpt
${B3}/step_00080000.ckpt
${B3}/step_00100000.ckpt
${B3}/step_00120000.ckpt
${B3}/latest.ckpt
EOF

cat >"${ROOT}/eval_warning.txt" <<EOF
Line B eval must use a LoRA-regenerated navtest hidden cache. Do not evaluate Line B on the original navtest hidden cache.
EOF

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
[[ -d "${FULL_GEOMETRY_CHUNK_ROOT}" ]] || { echo "FULL_GEOMETRY_CHUNK_ROOT missing: ${FULL_GEOMETRY_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }
"${cmd_b1[@]}" >"${ROOT}/logs/vlm_lora_cot_alignment.log" 2>&1
"${cmd_extract[@]}" >"${ROOT}/logs/extract_adapters.log" 2>&1
"${cmd_cache[@]}" >"${ROOT}/logs/regenerate_hidden_cache.log" 2>&1
"${cmd_b3[@]}" >"${ROOT}/logs/progressive_bottleneck.log" 2>&1
