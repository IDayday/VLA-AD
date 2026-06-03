#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_GEOMETRY_CHUNK_ROOT A0_INIT_CHECKPOINT OUT_ROOT MASTER_PORT VLM_PATH NAVSIM_LOG_PATH SENSOR_BLOBS_PATH)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
[[ -d "${FULL_GEOMETRY_CHUNK_ROOT}" ]] || { echo "FULL_GEOMETRY_CHUNK_ROOT missing: ${FULL_GEOMETRY_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
ROOT="${OUT_ROOT}/serverB_lora"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_hidden_cache"
B3="${ROOT}/progressive_bottleneck"
mkdir -p "${ROOT}/logs" "${B1}" "${B2}" "${B3}"
COMMANDS_LOG="${ROOT}/commands.log"

cmd_b1=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_vlm_lora_cot_alignment
  use_cache_without_dataset=false
  force_cache_computation=false
  "output_dir=${B1}"
  agent.cache_hidden_state=false
  agent.train_backbone=true
  agent.last_vla_train_vlm_lora=true
  agent.vlm_path="${VLM_PATH}"
  agent.vlm_type="${VLM_TYPE:-internvl}"
  agent.use_expert_features=true
  agent.expert_feature_source=chunk
  agent.expert_cache_dir="${FULL_GEOMETRY_CHUNK_ROOT}"
  agent.allow_expert_target_features=true
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  agent.last_vla_require_full_geometry=true
  agent.last_vla_allow_patch_geometry_fallback=false
  agent.last_vla_geometry_teacher_dim="${GEOMETRY_TEACHER_DIM:-512}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
cmd_extract=(
  "${PYTHON_BIN}" scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py
  --checkpoint "${B1}/latest.ckpt"
  --output-dir "${B1}/adapters"
)
cmd_cache=(
  "${PYTHON_BIN}" scripts/build_recogdrive_hidden_cache_with_lora.py
  --base-chunk-root "${FULL_GEOMETRY_CHUNK_ROOT}"
  --output-chunk-root "${B2}"
  --vlm-path "${VLM_PATH}"
  --vlm-type "${VLM_TYPE:-internvl}"
  --vlm-lora-adapter "${B1}/adapters/vlm_lora_adapter_state.pt"
  --precision "${PRECISION:-bf16}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --batch-size "${LORA_CACHE_BATCH_SIZE:-1}"
  --shard-index "${SHARD_INDEX:-0}"
  --num-shards "${NUM_SHARDS:-1}"
)
cmd_b3=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "$((MASTER_PORT + 1))"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_progressive_bottleneck
  "cache_path=${B2}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "output_dir=${B3}"
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  agent.last_vla_adapter_checkpoint="${B1}/adapters/last_vla_cot_adapter.pt"
  agent.last_vla_require_full_geometry=true
  agent.last_vla_allow_patch_geometry_fallback=false
  agent.last_vla_geometry_teacher_dim="${GEOMETRY_TEACHER_DIM:-512}"
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

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi

"${cmd_b1[@]}" >"${ROOT}/logs/vlm_lora_cot_alignment.log" 2>&1
"${cmd_extract[@]}" >"${ROOT}/logs/extract_adapters.log" 2>&1
"${cmd_cache[@]}" >"${ROOT}/logs/regenerate_hidden_cache.log" 2>&1
"${cmd_b3[@]}" >"${ROOT}/logs/progressive_bottleneck.log" 2>&1
