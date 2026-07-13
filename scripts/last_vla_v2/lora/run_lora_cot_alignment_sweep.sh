#!/usr/bin/env bash
set -Eeuo pipefail

required=(NAVSIM_LOG_PATH SENSOR_BLOBS_PATH VLM_PATH OUT_ROOT MASTER_PORT_BASE)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

TEACHER_CACHE_ROOT="${EXPERT_TEACHER_CACHE_ROOT:-${TRAIN_CHUNK_CACHE_ROOT:-}}"
if [[ -z "${TEACHER_CACHE_ROOT}" ]]; then
  echo "Missing EXPERT_TEACHER_CACHE_ROOT or TRAIN_CHUNK_CACHE_ROOT" >&2
  exit 2
fi

if [[ "${RUN_TRAIN:-0}" == "1" || "${LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS:-0}" != "1" ]]; then
  [[ -d "${TEACHER_CACHE_ROOT}" ]] || { echo "Teacher cache root missing: ${TEACHER_CACHE_ROOT}" >&2; exit 2; }
  [[ -e "${VLM_PATH}" ]] || { echo "VLM_PATH missing: ${VLM_PATH}" >&2; exit 2; }
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
NUM_GPUS="${NUM_GPUS:-8}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
IFS=',' read -r -a configs <<<"${LORA_SWEEP_CONFIGS:-last_vla_vlm_lora_attention_only_r16,last_vla_vlm_lora_attention_mlp_r32,last_vla_vlm_lora_all_linear_r64}"

mkdir -p "${OUT_ROOT}"
for idx in "${!configs[@]}"; do
  config="$(echo "${configs[$idx]}" | xargs)"
  [[ -z "${config}" ]] && continue
  lora_preset="attention_mlp"
  lora_scope="llm"
  lora_r="32"
  lora_alpha="64"
  lora_dropout="0.05"
  lora_use_rslora="true"
  lora_hidden_anchor_weight="0.01"
  lr_lora="1e-5"
  case "${config}" in
    *attention_only_r16*)
      lora_preset="attention_only"
      lora_r="16"
      lora_alpha="32"
      lora_use_rslora="false"
      ;;
    *attention_mlp_r32*)
      lora_preset="attention_mlp"
      lora_r="32"
      lora_alpha="64"
      lora_use_rslora="true"
      ;;
    *all_linear_r64*)
      lora_preset="all_linear"
      lora_r="64"
      lora_alpha="128"
      lora_use_rslora="true"
      lora_hidden_anchor_weight="0.02"
      lr_lora="5e-6"
      ;;
  esac
  run_dir="${OUT_ROOT}/${config}"
  mkdir -p "${run_dir}/logs"
  commands_log="${run_dir}/commands.log"
  port="$((MASTER_PORT_BASE + idx))"
  cmd_train=(
    "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" --master_port "${port}"
    navsim/planning/script/run_training_recogdrive.py
    "+experiment=${config}"
    "train_test_split=${TRAIN_TEST_SPLIT}"
    cache_path=null
    use_cache_without_dataset=false
    force_cache_computation=false
    "navsim_log_path=${NAVSIM_LOG_PATH}"
    "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
    "output_dir=${run_dir}"
    agent.cache_hidden_state=false
    agent.train_backbone=true
    agent.last_vla_train_vlm_lora=true
    "agent.last_vla_vlm_lora_preset=${lora_preset}"
    "agent.last_vla_vlm_lora_scope=${lora_scope}"
    "agent.last_vla_vlm_lora_r=${lora_r}"
    "agent.last_vla_vlm_lora_alpha=${lora_alpha}"
    "agent.last_vla_vlm_lora_dropout=${lora_dropout}"
    "agent.last_vla_vlm_lora_use_rslora=${lora_use_rslora}"
    agent.last_vla_vlm_lora_use_dora=false
    "agent.last_vla_hidden_anchor_weight=${lora_hidden_anchor_weight}"
    agent.last_vla_hidden_anchor_mode=summary_cosine
    "agent.lr_vlm_lora=${lr_lora}"
    agent.lr_last_vla_cot=1e-4
    agent.use_expert_features=true
    agent.expert_feature_source=chunk
    "agent.expert_cache_dir=${TEACHER_CACHE_ROOT}"
    "agent.vlm_path=${VLM_PATH}"
    "agent.vlm_type=${VLM_TYPE:-internvl}"
    trainer.params.devices="${NUM_GPUS}"
    trainer.params.strategy=ddp_find_unused_parameters_true
  )
  if [[ -n "${MAX_EPOCHS:-}" ]]; then cmd_train+=(trainer.params.max_epochs="${MAX_EPOCHS}"); fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd_train+=(dataloader.params.max_samples="${MAX_SAMPLES}"); fi
  cmd_extract=(
    "${PYTHON_BIN}" scripts/last_vla_v2/extract_vlm_lora_and_cot_adapters.py
    --checkpoint "${run_dir}/latest.ckpt"
    --output-dir "${run_dir}/adapters"
    --base-vlm-path "${VLM_PATH}"
    --vlm-type "${VLM_TYPE:-internvl}"
    --preset "${lora_preset}"
    --scope "${lora_scope}"
    --r "${lora_r}"
    --alpha "${lora_alpha}"
    --dropout "${lora_dropout}"
    --bias none
    --lora-target-report "${run_dir}/lora_target_report.json"
    --lora-training-config "${run_dir}/lora_training_config.json"
  )
  if [[ "${lora_use_rslora}" == "true" ]]; then
    cmd_extract+=(--use-rslora)
  else
    cmd_extract+=(--no-rslora)
  fi
  {
    date -Is
    printf '%q ' "${cmd_train[@]}"; printf '\n'
    printf '%q ' "${cmd_extract[@]}"; printf '\n'
  } >>"${commands_log}"
  printf '%q ' "${cmd_train[@]}"; printf '\n'
  printf '%q ' "${cmd_extract[@]}"; printf '\n'
  if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
    "${cmd_train[@]}" >"${run_dir}/logs/train.log" 2>&1
    "${cmd_extract[@]}" >"${run_dir}/logs/extract_adapters.log" 2>&1
  fi
done

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written under ${OUT_ROOT}"
fi
