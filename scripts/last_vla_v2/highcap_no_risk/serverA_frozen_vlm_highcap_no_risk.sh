#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_GEOMETRY_CHUNK_ROOT A0_INIT_CHECKPOINT OUT_ROOT MASTER_PORT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
ROOT="${OUT_ROOT}/serverA_frozen_highcap_no_risk"
mkdir -p "${ROOT}/logs"
COMMANDS_LOG="${ROOT}/commands.log"

A1="${ROOT}/cot_alignment"
A2="${ROOT}/progressive_bottleneck"
mkdir -p "${A1}" "${A2}"

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

cmd_a1=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_cot_alignment_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache_path=${FULL_GEOMETRY_CHUNK_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "output_dir=${A1}"
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  "${common_overrides[@]}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
cmd_a2=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "$((MASTER_PORT + 1))"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_progressive_bottleneck_highcap_no_risk
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "cache_path=${FULL_GEOMETRY_CHUNK_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "output_dir=${A2}"
  agent.checkpoint_path="${A0_INIT_CHECKPOINT}"
  agent.last_vla_adapter_checkpoint="${COT_ALIGNMENT_CHECKPOINT:-${A1}/latest.ckpt}"
  "${common_overrides[@]}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)

{
  date -Is
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf '%q ' "${cmd_a1[@]}"; printf '\n'
  printf '%q ' "${cmd_a2[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

cat >"${ROOT}/checkpoints_to_eval.txt" <<EOF
${A2}/step_00050000.ckpt
${A2}/step_00060000.ckpt
${A2}/step_00080000.ckpt
${A2}/step_00100000.ckpt
${A2}/step_00120000.ckpt
${A2}/latest.ckpt
EOF

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
[[ -d "${FULL_GEOMETRY_CHUNK_ROOT}" ]] || { echo "FULL_GEOMETRY_CHUNK_ROOT missing: ${FULL_GEOMETRY_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }
"${cmd_a1[@]}" >"${ROOT}/logs/cot_alignment.log" 2>&1
"${cmd_a2[@]}" >"${ROOT}/logs/progressive_bottleneck.log" 2>&1
