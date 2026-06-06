#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/highcap_no_risk_full_sft_remote/train_remote_retry_20260605T050837Z_peftgeneric_vggt12}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete/step_00100000.ckpt}"
MASTER_PORT="${MASTER_PORT:-29843}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

ROOT="${OUT_ROOT}/serverB_lora_highcap_no_risk"
B1="${ROOT}/vlm_lora_cot_alignment"
B2="${ROOT}/lora_hidden_cache_highcap_no_risk"
B3="${ROOT}/progressive_bottleneck"
LOG_DIR="${ROOT}/logs"
RESUME_CKPT="${RESUME_CKPT:-${B3}/lightning_logs/version_0/checkpoints/last.ckpt}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME_LOG="${LOG_DIR}/progressive_bottleneck_resume_${RUN_ID}.log"
SUMMARY_LOG="${LOG_DIR}/progressive_bottleneck_resume.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "Missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 2
fi

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  echo "Resuming B progressive from ${RESUME_CKPT}"
  echo "repo=${REPO_ROOT}"
  echo "root=${ROOT}"
  echo "cache=${B2}"
  echo "output=${B3}"
  echo "log=${RESUME_LOG}"
} >>"${SUMMARY_LOG}"

"${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}" \
  navsim/planning/script/run_training_recogdrive.py \
  +experiment=last_vla_progressive_bottleneck_highcap_no_risk \
  "+resume_ckpt_path=${RESUME_CKPT}" \
  "train_test_split=${TRAIN_TEST_SPLIT}" \
  "cache_path=${B2}" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  "output_dir=${B3}" \
  "agent.checkpoint_path=${A0_INIT_CHECKPOINT}" \
  "agent.last_vla_adapter_checkpoint=${B1}/adapters/last_vla_cot_adapter.pt" \
  agent.num_jepa_tokens=128 \
  agent.num_dynamic_tokens=128 \
  agent.num_vggt_tokens=12 \
  agent.num_geometry_tokens=192 \
  agent.num_risk_tokens=0 \
  agent.last_vla_cot_num_tokens=192 \
  agent.last_vla_vlm_summary_tokens=64 \
  agent.last_vla_use_risk_head=false \
  agent.last_vla_risk_loss_weight=0.0 \
  agent.last_vla_require_full_geometry=true \
  agent.last_vla_allow_patch_geometry_fallback=false \
  agent.last_vla_geometry_teacher_dim=512 \
  agent.last_vla_geometry_grid_rows=12 \
  agent.last_vla_geometry_grid_cols=16 \
  agent.last_vla_raw_vlm_context_to_dit=false \
  trainer.params.devices=8 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  >"${RESUME_LOG}" 2>&1
