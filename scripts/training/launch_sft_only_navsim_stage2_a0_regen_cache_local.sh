#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"

RECOGDRIVE_VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD_last_vla_dev/outputs/sft_only_navsim/hf_eval_model}"
CACHE_PATH="${CACHE_PATH:-/mnt/project/VLA-AD/cache/sft_only_navsim_stage1_hidden_navtrain_2b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/project/VLA-AD/outputs/sft_only_navsim_stage2_a0_official_aligned_20260707}"
CACHE_OUTPUT_DIR="${CACHE_OUTPUT_DIR:-${OUTPUT_ROOT}/cache_full}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/a0_official_aligned_sftonly_navsim_stage1}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-/mnt/navsim/trainval_sensor_blobs/trainval}"

CACHE_MASTER_PORT="${CACHE_MASTER_PORT:-29671}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29672}"
SEED="${SEED:-0}"
KEY_STEPS="${KEY_STEPS:-50000,60000,80000,100000,120000,140000,160000}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD/cache}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(pwd)}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/navsim}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${RECOGDRIVE_VLM_PATH}/model.safetensors" ]]; then
  echo "Missing Stage1 model.safetensors under RECOGDRIVE_VLM_PATH: ${RECOGDRIVE_VLM_PATH}" >&2
  exit 2
fi
if [[ ! -d "${NAVSIM_LOG_PATH}" ]]; then
  echo "Missing NAVSIM_LOG_PATH: ${NAVSIM_LOG_PATH}" >&2
  exit 2
fi
if [[ ! -d "${SENSOR_BLOBS_PATH}" ]]; then
  echo "Missing SENSOR_BLOBS_PATH: ${SENSOR_BLOBS_PATH}" >&2
  exit 2
fi
if [[ -d "${CACHE_PATH}" ]] && find "${CACHE_PATH}" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${ALLOW_EXISTING_CACHE:-0}" != "1" ]]; then
    echo "Refusing to reuse existing cache: ${CACHE_PATH}" >&2
    echo "Use a fresh CACHE_PATH, or set ALLOW_EXISTING_CACHE=1 only for intentional resume." >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_ROOT}" "${CACHE_OUTPUT_DIR}"
COMMANDS_LOG="${OUTPUT_ROOT}/commands.log"
CACHE_LOG="${CACHE_OUTPUT_DIR}/cache_full.log"

cache_cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${CACHE_MASTER_PORT}"
  navsim/planning/script/run_dataset_caching_multi_node.py
  agent=recogdrive_agent
  experiment_name=sft_only_navsim_stage1_hidden_cache
  agent.cam_type=single
  agent.cache_hidden_state=True
  agent.cache_mode=True
  agent.vlm_type=internvl
  "agent.vlm_path=${RECOGDRIVE_VLM_PATH}"
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  agent.use_expert_features=False
  agent.use_jepa=False
  agent.use_vggt=False
  "train_test_split=${TRAIN_TEST_SPLIT}"
  "navsim_log_path=${NAVSIM_LOG_PATH}"
  "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  "cache_path=${CACHE_PATH}"
  force_cache_computation=False
  "output_dir=${CACHE_OUTPUT_DIR}"
)

{
  printf '[%s] regenerate_hidden_cache ' "$(date -Is)"
  printf '%q ' "${cache_cmd[@]}"
  printf '\n'
} >>"${COMMANDS_LOG}"

echo "Regenerating hidden cache with NavSim-only Stage1 VLM. Log: ${CACHE_LOG}"
"${cache_cmd[@]}" >"${CACHE_LOG}" 2>&1

feature_count="$(find "${CACHE_PATH}" -name internvl_feature.gz | wc -l | tr -d ' ')"
target_count="$(find "${CACHE_PATH}" -name trajectory_target.gz | wc -l | tr -d ' ')"
echo "Generated cache files: internvl_feature=${feature_count}, trajectory_target=${target_count}" | tee -a "${CACHE_LOG}"
if [[ "${feature_count}" == "0" || "${target_count}" == "0" ]]; then
  echo "Cache generation produced no usable feature/target files." >&2
  exit 2
fi

echo "Starting A0 official-aligned Stage2 training from regenerated cache."
RECOGDRIVE_VLM_PATH="${RECOGDRIVE_VLM_PATH}" \
CACHE_PATH="${CACHE_PATH}" \
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT}" \
OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
MASTER_PORT="${TRAIN_MASTER_PORT}" \
SEED="${SEED}" \
KEY_STEPS="${KEY_STEPS}" \
PYTHON_BIN="${PYTHON_BIN}" \
TORCHRUN_BIN="${TORCHRUN_BIN}" \
bash scripts/run_a0_official_aligned_8gpu.sh
