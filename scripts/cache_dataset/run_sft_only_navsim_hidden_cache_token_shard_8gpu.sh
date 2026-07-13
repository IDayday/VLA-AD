#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${SHARD_INDEX:-}" || -z "${SHARD_COUNT:-}" ]]; then
  echo "Set SHARD_INDEX and SHARD_COUNT." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
RECOGDRIVE_VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD_last_vla_dev/outputs/sft_only_navsim/hf_eval_model}"
CACHE_PATH="${CACHE_PATH:-/mnt/project/VLA-AD/cache/sft_only_navsim_stage1_hidden_navtrain_2b}"
TOKEN_MANIFEST="${TOKEN_MANIFEST:-/mnt/project/VLA-AD/outputs/sft_only_navsim_stage2_a0_official_aligned_20260707/cache_full/navtrain_token_manifest.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/project/VLA-AD/outputs/sft_only_navsim_stage2_a0_official_aligned_20260707}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/cache_full/sharded}"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
MASTER_PORT="${MASTER_PORT:-29680}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-/mnt/navsim/trainval_sensor_blobs/trainval}"

if [[ "${CACHE_PATH}" == *recogdrive_official* ]]; then
  echo "Refusing to write this experiment into an official ReCogDrive cache path: ${CACHE_PATH}" >&2
  exit 2
fi

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/mnt/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD/cache}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(pwd)}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/mnt/navsim}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NAVSIM_DISABLE_TQDM="${NAVSIM_DISABLE_TQDM:-1}"

mkdir -p "${LOG_DIR}"
log_path="${LOG_DIR}/cache_shard_${SHARD_INDEX}_of_${SHARD_COUNT}_${HOST_TAG}.log"
done_path="${LOG_DIR}/cache_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.done"
fail_path="${LOG_DIR}/cache_shard_${SHARD_INDEX}_of_${SHARD_COUNT}.fail"
rm -f "${done_path}" "${fail_path}"

on_error() {
  echo "failed_at=$(date -Is)" >"${fail_path}"
}
trap on_error ERR

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_dataset_caching_multi_node.py
  agent=recogdrive_agent
  experiment_name=sft_only_navsim_stage1_hidden_cache_sharded
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
  "output_dir=${LOG_DIR}/hydra_shard_${SHARD_INDEX}"
  "+cache_token_manifest=${TOKEN_MANIFEST}"
  "+cache_token_shard_count=${SHARD_COUNT}"
  "+cache_token_shard_index=${SHARD_INDEX}"
)

{
  printf '[%s] host=%s shard=%s/%s ' "$(date -Is)" "${HOST_TAG}" "${SHARD_INDEX}" "${SHARD_COUNT}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >>"${LOG_DIR}/commands.log"

echo "Starting cache shard ${SHARD_INDEX}/${SHARD_COUNT} on ${HOST_TAG}. Log: ${log_path}"
"${cmd[@]}" >"${log_path}" 2>&1
echo "done_at=$(date -Is)" >"${done_path}"
trap - ERR
