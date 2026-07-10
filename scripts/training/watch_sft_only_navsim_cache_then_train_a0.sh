#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
RECOGDRIVE_VLM_PATH="${RECOGDRIVE_VLM_PATH:-/mnt/project/VLA-AD_last_vla_dev/outputs/sft_only_navsim/hf_eval_model}"
CACHE_PATH="${CACHE_PATH:-/mnt/project/VLA-AD/cache/sft_only_navsim_stage1_hidden_navtrain_2b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/project/VLA-AD/outputs/sft_only_navsim_stage2_a0_official_aligned_20260707}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/cache_full/sharded}"
TOKEN_MANIFEST="${TOKEN_MANIFEST:-${OUTPUT_ROOT}/cache_full/navtrain_token_manifest.tsv}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${OUTPUT_ROOT}/a0_official_aligned_sftonly_navsim_stage1}"
SHARD_COUNT="${SHARD_COUNT:-4}"
POLL_SECONDS="${POLL_SECONDS:-300}"

TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29672}"
SEED="${SEED:-0}"
KEY_STEPS="${KEY_STEPS:-50000,60000,80000,100000,120000,140000,160000}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
watch_log="${OUTPUT_ROOT}/cache_then_train_watcher.log"

echo "Watcher started at $(date -Is), waiting for ${SHARD_COUNT} cache shards." | tee -a "${watch_log}"

while true; do
  fail_count="$(find "${LOG_DIR}" -maxdepth 1 -name 'cache_shard_*_of_*.fail' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${fail_count}" != "0" ]]; then
    echo "Detected failed cache shard(s):" | tee -a "${watch_log}"
    find "${LOG_DIR}" -maxdepth 1 -name 'cache_shard_*_of_*.fail' -print | sort | tee -a "${watch_log}"
    exit 1
  fi

  done_count=0
  for idx in $(seq 0 $((SHARD_COUNT - 1))); do
    if [[ -f "${LOG_DIR}/cache_shard_${idx}_of_${SHARD_COUNT}.done" ]]; then
      done_count=$((done_count + 1))
    fi
  done
  echo "[$(date -Is)] cache shard done ${done_count}/${SHARD_COUNT}" | tee -a "${watch_log}"
  if [[ "${done_count}" == "${SHARD_COUNT}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

expected_count="$(wc -l <"${TOKEN_MANIFEST}" | tr -d ' ')"
feature_count="$(find "${CACHE_PATH}" -name internvl_feature.gz | wc -l | tr -d ' ')"
target_count="$(find "${CACHE_PATH}" -name trajectory_target.gz | wc -l | tr -d ' ')"
{
  echo "Cache shards complete at $(date -Is)."
  echo "expected=${expected_count} internvl_feature=${feature_count} trajectory_target=${target_count}"
} | tee -a "${watch_log}"

if [[ "${feature_count}" -lt "${expected_count}" || "${target_count}" -lt "${expected_count}" ]]; then
  echo "Cache count is incomplete; refusing to start Stage2 training." | tee -a "${watch_log}"
  exit 1
fi

echo "Starting A0 official-aligned Stage2 training at $(date -Is)." | tee -a "${watch_log}"
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
