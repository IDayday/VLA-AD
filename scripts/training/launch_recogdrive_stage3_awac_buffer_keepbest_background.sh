#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_awac_keepbest_buffer_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train_v2_keepbest}"

GPU_LIST="${GPU_LIST:-0}"
SHARD_COUNT="${SHARD_COUNT:-8}"
SHARD_INDICES="${SHARD_INDICES:-0}"
BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE:-1}"
ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES:-32}"
ELITE_TOP_M="${ELITE_TOP_M:-8}"
MERGE_KEEP_TOP_K="${MERGE_KEEP_TOP_K:-${ELITE_TOP_M}}"

CYCLES="${CYCLES:-0}"  # 0 means run until stopped.
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_MAX_MEM_USED_MB="${GPU_MAX_MEM_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"
VALIDATE_AFTER_BUFFER="${VALIDATE_AFTER_BUFFER:-0}"

IL_CHECKPOINT="${IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}"
AUTO_POLICY_CHECKPOINT_DIR="${AUTO_POLICY_CHECKPOINT_DIR:-}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${ELITE_BUFFER_DIR}"

gpu_busy_report() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -F, -v gpu_list="${GPU_LIST}" -v max_mem="${GPU_MAX_MEM_USED_MB}" -v max_util="${GPU_MAX_UTIL}" '
      BEGIN {
        n = split(gpu_list, wanted, ",");
        for (i = 1; i <= n; ++i) allow[wanted[i] + 0] = 1;
      }
      {
        idx = $1 + 0;
        mem = $2 + 0;
        util = $3 + 0;
        if (idx in allow && (mem > max_mem || util > max_util)) {
          printf("gpu=%d mem=%dMB util=%d%%\n", idx, mem, util);
        }
      }'
}

wait_for_gpus() {
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot wait for GPUs." >&2
    exit 3
  fi
  while true; do
    local blocked
    blocked="$(gpu_busy_report || true)"
    if [[ -z "${blocked}" ]]; then
      return 0
    fi
    {
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for GPUs ${GPU_LIST}"
      echo "${blocked}"
    } | tee -a "${OUT_ROOT}/gpu_wait.log"
    sleep "${GPU_WAIT_POLL_SECONDS}"
  done
}

resolve_policy_checkpoint() {
  if [[ -n "${AUTO_POLICY_CHECKPOINT_DIR}" ]]; then
    find "${AUTO_POLICY_CHECKPOINT_DIR}" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | awk 'NR == 1 {print $2}'
    return 0
  fi
  printf '%s\n' "${POLICY_CHECKPOINT}"
}

cycle=0
while true; do
  cycle=$((cycle + 1))
  if [[ "${CYCLES}" != "0" && "${cycle}" -gt "${CYCLES}" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) completed ${CYCLES} cycles." | tee -a "${OUT_ROOT}/keepbest_buffer.log"
    exit 0
  fi

  wait_for_gpus
  cycle_out="${OUT_ROOT}/cycle_$(printf '%04d' "${cycle}")"
  mkdir -p "${cycle_out}"
  resolved_policy="$(resolve_policy_checkpoint)"
  {
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "cycle=${cycle}"
    echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
    echo "gpu_list=${GPU_LIST}"
    echo "shard_count=${SHARD_COUNT}"
    echo "shard_indices=${SHARD_INDICES}"
    echo "online_policy_samples=${ONLINE_POLICY_SAMPLES}"
    echo "elite_top_m=${ELITE_TOP_M}"
    echo "merge_keep_top_k=${MERGE_KEEP_TOP_K}"
    echo "il_checkpoint=${IL_CHECKPOINT}"
    echo "policy_checkpoint=${resolved_policy}"
  } > "${cycle_out}/cycle_config.txt"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting keep-best buffer cycle=${cycle}" | tee -a "${OUT_ROOT}/keepbest_buffer.log"
  (
    cd "${REPO_ROOT}"
    env \
      RUN_NAME="${RUN_NAME}_cycle_${cycle}" \
      OUT_ROOT="${cycle_out}" \
      ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR}" \
      GPU_LIST="${GPU_LIST}" \
      SHARD_COUNT="${SHARD_COUNT}" \
      SHARD_INDICES="${SHARD_INDICES}" \
      BUFFER_BATCH_SIZE="${BUFFER_BATCH_SIZE}" \
      ONLINE_POLICY_SAMPLES="${ONLINE_POLICY_SAMPLES}" \
      ELITE_TOP_M="${ELITE_TOP_M}" \
      SKIP_EXISTING_RECORDS=0 \
      VALIDATE_EXISTING_RECORDS=1 \
      PREFILTER_EXISTING_RECORDS=0 \
      MERGE_EXISTING_RECORDS=1 \
      MERGE_KEEP_TOP_K="${MERGE_KEEP_TOP_K}" \
      MERGE_KEEP_SUPPORT=1 \
      RUN_BUFFER=1 \
      RUN_TRAIN_AFTER_BUFFER=0 \
      VALIDATE_AFTER_BUFFER="${VALIDATE_AFTER_BUFFER}" \
      DRY_RUN=0 \
      RUN_STAGE3=1 \
      IL_CHECKPOINT="${IL_CHECKPOINT}" \
      POLICY_CHECKPOINT="${resolved_policy}" \
      AWAC_USE_BATCHED_PDM_SCORING=true \
      AWAC_USE_EXACT_ARRAY_PDM_STATE_CONVERSION=true \
      AWAC_USE_FAST_PDM_SCORER=true \
      AWAC_PDM_SHADOW_CHECK=false \
      bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_awac_iql_full_pipeline_zt2.sh"
  ) > "${cycle_out}/pipeline.log" 2>&1
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) finished keep-best buffer cycle=${cycle}" | tee -a "${OUT_ROOT}/keepbest_buffer.log"
  if [[ "${CYCLES}" != "0" && "${cycle}" -ge "${CYCLES}" ]]; then
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done
