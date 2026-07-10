#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON=${PYTHON:-/root/miniconda3/envs/navsim/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}
OUT_ROOT=${OUT_ROOT:-/mnt/project/VLA-AD/outputs/recogdrive_official_stage1_vlm_2b_direct_text_navtest_$(date -u +%Y%m%dT%H%M%SZ)}
PRECISION=${PRECISION:-bf16}
SAVE_EVERY=${SAVE_EVERY:-100}
MAX_SAMPLES=${MAX_SAMPLES:-}
STAGGER_SECONDS=${STAGGER_SECONDS:-8}
ALLOW_TOLERANT_PARSE=${ALLOW_TOLERANT_PARSE:-1}
PROMPT_TYPE=${PROMPT_TYPE:-base}
CAM_TYPE=${CAM_TYPE:-single}
NUM_SHARDS=${NUM_SHARDS:-8}
SHARD_START=${SHARD_START:-0}
SHARD_COUNT=${SHARD_COUNT:-${NUM_SHARDS}}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-1}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/base"
cd "${REPO_ROOT}"
echo "$$" > "${OUT_ROOT}/driver.pid"

declare -a PIDS=()

{
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "driver_pid=$$"
  echo "model_path=${MODEL_PATH}"
  echo "trajectory_output_key=direct_text_pt"
  echo "prompt_type=${PROMPT_TYPE}"
  echo "cam_type=${CAM_TYPE}"
  echo "precision=${PRECISION}"
  echo "num_shards=${NUM_SHARDS}"
  echo "shard_start=${SHARD_START}"
  echo "shard_count=${SHARD_COUNT}"
  echo "gpus=${GPUS}"
  echo "shards_per_gpu=${SHARDS_PER_GPU}"
  echo "allow_tolerant_parse=${ALLOW_TOLERANT_PARSE}"
  echo "save_every=${SAVE_EVERY}"
  echo "max_samples=${MAX_SAMPLES}"
} | tee "${OUT_ROOT}/commands.log"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "GPUS must contain at least one GPU id" >&2
  exit 2
fi
if [[ "${SHARDS_PER_GPU}" -le 0 ]]; then
  echo "SHARDS_PER_GPU must be positive" >&2
  exit 2
fi

for local_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  shard=$((SHARD_START + local_idx))
  if [[ "${shard}" -ge "${NUM_SHARDS}" ]]; then
    echo "computed shard ${shard} >= NUM_SHARDS ${NUM_SHARDS}" >&2
    exit 2
  fi
  shard_name=$(printf "%02d" "${shard}")
  shard_dir="${OUT_ROOT}/base/shard_${shard_name}"
  mkdir -p "${shard_dir}"
  gpu_slot=$(((local_idx / SHARDS_PER_GPU) % ${#GPU_LIST[@]}))
  gpu="${GPU_LIST[${gpu_slot}]}"
  cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1
    "${PYTHON}" scripts/eval_vlm_direct_text_pdm.py
    --model-path "${MODEL_PATH}"
    --output-dir "${shard_dir}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${shard}"
    --precision "${PRECISION}"
    --prompt-type "${PROMPT_TYPE}"
    --cam-type "${CAM_TYPE}"
    --save-every "${SAVE_EVERY}"
    --progress-every 10
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ "${ALLOW_TOLERANT_PARSE}" == "1" ]]; then
    cmd+=(--allow-tolerant-parse)
  fi

  printf '[%s] shard=%s gpu=%s\n' "$(date -Is)" "${shard_name}" "${gpu}" | tee -a "${OUT_ROOT}/commands.log"
  printf '%q ' "${cmd[@]}" | tee -a "${OUT_ROOT}/commands.log"
  printf '\n' | tee -a "${OUT_ROOT}/commands.log"
  "${cmd[@]}" > "${OUT_ROOT}/logs/base_shard_${shard_name}.log" 2>&1 &
  pid=$!
  PIDS+=("${pid}")
  echo "${pid}" > "${OUT_ROOT}/logs/base_shard_${shard_name}.pid"
  echo "[base_shard_${shard_name}] pid=${pid}" | tee -a "${OUT_ROOT}/commands.log"
  sleep "${STAGGER_SECONDS}"
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  shard_name=$(printf "%02d" "${idx}")
  if wait "${pid}"; then
    echo "[base_shard_${shard_name}] exit=0" | tee -a "${OUT_ROOT}/commands.log"
  else
    rc=$?
    echo "[base_shard_${shard_name}] exit=${rc}" | tee -a "${OUT_ROOT}/commands.log"
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "one or more Stage1 VLM direct-text eval shards failed; skipping aggregation" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/aggregate_vlm_direct_text_eval.py --out-root "${OUT_ROOT}" --run base \
  > "${OUT_ROOT}/aggregate.log" 2>&1

echo "completed ${OUT_ROOT}"
