#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT}"
CONFIG="${CONFIG:?Set CONFIG to the planner evaluation YAML}"
MANIFEST="${MANIFEST:?Set MANIFEST}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
HIDDEN_CACHE_ROOT="${HIDDEN_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}"
SUPPORT_ROOT="${SUPPORT_ROOT:-${REPO_ROOT}/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-32}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-4}"
SEED="${SEED:-260711}"
PRECISION="${PRECISION:-fp32}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST produced no GPUs" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs"
cd "${REPO_ROOT}"
pids=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
  shard_dir="${OUT_ROOT}/shard_$(printf '%02d' "${shard}")"
  mkdir -p "${shard_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/evaluation/evaluate_recogdrive_support_aligned_diversity.py
    --checkpoint "${CHECKPOINT}"
    --config "${CONFIG}"
    --hidden-cache-root "${HIDDEN_CACHE_ROOT}"
    --support-root "${SUPPORT_ROOT}"
    --manifest "${MANIFEST}"
    --output-dir "${shard_dir}"
    --samples-per-scene "${SAMPLES_PER_SCENE}"
    --sample-batch-size "${SAMPLE_BATCH_SIZE}"
    --seed "${SEED}"
    --num-shards "${NUM_SHARDS}"
    --shard-index "${shard}"
    --precision "${PRECISION}"
  )
  if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
    cmd+=(--save-predictions)
  fi
  {
    printf '[%s] CUDA_VISIBLE_DEVICES=%q ' "$(date -Is)" "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${OUT_ROOT}/commands.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" \
    >"${OUT_ROOT}/logs/shard_$(printf '%02d' "${shard}").log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  tail -n 80 "${OUT_ROOT}"/logs/*.log >&2 || true
  exit "${status}"
fi
echo "support-aligned diversity evaluation complete: ${OUT_ROOT}"
