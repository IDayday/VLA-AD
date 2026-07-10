#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}

VLM_PATH=${VLM_PATH:?Set VLM_PATH to the completed Stage1 checkpoint directory}
STAGE1_DATA_DIR=${STAGE1_DATA_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_sft_data_v1}
DATA_ROOT=${DATA_ROOT:-/mnt/data/Bench2Drive-Base}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_cache_${RUN_ID}}
SPLITS=${SPLITS:-train,val}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_SHARDS=${N_SHARDS:-8}
LOG_EVERY=${LOG_EVERY:-100}
OVERWRITE=${OVERWRITE:-0}

if [[ ! -d "${VLM_PATH}" ]]; then
  echo "Stage1 VLM checkpoint not found: ${VLM_PATH}" >&2
  exit 2
fi
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} < N_SHARDS )); then
  echo "N_SHARDS=${N_SHARDS} requires at least that many entries in GPU_LIST=${GPU_LIST}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
GIT_COMMIT=$(cd "${VLA_AD_ROOT}" && git rev-parse HEAD)
cat > "${OUTPUT_ROOT}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=${GIT_COMMIT}
vlm_path=${VLM_PATH}
stage1_data_dir=${STAGE1_DATA_DIR}
data_root=${DATA_ROOT}
output_root=${OUTPUT_ROOT}
splits=${SPLITS}
gpu_list=${GPU_LIST}
n_shards=${N_SHARDS}
EOF

run_split() {
  local split="$1"
  local clip_list="${STAGE1_DATA_DIR}/${split}_clips.txt"
  if [[ ! -f "${clip_list}" ]]; then
    echo "Missing Stage1 ${split} clip list: ${clip_list}" >&2
    return 2
  fi
  mapfile -t clips < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "${clip_list}")
  local total=${#clips[@]}
  if (( total == 0 )); then
    echo "Empty clip list: ${clip_list}" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/${split}"
  local -a pids=()
  for (( shard=0; shard<N_SHARDS; shard++ )); do
    local start=$(( (total * shard) / N_SHARDS ))
    local stop=$(( (total * (shard + 1)) / N_SHARDS ))
    if (( start == stop )); then
      continue
    fi
    local shard_name
    shard_name=$(printf 'shard_%02d' "${shard}")
    local shard_dir="${OUTPUT_ROOT}/${split}/${shard_name}"
    local log_path="${OUTPUT_ROOT}/${split}/${shard_name}.log"
    local -a args=(
      python scripts/build_bench2drive_recogdrive_chunk_cache.py
      --data-root "${DATA_ROOT}"
      --output-dir "${shard_dir}"
      --clip-list "${clip_list}"
      --clip-start "${start}"
      --clip-stop "${stop}"
      --hidden-source recogdrive-vlm
      --recogdrive-vlm-path "${VLM_PATH}"
      --system-prompt-profile bench2drive
      --device cuda
      --frame-step 5
      --sample-stride 5
      --history-frames 4
      --future-frames 8
      --log-every "${LOG_EVERY}"
    )
    if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "TRUE" ]]; then
      args+=(--overwrite)
    fi
    echo "[cache] split=${split} shard=${shard_name} clips=[${start},${stop}) gpu=${GPUS[shard]}"
    (
      cd "${VLA_AD_ROOT}"
      env CUDA_VISIBLE_DEVICES="${GPUS[shard]}" \
        PYTHONPATH="${VLA_AD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" "${args[@]}"
    ) > "${log_path}" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "One or more ${split} cache workers failed; inspect ${OUTPUT_ROOT}/${split}/shard_*.log" >&2
    return 1
  fi
  echo "[cache] completed split=${split}"
}

IFS=',' read -r -a requested_splits <<< "${SPLITS}"
for split in "${requested_splits[@]}"; do
  split=${split//[[:space:]]/}
  case "${split}" in
    train|val) run_split "${split}" ;;
    *) echo "Unsupported split ${split}; use train,val" >&2; exit 2 ;;
  esac
done
