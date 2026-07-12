#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CACHE_PYTHON=${CACHE_PYTHON:-/root/miniconda3/envs/navsim/bin/python}

VLM_PATH=${VLM_PATH:?Set VLM_PATH to the completed Stage1 checkpoint directory}
DATA_ROOT=${DATA_ROOT:-/mnt/data/Bench2Drive-Base}
VLM_CODE_SOURCE=${VLM_CODE_SOURCE:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_cache_closest_public_${RUN_ID}}
SPLITS=${SPLITS:-train}
TRAIN_CLIP_LIST=${TRAIN_CLIP_LIST:-}
VAL_CLIP_LIST=${VAL_CLIP_LIST:-}
EXPECTED_TRAIN_CLIPS=${EXPECTED_TRAIN_CLIPS:-1000}
EXPECTED_VAL_CLIPS=${EXPECTED_VAL_CLIPS:-}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_SHARDS=${N_SHARDS:-8}
CACHE_CPU_THREADS=${CACHE_CPU_THREADS:-4}
LOG_EVERY=${LOG_EVERY:-100}
OVERWRITE=${OVERWRITE:-0}
CONTRACT_AUDIT_SAMPLES=${CONTRACT_AUDIT_SAMPLES:-128}

if [[ ! -d "${VLM_PATH}" ]]; then
  echo "Stage1 VLM checkpoint not found: ${VLM_PATH}" >&2
  exit 2
fi
if [[ ! -f "${VLM_PATH}/model.safetensors" || ! -f "${VLM_PATH}/config.json" ]]; then
  echo "Stage1 VLM is incomplete (need model.safetensors and config.json): ${VLM_PATH}" >&2
  exit 2
fi
if [[ ! -x "${CACHE_PYTHON}" ]]; then
  echo "Cache Python is missing or not executable: ${CACHE_PYTHON}" >&2
  exit 2
fi
"${CACHE_PYTHON}" "${VLA_AD_ROOT}/scripts/bench2drive/materialize_recogdrive_vlm_remote_code.py" \
  --checkpoint "${VLM_PATH}" \
  --source "${VLM_CODE_SOURCE}" \
  > "${VLM_PATH}/recogdrive_remote_code_materialization.log"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
if (( N_SHARDS <= 0 )); then
  echo "N_SHARDS must be positive: ${N_SHARDS}" >&2
  exit 2
fi
if (( CACHE_CPU_THREADS <= 0 )); then
  echo "CACHE_CPU_THREADS must be positive: ${CACHE_CPU_THREADS}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -z "${TRAIN_CLIP_LIST}" ]]; then
  TRAIN_CLIP_LIST="${OUTPUT_ROOT}/all_1000_clips.txt"
  DATA_ROOT="${DATA_ROOT}" TRAIN_CLIP_LIST="${TRAIN_CLIP_LIST}" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATA_ROOT"])
clips = sorted(
    path.name
    for path in root.iterdir()
    if path.is_dir() and path.name != "maps" and (path / "anno").is_dir()
)
if len(clips) != 1000:
    raise SystemExit(f"Expected 1000 raw Bench2Drive clips, found {len(clips)} under {root}")
Path(os.environ["TRAIN_CLIP_LIST"]).write_text("\n".join(clips) + "\n", encoding="utf-8")
PY
fi
"${CACHE_PYTHON}" "${VLA_AD_ROOT}/scripts/bench2drive/audit_recogdrive_b2d_stage2_contract.py" \
  --data-root "${DATA_ROOT}" \
  --sample-count "${CONTRACT_AUDIT_SAMPLES}" \
  --report "${OUTPUT_ROOT}/released_contract_audit.json" \
  > "${OUTPUT_ROOT}/released_contract_audit.log"
GIT_COMMIT=$(cd "${VLA_AD_ROOT}" && git rev-parse HEAD)
cat > "${OUTPUT_ROOT}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=${GIT_COMMIT}
vlm_path=${VLM_PATH}
data_root=${DATA_ROOT}
output_root=${OUTPUT_ROOT}
splits=${SPLITS}
train_clip_list=${TRAIN_CLIP_LIST}
val_clip_list=${VAL_CLIP_LIST}
expected_train_clips=${EXPECTED_TRAIN_CLIPS}
expected_val_clips=${EXPECTED_VAL_CLIPS}
cache_python=${CACHE_PYTHON}
vlm_code_source=${VLM_CODE_SOURCE}
gpu_list=${GPU_LIST}
n_shards=${N_SHARDS}
workers_per_gpu=$(( (N_SHARDS + ${#GPUS[@]} - 1) / ${#GPUS[@]} ))
cache_cpu_threads=${CACHE_CPU_THREADS}
contract_audit_samples=${CONTRACT_AUDIT_SAMPLES}
EOF

run_split() {
  local split="$1"
  local clip_list
  local expected_clips
  if [[ "${split}" == "train" ]]; then
    clip_list="${TRAIN_CLIP_LIST}"
    expected_clips="${EXPECTED_TRAIN_CLIPS}"
  else
    clip_list="${VAL_CLIP_LIST}"
    expected_clips="${EXPECTED_VAL_CLIPS}"
  fi
  if [[ ! -f "${clip_list}" ]]; then
    echo "Missing ${split} clip list: ${clip_list}" >&2
    return 2
  fi
  mapfile -t clips < <(sed -e 's/#.*$//' -e '/^[[:space:]]*$/d' "${clip_list}")
  local total=${#clips[@]}
  if (( total == 0 )); then
    echo "Empty clip list: ${clip_list}" >&2
    return 2
  fi
  if [[ -n "${expected_clips}" && "${total}" -ne "${expected_clips}" ]]; then
    echo "${split} clip list has ${total} entries; expected ${expected_clips}: ${clip_list}" >&2
    return 2
  fi
  local unique_total
  unique_total=$(printf '%s\n' "${clips[@]}" | sort -u | wc -l)
  if (( unique_total != total )); then
    echo "${split} clip list contains duplicate entries: ${clip_list}" >&2
    return 2
  fi
  mkdir -p "${OUTPUT_ROOT}/${split}"
  local -a pids=()
  local gpu_count=${#GPUS[@]}
  for (( shard=0; shard<N_SHARDS; shard++ )); do
    local start=$(( (total * shard) / N_SHARDS ))
    local stop=$(( (total * (shard + 1)) / N_SHARDS ))
    if (( start == stop )); then
      continue
    fi
    local shard_name
    shard_name=$(printf 'shard_%02d' "${shard}")
    local gpu_index=$(( shard % gpu_count ))
    local gpu=${GPUS[gpu_index]}
    local worker_slot=$(( shard / gpu_count ))
    local shard_dir="${OUTPUT_ROOT}/${split}/${shard_name}"
    local log_path="${OUTPUT_ROOT}/${split}/${shard_name}.log"
    local -a args=(
      "${CACHE_PYTHON}" scripts/build_bench2drive_recogdrive_chunk_cache.py
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
      --sample-stride 1
      --history-frames 4
      --future-frames 6
      --cache-hidden-dtype bfloat16
      --cpu-threads "${CACHE_CPU_THREADS}"
      --log-every "${LOG_EVERY}"
    )
    if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "TRUE" ]]; then
      args+=(--overwrite)
    fi
    echo "[cache] split=${split} shard=${shard_name} clips=[${start},${stop}) gpu=${gpu} worker_slot=${worker_slot} cpu_threads=${CACHE_CPU_THREADS}"
    (
      cd "${VLA_AD_ROOT}"
      env CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS="${CACHE_CPU_THREADS}" \
        MKL_NUM_THREADS="${CACHE_CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${CACHE_CPU_THREADS}" \
        NUMEXPR_NUM_THREADS="${CACHE_CPU_THREADS}" \
        OMP_WAIT_POLICY=PASSIVE \
        TOKENIZERS_PARALLELISM=false \
        PYTHONPATH="${VLA_AD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${args[@]}"
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
