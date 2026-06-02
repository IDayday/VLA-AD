#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUTPUT_CACHE_ROOT VGGT_MODEL_PATH CHUNK_INDEX CHUNK_SIZE)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
NAVSIM_ROOT="${NAVSIM_ROOT:-/mnt/navsim}"
SPLIT="${SPLIT:-navtrain}"
PRECISION="${PRECISION:-fp32}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
CHUNK_NAME="${CHUNK_NAME:-train_geometry_chunk_$(printf '%06d' "${CHUNK_INDEX}")}"
OUT_CHUNK="${OUTPUT_CACHE_ROOT}/${CHUNK_NAME}"

mkdir -p "${OUT_CHUNK}"
cmd=(
  "${PYTHON_BIN}" scripts/build_recogdrive_chunk_cache.py
  --navsim-root "${NAVSIM_ROOT}"
  --split "${SPLIT}"
  --chunk-index "${CHUNK_INDEX}"
  --chunk-size "${CHUNK_SIZE}"
  --output-dir "${OUT_CHUNK}"
  --build-vggt
  --require-vggt-geometry
  --vggt-model-path "${VGGT_MODEL_PATH}"
  --precision "${PRECISION}"
  --device "${DEVICE}"
  --num-gpus "${NUM_GPUS}"
  --workers-per-gpu "${WORKERS_PER_GPU}"
  --allow-partial-final-chunk
  --log-every "${LOG_EVERY}"
)
if [[ -n "${CHUNK_START:-}" ]]; then cmd+=(--chunk-start "${CHUNK_START}"); fi
if [[ -n "${CHUNK_STOP:-}" ]]; then cmd+=(--chunk-stop "${CHUNK_STOP}"); fi
if [[ "${STRICT_TOKEN_WINDOW:-0}" == "1" ]]; then cmd+=(--strict-token-window); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then cmd+=(--overwrite); fi

printf '%q ' "${cmd[@]}" >>"${OUT_CHUNK}/commands.log"; printf '\n' >>"${OUT_CHUNK}/commands.log"
"${cmd[@]}" 2>&1 | tee "${OUT_CHUNK}/generate.log"
