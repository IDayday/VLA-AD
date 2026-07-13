#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUT_ROOT NAVTEST_CHUNK_CACHE_ROOT METRIC_CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
EVAL_ROOT="${OUT_ROOT}/round2_eval"
mkdir -p "${EVAL_ROOT}/logs"
COMMANDS_LOG="${EVAL_ROOT}/commands.log"
CONFIG="${CONFIG:-configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml}"

write_eval_cmd() {
  local line_name="$1"
  local cache_root="$2"
  local ckpt="$3"
  local out_dir="${EVAL_ROOT}/${line_name}/$(basename "${ckpt}" .ckpt)"
  mkdir -p "${out_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${ckpt}"
    --chunk-cache-root "${cache_root}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision fp32
    --output-dir "${out_dir}"
  )
  printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
  if [[ "${RUN_EVAL:-0}" == "1" ]]; then
    "${cmd[@]}" >"${out_dir}/eval.log" 2>&1
  fi
}

for line in serverA_frozen serverB_lora; do
  list="${OUT_ROOT}/${line}/checkpoints_to_eval.txt"
  [[ -f "${list}" ]] || continue
  cache="${NAVTEST_CHUNK_CACHE_ROOT}"
  if [[ "${line}" == "serverB_lora" ]]; then
    if [[ -z "${LORA_NAVTEST_CHUNK_CACHE_ROOT:-}" ]]; then
      echo "Skipping serverB_lora eval commands: LORA_NAVTEST_CHUNK_CACHE_ROOT is required." >&2
      continue
    fi
    cache="${LORA_NAVTEST_CHUNK_CACHE_ROOT}"
  fi
  while IFS= read -r ckpt; do
    [[ -z "${ckpt}" ]] && continue
    write_eval_cmd "${line}" "${cache}" "${ckpt}"
  done <"${list}"
done

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  echo "RUN_EVAL is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
fi
