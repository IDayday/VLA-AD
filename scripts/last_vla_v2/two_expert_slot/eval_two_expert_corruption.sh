#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUT_ROOT CHECKPOINT NAVTEST_CHUNK_CACHE_ROOT METRIC_CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CONFIG="${CONFIG:-configs/last_vla_v2/two_expert_slot/stage2_dit_sft.yaml}"
CORR_ROOT="${OUT_ROOT}/two_expert_slot_corruption"
COMMANDS_LOG="${CORR_ROOT}/commands.log"
mkdir -p "${CORR_ROOT}"

modes=(normal zero_h_dyn zero_h_geo zero_all_experts raw_vlm_only dyn_only geo_only)
for mode in "${modes[@]}"; do
  out_dir="${CORR_ROOT}/${mode}"
  mkdir -p "${out_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision "${PRECISION:-fp32}"
    --two-expert-corruption-mode "${mode}"
    --output-dir "${out_dir}"
  )
  if [[ "${FULL_CORRUPTION:-0}" != "1" ]]; then cmd+=(--max-samples "${MAX_SAMPLES:-1000}"); fi
  printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
  if [[ "${RUN_EVAL:-0}" == "1" ]]; then
    "${cmd[@]}" >"${out_dir}/corruption.log" 2>&1
  fi
done

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  echo "RUN_EVAL is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
fi
