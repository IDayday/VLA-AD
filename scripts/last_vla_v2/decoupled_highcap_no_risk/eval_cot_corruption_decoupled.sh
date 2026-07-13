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
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval.yaml}"
CORR_ROOT="${OUT_ROOT}/decoupled_highcap_no_risk_corruption"
mkdir -p "${CORR_ROOT}"
COMMANDS_LOG="${CORR_ROOT}/commands.log"

modes=(
  normal
  zero_all_cot
  zero_scene_cot
  zero_geometry_cot
  zero_dynamic_cot
  zero_fusion_cot
  zero_action_refine_cot
  zero_coarse_prior
  zero_cot_condition_branch
  raw_vlm_only
  cot_only_for_debug_only
)
for mode in "${modes[@]}"; do
  out_dir="${CORR_ROOT}/${mode}"
  mkdir -p "${out_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/eval_last_vla_cot_corruption_pdm.py
    --config "${CONFIG}"
    --checkpoint "${CHECKPOINT}"
    --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --score-mode pdm
    --precision fp32
    --output-dir "${out_dir}"
  )
  if [[ "${FULL_CORRUPTION:-0}" != "1" ]]; then
    cmd+=(--max-samples "${MAX_SAMPLES:-1000}")
  fi
  case "${mode}" in
    zero_all_cot) cmd+=(--zero-all-cot) ;;
    zero_scene_cot) cmd+=(--zero-scene-cot) ;;
    zero_geometry_cot) cmd+=(--zero-geometry-cot) ;;
    zero_dynamic_cot) cmd+=(--zero-dynamic-cot) ;;
    zero_fusion_cot) cmd+=(--zero-fusion-cot) ;;
    zero_action_refine_cot) cmd+=(--zero-action-refine-cot) ;;
    zero_coarse_prior) cmd+=(--zero-coarse-prior) ;;
    zero_cot_condition_branch) cmd+=(--zero-cot-condition-branch) ;;
    raw_vlm_only) cmd+=(--raw-vlm-only) ;;
    cot_only_for_debug_only) cmd+=(--cot-only-for-debug-only) ;;
  esac
  printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
  if [[ "${RUN_EVAL:-0}" == "1" ]]; then
    "${cmd[@]}" >"${out_dir}/corruption.log" 2>&1
  fi
done

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  echo "RUN_EVAL is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
fi
