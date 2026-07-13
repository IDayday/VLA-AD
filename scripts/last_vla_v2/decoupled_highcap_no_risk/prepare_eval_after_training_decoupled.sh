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
ROOT="${OUT_ROOT}/decoupled_highcap_no_risk_eval_prep"
mkdir -p "${ROOT}"
COMMANDS_LOG="${ROOT}/commands.log"

cmd_sweep=(
  scripts/last_vla_v2/decoupled_highcap_no_risk/eval_checkpoint_sweep_decoupled.sh
)
cmd_summary=(
  "${PYTHON_BIN}" scripts/last_vla_v2/decoupled_highcap_no_risk/summarize_decoupled_highcap_no_risk.py
  --out-root "${OUT_ROOT}"
)

{
  date -Is
  printf 'OUT_ROOT=%q NAVTEST_CHUNK_CACHE_ROOT=%q METRIC_CACHE_DIR=%q ' "${OUT_ROOT}" "${NAVTEST_CHUNK_CACHE_ROOT}" "${METRIC_CACHE_DIR}"
  if [[ -n "${LORA_NAVTEST_CHUNK_CACHE_ROOT:-}" ]]; then
    printf 'LORA_NAVTEST_CHUNK_CACHE_ROOT=%q ' "${LORA_NAVTEST_CHUNK_CACHE_ROOT}"
  fi
  printf '%q\n' "${cmd_sweep[@]}"
  printf '%q ' "${cmd_summary[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  "${cmd_sweep[@]}"
  "${cmd_summary[@]}"
else
  echo "RUN_EVAL is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
fi
