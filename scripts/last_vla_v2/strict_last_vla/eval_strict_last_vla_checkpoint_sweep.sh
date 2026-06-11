#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/strict_last_vla_navtest_sweep}"

cmd=(
  "${PYTHON_BIN}" scripts/run_recogdrive_full_pdm_suite.py
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  printf 'RUN_EVAL is not 1; strict checkpoint sweep not launched.\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${CHECKPOINT_ROOT}" ]] || { echo "Missing CHECKPOINT_ROOT." >&2; exit 2; }
"${cmd[@]}"
