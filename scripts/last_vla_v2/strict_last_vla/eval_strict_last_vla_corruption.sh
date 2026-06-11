#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
CACHE_ROOT="${CACHE_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/strict_last_vla_corruption}"
MODES="${MODES:-zero_h_dyn,zero_h_geo,zero_h_plan,zero_all_latent,raw_vlm_only}"

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/strict_last_vla/eval_strict_last_vla_corruption.py
  --checkpoint "${CHECKPOINT}"
  --cache-root "${CACHE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --modes "${MODES}"
)

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  printf 'RUN_EVAL is not 1; strict corruption eval not launched.\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

[[ -n "${CHECKPOINT}" ]] || { echo "Missing CHECKPOINT." >&2; exit 2; }
[[ -n "${CACHE_ROOT}" ]] || { echo "Missing CACHE_ROOT." >&2; exit 2; }
"${cmd[@]}"
