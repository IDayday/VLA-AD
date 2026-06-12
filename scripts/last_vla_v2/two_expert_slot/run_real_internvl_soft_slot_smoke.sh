#!/usr/bin/env bash
set -Eeuo pipefail

REPORT_DIR="${REPORT_DIR:-reports/two_expert_slot}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPORT_DIR}/internvl_soft_slot_smoke.json}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
mkdir -p "${REPORT_DIR}"

cmd=(
  "${PYTHON_BIN}" scripts/last_vla_v2/two_expert_slot/smoke_internvl_soft_slots.py
  --output-json "${OUTPUT_JSON}"
  --device "${DEVICE:-cuda}"
  --vlm-hidden-dim "${VLM_HIDDEN_DIM:-1536}"
  --image-size "${IMAGE_SIZE:-448}"
  --max-image-patches "${MAX_IMAGE_PATCHES:-12}"
  --threshold "${THRESHOLD:-1e-6}"
)

if [[ -n "${VLM_PATH:-}" ]]; then cmd+=(--vlm-path "${VLM_PATH}"); fi
if [[ -n "${IMAGE_PATH:-}" ]]; then cmd+=(--image-path "${IMAGE_PATH}"); fi
if [[ -n "${BASE_CACHE_ROOT:-}" ]]; then cmd+=(--base-cache-root "${BASE_CACHE_ROOT}"); fi
if [[ -n "${SAMPLE_TOKEN:-}" ]]; then cmd+=(--sample-token "${SAMPLE_TOKEN}"); fi
if [[ -n "${MAX_RECORDS:-}" ]]; then cmd+=(--max-records "${MAX_RECORDS}"); fi
if [[ "${ALLOW_MINIMAL_PROMPT:-0}" == "1" ]]; then cmd+=(--allow-minimal-prompt); fi

printf '%q ' "${cmd[@]}" >"${REPORT_DIR}/internvl_soft_slot_smoke.command"
printf '\n' >>"${REPORT_DIR}/internvl_soft_slot_smoke.command"

if [[ "${RUN_SMOKE:-0}" != "1" ]]; then
  echo "RUN_SMOKE is not 1; dry-run only. Command written to ${REPORT_DIR}/internvl_soft_slot_smoke.command"
  "${cmd[@]}"
  exit 0
fi

"${cmd[@]}"
