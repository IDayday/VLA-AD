#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-${BIT_WORK_ROOT}/experiments/risk_vla_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-${BIT_EXP_ROOT}/vlm_lora}"
DRY_RUN=0
for arg in "$@"; do
  [[ "${arg}" == "--dry-run" ]] && DRY_RUN=1
done

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
if [[ "${ENABLE_VLM_LORA:-0}" != "1" ]]; then
  cat > "${OUTPUT_DIR}/VLM_LORA_NOT_RUN.md" <<EOF
# VLM Risk LoRA Not Run

This optional path is disabled by default. Enable only if critic/router evidence shows perception or latent grounding is the bottleneck.
Set ENABLE_VLM_LORA=1 plus explicit train/val instruction-data paths to run.
EOF
  echo "VLM LoRA disabled by default"
  exit 0
fi

if [[ -z "${VLM_RISK_INSTRUCTION_DIR:-}" ]]; then
  echo "Blocked: missing VLM_RISK_INSTRUCTION_DIR"
  exit 1
fi

args=(--instruction-data-dir "${VLM_RISK_INSTRUCTION_DIR}" --output-dir "${OUTPUT_DIR}" --enable-train)
[[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
python scripts/risk_vla/train_vlm_risk_lora.py "${args[@]}"
