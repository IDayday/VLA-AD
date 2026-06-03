#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
SHARED_CHUNK_CACHE_ROOT="${SHARED_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
SOURCE_CACHE_DIR="${SOURCE_CACHE_DIR:-${SHARED_CHUNK_CACHE_ROOT}}"
OUTPUT_CACHE_DIR="${OUTPUT_CACHE_DIR:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
LABELS_JSONL="${LABELS_JSONL:-${ROUND_DIR}/risk_labels/risk_labels.jsonl}"
MODE="${MODE:-copy}"
MAX_SAMPLES="${MAX_SAMPLES:-4096}"

CMD=(python scripts/risk_vla/create_labeled_chunk_cache_overlay.py --source-cache-dir "${SOURCE_CACHE_DIR}" --output-cache-dir "${OUTPUT_CACHE_DIR}" --labels-jsonl "${LABELS_JSONL}" --mode "${MODE}" --max-samples "${MAX_SAMPLES}")
if [[ "${EXECUTE}" == "1" ]]; then
  CMD+=(--write --overwrite)
else
  CMD+=(--dry-run)
fi

echo "[run_02] source cache: ${SOURCE_CACHE_DIR}"
echo "[run_02] output overlay: ${OUTPUT_CACHE_DIR}"
echo "[run_02] labels: ${LABELS_JSONL}"
printf '[run_02] command: '; printf '%q ' "${CMD[@]}"; echo

if [[ "${EXECUTE}" == "1" ]]; then
  "${CMD[@]}"
else
  echo "[run_02] DRY_RUN=${DRY_RUN}; set EXECUTE=1 to create overlay."
fi
