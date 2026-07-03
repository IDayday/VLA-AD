#!/usr/bin/env bash
set -euo pipefail

CANDIDATE_ARCHIVE_PATH=${CANDIDATE_ARCHIVE_PATH:?set CANDIDATE_ARCHIVE_PATH}
SCORER_CHECKPOINT_PATH=${SCORER_CHECKPOINT_PATH:?set SCORER_CHECKPOINT_PATH}
OUTPUT_PATH=${OUTPUT_PATH:?set OUTPUT_PATH}

python scripts/tools/refine_support_with_scorer.py \
  --candidate_archive_path "${CANDIDATE_ARCHIVE_PATH}" \
  --scorer_checkpoint_path "${SCORER_CHECKPOINT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --pred_top_k "${PRED_TOP_K:-24}" \
  --support_top_m "${SUPPORT_TOP_M:-12}" \
  --device "${DEVICE:-cuda}"
