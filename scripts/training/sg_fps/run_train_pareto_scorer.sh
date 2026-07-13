#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_PATH=${ARCHIVE_PATH:?set ARCHIVE_PATH}
OUTPUT_PATH=${OUTPUT_PATH:?set OUTPUT_PATH}

python scripts/training/run_train_pareto_vector_scorer.py \
  --archive_path "${ARCHIVE_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --epochs "${EPOCHS:-1}" \
  --batch_size "${BATCH_SIZE:-128}" \
  --lr "${LR:-1e-3}" \
  --device "${DEVICE:-cuda}"
