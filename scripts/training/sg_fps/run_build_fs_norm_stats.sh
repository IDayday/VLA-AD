#!/usr/bin/env bash
set -euo pipefail

SUPPORT_ARCHIVE_PATH=${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH}
OUTPUT_PATH=${OUTPUT_PATH:?set OUTPUT_PATH}

python scripts/tools/build_fs_norm_stats.py \
  --support_archive_path "${SUPPORT_ARCHIVE_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --use_robust "${USE_ROBUST:-true}" \
  --clip "${FS_NORM_CLIP:-5.0}"
