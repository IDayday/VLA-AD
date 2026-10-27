#!/usr/bin/env bash
set -euo pipefail
: "${ROUND1_PATHS_CONFIG:?Set ROUND1_PATHS_CONFIG to the verified local paths/calibration overlay}"
: "${ROUND1_RESOURCES:?Set ROUND1_RESOURCES to the verified eight-run host/GPU UUID resource manifest}"
: "${ROUND1_OUTPUT:?Set ROUND1_OUTPUT to the dedicated experiment output root}"
python -B -m student_distillation.run_suite \
  --config-dir configs/student_distillation \
  --paths-config "$ROUND1_PATHS_CONFIG" \
  --experiments H_full H_gain1 H_mu_teacher H_mu_student \
  --training-seeds 20261027 20261028 --epochs 40 --global-batch-size 64 \
  --resources "$ROUND1_RESOURCES" --resume auto --complete-through report \
  --out "$ROUND1_OUTPUT"
