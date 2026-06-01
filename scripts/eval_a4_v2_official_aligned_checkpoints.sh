#!/usr/bin/env bash
set -Eeuo pipefail

# Thin A4 wrapper around the generic official-aligned checkpoint evaluator.
# Set A4_EVAL_CONFIG to the matching planner YAML for the run being evaluated.

export CONFIG="${A4_EVAL_CONFIG:-configs/ablations/recogdrive2b_A4_v2.yaml}"
exec bash scripts/eval_a0_stage2_checkpoints.sh
