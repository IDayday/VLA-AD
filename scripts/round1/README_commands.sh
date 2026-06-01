#!/usr/bin/env bash
set -euo pipefail

# LaST-RD Round1 command index.
#
# Default behavior is read-only: print the exact commands and exit.
# To execute one step intentionally, set:
#   RUN_ROUND1=1 ROUND1_STEP=<step> bash scripts/round1/README_commands.sh
#
# Supported ROUND1_STEP values:
#   preflight
#   stage1_full
#   stage1_jepa_only
#   progressive_lastrd_only
#   progressive_hybrid
#   eval_sweep
#   corruption
#   summarize
#
# This file is not a cross-server orchestrator. Run the server-specific training
# steps on the intended server only.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_commands() {
  cat <<'EOF'
LaST-RD Round1 commands
=======================

Baseline:
  A0-official-aligned checkpoint: step_00100000
  A0-official-aligned full navtest PDMS: 0.864891

Wave 0: Preflight
-----------------
export TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks
export OUT_ROOT=/path/to/outputs/last_rd_round1
export TRAIN_CHUNK_NAME_PATTERN='navtrain_chunk_*'
export MAX_MANIFEST_SAMPLES=128
export FULL_MANIFEST=0
bash scripts/run_last_rd_round1_preflight.sh

Wave 1: Stage1.5, Server 1 full adapter
----------------------------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks
export TRAIN_CHUNK_NAME_PATTERN='navtrain_chunk_*'
export TRAIN_TEST_SPLIT=navtrain
export MASTER_PORT=29531
bash scripts/round1/run_stage1_5_full_server1.sh

Wave 1: Stage1.5, Server 2 JEPA-only backup adapter
----------------------------------------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks
export TRAIN_TEST_SPLIT=navtrain
export MASTER_PORT=29541
bash scripts/round1/run_stage1_5_jepa_only_server2.sh

Wave 2: Progressive SFT, Server 1 LastRD-only
---------------------------------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks
export TRAIN_TEST_SPLIT=navtrain
export A0_INIT_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt
export A0_REFERENCE_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt
export MASTER_PORT=29532
bash scripts/round1/run_progressive_lastrd_only_server1.sh

Wave 2: Progressive SFT, Server 2 hybrid
----------------------------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export TRAIN_CHUNK_CACHE_ROOT=/path/to/train_chunks
export TRAIN_TEST_SPLIT=navtrain
export A0_INIT_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt
export A0_REFERENCE_CHECKPOINT=/path/to/a0_official_aligned/step_00100000.ckpt
export MASTER_PORT=29542
bash scripts/round1/run_progressive_hybrid_server2.sh

Wave 3: checkpoint sweep full eval
----------------------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_chunks
export NAVTEST_CHUNK_NAME_PATTERN='navtest_full_chunk_*'
export METRIC_CACHE_DIR=/path/to/navtest_metric_cache
export RUN_NAME=all
export GPU_ID=0
bash scripts/round1/eval_progressive_checkpoint_sweep.sh

Wave 4: corruption eval
-----------------------
export OUT_ROOT=/path/to/outputs/last_rd_round1
export NAVTEST_CHUNK_CACHE_ROOT=/path/to/navtest_chunks
export NAVTEST_CHUNK_NAME_PATTERN='navtest_full_chunk_*'
export METRIC_CACHE_DIR=/path/to/navtest_metric_cache
export EVAL_MAX_SAMPLES=1000
export FULL_CORRUPTION=0
export GPU_ID=0
bash scripts/round1/eval_best_corruption.sh

Summarize
---------
python scripts/round1/summarize_round1_results.py \
  --root "${OUT_ROOT}" \
  --baseline-pdms 0.864891

Optional single-step execution through this index
-------------------------------------------------
RUN_ROUND1=1 ROUND1_STEP=preflight bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=stage1_full bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=stage1_jepa_only bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=progressive_lastrd_only bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=progressive_hybrid bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=eval_sweep bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=corruption bash scripts/round1/README_commands.sh
RUN_ROUND1=1 ROUND1_STEP=summarize bash scripts/round1/README_commands.sh
EOF
}

if [[ "${RUN_ROUND1:-0}" != "1" ]]; then
  print_commands
  exit 0
fi

case "${ROUND1_STEP:-}" in
  preflight)
    bash "${REPO_ROOT}/scripts/run_last_rd_round1_preflight.sh"
    ;;
  stage1_full)
    bash "${REPO_ROOT}/scripts/round1/run_stage1_5_full_server1.sh"
    ;;
  stage1_jepa_only)
    bash "${REPO_ROOT}/scripts/round1/run_stage1_5_jepa_only_server2.sh"
    ;;
  progressive_lastrd_only)
    bash "${REPO_ROOT}/scripts/round1/run_progressive_lastrd_only_server1.sh"
    ;;
  progressive_hybrid)
    bash "${REPO_ROOT}/scripts/round1/run_progressive_hybrid_server2.sh"
    ;;
  eval_sweep)
    bash "${REPO_ROOT}/scripts/round1/eval_progressive_checkpoint_sweep.sh"
    ;;
  corruption)
    bash "${REPO_ROOT}/scripts/round1/eval_best_corruption.sh"
    ;;
  summarize)
    python "${REPO_ROOT}/scripts/round1/summarize_round1_results.py" \
      --root "${OUT_ROOT:?Set OUT_ROOT.}" \
      --baseline-pdms "${BASELINE_PDMS:-0.864891}"
    ;;
  ""|help|print)
    print_commands
    ;;
  *)
    echo "Unknown ROUND1_STEP=${ROUND1_STEP}" >&2
    print_commands >&2
    exit 2
    ;;
esac
