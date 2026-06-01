#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
TRAIN_CHUNK_NAME_PATTERN="${TRAIN_CHUNK_NAME_PATTERN:-navtrain_chunk_*}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

export BASE_CONFIG=last_rd_stage1_5
export LAST_RD_RISK_LOSS_WEIGHT=0.0
export TRAIN_CHUNK_CACHE_ROOT
export TRAIN_CHUNK_NAME_PATTERN
export TRAIN_TEST_SPLIT
export OUTPUT_DIR="${OUT_ROOT}/wave1_stage1_5/full"
export MASTER_PORT

mkdir -p "${OUTPUT_DIR}"
{
  printf 'Round1 Wave1 Stage1.5 full adapter on Server 1\n'
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_CHUNK_NAME_PATTERN=%s\n' "${TRAIN_CHUNK_NAME_PATTERN}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
} > "${OUTPUT_DIR}/wrapper_commands.log"

bash "${REPO_ROOT}/scripts/run_last_rd_stage1_5_8gpu.sh"

if [[ -f "${OUTPUT_DIR}/last_rd_adapter.pt" ]]; then
  python "${REPO_ROOT}/scripts/audit_last_rd_adapter_checkpoint.py" \
    --checkpoint "${OUTPUT_DIR}/last_rd_adapter.pt" \
    --output "${OUTPUT_DIR}/adapter_audit.json"
fi
