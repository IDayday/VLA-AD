#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CHUNK_CACHE_ROOT:?Set TRAIN_CHUNK_CACHE_ROOT.}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT.}"
MASTER_PORT="${MASTER_PORT:?Set MASTER_PORT.}"
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:?Set A0_INIT_CHECKPOINT.}"
A0_REFERENCE_CHECKPOINT="${A0_REFERENCE_CHECKPOINT:?Set A0_REFERENCE_CHECKPOINT.}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
STAGE1_5_CHECKPOINT="${STAGE1_5_CHECKPOINT:-${OUT_ROOT}/wave1_stage1_5/full/last_rd_adapter.pt}"

[[ -f "${STAGE1_5_CHECKPOINT}" ]] || { echo "Missing STAGE1_5_CHECKPOINT=${STAGE1_5_CHECKPOINT}" >&2; exit 1; }
[[ -f "${A0_INIT_CHECKPOINT}" ]] || { echo "Missing A0_INIT_CHECKPOINT=${A0_INIT_CHECKPOINT}" >&2; exit 1; }
[[ -f "${A0_REFERENCE_CHECKPOINT}" ]] || { echo "Missing A0_REFERENCE_CHECKPOINT=${A0_REFERENCE_CHECKPOINT}" >&2; exit 1; }

export BASE_CONFIG=last_rd_progressive_sft
export STAGE1_5_CHECKPOINT
export A0_INIT_CHECKPOINT
export A0_REFERENCE_CHECKPOINT
export TRAIN_CHUNK_CACHE_ROOT
export TRAIN_TEST_SPLIT
export OUTPUT_DIR="${OUT_ROOT}/wave2_progressive/hybrid"
export MASTER_PORT
export LAST_RD_POLICY_KD_WEIGHT=0.05
export LAST_RD_POLICY_KD_MODE=noise
export LAST_RD_RISK_LOSS_WEIGHT=0.0

mkdir -p "${OUTPUT_DIR}"
{
  printf 'Round1 Wave2 Progressive SFT hybrid on Server 2\n'
  printf 'BASE_CONFIG=%s\n' "${BASE_CONFIG}"
  printf 'TRAIN_CHUNK_CACHE_ROOT=%s\n' "${TRAIN_CHUNK_CACHE_ROOT}"
  printf 'TRAIN_TEST_SPLIT=%s\n' "${TRAIN_TEST_SPLIT}"
  printf 'STAGE1_5_CHECKPOINT=%s\n' "${STAGE1_5_CHECKPOINT}"
  printf 'A0_INIT_CHECKPOINT=%s\n' "${A0_INIT_CHECKPOINT}"
  printf 'A0_REFERENCE_CHECKPOINT=%s\n' "${A0_REFERENCE_CHECKPOINT}"
  printf 'OUTPUT_DIR=%s\n' "${OUTPUT_DIR}"
  printf 'MASTER_PORT=%s\n' "${MASTER_PORT}"
} > "${OUTPUT_DIR}/wrapper_commands.log"

bash "${REPO_ROOT}/scripts/run_last_rd_progressive_sft_8gpu.sh"
