#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

bash -n "${REPO_ROOT}/scripts/run_last_rd_stage1_5_8gpu.sh"
bash -n "${REPO_ROOT}/scripts/run_last_rd_progressive_sft_8gpu.sh"

STAGE1_OUT="${TMP_ROOT}/stage1_5"
BASE_CONFIG=last_rd_stage1_5 \
TRAIN_CHUNK_CACHE_ROOT="${TMP_ROOT}/fake_train_cache" \
TRAIN_CHUNK_NAME_PATTERN='navtrain_chunk_*' \
OUTPUT_DIR="${STAGE1_OUT}" \
MASTER_PORT=29531 \
LAST_RD_DRY_RUN=1 \
bash "${REPO_ROOT}/scripts/run_last_rd_stage1_5_8gpu.sh" >/tmp/last_rd_stage1_5_launcher_dryrun.log

grep -q "Recommended preflight: python scripts/audit_last_rd_cache_manifest.py" "${STAGE1_OUT}/commands.log"

set +e
BASE_CONFIG=last_rd_progressive_sft \
STAGE1_5_CHECKPOINT="${TMP_ROOT}/fake_adapter.pt" \
TRAIN_CHUNK_CACHE_ROOT="${TMP_ROOT}/fake_train_cache" \
OUTPUT_DIR="${TMP_ROOT}/progressive_fail" \
MASTER_PORT=29532 \
LAST_RD_DRY_RUN=1 \
bash "${REPO_ROOT}/scripts/run_last_rd_progressive_sft_8gpu.sh" >/tmp/last_rd_progressive_launcher_fail.log 2>&1
FAIL_STATUS=$?
set -e
if [[ "${FAIL_STATUS}" -eq 0 ]]; then
  echo "Progressive launcher did not fail fast when KD was enabled without A0_REFERENCE_CHECKPOINT." >&2
  exit 1
fi
grep -q "policy KD is enabled" /tmp/last_rd_progressive_launcher_fail.log

PROGRESSIVE_OUT="${TMP_ROOT}/progressive_ok"
BASE_CONFIG=last_rd_progressive_sft \
STAGE1_5_CHECKPOINT="${TMP_ROOT}/fake_adapter.pt" \
TRAIN_CHUNK_CACHE_ROOT="${TMP_ROOT}/fake_train_cache" \
OUTPUT_DIR="${PROGRESSIVE_OUT}" \
MASTER_PORT=29532 \
LAST_RD_POLICY_KD_WEIGHT=0 \
LAST_RD_DRY_RUN=1 \
bash "${REPO_ROOT}/scripts/run_last_rd_progressive_sft_8gpu.sh" >/tmp/last_rd_progressive_launcher_dryrun.log

grep -q "LAST_RD_POLICY_KD_MODE=none" "${PROGRESSIVE_OUT}/commands.log"
grep -q "Recommended preflight: python scripts/audit_last_rd_cache_manifest.py" "${PROGRESSIVE_OUT}/commands.log"

echo "LaST-RD launcher dry-run checks passed."
