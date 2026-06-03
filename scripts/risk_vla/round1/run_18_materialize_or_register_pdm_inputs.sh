#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
DRY_RUN="${DRY_RUN:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
BIT_WORK_ROOT="${BIT_WORK_ROOT:-/mnt/project/bit_drive_left_tail}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
SHARED_CHUNK_CACHE_ROOT="${SHARED_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/project/VLA-AD/checkpoints}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
STAGE_DIR="${STAGE_DIR:-${ROUND_DIR}/stage7_pdm_inputs}"
PDM_INPUT_DIR="${PDM_INPUT_DIR:-${ROUND_DIR}/pdm_inputs}"
MANIFEST_YAML="${MANIFEST_YAML:-${ROUND_DIR}/round1_input_manifest.local.yaml}"

mkdir -p "${STAGE_DIR}" "${PDM_INPUT_DIR}" "${ROUND_DIR}"

echo "[run_18] EXECUTE=${EXECUTE}; default is dry-run."
echo "[run_18] stage dir: ${STAGE_DIR}"

python scripts/risk_vla/find_pdm_csv_candidates.py \
  --search-root "${BIT_EXP_ROOT}" \
  --search-root "${BIT_WORK_ROOT}" \
  --search-root "${CHECKPOINT_ROOT}" \
  --output-csv "${STAGE_DIR}/pdm_csv_candidates.csv" \
  --output-md "${STAGE_DIR}/pdm_csv_candidates.md" \
  --max-depth "${MAX_DEPTH:-5}" || true

ENV_A0="${A0_PDM_CSV:-}"
ENV_B3="${B3_PDM_CSV:-}"
ENV_TRAIN="${TRAIN_PDM_CSV:-}"
ENV_VAL="${VAL_PDM_CSV:-}"

GEN_A0="${PDM_INPUT_DIR}/A0_base/pdm.csv"
GEN_B3="${PDM_INPUT_DIR}/B3_direct_bit/pdm.csv"
GEN_TRAIN="${PDM_INPUT_DIR}/train/pdm.csv"
GEN_VAL="${PDM_INPUT_DIR}/val/pdm.csv"

if [[ ! -f "${ENV_A0}" && -f "${GEN_A0}" ]]; then ENV_A0="${GEN_A0}"; fi
if [[ ! -f "${ENV_B3}" && -f "${GEN_B3}" ]]; then ENV_B3="${GEN_B3}"; fi
if [[ ! -f "${ENV_TRAIN}" && -f "${GEN_TRAIN}" ]]; then ENV_TRAIN="${GEN_TRAIN}"; fi
if [[ ! -f "${ENV_VAL}" && -f "${GEN_VAL}" ]]; then ENV_VAL="${GEN_VAL}"; fi

if [[ -f "${ENV_A0}" || -f "${ENV_B3}" || -f "${ENV_TRAIN}" || -f "${ENV_VAL}" ]]; then
  python scripts/risk_vla/build_round1_manifest_from_env.py \
    --output-yaml "${MANIFEST_YAML}" \
    --a0-pdm-csv "${ENV_A0}" \
    --b3-pdm-csv "${ENV_B3}" \
    --train-pdm-csv "${ENV_TRAIN}" \
    --val-pdm-csv "${ENV_VAL}" \
    --source-chunk-cache-dir "${SHARED_CHUNK_CACHE_ROOT}" \
    --overlay-output-dir "${BIT_CACHE_ROOT}/risk_vla_round1_overlay" \
    --checkpoint "${B3_CHECKPOINT:-${A0_CHECKPOINT:-}}"

  set +e
  python scripts/risk_vla/validate_round1_input_manifest.py \
    --input-yaml "${MANIFEST_YAML}" \
    --output-json "${ROUND_DIR}/input_manifest_resolved.json" \
    --output-md "${ROUND_DIR}/input_manifest_resolved.md"
  manifest_status=$?
  set -e
else
  manifest_status=99
fi

python scripts/risk_vla/build_small_pdm_eval_commands.py \
  --output-dir "${PDM_INPUT_DIR}" \
  --a0-checkpoint "${A0_CHECKPOINT:-}" \
  --b3-checkpoint "${B3_CHECKPOINT:-}" \
  --cache-path "${SHARED_CHUNK_CACHE_ROOT}" \
  --metric-cache-path "${METRIC_CACHE_PATH:-}" \
  --navsim-log-path "${NAVSIM_LOG_PATH:-}" \
  --sensor-blobs-path "${SENSOR_BLOBS_PATH:-}" \
  --vlm-path "${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-}}" \
  --split "${PDM_ANALYSIS_SPLIT:-navval}" \
  --train-split "${PDM_TRAIN_SPLIT:-navtrain}" \
  --val-split "${PDM_VAL_SPLIT:-navval}" \
  --max-samples "${MAX_SAMPLES}" \
  --devices "${DEVICES:-1}" \
  --master-port "${MASTER_PORT:-29671}"

if [[ "${EXECUTE}" == "1" ]]; then
  if grep -q '^-' "${PDM_INPUT_DIR}/pdm_generation_blockers.md"; then
    echo "[run_18] PDM generation blocked; see ${PDM_INPUT_DIR}/pdm_generation_blockers.md" >&2
    exit 1
  fi
  EXECUTE=1 MAX_SAMPLES="${MAX_SAMPLES}" bash "${PDM_INPUT_DIR}/commands/generate_a0_pdm.sh"
  EXECUTE=1 MAX_SAMPLES="${MAX_SAMPLES}" bash "${PDM_INPUT_DIR}/commands/generate_b3_pdm.sh"
  EXECUTE=1 MAX_SAMPLES="${MAX_SAMPLES}" bash "${PDM_INPUT_DIR}/commands/generate_train_pdm.sh"
  EXECUTE=1 MAX_SAMPLES="${MAX_SAMPLES}" bash "${PDM_INPUT_DIR}/commands/generate_val_pdm.sh"
  A0_PDM_CSV="${GEN_A0}" B3_PDM_CSV="${GEN_B3}" TRAIN_PDM_CSV="${GEN_TRAIN}" VAL_PDM_CSV="${GEN_VAL}" EXECUTE=0 DRY_RUN=1 bash "$0"
  exit 0
fi

{
  echo "# RISK-VLA Stage 7 PDM Input Status"
  echo
  echo "PDM CSVs are generated evaluation artifacts. This wrapper registers existing CSVs when provided and writes small-scale generation commands when they are missing."
  echo
  echo "- execute: \`${EXECUTE}\`"
  echo "- max samples: \`${MAX_SAMPLES}\`"
  echo "- manifest: \`${MANIFEST_YAML}\`"
  echo
  echo "## Candidate Search"
  echo
  cat "${STAGE_DIR}/pdm_csv_candidates.md"
  echo
  echo "## Manifest Status"
  echo
  if [[ -f "${ROUND_DIR}/input_manifest_resolved.md" ]]; then
    cat "${ROUND_DIR}/input_manifest_resolved.md"
  else
    echo "_No manifest was created because no existing/generated PDM CSVs were found._"
  fi
  echo
  echo "## PDM Generation Commands"
  echo
  cat "${PDM_INPUT_DIR}/small_pdm_eval_commands.md"
  echo
  echo "## Status"
  echo
  if [[ "${manifest_status}" == "0" ]]; then
    echo "A manifest was created and passed validation. Check whether all required A0/B3/train/val CSVs are present before R0."
  else
    echo "No complete validated manifest is ready. Provide existing PDM CSVs or resolve blockers and run small-scale PDM generation with EXECUTE=1."
  fi
} >"${STAGE_DIR}/stage7_pdm_input_status.md"

echo "[run_18] status report: ${STAGE_DIR}/stage7_pdm_input_status.md"
