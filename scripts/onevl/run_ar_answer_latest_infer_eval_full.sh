#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANSWER_SWIFT_DIR=${ANSWER_SWIFT_DIR:-/mnt/project/onevl_navsim_exp/answer_full_20260624_184133/swift_output/v0-20260624-184217}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
CHECKPOINT_VALIDATION_DIR=${CHECKPOINT_VALIDATION_DIR:-/tmp/onevl_ar_answer_latest_checkpoint_validation}

if [ ! -d "${ANSWER_SWIFT_DIR}" ]; then
  echo "Missing ANSWER_SWIFT_DIR: ${ANSWER_SWIFT_DIR}" >&2
  exit 1
fi

mapfile -t checkpoints < <(
  find "${ANSWER_SWIFT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | sort -Vr
)

if [ "${#checkpoints[@]}" -eq 0 ]; then
  echo "No checkpoint-* directories found under ${ANSWER_SWIFT_DIR}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_VALIDATION_DIR}"
latest_checkpoint=""
for candidate in "${checkpoints[@]}"; do
  candidate_path="${ANSWER_SWIFT_DIR}/${candidate}"
  report_path="${CHECKPOINT_VALIDATION_DIR}/${candidate}.validation.json"
  log_path="${CHECKPOINT_VALIDATION_DIR}/${candidate}.validation.log"
  if TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_hf_checkpoint.py" \
    --model-path "${candidate_path}" \
    --require-tokenizer \
    --transformers-smoke \
    --report-json "${report_path}" \
    > "${log_path}" 2>&1; then
    latest_checkpoint="${candidate}"
    break
  fi
  echo "Skipping invalid checkpoint ${candidate}; see ${log_path}" >&2
done

if [ -z "${latest_checkpoint}" ]; then
  echo "No valid checkpoint found under ${ANSWER_SWIFT_DIR}" >&2
  exit 1
fi

export MODEL_PATH="${ANSWER_SWIFT_DIR}/${latest_checkpoint}"
export OUT_ROOT=${OUT_ROOT:-/mnt/project/onevl_navsim_exp/ar_answer_${latest_checkpoint}_navtest_eval_$(date +%Y%m%d_%H%M%S)}

echo "Using latest valid AR Answer checkpoint: ${MODEL_PATH}"
echo "Output root: ${OUT_ROOT}"
exec "${SCRIPT_DIR}/run_ar_answer_infer_eval_full.sh"
