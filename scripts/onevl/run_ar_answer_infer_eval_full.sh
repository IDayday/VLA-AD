#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
REPO_ROOT=${REPO_ROOT:-/mnt/project/OneVL_training}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to an AR Answer checkpoint, for example /mnt/project/onevl_navsim_exp/answer_full_20260624_184133/swift_output/v0-20260624-184217/checkpoint-1000}
TEST_SET_PATH=${TEST_SET_PATH:-/mnt/project/onevl/test_data/navsim_test.json}
OUT_ROOT=${OUT_ROOT:-/mnt/project/onevl_navsim_exp/ar_answer_infer_eval_$(date +%Y%m%d_%H%M%S)}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
NAVSIM_LOG_PATH=${NAVSIM_LOG_PATH:-/mnt/project/onevl_navsim_data/navsim_logs/test}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/mnt/project/onevl_navsim_exp/metric_cache}
RUN_PDM_EVAL=${RUN_PDM_EVAL:-1}
PDM_NUM_WORKERS=${PDM_NUM_WORKERS:-${NPROC_PER_NODE}}
PDM_CHUNK_MULTIPLIER=${PDM_CHUNK_MULTIPLIER:-4}
SKIP_INPUT_VALIDATION=${SKIP_INPUT_VALIDATION:-0}
SKIP_CHECKPOINT_VALIDATION=${SKIP_CHECKPOINT_VALIDATION:-0}
INVALID_TRAJECTORY_POLICY=${INVALID_TRAJECTORY_POLICY:-error}

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing MODEL_PATH directory: ${MODEL_PATH}" >&2
  exit 1
fi
if compgen -G "${MODEL_PATH}/*.aria2" > /dev/null; then
  echo "MODEL_PATH has incomplete .aria2 files: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${TEST_SET_PATH}" ]; then
  echo "Missing TEST_SET_PATH: ${TEST_SET_PATH}" >&2
  exit 1
fi
if [ ! -d "${NAVSIM_LOG_PATH}" ]; then
  echo "Missing NAVSIM_LOG_PATH: ${NAVSIM_LOG_PATH}" >&2
  exit 1
fi
if [ ! -d "${METRIC_CACHE_PATH}" ]; then
  echo "Missing METRIC_CACHE_PATH: ${METRIC_CACHE_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/shards"
{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "$0" "$@"
  printf '\n'
  env | sort | grep -E '^(PYTHON_BIN|REPO_ROOT|MODEL_PATH|TEST_SET_PATH|OUT_ROOT|NPROC_PER_NODE|MAX_NEW_TOKENS|NAVSIM_LOG_PATH|METRIC_CACHE_PATH|RUN_PDM_EVAL|PDM_NUM_WORKERS|PDM_CHUNK_MULTIPLIER|SKIP_INPUT_VALIDATION|SKIP_CHECKPOINT_VALIDATION|INVALID_TRAJECTORY_POLICY)=' || true
} >> "${OUT_ROOT}/commands.log"

if [ "${SKIP_CHECKPOINT_VALIDATION}" != "1" ]; then
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_hf_checkpoint.py" \
    --model-path "${MODEL_PATH}" \
    --require-tokenizer \
    --transformers-smoke \
    --report-json "${OUT_ROOT}/checkpoint.validation.json" \
    > "${OUT_ROOT}/logs/validate_checkpoint.log" 2>&1
fi

if [ "${SKIP_INPUT_VALIDATION}" != "1" ]; then
  PYTHONPATH="/mnt:${PYTHONPATH:-}" "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_navsim_infer_inputs.py" \
    --test-set-path "${TEST_SET_PATH}" \
    --repo-root "${REPO_ROOT}" \
    --navsim-log-path "${NAVSIM_LOG_PATH}" \
    --metric-cache-path "${METRIC_CACHE_PATH}" \
    --report-json "${OUT_ROOT}/navsim_input_alignment.report.json" \
    > "${OUT_ROOT}/logs/validate_inputs.log" 2>&1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/split_infer_dataset.py" \
  --input "${TEST_SET_PATH}" \
  --out-dir "${OUT_ROOT}/shards" \
  --num-shards "${NPROC_PER_NODE}" \
  > "${OUT_ROOT}/logs/split.log" 2>&1

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PIDS=()
for shard_id in $(seq 0 $((NPROC_PER_NODE - 1))); do
  shard_input="${OUT_ROOT}/shards/shard_${shard_id}.json"
  shard_output="${OUT_ROOT}/shards/predict_${shard_id}.json"
  if [ ! -s "${shard_input}" ]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${shard_id}" "${PYTHON_BIN}" \
    "${REPO_ROOT}/run_script/infer/qwen3_vl_infer.py" \
    --model_path "${MODEL_PATH}" \
    --test_set_path "${shard_input}" \
    --output_path "${shard_output}" \
    --device cuda:0 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    > "${OUT_ROOT}/logs/infer_${shard_id}.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done
if [ "${FAIL}" -ne 0 ]; then
  echo "one or more AR Answer inference shards failed; see ${OUT_ROOT}/logs" >&2
  exit 1
fi

MERGED_JSON="${OUT_ROOT}/ar_answer_predictions_merged.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_sharded_predictions.py" \
  --shard-dir "${OUT_ROOT}/shards" \
  --output-json "${MERGED_JSON}" \
  > "${OUT_ROOT}/logs/merge.log" 2>&1

SUBMISSION_PKL="${OUT_ROOT}/navsim_results_submission.pkl"
PYTHONPATH="/mnt:${PYTHONPATH:-}" "${PYTHON_BIN}" "${SCRIPT_DIR}/navsim_predictions_to_submission.py" \
  --input-json "${MERGED_JSON}" \
  --output-pkl "${SUBMISSION_PKL}" \
  --navsim-log-path "${NAVSIM_LOG_PATH}" \
  --metric-cache-path "${METRIC_CACHE_PATH}" \
  --invalid-trajectory-policy "${INVALID_TRAJECTORY_POLICY}" \
  > "${OUT_ROOT}/logs/convert_submission.log" 2>&1

if [ "${RUN_PDM_EVAL}" = "1" ]; then
  EVAL_DIR="${OUT_ROOT}/pdm_eval"
  mkdir -p "${EVAL_DIR}"
  if [ "${PDM_NUM_WORKERS}" -gt 1 ]; then
    env \
      PYTHONPATH="/mnt:${PYTHONPATH:-}" \
      OPENSCENE_DATA_ROOT=/mnt/project/onevl_navsim_data \
      NAVSIM_EXP_ROOT=/mnt/project/onevl_navsim_exp \
      NUPLAN_MAPS_ROOT=/mnt/navsim/maps \
      NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
      "${PYTHON_BIN}" "${SCRIPT_DIR}/pdm_score_from_submission_parallel.py" \
        --metric-cache-path "${METRIC_CACHE_PATH}" \
        --submission-file-path "${SUBMISSION_PKL}" \
        --output-dir "${EVAL_DIR}" \
        --num-workers "${PDM_NUM_WORKERS}" \
        --chunk-multiplier "${PDM_CHUNK_MULTIPLIER}" \
        --report-json "${OUT_ROOT}/pdm_eval.report.json" \
        > "${OUT_ROOT}/logs/pdm_eval.log" 2>&1
  else
    env \
      PYTHONPATH="/mnt:${PYTHONPATH:-}" \
      OPENSCENE_DATA_ROOT=/mnt/project/onevl_navsim_data \
      NAVSIM_EXP_ROOT=/mnt/project/onevl_navsim_exp \
      NUPLAN_MAPS_ROOT=/mnt/navsim/maps \
      NUPLAN_MAP_VERSION=nuplan-maps-v1.0 \
      "${PYTHON_BIN}" /mnt/navsim/planning/script/run_pdm_score_from_submission.py \
        metric_cache_path="${METRIC_CACHE_PATH}" \
        submission_file_path="${SUBMISSION_PKL}" \
        output_dir="${EVAL_DIR}" \
        > "${OUT_ROOT}/logs/pdm_eval.log" 2>&1
  fi
fi

ln -sfn "${OUT_ROOT}" /mnt/project/onevl_navsim_exp/ar_answer_infer_eval_latest
echo "merged predictions: ${MERGED_JSON}"
echo "submission: ${SUBMISSION_PKL}"
if [ "${RUN_PDM_EVAL}" = "1" ]; then
  echo "pdm eval dir: ${OUT_ROOT}/pdm_eval"
fi
