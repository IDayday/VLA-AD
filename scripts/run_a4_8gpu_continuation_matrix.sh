#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
SERVER_GROUP="${SERVER_GROUP:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-outputs/a4_8gpu_cont_${SERVER_GROUP:-unset}_${STAMP}}"
OPT_STEPS="${OPT_STEPS:-5000}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SEED="${SEED:-20260530}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
START_EPOCH_120K="${START_EPOCH_120K:-149}"
START_EPOCH_160K="${START_EPOCH_160K:-198}"
MASTER_PORT="${MASTER_PORT:-29571}"
SMOKE="${SMOKE:-0}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  OPT_STEPS=2
  EVAL_MAX_SAMPLES=8
fi

required_env=(
  BASE_CONFIG
  CKPT_120K
  CKPT_160K
  TRAIN_CHUNK_CACHE_ROOT
  TRAIN_CHUNK_NAME_PATTERN
  EVAL_CHUNK_CACHE_ROOT
  EVAL_CHUNK_NAME_PATTERN
  METRIC_CACHE_DIR
  SERVER_GROUP
)
missing_env=()
for key in "${required_env[@]}"; do
  if [[ -z "${!key:-}" ]]; then
    missing_env+=("${key}")
  fi
done
if (( ${#missing_env[@]} > 0 )); then
  printf 'Missing required environment variables:\n' >&2
  printf '  - %s\n' "${missing_env[@]}" >&2
  exit 2
fi

case "${SERVER_GROUP}" in
  start120)
    START_KEY="120"
    CKPT="${CKPT_120K}"
    START_EPOCH="${START_EPOCH_120K}"
    ;;
  start160)
    START_KEY="160"
    CKPT="${CKPT_160K}"
    START_EPOCH="${START_EPOCH_160K}"
    ;;
  *)
    printf 'SERVER_GROUP must be start120 or start160; got %s\n' "${SERVER_GROUP}" >&2
    exit 2
    ;;
esac

if [[ -e "${OUT_ROOT}" && "${ALLOW_OVERWRITE}" != "1" ]]; then
  printf 'OUT_ROOT already exists: %s\nSet ALLOW_OVERWRITE=1 to reuse it.\n' "${OUT_ROOT}" >&2
  exit 3
fi

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/runs" "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
if [[ ! -f "${COMMANDS_LOG}" || "${ALLOW_OVERWRITE}" != "1" ]]; then
  : > "${COMMANDS_LOG}"
fi
CONT_CONFIG="${OUT_ROOT}/configs/a4_cont_noalign.yaml"

"${PYTHON}" scripts/make_continuation_noalign_config.py \
  --base-config "${BASE_CONFIG}" \
  --output-config "${CONT_CONFIG}"

"${PYTHON}" -m py_compile \
  scripts/train_recogdrive_expert_chunked.py \
  scripts/eval_recogdrive_expert_pdm.py \
  scripts/make_continuation_noalign_config.py \
  scripts/summarize_a4_8gpu_continuation.py

export OUT_ROOT SMOKE OPT_STEPS PER_GPU_BATCH GRAD_ACC NUM_WORKERS PREFETCH_FACTOR SEED SKIP_COMPLETED
export EVAL_MAX_SAMPLES START_EPOCH_120K START_EPOCH_160K SERVER_GROUP
export BASE_CONFIG CKPT_120K CKPT_160K TRAIN_CHUNK_CACHE_ROOT TRAIN_CHUNK_NAME_PATTERN
export EVAL_CHUNK_CACHE_ROOT EVAL_CHUNK_NAME_PATTERN METRIC_CACHE_DIR
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

def none_if_empty(value):
    return None if value == "" else value

out_root = Path(os.environ["OUT_ROOT"])
meta = {
    "out_root": str(out_root),
    "server_group": os.environ["SERVER_GROUP"],
	    "smoke": os.environ["SMOKE"] == "1",
	    "skip_completed": os.environ["SKIP_COMPLETED"] == "1",
    "base_config": os.environ["BASE_CONFIG"],
    "continuation_config": str(out_root / "configs" / "a4_cont_noalign.yaml"),
    "ckpt_120k": os.environ["CKPT_120K"],
    "ckpt_160k": os.environ["CKPT_160K"],
    "train_chunk_cache_root": os.environ["TRAIN_CHUNK_CACHE_ROOT"],
    "train_chunk_name_pattern": os.environ["TRAIN_CHUNK_NAME_PATTERN"],
    "eval_chunk_cache_root": os.environ["EVAL_CHUNK_CACHE_ROOT"],
    "eval_chunk_name_pattern": os.environ["EVAL_CHUNK_NAME_PATTERN"],
    "metric_cache_dir": os.environ["METRIC_CACHE_DIR"],
    "opt_steps": int(os.environ["OPT_STEPS"]),
    "per_gpu_batch": int(os.environ["PER_GPU_BATCH"]),
    "grad_acc": int(os.environ["GRAD_ACC"]),
    "world_size": 8,
    "effective_batch_size": int(os.environ["PER_GPU_BATCH"]) * int(os.environ["GRAD_ACC"]) * 8,
    "num_workers": int(os.environ["NUM_WORKERS"]),
    "prefetch_factor": int(os.environ["PREFETCH_FACTOR"]),
    "seed": int(os.environ["SEED"]),
    "eval_max_samples": none_if_empty(os.environ["EVAL_MAX_SAMPLES"]),
    "start_epoch_120k": int(os.environ["START_EPOCH_120K"]),
    "start_epoch_160k": int(os.environ["START_EPOCH_160K"]),
}
(out_root / "matrix_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log_command() {
  {
    printf '%q ' "$@"
    printf '\n'
  } >> "${COMMANDS_LOG}"
}

append_eval_max_samples() {
  if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    eval_cmd+=(--max-samples "${EVAL_MAX_SAMPLES}")
  fi
}

run_name_for() {
  case "$1" in
    C1) printf 'start%s_C1_truebf16_tail\n' "${START_KEY}" ;;
    C2) printf 'start%s_C2_mixed_tail\n' "${START_KEY}" ;;
    C3) printf 'start%s_C3_mixed_const_expertlr\n' "${START_KEY}" ;;
    *) printf 'Unknown condition: %s\n' "$1" >&2; exit 4 ;;
  esac
}

condition_args() {
  case "$1" in
    C1)
      printf '%s\n' \
        --true-bf16-weights \
        --lr-scheduler official-cosine \
        --lr-scheduler-epochs 200 \
        --lr-scheduler-start-epoch "${START_EPOCH}" \
        --lr-warmup-epochs 3 \
        --min-lr 1e-6 \
        --lr-action-head 1e-4 \
        --lr-expert 1e-4
      ;;
    C2)
      printf '%s\n' \
        --lr-scheduler official-cosine \
        --lr-scheduler-epochs 200 \
        --lr-scheduler-start-epoch "${START_EPOCH}" \
        --lr-warmup-epochs 3 \
        --min-lr 1e-6 \
        --lr-action-head 1e-4 \
        --lr-expert 1e-4
      ;;
    C3)
      printf '%s\n' \
        --lr-scheduler none \
        --lr-action-head 3e-5 \
        --lr-expert 1e-4 \
        --lr-expert-gate 5e-4
      ;;
  esac
}

train_one() {
  local condition="$1"
  local idx="$2"
	  local run_name log_path port
	  run_name="$(run_name_for "${condition}")"
	  log_path="${OUT_ROOT}/logs/${run_name}.train.log"
	  port=$((MASTER_PORT + idx))
	  if [[ "${SKIP_COMPLETED}" == "1" && -f "${OUT_ROOT}/runs/${run_name}/latest.ckpt" ]]; then
	    printf 'Skipping completed train: %s\n' "${run_name}" | tee -a "${log_path}"
	    return 0
	  fi

  local -a train_cmd=(
    "${TORCHRUN}" --nproc_per_node=8 --master_port "${port}"
    scripts/train_recogdrive_expert_chunked.py
    --config "${CONT_CONFIG}"
    --resume-from "${CKPT}"
    --resume-mode weights-only
    --chunk-cache-root "${TRAIN_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN}"
    --global-epochs 999
    --flat-global-dataset
    --num-optimizer-steps "${OPT_STEPS}"
    --batch-size "${PER_GPU_BATCH}"
    --gradient-accumulation-steps "${GRAD_ACC}"
    --num-workers "${NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --precision bf16
    --final-check-precision fp32
    --save-every 10000
    --seed "${SEED}"
    --output-dir "${OUT_ROOT}/runs/${run_name}"
  )
  while IFS= read -r arg; do
    train_cmd+=("${arg}")
  done < <(condition_args "${condition}")

  log_command "${train_cmd[@]}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${train_cmd[@]}" > "${log_path}" 2>&1
}

eval_one() {
	  local run_name="$1"
	  local checkpoint="$2"
	  local log_path="${OUT_ROOT}/logs/${run_name}.eval.log"
	  if [[ "${SKIP_COMPLETED}" == "1" && -f "${OUT_ROOT}/eval/${run_name}/metrics.json" ]]; then
	    printf 'Skipping completed eval: %s\n' "${run_name}" | tee -a "${log_path}"
	    return 0
	  fi
  eval_cmd=(
    "${PYTHON}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONT_CONFIG}"
    --checkpoint "${checkpoint}"
    --chunk-cache-root "${EVAL_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${EVAL_CHUNK_NAME_PATTERN}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision fp32
    --output-dir "${OUT_ROOT}/eval/${run_name}"
  )
  append_eval_max_samples
  log_command "${eval_cmd[@]}"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${eval_cmd[@]}" > "${log_path}" 2>&1
}

idx=0
for condition in C1 C2 C3; do
  run_name="$(run_name_for "${condition}")"
  train_one "${condition}" "${idx}"
  eval_one "${run_name}" "${OUT_ROOT}/runs/${run_name}/latest.ckpt"
  idx=$((idx + 1))
done

eval_one "start${START_KEY}_baseline" "${CKPT}"
"${PYTHON}" scripts/summarize_a4_8gpu_continuation.py --root "${OUT_ROOT}"

printf 'OUT_ROOT=%s\nsummary.md=%s\nsummary.csv=%s\n' "${OUT_ROOT}" "${OUT_ROOT}/summary.md" "${OUT_ROOT}/summary.csv"
sed -n '1,180p' "${OUT_ROOT}/summary.md"
