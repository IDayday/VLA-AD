#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
REMOTE_HOST="${REMOTE_HOST:-training-vla-zt-peer}"
REMOTE_PROJECT_ROOT="${REMOTE_PROJECT_ROOT:-${PROJECT_ROOT}}"
SMOKE="${SMOKE:-0}"
OUT_ROOT="${OUT_ROOT:-outputs/a4_min_cont_8gpu_dist_$(date +%Y%m%d_%H%M%S)}"
OPT_STEPS="${OPT_STEPS:-5000}"
PER_GPU_BATCH="${PER_GPU_BATCH:-16}"
GRAD_ACC="${GRAD_ACC:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SEED="${SEED:-20260530}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
START_EPOCH_120K="${START_EPOCH_120K:-149}"
START_EPOCH_160K="${START_EPOCH_160K:-198}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
EVAL_GPU="${EVAL_GPU:-0}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29831}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
WORKER_MODE="${WORKER_MODE:-0}"
WORKER_START="${WORKER_START:-}"
WORKER_LABEL="${WORKER_LABEL:-local}"

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

CONT_CONFIG="${OUT_ROOT}/configs/a4_cont_noalign.yaml"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

q() {
  printf '%q' "$1"
}

log_command() {
  local host="$1"
  local gpu="$2"
  shift 2
  {
    printf '[%s] CUDA_VISIBLE_DEVICES=%q ' "${host}" "${gpu}"
    printf '%q ' "$@"
    printf '\n'
  } >> "${COMMANDS_LOG}"
}

append_eval_max_samples() {
  if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
    eval_cmd+=(--max-samples "${EVAL_MAX_SAMPLES}")
  fi
}

start_epoch_for() {
  case "$1" in
    120) printf '%s\n' "${START_EPOCH_120K}" ;;
    160) printf '%s\n' "${START_EPOCH_160K}" ;;
    *) printf 'Unknown start key: %s\n' "$1" >&2; exit 4 ;;
  esac
}

ckpt_for() {
  case "$1" in
    120) printf '%s\n' "${CKPT_120K}" ;;
    160) printf '%s\n' "${CKPT_160K}" ;;
    *) printf 'Unknown start key: %s\n' "$1" >&2; exit 4 ;;
  esac
}

run_name_for() {
  local start="$1"
  local condition="$2"
  case "${condition}" in
    C1) printf 'start%s_C1_truebf16_tail\n' "${start}" ;;
    C2) printf 'start%s_C2_mixed_tail\n' "${start}" ;;
    C3) printf 'start%s_C3_mixed_const_expertlr\n' "${start}" ;;
    *) printf 'Unknown condition: %s\n' "${condition}" >&2; exit 4 ;;
  esac
}

condition_args() {
  local condition="$1"
  local start_epoch="$2"
  case "${condition}" in
    C1)
      printf '%s\n' \
        --true-bf16-weights \
        --lr-scheduler official-cosine \
        --lr-scheduler-epochs 200 \
        --lr-scheduler-start-epoch "${start_epoch}" \
        --lr-warmup-epochs 3 \
        --min-lr 1e-6 \
        --lr-action-head 1e-4 \
        --lr-expert 1e-4
      ;;
    C2)
      printf '%s\n' \
        --lr-scheduler official-cosine \
        --lr-scheduler-epochs 200 \
        --lr-scheduler-start-epoch "${start_epoch}" \
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
    *)
      printf 'Unknown condition: %s\n' "${condition}" >&2
      exit 4
      ;;
  esac
}

run_train() {
  local start="$1"
  local condition="$2"
  local condition_index="$3"
  local run_name ckpt start_epoch log_path port
  run_name="$(run_name_for "${start}" "${condition}")"
  ckpt="$(ckpt_for "${start}")"
  start_epoch="$(start_epoch_for "${start}")"
  log_path="${OUT_ROOT}/logs/${run_name}.train.log"
  port=$((MASTER_PORT_BASE + condition_index))

  local -a train_cmd=(
    "${TORCHRUN}"
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_port "${port}"
    scripts/train_recogdrive_expert_chunked.py
    --config "${CONT_CONFIG}"
    --resume-from "${ckpt}"
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
    --save-every "${SAVE_EVERY}"
    --seed "${SEED}"
    --output-dir "${OUT_ROOT}/runs/${run_name}"
  )
  while IFS= read -r arg; do
    train_cmd+=("${arg}")
  done < <(condition_args "${condition}" "${start_epoch}")

  log_command "${WORKER_LABEL}:${run_name}:train" "${GPUS}" "${train_cmd[@]}"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  "${train_cmd[@]}" > "${log_path}" 2>&1
}

run_eval() {
  local start="$1"
  local run_name="$2"
  local checkpoint="$3"
  local log_path="${OUT_ROOT}/logs/${run_name}.eval.log"
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
  log_command "${WORKER_LABEL}:${run_name}:eval" "${EVAL_GPU}" "${eval_cmd[@]}"
  PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" \
  "${eval_cmd[@]}" > "${log_path}" 2>&1
}

run_worker() {
  if [[ "${WORKER_START}" != "120" && "${WORKER_START}" != "160" ]]; then
    printf 'WORKER_START must be 120 or 160 in WORKER_MODE.\n' >&2
    exit 5
  fi
  mkdir -p "${OUT_ROOT}/runs" "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
  local condition idx run_name baseline_name ckpt
  idx=0
  for condition in C1 C2 C3; do
    run_train "${WORKER_START}" "${condition}" "${idx}"
    run_name="$(run_name_for "${WORKER_START}" "${condition}")"
    run_eval "${WORKER_START}" "${run_name}" "${OUT_ROOT}/runs/${run_name}/latest.ckpt"
    idx=$((idx + 1))
  done
  ckpt="$(ckpt_for "${WORKER_START}")"
  baseline_name="start${WORKER_START}_baseline"
  run_eval "${WORKER_START}" "${baseline_name}" "${ckpt}"
}

write_meta() {
  export OUT_ROOT SMOKE OPT_STEPS PER_GPU_BATCH GRAD_ACC NUM_WORKERS PREFETCH_FACTOR SEED
  export EVAL_MAX_SAMPLES START_EPOCH_120K START_EPOCH_160K SAVE_EVERY
  export BASE_CONFIG CKPT_120K CKPT_160K TRAIN_CHUNK_CACHE_ROOT TRAIN_CHUNK_NAME_PATTERN
  export EVAL_CHUNK_CACHE_ROOT EVAL_CHUNK_NAME_PATTERN METRIC_CACHE_DIR NPROC_PER_NODE GPUS
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

def none_if_empty(value):
    return None if value == "" else value

out_root = Path(os.environ["OUT_ROOT"])
meta = {
    "out_root": str(out_root),
    "smoke": os.environ["SMOKE"] == "1",
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
    "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
    "effective_batch_size": int(os.environ["PER_GPU_BATCH"]) * int(os.environ["GRAD_ACC"]) * int(os.environ["NPROC_PER_NODE"]),
    "num_workers": int(os.environ["NUM_WORKERS"]),
    "prefetch_factor": int(os.environ["PREFETCH_FACTOR"]),
    "seed": int(os.environ["SEED"]),
    "eval_max_samples": none_if_empty(os.environ["EVAL_MAX_SAMPLES"]),
    "start_epoch_120k": int(os.environ["START_EPOCH_120K"]),
    "start_epoch_160k": int(os.environ["START_EPOCH_160K"]),
    "save_every": int(os.environ["SAVE_EVERY"]),
    "gpu_assignment": {
        "local_start120": os.environ["GPUS"],
        "remote_start160": os.environ["GPUS"],
    },
}
(out_root / "matrix_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

preflight_remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd $(q "${REMOTE_PROJECT_ROOT}") && $(q "${PYTHON}") - <<'PY'
import torch
print(torch.cuda.device_count())
PY"
}

launch_remote_worker() {
  local remote_cmd
  remote_cmd=$(
    cat <<EOF
cd $(q "${REMOTE_PROJECT_ROOT}") && \
BASE_CONFIG=$(q "${BASE_CONFIG}") \
CKPT_120K=$(q "${CKPT_120K}") \
CKPT_160K=$(q "${CKPT_160K}") \
TRAIN_CHUNK_CACHE_ROOT=$(q "${TRAIN_CHUNK_CACHE_ROOT}") \
TRAIN_CHUNK_NAME_PATTERN=$(q "${TRAIN_CHUNK_NAME_PATTERN}") \
EVAL_CHUNK_CACHE_ROOT=$(q "${EVAL_CHUNK_CACHE_ROOT}") \
EVAL_CHUNK_NAME_PATTERN=$(q "${EVAL_CHUNK_NAME_PATTERN}") \
METRIC_CACHE_DIR=$(q "${METRIC_CACHE_DIR}") \
OUT_ROOT=$(q "${OUT_ROOT}") \
PYTHON=$(q "${PYTHON}") \
TORCHRUN=$(q "${TORCHRUN}") \
SMOKE=$(q "${SMOKE}") \
OPT_STEPS=$(q "${OPT_STEPS}") \
PER_GPU_BATCH=$(q "${PER_GPU_BATCH}") \
GRAD_ACC=$(q "${GRAD_ACC}") \
NUM_WORKERS=$(q "${NUM_WORKERS}") \
PREFETCH_FACTOR=$(q "${PREFETCH_FACTOR}") \
SEED=$(q "${SEED}") \
EVAL_MAX_SAMPLES=$(q "${EVAL_MAX_SAMPLES}") \
START_EPOCH_120K=$(q "${START_EPOCH_120K}") \
START_EPOCH_160K=$(q "${START_EPOCH_160K}") \
SAVE_EVERY=$(q "${SAVE_EVERY}") \
NPROC_PER_NODE=$(q "${NPROC_PER_NODE}") \
GPUS=$(q "${GPUS}") \
EVAL_GPU=$(q "${EVAL_GPU}") \
MASTER_PORT_BASE=$(q "$((MASTER_PORT_BASE + 100))") \
WORKER_MODE=1 \
WORKER_START=160 \
WORKER_LABEL=remote:${REMOTE_HOST} \
bash scripts/run_a4_minimal_continuation_matrix.sh
EOF
  )
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${REMOTE_HOST}" "${remote_cmd}" \
    > "${OUT_ROOT}/logs/worker_start160.remote.log" 2>&1 &
  remote_pid=$!
}

main() {
  if [[ "${WORKER_MODE}" == "1" ]]; then
    run_worker
    return
  fi

  if [[ -e "${OUT_ROOT}" && "${ALLOW_OVERWRITE}" != "1" ]]; then
    printf 'OUT_ROOT already exists: %s\nSet ALLOW_OVERWRITE=1 to reuse it.\n' "${OUT_ROOT}" >&2
    exit 3
  fi

  mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/runs" "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
  : > "${COMMANDS_LOG}"
  "${PYTHON}" scripts/make_continuation_noalign_config.py \
    --base-config "${BASE_CONFIG}" \
    --output-config "${CONT_CONFIG}"
  "${PYTHON}" -m py_compile \
    scripts/train_recogdrive_expert_chunked.py \
    scripts/eval_recogdrive_expert_pdm.py \
    scripts/make_continuation_noalign_config.py \
    scripts/summarize_a4_minimal_continuation.py
  write_meta
  preflight_remote > "${OUT_ROOT}/logs/remote_preflight.log" 2>&1

  WORKER_MODE=1 WORKER_START=120 WORKER_LABEL="local:$(hostname)" \
    bash scripts/run_a4_minimal_continuation_matrix.sh \
    > "${OUT_ROOT}/logs/worker_start120.local.log" 2>&1 &
  local_pid=$!
  launch_remote_worker

  local failed=0
  wait "${local_pid}" || failed=1
  wait "${remote_pid}" || failed=1
  if (( failed != 0 )); then
    printf 'One or more worker containers failed. Inspect %s/logs.\n' "${OUT_ROOT}" >&2
    exit 1
  fi

  "${PYTHON}" scripts/summarize_a4_minimal_continuation.py --root "${OUT_ROOT}"
  printf 'OUT_ROOT=%s\nsummary.md=%s\nsummary.csv=%s\n' "${OUT_ROOT}" "${OUT_ROOT}/summary.md" "${OUT_ROOT}/summary.csv"
}

main "$@"
