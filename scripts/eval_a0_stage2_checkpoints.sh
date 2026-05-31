#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  CONFIG
  CHECKPOINT_DIR
  EVAL_CHUNK_CACHE_ROOT
  EVAL_CHUNK_NAME_PATTERN
  METRIC_CACHE_DIR
  EVAL_OUTPUT_DIR
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
COMMANDS_LOG="${COMMANDS_LOG:-$(dirname "${EVAL_OUTPUT_DIR}")/commands.log}"
LOG_DIR="${LOG_DIR:-$(dirname "${EVAL_OUTPUT_DIR}")/logs}"
WAIT_FOR_CHECKPOINTS="${WAIT_FOR_CHECKPOINTS:-0}"
ALLOW_MISSING_CHECKPOINTS="${ALLOW_MISSING_CHECKPOINTS:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PARALLEL_EVAL="${PARALLEL_EVAL:-1}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
INCLUDE_TOPK_CHECKPOINTS="${INCLUDE_TOPK_CHECKPOINTS:-1}"
TOPK_CHECKPOINT_DIR="${TOPK_CHECKPOINT_DIR:-${CHECKPOINT_DIR}/lightning_logs/version_0/checkpoints}"

mkdir -p "${EVAL_OUTPUT_DIR}" "${LOG_DIR}" "$(dirname "${COMMANDS_LOG}")"

resolve_ckpt() {
  local step_name="$1"
  local ckpt=""
  case "${step_name}" in
    topk_*)
      local topk_base="${step_name#topk_}"
      ckpt="${TOPK_CHECKPOINT_DIR}/${topk_base}.ckpt"
      ;;
    final)
      for candidate in \
        "${CHECKPOINT_DIR}/latest.ckpt" \
        "${CHECKPOINT_DIR}/final.ckpt" \
        "${CHECKPOINT_DIR}/last.ckpt"; do
        if [[ -f "${candidate}" ]]; then
          ckpt="${candidate}"
          break
        fi
      done
      ;;
    *)
      local step_num="${step_name#step_}"
      ckpt="${CHECKPOINT_DIR}/step_${step_num}.ckpt"
      if [[ ! -f "${ckpt}" ]]; then
        ckpt="$(find "${CHECKPOINT_DIR}" -maxdepth 4 -type f -name "step_${step_num}*.ckpt" | sort | head -n 1 || true)"
      fi
      ;;
  esac
  printf '%s' "${ckpt}"
}

wait_for_ckpt() {
  local name="$1"
  local ckpt
  ckpt="$(resolve_ckpt "${name}")"
  while [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; do
    if [[ "${WAIT_FOR_CHECKPOINTS}" != "1" ]]; then
      if [[ "${ALLOW_MISSING_CHECKPOINTS}" == "1" ]]; then
        echo "Skipping missing checkpoint ${name} under ${CHECKPOINT_DIR}" >&2
        return 2
      fi
      echo "Missing checkpoint for ${name} under ${CHECKPOINT_DIR}" >&2
      return 1
    fi
    echo "Waiting for checkpoint ${name} under ${CHECKPOINT_DIR}" >&2
    sleep "${POLL_SECONDS}"
    ckpt="$(resolve_ckpt "${name}")"
  done
  printf '%s' "${ckpt}"
}

names=(
  step_00050000
  step_00060000
  step_00080000
  step_00100000
  step_00120000
  step_00140000
  step_00160000
  final
)

discover_topk_names() {
  if [[ "${INCLUDE_TOPK_CHECKPOINTS}" != "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${CHECKPOINT_DIR}" "${TOPK_CHECKPOINT_DIR}" <<'PY'
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
topk_dir = Path(sys.argv[2])
paths = []

state_candidates = [
    checkpoint_dir / "latest.ckpt",
    checkpoint_dir / "final.ckpt",
    checkpoint_dir / "last.ckpt",
    topk_dir / "last.ckpt",
]

for state_path in state_candidates:
    if not state_path.is_file():
        continue
    try:
        import torch
        ckpt = torch.load(state_path, map_location="cpu")
    except Exception:
        continue
    callbacks = ckpt.get("callbacks", {}) if isinstance(ckpt, dict) else {}
    for state in callbacks.values():
        if not isinstance(state, dict):
            continue
        best_k = state.get("best_k_models")
        if not isinstance(best_k, dict):
            continue
        for raw_path in best_k:
            p = Path(raw_path)
            if not p.is_file():
                p = topk_dir / p.name
            if p.is_file() and p.name != "last.ckpt":
                paths.append(p)
    if paths:
        break

if not paths and topk_dir.is_dir():
    paths = sorted(p for p in topk_dir.glob("epoch=*-step=*.ckpt") if p.is_file())

seen = set()
for p in sorted(paths, key=lambda item: item.name):
    if p in seen:
        continue
    seen.add(p)
    print(f"topk_{p.stem}")
PY
}

if [[ "${INCLUDE_TOPK_CHECKPOINTS}" == "1" ]]; then
  while IFS= read -r topk_name; do
    [[ -n "${topk_name}" ]] || continue
    names+=("${topk_name}")
  done < <(discover_topk_names)
fi

run_eval_one() {
  local name="$1"
  local gpu="${2:-}"
  local ckpt
  if ! ckpt="$(wait_for_ckpt "${name}")"; then
    if [[ "${ALLOW_MISSING_CHECKPOINTS}" == "1" ]]; then
      return 0
    fi
    return 1
  fi
  local out_dir="${EVAL_OUTPUT_DIR}/${name}"
  local log_file="${LOG_DIR}/${name}.eval.log"
  if [[ -f "${out_dir}/metrics.json" && "${SKIP_COMPLETED:-1}" == "1" ]]; then
    echo "Skipping completed eval: ${out_dir}/metrics.json"
    return 0
  fi
  local cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${ckpt}"
    --chunk-cache-root "${EVAL_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${EVAL_CHUNK_NAME_PATTERN}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision fp32
    --output-dir "${out_dir}"
  )
  {
    printf '[%s] ' "$(date -Is)"
    if [[ -n "${gpu}" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${COMMANDS_LOG}"
  echo "Evaluating ${name}: ${ckpt}${gpu:+ on GPU ${gpu}}"
  if [[ -n "${gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1
  else
    "${cmd[@]}" >"${log_file}" 2>&1
  fi
}

if [[ "${PARALLEL_EVAL}" == "1" ]]; then
  IFS=',' read -r -a gpu_list <<<"${EVAL_GPUS}"
  if [[ "${#gpu_list[@]}" -eq 0 ]]; then
    echo "PARALLEL_EVAL=1 requires EVAL_GPUS to contain at least one GPU id." >&2
    exit 2
  fi
  pids=()
  for i in "${!names[@]}"; do
    gpu="${gpu_list[$((i % ${#gpu_list[@]}))]}"
    run_eval_one "${names[$i]}" "${gpu}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  exit "${failed}"
else
  for name in "${names[@]}"; do
    run_eval_one "${name}" ""
  done
fi
