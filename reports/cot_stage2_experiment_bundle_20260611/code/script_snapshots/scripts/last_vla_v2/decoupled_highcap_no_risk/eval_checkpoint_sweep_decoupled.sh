#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUT_ROOT NAVTEST_CHUNK_CACHE_ROOT METRIC_CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/decoupled_highcap_no_risk_eval}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval.yaml}"
NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
PRECISION="${PRECISION:-fp32}"
INCLUDE_TOPK_CHECKPOINTS="${INCLUDE_TOPK_CHECKPOINTS:-1}"
INCLUDE_CHECKPOINTS_TO_EVAL="${INCLUDE_CHECKPOINTS_TO_EVAL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PARALLEL_EVAL="${PARALLEL_EVAL:-1}"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
MAX_PARALLEL_EVAL="${MAX_PARALLEL_EVAL:-}"
B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH:-}"
B_ONLINE_VLM_TYPE="${B_ONLINE_VLM_TYPE:-internvl}"
B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR:-}"
VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT:-}"

mkdir -p "${EVAL_ROOT}/logs"
COMMANDS_LOG="${EVAL_ROOT}/commands.log"

safe_name() {
  basename "$1" .ckpt | tr '/ :' '___'
}

line_specs() {
  # Current no-residual stage2 direct layout.
  if [[ -d "${OUT_ROOT}/A_frozen_vlm_stage2_progressive" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "A_frozen_vlm_stage2_progressive" \
      "${OUT_ROOT}/A_frozen_vlm_stage2_progressive" \
      "${OUT_ROOT}/A_frozen_vlm_stage2_progressive/checkpoints_to_eval.txt" \
      "${NAVTEST_CHUNK_CACHE_ROOT}" \
      "0"
  fi
  if [[ -d "${OUT_ROOT}/B_lora_stage2_progressive" ]]; then
    local b_cache="${LORA_NAVTEST_CHUNK_CACHE_ROOT:-${NAVTEST_CHUNK_CACHE_ROOT}}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "B_lora_stage2_progressive" \
      "${OUT_ROOT}/B_lora_stage2_progressive" \
      "${OUT_ROOT}/B_lora_stage2_progressive/checkpoints_to_eval.txt" \
      "${b_cache}" \
      "1"
  fi

  # Historical decoupled runbook layout.
  if [[ -d "${OUT_ROOT}/serverA_frozen_vlm_decoupled_highcap_no_risk/progressive_sft_decoupled" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "serverA_frozen_vlm_decoupled_highcap_no_risk" \
      "${OUT_ROOT}/serverA_frozen_vlm_decoupled_highcap_no_risk/progressive_sft_decoupled" \
      "${OUT_ROOT}/serverA_frozen_vlm_decoupled_highcap_no_risk/checkpoints_to_eval.txt" \
      "${NAVTEST_CHUNK_CACHE_ROOT}" \
      "0"
  fi
  if [[ -d "${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk/progressive_sft_decoupled" ]]; then
    local b_cache="${LORA_NAVTEST_CHUNK_CACHE_ROOT:-${NAVTEST_CHUNK_CACHE_ROOT}}"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "serverB_vlm_lora_decoupled_highcap_no_risk" \
      "${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk/progressive_sft_decoupled" \
      "${OUT_ROOT}/serverB_vlm_lora_decoupled_highcap_no_risk/checkpoints_to_eval.txt" \
      "${b_cache}" \
      "1"
  fi
}

discover_topk_ckpts() {
  local run_dir="$1"
  if [[ "${INCLUDE_TOPK_CHECKPOINTS}" != "1" && "${INCLUDE_TOPK_CHECKPOINTS}" != "true" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${run_dir}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
paths = []

state_candidates = [
    run_dir / "latest.ckpt",
    run_dir / "final.ckpt",
    run_dir / "last.ckpt",
]
state_candidates.extend(run_dir.glob("lightning_logs/version_*/checkpoints/last.ckpt"))

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
                candidate = run_dir / "lightning_logs" / "version_0" / "checkpoints" / p.name
                p = candidate if candidate.is_file() else p
            if p.is_file() and p.name != "last.ckpt":
                paths.append(p)
    if paths:
        break

if not paths:
    paths = sorted(run_dir.glob("lightning_logs/version_*/checkpoints/epoch=*-step=*.ckpt"))

seen = set()
for p in sorted(paths, key=lambda item: item.name):
    resolved = str(p.resolve())
    if resolved in seen:
        continue
    seen.add(resolved)
    print(resolved)
PY
}

collect_ckpts_for_line() {
  local run_dir="$1"
  local list_path="$2"
  if [[ "${INCLUDE_CHECKPOINTS_TO_EVAL}" == "1" || "${INCLUDE_CHECKPOINTS_TO_EVAL}" == "true" ]]; then
    if [[ -f "${list_path}" ]]; then
      while IFS= read -r ckpt; do
        [[ -n "${ckpt}" ]] && printf '%s\n' "${ckpt}"
      done <"${list_path}"
    fi
  fi
  discover_topk_ckpts "${run_dir}"
}

run_eval_one() {
  local line_name="$1"
  local run_dir="$2"
  local cache_root="$3"
  local is_b="$4"
  local ckpt="$5"
  local gpu="${6:-}"
  local label
  label="$(safe_name "${ckpt}")"

  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt}" >&2
    return 0
  fi

  local out_dir="${EVAL_ROOT}/${line_name}/${label}"
  local log_dir="${EVAL_ROOT}/logs/${line_name}"
  local log_file="${log_dir}/${label}.eval.log"
  mkdir -p "${out_dir}" "${log_dir}"
  if [[ -f "${out_dir}/metrics.json" && "${SKIP_COMPLETED}" == "1" ]]; then
    echo "Skipping completed eval: ${out_dir}/metrics.json"
    return 0
  fi

  local cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${ckpt}"
    --chunk-cache-root "${cache_root}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision "${PRECISION}"
    --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}"
    --output-dir "${out_dir}"
  )
  if [[ -n "${VLM_TEXT_ANCHOR_CACHE_ROOT}" ]]; then
    cmd+=(--vlm-text-anchor-cache-root "${VLM_TEXT_ANCHOR_CACHE_ROOT}")
  fi

  if [[ "${is_b}" == "1" && -n "${B_ONLINE_VLM_PATH}" && -n "${B_ONLINE_VLM_LORA_ADAPTER_DIR}" ]]; then
    cmd+=(
      --feature-source chunk_or_online
      --online-vlm-path "${B_ONLINE_VLM_PATH}"
      --online-vlm-type "${B_ONLINE_VLM_TYPE}"
      --online-vlm-lora-adapter-dir "${B_ONLINE_VLM_LORA_ADAPTER_DIR}"
    )
  elif [[ "${is_b}" == "1" && -z "${LORA_NAVTEST_CHUNK_CACHE_ROOT:-}" ]]; then
    echo "Skipping B eval for ${ckpt}: set LORA_NAVTEST_CHUNK_CACHE_ROOT or B_ONLINE_VLM_PATH+B_ONLINE_VLM_LORA_ADAPTER_DIR." >&2
    return 0
  fi

  {
    printf '[%s] line=%q checkpoint=%q ' "$(date -Is)" "${line_name}" "${ckpt}"
    if [[ -n "${gpu}" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${COMMANDS_LOG}"

  echo "Evaluating ${line_name}/${label}: ${ckpt}${gpu:+ on GPU ${gpu}}"
  if [[ -n "${gpu}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1
  else
    "${cmd[@]}" >"${log_file}" 2>&1
  fi
}

mapfile -t specs < <(line_specs)
if [[ "${#specs[@]}" -eq 0 ]]; then
  echo "No A/B run directories found under OUT_ROOT=${OUT_ROOT}" >&2
  exit 2
fi

tasks_file="${EVAL_ROOT}/tasks.tsv"
: >"${tasks_file}"
for spec in "${specs[@]}"; do
  IFS=$'\t' read -r line_name run_dir list_path cache_root is_b <<<"${spec}"
  seen=""
  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    if [[ "${seen}" == *"|${ckpt}|"* ]]; then
      continue
    fi
    seen="${seen}|${ckpt}|"
    printf '%s\t%s\t%s\t%s\t%s\n' "${line_name}" "${run_dir}" "${cache_root}" "${is_b}" "${ckpt}" >>"${tasks_file}"
  done < <(collect_ckpts_for_line "${run_dir}" "${list_path}")
done

if [[ ! -s "${tasks_file}" ]]; then
  echo "No checkpoints discovered for eval under OUT_ROOT=${OUT_ROOT}" >&2
  exit 0
fi

if [[ "${PARALLEL_EVAL}" == "1" || "${PARALLEL_EVAL}" == "true" ]]; then
  IFS=',' read -r -a gpu_list <<<"${EVAL_GPUS}"
  if [[ "${#gpu_list[@]}" -eq 0 ]]; then
    echo "PARALLEL_EVAL=1 requires EVAL_GPUS to contain at least one GPU id." >&2
    exit 2
  fi
  if [[ -z "${MAX_PARALLEL_EVAL}" ]]; then
    MAX_PARALLEL_EVAL="${#gpu_list[@]}"
  fi
  pids=()
  failed=0
  task_index=0
  while IFS=$'\t' read -r line_name run_dir cache_root is_b ckpt; do
    gpu="${gpu_list[$((task_index % ${#gpu_list[@]}))]}"
    run_eval_one "${line_name}" "${run_dir}" "${cache_root}" "${is_b}" "${ckpt}" "${gpu}" &
    pids+=("$!")
    task_index=$((task_index + 1))
    if [[ "${#pids[@]}" -ge "${MAX_PARALLEL_EVAL}" ]]; then
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          failed=1
        fi
      done
      pids=()
    fi
  done <"${tasks_file}"
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  exit "${failed}"
fi

while IFS=$'\t' read -r line_name run_dir cache_root is_b ckpt; do
  run_eval_one "${line_name}" "${run_dir}" "${cache_root}" "${is_b}" "${ckpt}" ""
done <"${tasks_file}"
