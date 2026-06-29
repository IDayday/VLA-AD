#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"

OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
STAGE1_CKPT="${STAGE1_CKPT:-/mnt/project/onevl_navsim_exp/answer_bs64_prompt4hist_20260627_160112/swift_output/v0-20260627-160133/checkpoint-3228}"
VAL_DATA_JSONL="${VAL_DATA_JSONL:-/mnt/project/onevl_navsim_data/navsim_answer_prompt4hist_val6000_from_metric_cache_20260628.jsonl}"
NAV_DATA_JSON="${NAV_DATA_JSON:-/mnt/project/onevl/test_data/navsim_test_prompt4hist_onevl.json}"
IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-/mnt/project/OneVL_training}"
VAL_NAVSIM_LOG_PATH="${VAL_NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs}"
NAV_NAVSIM_LOG_PATH="${NAV_NAVSIM_LOG_PATH:-/mnt/navsim/test_navsim_logs}"
VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-${OUT_ROOT}/cache_val6000}"
NAV_CACHE_ROOT="${NAV_CACHE_ROOT:-${OUT_ROOT}/cache_navtest}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DTYPE="${DTYPE:-bfloat16}"
MAX_IMAGE_SIZE="${MAX_IMAGE_SIZE:-1792}"
HIDDEN_MAX_LENGTH="${HIDDEN_MAX_LENGTH:-2800}"
CURRENT_IMAGE_POLICY="${CURRENT_IMAGE_POLICY:-single_or_last}"
PROMPT_SOURCE="${PROMPT_SOURCE:-}"
VAL_PROMPT_SOURCE="${VAL_PROMPT_SOURCE:-${PROMPT_SOURCE:-row}}"
NAV_PROMPT_SOURCE="${NAV_PROMPT_SOURCE:-${PROMPT_SOURCE:-row}}"
PLANNER_SOURCE="${PLANNER_SOURCE:-}"
VAL_PLANNER_SOURCE="${VAL_PLANNER_SOURCE:-${PLANNER_SOURCE:-json}}"
NAV_PLANNER_SOURCE="${NAV_PLANNER_SOURCE:-${PLANNER_SOURCE:-json}}"
INCLUDE_CONTROL_CONVENTION="${INCLUDE_CONTROL_CONVENTION:-0}"
EVAL_SPLITS="${EVAL_SPLITS:-val6000,navtest}"
WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-0}"
GPU_USED_MAX_MB="${GPU_USED_MAX_MB:-1024}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-120}"

mkdir -p "${OUT_ROOT}/logs" "${VAL_CACHE_ROOT}" "${NAV_CACHE_ROOT}"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

record_cmd() {
  {
    printf '[%s] ' "$(date -Is)"
    printf '%q ' "$@"
    printf '\n'
  } >> "${COMMANDS_LOG}"
}

wait_for_gpus() {
  if [[ "${WAIT_FOR_GPU_FREE}" != "1" ]]; then
    return 0
  fi
  while true; do
    mapfile -t used_values < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "${#used_values[@]}" -lt "${NUM_SHARDS}" ]]; then
      echo "Only ${#used_values[@]} GPUs visible; need ${NUM_SHARDS}." >&2
      sleep "${GPU_WAIT_POLL_SECONDS}"
      continue
    fi
    busy=0
    for gpu in $(echo "${GPU_LIST}" | tr ',' ' '); do
      used="${used_values[$gpu]//[[:space:]]/}"
      if [[ -z "${used}" || "${used}" -gt "${GPU_USED_MAX_MB}" ]]; then
        busy=1
      fi
    done
    if [[ "${busy}" -eq 0 ]]; then
      echo "remote GPUs are below ${GPU_USED_MAX_MB}MB; starting cache generation at $(date -Is)"
      return 0
    fi
    echo "remote GPUs busy; waiting ${GPU_WAIT_POLL_SECONDS}s before cache generation. used_mb=${used_values[*]} threshold=${GPU_USED_MAX_MB} $(date -Is)"
    sleep "${GPU_WAIT_POLL_SECONDS}"
  done
}

run_split() {
  local split="$1"
  local data_path="$2"
  local log_path="$3"
  local cache_root="$4"
  local chunk_prefix="$5"
  local prompt_source="$6"
  local planner_source="$7"

  IFS=',' read -r -a gpus <<< "${GPU_LIST}"
  if [[ "${#gpus[@]}" -lt "${NUM_SHARDS}" ]]; then
    echo "Need at least NUM_SHARDS GPUs in GPU_LIST; NUM_SHARDS=${NUM_SHARDS} GPU_LIST=${GPU_LIST}" >&2
    exit 2
  fi

  echo "== ${split} cache start $(date -Is) =="
  pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${gpus[$shard]}"
    local chunk_name="${chunk_prefix}_$(printf '%02d' "${shard}")"
    local chunk_dir="${cache_root}/${chunk_name}"
    mkdir -p "${chunk_dir}"
    cmd=(
      "${PYTHON_BIN}"
      "${SCRIPT_DIR}/bridge_ar_answer_to_recogdrive_dit.py"
      --model-path "${STAGE1_CKPT}"
      --data-jsonl "${data_path}"
      --navsim-log-path "${log_path}"
      --image-base-path "${IMAGE_BASE_PATH}"
      --output-dir "${chunk_dir}"
      --max-samples 0
      --num-shards "${NUM_SHARDS}"
      --shard-id "${shard}"
      --device cuda:0
      --dtype "${DTYPE}"
      --max-image-size "${MAX_IMAGE_SIZE}"
      --cache-format flat
      --hidden-padding max_length
      --hidden-max-length "${HIDDEN_MAX_LENGTH}"
      --hidden-padding-side left
      --no-hidden-truncation
      --current-image-policy "${CURRENT_IMAGE_POLICY}"
      --prompt-source "${prompt_source}"
      --planner-source "${planner_source}"
      --dit-type small
      --sampling-method ddim
      --no-dit-forward
      --no-skip-existing
    )
    if [[ "${INCLUDE_CONTROL_CONVENTION}" == "1" ]]; then
      cmd+=(--include-control-convention)
    fi
    record_cmd env "CUDA_VISIBLE_DEVICES=${gpu}" "${cmd[@]}"
    (
      set +e
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${OUT_ROOT}/logs/cache_${split}_${chunk_name}.log" 2>&1
      status=$?
      echo "${status}" > "${OUT_ROOT}/logs/cache_${split}_${chunk_name}.status"
      exit "${status}"
    ) &
    pids+=("$!")
    echo "$!" > "${OUT_ROOT}/logs/cache_${split}_${chunk_name}.pid"
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  echo "${status}" > "${OUT_ROOT}/cache_${split}.status"
  if [[ "${status}" -ne 0 ]]; then
    echo "${split} cache generation failed; see ${OUT_ROOT}/logs/cache_${split}_*.log" >&2
    return "${status}"
  fi

  "${PYTHON_BIN}" - "${cache_root}" "${data_path}" "${NUM_SHARDS}" "${HIDDEN_MAX_LENGTH}" "${split}" "${planner_source}" <<'PY'
import json
import sys
from pathlib import Path

import torch

cache_root = Path(sys.argv[1])
data_path = Path(sys.argv[2])
num_shards = int(sys.argv[3])
hidden_max_length = int(sys.argv[4])
split = sys.argv[5]
planner_source = sys.argv[6]
expected_planner_state_source = "json_prompt_fields" if planner_source != "scene" else "navsim_scene_frames"

if data_path.suffix == ".jsonl":
    expected = sum(1 for line in data_path.open("r", encoding="utf-8") if line.strip())
else:
    expected = len(json.loads(data_path.read_text()))

chunks = []
sample_checks = []
errors = []
total = 0
for chunk_dir in sorted(path for path in cache_root.iterdir() if path.is_dir()):
    index_path = chunk_dir / "index.jsonl"
    if not index_path.is_file():
        errors.append(f"missing index: {index_path}")
        count = 0
        records = []
    else:
        records = [json.loads(line) for line in index_path.open("r", encoding="utf-8") if line.strip()]
        count = len(records)
    chunks.append({"chunk": chunk_dir.name, "count": count, "index": str(index_path)})
    total += count
    if records:
        sample_path = chunk_dir / records[0]["path"]
        sample = torch.load(sample_path, map_location="cpu")
        check = {
            "chunk": chunk_dir.name,
            "path": str(sample_path),
            "last_hidden_state": list(sample["last_hidden_state"].shape),
            "history_trajectory": list(sample["history_trajectory"].shape),
            "status_feature": list(sample["status_feature"].shape),
            "high_command_one_hot": list(sample["high_command_one_hot"].shape),
            "trajectory": list(sample["trajectory"].shape),
            "target_source": sample.get("meta", {}).get("target_source"),
            "prompt_source": sample.get("meta", {}).get("prompt_source"),
            "planner_state_source": sample.get("meta", {}).get("planner_state_source"),
            "planner_source_mode": sample.get("meta", {}).get("planner_source_mode"),
            "prompt_scene_alignment_pass": sample.get("meta", {}).get("prompt_scene_alignment_pass"),
            "hidden_padding": sample.get("meta", {}).get("hidden_padding"),
            "hidden_truncation": sample.get("meta", {}).get("hidden_truncation"),
        }
        sample_checks.append(check)
        if check["last_hidden_state"] != [hidden_max_length, 2560]:
            errors.append(f"{sample_path} hidden shape {check['last_hidden_state']}")
        if check["history_trajectory"] != [4, 3]:
            errors.append(f"{sample_path} history shape {check['history_trajectory']}")
        if check["status_feature"] != [8] or check["high_command_one_hot"] != [3]:
            errors.append(f"{sample_path} planner state shape mismatch")
        if check["trajectory"] != [8, 3]:
            errors.append(f"{sample_path} trajectory shape {check['trajectory']}")
        if check["hidden_padding"] != "max_length" or check["hidden_truncation"] is not False:
            errors.append(f"{sample_path} hidden padding/truncation mismatch")
        if check["planner_state_source"] != expected_planner_state_source:
            errors.append(f"{sample_path} planner_state_source {check['planner_state_source']}")
        if check["planner_source_mode"] != planner_source:
            errors.append(f"{sample_path} planner_source_mode {check['planner_source_mode']}")

summary = {
    "split": split,
    "cache_root": str(cache_root),
    "data_path": str(data_path),
    "expected_records": expected,
    "total_records": total,
    "num_chunks": len(chunks),
    "chunks": chunks,
    "sample_checks": sample_checks,
    "errors": errors,
}
(cache_root / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if total != expected:
    raise SystemExit(f"{split} cache count mismatch: total={total} expected={expected}")
if len(chunks) != num_shards:
    raise SystemExit(f"{split} chunk count mismatch: chunks={len(chunks)} expected={num_shards}")
if errors:
    raise SystemExit(f"{split} cache validation failed")
PY
  echo "== ${split} cache done $(date -Is) =="
}

{
  echo "host=$(hostname)"
  echo "out_root=${OUT_ROOT}"
  echo "stage1_ckpt=${STAGE1_CKPT}"
  echo "val_data_jsonl=${VAL_DATA_JSONL}"
  echo "nav_data_json=${NAV_DATA_JSON}"
  echo "val_cache_root=${VAL_CACHE_ROOT}"
  echo "nav_cache_root=${NAV_CACHE_ROOT}"
  echo "num_shards=${NUM_SHARDS}"
  echo "gpu_list=${GPU_LIST}"
  echo "eval_splits=${EVAL_SPLITS}"
  echo "prompt_source=${PROMPT_SOURCE}"
  echo "val_prompt_source=${VAL_PROMPT_SOURCE}"
  echo "nav_prompt_source=${NAV_PROMPT_SOURCE}"
  echo "planner_source=${PLANNER_SOURCE}"
  echo "val_planner_source=${VAL_PLANNER_SOURCE}"
  echo "nav_planner_source=${NAV_PLANNER_SOURCE}"
  echo "wait_for_gpu_free=${WAIT_FOR_GPU_FREE}"
  echo "gpu_used_max_mb=${GPU_USED_MAX_MB}"
} > "${OUT_ROOT}/eval_cache.env"

split_enabled() {
  local split="$1"
  [[ ",${EVAL_SPLITS}," == *",${split},"* ]]
}

status=0
wait_for_gpus
if split_enabled "val6000"; then
  run_split "val6000" "${VAL_DATA_JSONL}" "${VAL_NAVSIM_LOG_PATH}" "${VAL_CACHE_ROOT}" "val6000_chunk" "${VAL_PROMPT_SOURCE}" "${VAL_PLANNER_SOURCE}" || status=1
fi
if split_enabled "navtest"; then
  run_split "navtest" "${NAV_DATA_JSON}" "${NAV_NAVSIM_LOG_PATH}" "${NAV_CACHE_ROOT}" "shard" "${NAV_PROMPT_SOURCE}" "${NAV_PLANNER_SOURCE}" || status=1
fi

if [[ "${status}" -eq 0 ]]; then
  if split_enabled "val6000"; then
    ln -sfn "${VAL_CACHE_ROOT}" /mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_val6000_prompt4hist_latest
  fi
  if split_enabled "navtest"; then
    ln -sfn "${NAV_CACHE_ROOT}" /mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_navtest_prompt4hist_latest
  fi
fi
echo "${status}" > "${OUT_ROOT}/eval_cache.status"
exit "${status}"
