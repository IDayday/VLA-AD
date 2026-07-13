#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_vlm_pred_traj_eval_20260608T033900Z}"
B_LORA_ROOT="${B_LORA_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
BASE_VLM_PATH="${BASE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
EPOCHS_CSV="${EPOCHS_CSV:-002,003,005}"
LAUNCH_EPOCHS_CSV="${LAUNCH_EPOCHS_CSV:-${EPOCHS_CSV}}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-0,2,4,6}"
AGGREGATE_WAIT_SECONDS="${AGGREGATE_WAIT_SECONDS:-86400}"
POLL_SECONDS="${POLL_SECONDS:-60}"
OLD_SUPERVISOR_PIDS="${OLD_SUPERVISOR_PIDS:-}"

IFS=',' read -r -a EPOCHS <<<"${EPOCHS_CSV}"
IFS=',' read -r -a LAUNCH_EPOCHS <<<"${LAUNCH_EPOCHS_CSV}"
IFS=',' read -r -a EVAL_GPUS <<<"${EVAL_GPUS_CSV}"
NUM_SHARDS="${NUM_SHARDS:-${#EVAL_GPUS[@]}}"

mkdir -p "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
SUMMARY_JSONL="${OUT_ROOT}/summary.jsonl"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "run,checkpoint,trajectory_output_key,num_samples,num_pdm_valid,num_pdm_missing_metric_cache,num_pdm_failed,PDMS,NC,DAC,TTC,comfort,EP,DDC,trajectory_l1,online_vlm_lora_adapter_dir,metrics_path" >"${SUMMARY_CSV}"
fi

contains_epoch() {
  local needle="$1"
  local item
  for item in "${LAUNCH_EPOCHS[@]}"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

run_name_for_epoch() {
  local ep="$1"
  echo "B_epoch${ep}_lora_direct_online_pred_traj_full"
}

aggregate_one() {
  local ep="$1"
  local run_name ckpt adapter run_dir
  run_name="$(run_name_for_epoch "${ep}")"
  ckpt="${B_LORA_ROOT}/epoch_${ep}.ckpt"
  adapter="${OUT_ROOT}/adapters/B_epoch_${ep}/vlm_lora"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  RUN_NAME="${run_name}" CKPT="${ckpt}" ADAPTER="${adapter}" RUN_DIR="${run_dir}" SUMMARY_JSONL="${SUMMARY_JSONL}" SUMMARY_CSV="${SUMMARY_CSV}" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_name = os.environ["RUN_NAME"]
ckpt = os.environ["CKPT"]
adapter = os.environ["ADAPTER"]
run_dir = Path(os.environ["RUN_DIR"])
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]
shards = []
for idx in range(4):
    path = run_dir / f"shard_{idx:02d}" / "metrics.json"
    if not path.is_file():
        raise SystemExit(f"missing shard metrics: {path}")
    shards.append(json.loads(path.read_text()))

def wmean(key, weight_key):
    total = 0.0
    weight = 0
    for item in shards:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight else None

aggregate = {
    "run": run_name,
    "checkpoint": ckpt,
    "trajectory_output_key": "pred_traj",
    "online_vlm_path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B",
    "online_vlm_lora_adapter_dir": adapter,
    "num_shards": 4,
    "num_samples": sum(int(item.get("num_samples") or 0) for item in shards),
    "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shards),
    "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shards),
    "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shards),
    "trajectory_l1": wmean("trajectory_l1", "num_samples"),
    "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(4)],
}
for key in metric_keys:
    aggregate[key] = wmean(key, "num_pdm_valid")
aggregate["pdm_score"] = aggregate["PDMS"]
(run_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
with Path(os.environ["SUMMARY_JSONL"]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(aggregate, sort_keys=True) + "\n")
fields = [
    "run", "checkpoint", "trajectory_output_key", "num_samples", "num_pdm_valid",
    "num_pdm_missing_metric_cache", "num_pdm_failed", "PDMS", "NC", "DAC", "TTC",
    "comfort", "EP", "DDC", "trajectory_l1", "online_vlm_lora_adapter_dir",
    "metrics_path",
]
with Path(os.environ["SUMMARY_CSV"]).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    row = {key: aggregate.get(key) for key in fields}
    row["metrics_path"] = str(run_dir / "metrics.json")
    writer.writerow(row)
print(json.dumps(aggregate, sort_keys=True), flush=True)
PY
}

all_shard_metrics_exist() {
  local ep="$1"
  local run_name run_dir shard
  run_name="$(run_name_for_epoch "${ep}")"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    [[ -f "${run_dir}/shard_$(printf '%02d' "${shard}")/metrics.json" ]] || return 1
  done
  return 0
}

pids=()
for ep in "${EPOCHS[@]}"; do
  run_name="$(run_name_for_epoch "${ep}")"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  adapter="${OUT_ROOT}/adapters/B_epoch_${ep}/vlm_lora"
  ckpt="${B_LORA_ROOT}/epoch_${ep}.ckpt"
  mkdir -p "${run_dir}" "${OUT_ROOT}/logs/${run_name}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date -Is)] missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
  if [[ ! -f "${adapter}/adapter_model.bin" ]]; then
    echo "[$(date -Is)] missing adapter: ${adapter}/adapter_model.bin" >&2
    exit 2
  fi
  if ! contains_epoch "${ep}"; then
    echo "[$(date -Is)] not launching ep=${ep}; waiting for existing shards if needed"
    continue
  fi
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[$(date -Is)] skip completed aggregate ${run_name}"
    continue
  fi
  echo "[$(date -Is)] launch ${run_name}"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_dir="${run_dir}/shard_$(printf '%02d' "${shard}")"
    shard_log="${OUT_ROOT}/logs/${run_name}/shard_$(printf '%02d' "${shard}").log"
    mkdir -p "${shard_dir}"
    if [[ -f "${shard_dir}/metrics.json" ]]; then
      echo "[$(date -Is)] skip completed ${run_name} shard=${shard}" | tee -a "${shard_log}"
      continue
    fi
    echo "[$(date -Is)] start ${run_name} shard=${shard} gpu=${EVAL_GPUS[$shard]}" | tee -a "${shard_log}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$shard]}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py \
      --config "${CONFIG}" \
      --checkpoint "${ckpt}" \
      --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}" \
      --chunk-name-pattern "navtest_full_chunk_*" \
      --metric-cache-dir "${METRIC_CACHE_DIR}" \
      --precision fp32 \
      --trajectory-output-key pred_traj \
      --online-vlm-path "${BASE_VLM_PATH}" \
      --online-vlm-type internvl \
      --online-vlm-lora-adapter-dir "${adapter}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --output-dir "${shard_dir}" >>"${shard_log}" 2>&1 &
    pids+=("$!")
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

deadline=$((SECONDS + AGGREGATE_WAIT_SECONDS))
while true; do
  all_done=1
  for ep in "${EPOCHS[@]}"; do
    run_name="$(run_name_for_epoch "${ep}")"
    run_dir="${OUT_ROOT}/eval/${run_name}"
    if [[ -f "${run_dir}/metrics.json" ]]; then
      continue
    fi
    if all_shard_metrics_exist "${ep}"; then
      aggregate_one "${ep}" | tee -a "${OUT_ROOT}/logs/${run_name}/aggregate.log"
    else
      all_done=0
    fi
  done
  [[ "${all_done}" -eq 1 ]] && break
  if (( SECONDS >= deadline )); then
    echo "[$(date -Is)] timed out waiting for shard metrics." >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

if [[ -n "${OLD_SUPERVISOR_PIDS}" ]]; then
  for pid in ${OLD_SUPERVISOR_PIDS}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
fi

echo "[$(date -Is)] all requested B LoRA pred_traj evals complete"
