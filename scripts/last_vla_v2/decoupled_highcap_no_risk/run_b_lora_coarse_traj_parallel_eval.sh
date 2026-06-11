#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_b_lora_coarse_traj_eval_${RUN_ID}}"
B_LORA_ROOT="${B_LORA_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment}"
ADAPTER_ROOT="${ADAPTER_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_vlm_pred_traj_eval_20260608T033900Z/adapters}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
BASE_VLM_PATH="${BASE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_coarse_traj}"
EPOCHS_CSV="${EPOCHS_CSV:-002,005}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-4}"
PRECISION="${PRECISION:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

IFS=',' read -r -a EPOCHS <<<"${EPOCHS_CSV}"
IFS=',' read -r -a EVAL_GPUS <<<"${EVAL_GPUS_CSV}"
if [[ "${NUM_SHARDS}" -le 0 ]]; then
  echo "NUM_SHARDS must be positive." >&2
  exit 2
fi
if [[ "${#EVAL_GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "EVAL_GPUS_CSV must provide at least NUM_SHARDS GPU ids." >&2
  exit 2
fi
MAX_PARALLEL_CKPTS=$(( ${#EVAL_GPUS[@]} / NUM_SHARDS ))
if [[ "${MAX_PARALLEL_CKPTS}" -le 0 ]]; then
  echo "Not enough GPUs for one checkpoint: EVAL_GPUS_CSV=${EVAL_GPUS_CSV}, NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
SUMMARY_JSONL="${OUT_ROOT}/summary.jsonl"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "run,checkpoint,trajectory_output_key,num_samples,num_pdm_valid,num_pdm_missing_metric_cache,num_pdm_failed,PDMS,NC,DAC,TTC,comfort,EP,DDC,trajectory_l1,online_vlm_lora_adapter_dir,metrics_path" >"${SUMMARY_CSV}"
fi

cd "${PROJECT_ROOT}"

run_name_for_epoch() {
  local ep="$1"
  echo "B_epoch${ep}_lora_direct_online_${TRAJECTORY_OUTPUT_KEY}_full"
}

adapter_for_epoch() {
  local ep="$1"
  echo "${ADAPTER_ROOT}/B_epoch_${ep}/vlm_lora"
}

aggregate_one() {
  local ep="$1"
  local run_name ckpt adapter run_dir
  run_name="$(run_name_for_epoch "${ep}")"
  ckpt="${B_LORA_ROOT}/epoch_${ep}.ckpt"
  adapter="$(adapter_for_epoch "${ep}")"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  RUN_NAME="${run_name}" \
  CKPT="${ckpt}" \
  ADAPTER="${adapter}" \
  RUN_DIR="${run_dir}" \
  TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SUMMARY_JSONL="${SUMMARY_JSONL}" \
  SUMMARY_CSV="${SUMMARY_CSV}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_name = os.environ["RUN_NAME"]
ckpt = os.environ["CKPT"]
adapter = os.environ["ADAPTER"]
run_dir = Path(os.environ["RUN_DIR"])
trajectory_key = os.environ["TRAJECTORY_OUTPUT_KEY"]
num_shards = int(os.environ["NUM_SHARDS"])
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]

shards = []
for idx in range(num_shards):
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
    "trajectory_output_key": trajectory_key,
    "online_vlm_path": "/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B",
    "online_vlm_lora_adapter_dir": adapter,
    "num_shards": num_shards,
    "num_samples": sum(int(item.get("num_samples") or 0) for item in shards),
    "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shards),
    "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shards),
    "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shards),
    "trajectory_l1": wmean("trajectory_l1", "num_samples"),
    "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
}
for key in metric_keys:
    aggregate[key] = wmean(key, "num_pdm_valid")
aggregate["pdm_score"] = aggregate["PDMS"]
(run_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
(run_dir / "report.md").write_text(
    "\n".join([
        "# B LoRA Direct-Online Coarse Trajectory NAVTEST",
        "",
        f"Run: `{run_name}`",
        f"Checkpoint: `{ckpt}`",
        f"Trajectory output key: `{trajectory_key}`",
        f"Online VLM LoRA adapter: `{adapter}`",
        f"Samples: {aggregate['num_samples']}",
        f"Valid PDM: {aggregate['num_pdm_valid']}",
        f"PDMS: {aggregate['PDMS']}",
        f"Trajectory L1: {aggregate['trajectory_l1']}",
        "",
    ]),
    encoding="utf-8",
)
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

launch_epoch() {
  local ep="$1"
  local slot="$2"
  local run_name run_dir adapter ckpt shard shard_dir shard_log gpu_index
  run_name="$(run_name_for_epoch "${ep}")"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  adapter="$(adapter_for_epoch "${ep}")"
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
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[$(date -Is)] skip completed aggregate ${run_name}"
    return 0
  fi
  echo "[$(date -Is)] launch ${run_name} slot=${slot}"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_dir="${run_dir}/shard_$(printf '%02d' "${shard}")"
    shard_log="${OUT_ROOT}/logs/${run_name}/shard_$(printf '%02d' "${shard}").log"
    mkdir -p "${shard_dir}"
    if [[ -f "${shard_dir}/metrics.json" ]]; then
      echo "[$(date -Is)] skip completed ${run_name} shard=${shard}" | tee -a "${shard_log}"
      continue
    fi
    gpu_index=$((slot * NUM_SHARDS + shard))
    {
      echo "[$(date -Is)] start run=${run_name} shard=${shard} gpu=${EVAL_GPUS[$gpu_index]}"
      printf '%q ' CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$gpu_index]}" PYTHONUNBUFFERED=1 \
        "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py \
        --config "${CONFIG}" \
        --checkpoint "${ckpt}" \
        --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}" \
        --chunk-name-pattern "navtest_full_chunk_*" \
        --metric-cache-dir "${METRIC_CACHE_DIR}" \
        --precision "${PRECISION}" \
        --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}" \
        --online-vlm-path "${BASE_VLM_PATH}" \
        --online-vlm-type internvl \
        --online-vlm-lora-adapter-dir "${adapter}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${shard}" \
        --output-dir "${shard_dir}"
      printf '\n'
    } | tee -a "${shard_log}" >>"${COMMANDS_LOG}"
    cmd=(
      "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
      --config "${CONFIG}"
      --checkpoint "${ckpt}"
      --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "navtest_full_chunk_*"
      --metric-cache-dir "${METRIC_CACHE_DIR}"
      --precision "${PRECISION}"
      --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}"
      --online-vlm-path "${BASE_VLM_PATH}"
      --online-vlm-type internvl
      --online-vlm-lora-adapter-dir "${adapter}"
      --num-shards "${NUM_SHARDS}"
      --shard-index "${shard}"
      --output-dir "${shard_dir}"
    )
    if [[ -n "${MAX_SAMPLES}" ]]; then
      cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$gpu_index]}" PYTHONUNBUFFERED=1 "${cmd[@]}" >>"${shard_log}" 2>&1 &
    BATCH_PIDS+=("$!")
  done
}

wait_batch() {
  local failed pid ep
  failed=0
  for pid in "${BATCH_PIDS[@]}"; do
    if ! wait "${pid}"; then
      failed=$((failed + 1))
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date -Is)] shard eval failed: failed=${failed}" >&2
    exit 1
  fi
  for ep in "${BATCH_EPOCHS[@]}"; do
    if [[ -f "${OUT_ROOT}/eval/$(run_name_for_epoch "${ep}")/metrics.json" ]]; then
      continue
    fi
    if ! all_shard_metrics_exist "${ep}"; then
      echo "[$(date -Is)] missing shard metrics after wait for epoch ${ep}" >&2
      exit 1
    fi
    aggregate_one "${ep}" | tee -a "${OUT_ROOT}/logs/$(run_name_for_epoch "${ep}")/aggregate.log"
  done
  BATCH_PIDS=()
  BATCH_EPOCHS=()
}

BATCH_PIDS=()
BATCH_EPOCHS=()
batch_slot=0
for ep in "${EPOCHS[@]}"; do
  [[ -z "${ep}" ]] && continue
  if [[ "${batch_slot}" -ge "${MAX_PARALLEL_CKPTS}" ]]; then
    wait_batch
    batch_slot=0
  fi
  launch_epoch "${ep}" "${batch_slot}"
  BATCH_EPOCHS+=("${ep}")
  batch_slot=$((batch_slot + 1))
done

if [[ "${#BATCH_EPOCHS[@]}" -gt 0 ]]; then
  wait_batch
fi

echo "[$(date -Is)] all requested B LoRA ${TRAJECTORY_OUTPUT_KEY} evals complete"
