#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_a_progressive_ckpt_eval_${RUN_ID}}"
A_RUN_ROOT="${A_RUN_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/A/serverA_frozen_vlm_decoupled_highcap_no_risk/progressive_sft_decoupled}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-1,3,4,6}"
PRECISION="${PRECISION:-fp32}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

IFS=',' read -r -a EVAL_GPUS <<<"${EVAL_GPUS_CSV}"
NUM_SHARDS="${NUM_SHARDS:-${#EVAL_GPUS[@]}}"
if [[ "${NUM_SHARDS}" -le 0 ]]; then
  echo "NUM_SHARDS must be positive." >&2
  exit 2
fi
if [[ "${#EVAL_GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "EVAL_GPUS_CSV must provide at least NUM_SHARDS GPU ids." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/eval"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
SUMMARY_JSONL="${OUT_ROOT}/summary.jsonl"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

if [[ -n "${CKPTS_FILE:-}" ]]; then
  mapfile -t CKPTS <"${CKPTS_FILE}"
elif [[ -n "${CKPTS:-}" ]]; then
  read -r -a CKPTS <<<"${CKPTS}"
else
  CKPTS=(
    "${A_RUN_ROOT}/step_00050000.ckpt"
    "${A_RUN_ROOT}/step_00060000.ckpt"
    "${A_RUN_ROOT}/step_00080000.ckpt"
    "${A_RUN_ROOT}/lightning_logs/version_0/checkpoints/epoch=130-step=86984.ckpt"
    "${A_RUN_ROOT}/lightning_logs/version_0/checkpoints/epoch=131-step=87648.ckpt"
    "${A_RUN_ROOT}/lightning_logs/version_0/checkpoints/epoch=132-step=88312.ckpt"
    "${A_RUN_ROOT}/lightning_logs/version_0/checkpoints/epoch=133-step=88976.ckpt"
    "${A_RUN_ROOT}/lightning_logs/version_0/checkpoints/epoch=135-step=90304.ckpt"
  )
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "run,checkpoint,trajectory_output_key,num_samples,num_pdm_valid,num_pdm_missing_metric_cache,num_pdm_failed,PDMS,NC,DAC,TTC,comfort,EP,DDC,trajectory_l1,metrics_path" >"${SUMMARY_CSV}"
fi

safe_name() {
  local ckpt="$1"
  basename "${ckpt}" .ckpt | tr '/ :' '___'
}

aggregate_one() {
  local run_name="$1"
  local ckpt="$2"
  local run_dir="$3"
  RUN_NAME="${run_name}" CKPT="${ckpt}" RUN_DIR="${run_dir}" TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" NUM_SHARDS="${NUM_SHARDS}" "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_name = os.environ["RUN_NAME"]
ckpt = os.environ["CKPT"]
run_dir = Path(os.environ["RUN_DIR"])
trajectory_key = os.environ["TRAJECTORY_OUTPUT_KEY"]
num_shards = int(os.environ["NUM_SHARDS"])
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]

shard_metrics = []
for idx in range(num_shards):
    path = run_dir / f"shard_{idx:02d}" / "metrics.json"
    if not path.is_file():
        raise SystemExit(f"missing shard metrics: {path}")
    shard_metrics.append(json.loads(path.read_text()))

total_samples = sum(int(m.get("num_samples") or 0) for m in shard_metrics)
total_valid = sum(int(m.get("num_pdm_valid") or 0) for m in shard_metrics)
total_missing = sum(int(m.get("num_pdm_missing_metric_cache") or 0) for m in shard_metrics)
total_failed = sum(int(m.get("num_pdm_failed") or 0) for m in shard_metrics)

def weighted_mean(key, weight_key):
    total = 0.0
    weight = 0
    for item in shard_metrics:
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
    "num_shards": num_shards,
    "num_samples": total_samples,
    "num_pdm_valid": total_valid,
    "num_pdm_missing_metric_cache": total_missing,
    "num_pdm_failed": total_failed,
    "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
    "trajectory_l1": weighted_mean("trajectory_l1", "num_samples"),
}
for key in metric_keys:
    aggregate[key] = weighted_mean(key, "num_pdm_valid")
aggregate["pdm_score"] = aggregate["PDMS"]

(run_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
(run_dir / "report.md").write_text(
    "\n".join([
        "# A Progressive Checkpoint NAVTEST",
        "",
        f"Run: `{run_name}`",
        f"Checkpoint: `{ckpt}`",
        f"Trajectory output key: `{trajectory_key}`",
        f"Samples: {total_samples}",
        f"Valid PDM: {total_valid}",
        f"PDMS: {aggregate['PDMS']}",
        f"Trajectory L1: {aggregate['trajectory_l1']}",
        "",
    ]),
    encoding="utf-8",
)
print(json.dumps(aggregate, sort_keys=True))

summary_jsonl = Path(os.environ.get("SUMMARY_JSONL", run_dir.parent.parent / "summary.jsonl"))
with summary_jsonl.open("a", encoding="utf-8") as f:
    f.write(json.dumps(aggregate, sort_keys=True) + "\n")

summary_csv = Path(os.environ.get("SUMMARY_CSV", run_dir.parent.parent / "summary.csv"))
with summary_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run",
            "checkpoint",
            "trajectory_output_key",
            "num_samples",
            "num_pdm_valid",
            "num_pdm_missing_metric_cache",
            "num_pdm_failed",
            "PDMS",
            "NC",
            "DAC",
            "TTC",
            "comfort",
            "EP",
            "DDC",
            "trajectory_l1",
            "metrics_path",
        ],
    )
    row = {key: aggregate.get(key) for key in writer.fieldnames}
    row["metrics_path"] = str(run_dir / "metrics.json")
    writer.writerow(row)
PY
}

for ckpt in "${CKPTS[@]}"; do
  [[ -z "${ckpt}" ]] && continue
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skipping missing checkpoint: ${ckpt}" | tee -a "${OUT_ROOT}/logs/missing_ckpts.log"
    continue
  fi
  run_name="$(safe_name "${ckpt}")_${TRAJECTORY_OUTPUT_KEY}"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  mkdir -p "${run_dir}" "${OUT_ROOT}/logs/${run_name}"

  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "Skipping completed aggregate: ${run_dir}/metrics.json"
    continue
  fi

  pids=()
  failed=0
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_dir="${run_dir}/shard_$(printf '%02d' "${shard}")"
    shard_log="${OUT_ROOT}/logs/${run_name}/shard_$(printf '%02d' "${shard}").log"
    mkdir -p "${shard_dir}"
    if [[ -f "${shard_dir}/metrics.json" ]]; then
      echo "Skipping completed shard ${run_name} shard=${shard}" | tee -a "${shard_log}"
      continue
    fi
    cmd=(
      "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
      --config "${CONFIG}"
      --checkpoint "${ckpt}"
      --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "navtest_full_chunk_*"
      --metric-cache-dir "${METRIC_CACHE_DIR}"
      --precision "${PRECISION}"
      --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}"
      --num-shards "${NUM_SHARDS}"
      --shard-index "${shard}"
      --output-dir "${shard_dir}"
    )
    if [[ -n "${MAX_SAMPLES}" ]]; then
      cmd+=(--max-samples "${MAX_SAMPLES}")
    fi
    {
      echo "[$(date -Is)] start run=${run_name} shard=${shard} gpu=${EVAL_GPUS[$shard]}"
      printf '%q ' CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$shard]}" PYTHONUNBUFFERED=1 "${cmd[@]}"
      printf '\n'
    } | tee -a "${shard_log}" >>"${COMMANDS_LOG}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$shard]}" PYTHONUNBUFFERED=1 "${cmd[@]}" >>"${shard_log}" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=$((failed + 1))
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "Shard eval failed for ${run_name}: failed=${failed}" | tee -a "${OUT_ROOT}/logs/failures.log"
    exit 1
  fi
  aggregate_one "${run_name}" "${ckpt}" "${run_dir}" | tee -a "${OUT_ROOT}/logs/${run_name}/aggregate.log"
done
