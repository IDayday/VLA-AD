#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
B_ROOT="${B_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk}"
B_PROGRESSIVE_ROOT="${B_PROGRESSIVE_ROOT:-${B_ROOT}/progressive_sft_decoupled}"
B_ADAPTER_DIR="${B_ADAPTER_DIR:-${B_ROOT}/vlm_lora_cot_alignment/adapters/vlm_lora}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_b_progressive_parallel_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
BASE_VLM_PATH="${BASE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
CKPT_STEPS_CSV="${CKPT_STEPS_CSV:-00050000,00060000,00080000}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
PRECISION="${PRECISION:-fp32}"
STAGGER_SECONDS="${STAGGER_SECONDS:-10}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LAUNCHER="${LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"

IFS=',' read -r -a CKPT_STEPS <<<"${CKPT_STEPS_CSV}"
IFS=',' read -r -a EVAL_GPUS <<<"${EVAL_GPUS_CSV}"

if [[ "${NUM_SHARDS}" -le 0 ]]; then
  echo "NUM_SHARDS must be positive" >&2
  exit 2
fi
if [[ "${#EVAL_GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "EVAL_GPUS_CSV must provide at least NUM_SHARDS GPU ids" >&2
  exit 2
fi
[[ -f "${B_ADAPTER_DIR}/adapter_model.bin" ]] || { echo "missing B adapter: ${B_ADAPTER_DIR}/adapter_model.bin" >&2; exit 2; }
[[ -d "${NAVTEST_CHUNK_CACHE_ROOT}" ]] || { echo "missing navtest chunk cache: ${NAVTEST_CHUNK_CACHE_ROOT}" >&2; exit 2; }

mkdir -p "${OUT_ROOT}/jobs" "${OUT_ROOT}/logs" "${OUT_ROOT}/eval"
cd "${PROJECT_ROOT}"
echo "$$" > "${OUT_ROOT}/driver.pid"

JOBS_TSV="${OUT_ROOT}/jobs/b_progressive_${TRAJECTORY_OUTPUT_KEY}_${NUM_SHARDS}shard.tsv"
: > "${JOBS_TSV}"

for step in "${CKPT_STEPS[@]}"; do
  [[ -z "${step}" ]] && continue
  ckpt="${B_PROGRESSIVE_ROOT}/step_${step}.ckpt"
  [[ -f "${ckpt}" ]] || { echo "missing checkpoint: ${ckpt}" >&2; exit 2; }
  run_name="step_${step}_${TRAJECTORY_OUTPUT_KEY}_online_lora"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_name="$(printf "%02d" "${shard}")"
    gpu="${EVAL_GPUS[$((shard % ${#EVAL_GPUS[@]}))]}"
    shard_dir="${OUT_ROOT}/eval/${run_name}/shard_${shard_name}"
    mkdir -p "${shard_dir}"
    cmd="${PYTHON_BIN} scripts/eval_recogdrive_expert_pdm.py --config ${CONFIG} --checkpoint ${ckpt} --chunk-cache-root ${NAVTEST_CHUNK_CACHE_ROOT} --chunk-name-pattern navtest_full_chunk_* --metric-cache-dir ${METRIC_CACHE_DIR} --precision ${PRECISION} --trajectory-output-key ${TRAJECTORY_OUTPUT_KEY} --online-vlm-path ${BASE_VLM_PATH} --online-vlm-type internvl --online-vlm-lora-adapter-dir ${B_ADAPTER_DIR} --num-shards ${NUM_SHARDS} --shard-index ${shard} --output-dir ${shard_dir}"
    printf '%s\t%s\t%s\n' "${run_name}_shard_${shard_name}" "${gpu}" "${cmd}" >> "${JOBS_TSV}"
  done
done

{
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "driver_pid=$$"
  echo "mode=b_progressive_parallel_eval"
  echo "trajectory_output_key=${TRAJECTORY_OUTPUT_KEY}"
  echo "ckpt_steps=${CKPT_STEPS_CSV}"
  echo "num_shards=${NUM_SHARDS}"
  echo "eval_gpus=${EVAL_GPUS_CSV}"
  echo "precision=${PRECISION}"
  echo "jobs_tsv=${JOBS_TSV}"
} | tee "${OUT_ROOT}/commands.log"

"${PYTHON_BIN}" "${LAUNCHER}" \
  --jobs-tsv "${JOBS_TSV}" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds "${STAGGER_SECONDS}" \
  --poll-seconds "${POLL_SECONDS}"

"${PYTHON_BIN}" - <<'PY' "${OUT_ROOT}" "${B_ADAPTER_DIR}" "${TRAJECTORY_OUTPUT_KEY}" "${NUM_SHARDS}" "${CKPT_STEPS_CSV}" "${B_PROGRESSIVE_ROOT}"
import csv
import json
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
adapter_dir = sys.argv[2]
trajectory_key = sys.argv[3]
num_shards = int(sys.argv[4])
steps = [item for item in sys.argv[5].split(",") if item]
progressive_root = Path(sys.argv[6])
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]

def weighted_mean(items, key, weight_key):
    total = 0.0
    weight = 0
    for item in items:
        value = item.get(key)
        w = int(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight else None

fields = [
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
    "online_vlm_lora_adapter_dir",
    "metrics_path",
]
summary_rows = []
for step in steps:
    run_name = f"step_{step}_{trajectory_key}_online_lora"
    run_dir = out_root / "eval" / run_name
    shard_metrics = []
    for shard in range(num_shards):
        path = run_dir / f"shard_{shard:02d}" / "metrics.json"
        if not path.is_file():
            raise SystemExit(f"missing shard metrics: {path}")
        shard_metrics.append(json.loads(path.read_text()))
    aggregate = {
        "run": run_name,
        "checkpoint": str(progressive_root / f"step_{step}.ckpt"),
        "trajectory_output_key": trajectory_key,
        "online_vlm_lora_adapter_dir": adapter_dir,
        "num_shards": num_shards,
        "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
        "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
        "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
        "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
        "trajectory_l1": weighted_mean(shard_metrics, "trajectory_l1", "num_samples"),
        "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
    }
    for key in metric_keys:
        aggregate[key] = weighted_mean(shard_metrics, key, "num_pdm_valid")
    aggregate["pdm_score"] = aggregate["PDMS"]
    (run_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(
        "\n".join([
            "# B Progressive NAVTEST",
            "",
            f"Run: `{run_name}`",
            f"Checkpoint: `{aggregate['checkpoint']}`",
            f"Trajectory output key: `{trajectory_key}`",
            f"Online VLM LoRA adapter: `{adapter_dir}`",
            f"Samples: {aggregate['num_samples']}",
            f"Valid PDM: {aggregate['num_pdm_valid']}",
            f"PDMS: {aggregate['PDMS']}",
            f"Trajectory L1: {aggregate['trajectory_l1']}",
            "",
        ]),
        encoding="utf-8",
    )
    summary_rows.append(aggregate)

(out_root / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in summary_rows:
        output = {field: row.get(field) for field in fields}
        output["metrics_path"] = str(out_root / "eval" / row["run"] / "metrics.json")
        writer.writerow(output)
PY

echo "completed ${OUT_ROOT}"
