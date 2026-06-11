#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
SOURCE_OUT_ROOT="${SOURCE_OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_20260609T021059Z}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_ckpt_parallel_eval_${RUN_ID}}"
LINE="${LINE:-A}"

CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
PRECISION="${PRECISION:-fp32}"
VLM_TEXT_ANCHOR_CACHE_ROOT="${VLM_TEXT_ANCHOR_CACHE_ROOT:-}"

A_RUN_ROOT="${A_RUN_ROOT:-${SOURCE_OUT_ROOT}/A_frozen_vlm_stage2_progressive}"
B_RUN_ROOT="${B_RUN_ROOT:-${SOURCE_OUT_ROOT}/B_lora_stage2_progressive}"
B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
B_ONLINE_VLM_TYPE="${B_ONLINE_VLM_TYPE:-internvl}"
B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora}"
LORA_NAVTEST_CHUNK_CACHE_ROOT="${LORA_NAVTEST_CHUNK_CACHE_ROOT:-}"

A_NUM_SHARDS="${A_NUM_SHARDS:-8}"
B_NUM_SHARDS="${B_NUM_SHARDS:-1}"
A_GPUS_CSV="${A_GPUS_CSV:-0,1,2,3,4,5,6,7}"
B_GPUS_CSV="${B_GPUS_CSV:-0,1,2,3,4,5,6,7}"
STAGGER_SECONDS="${STAGGER_SECONDS:-6}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LAUNCHER="${LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"
MAX_CKPTS_PER_LINE="${MAX_CKPTS_PER_LINE:-0}"
INCLUDE_TOPK_CKPTS="${INCLUDE_TOPK_CKPTS:-1}"
INCLUDE_FIXED_STEP_CKPTS="${INCLUDE_FIXED_STEP_CKPTS:-0}"
FIXED_STEP_CKPTS_CSV="${FIXED_STEP_CKPTS_CSV:-00050000,00060000,00080000,00100000,00120000,00140000,00160000}"

mkdir -p "${OUT_ROOT}/jobs" "${OUT_ROOT}/logs" "${OUT_ROOT}/eval" "${OUT_ROOT}/checkpoint_snapshots"
cd "${PROJECT_ROOT}"

case "${LINE}" in
  A|B) ;;
  *) echo "LINE must be A or B, got ${LINE}" >&2; exit 2 ;;
esac

safe_name() {
  basename "$1" .ckpt | tr '/ :' '___'
}

snapshot_ckpt() {
  local line="$1"
  local ckpt="$2"
  local safe
  safe="$(safe_name "${ckpt}")"
  local snapshot_dir="${OUT_ROOT}/checkpoint_snapshots/${line}"
  local snapshot="${snapshot_dir}/${safe}.ckpt"
  mkdir -p "${snapshot_dir}"
  if [[ ! -f "${snapshot}" ]]; then
    ln "${ckpt}" "${snapshot}"
  fi
  printf '%s\n' "${snapshot}"
}

discover_ckpts() {
  local run_root="$1"
  {
    if [[ "${INCLUDE_TOPK_CKPTS}" == "1" || "${INCLUDE_TOPK_CKPTS}" == "true" ]]; then
      find "${run_root}/lightning_logs" -path '*/checkpoints/epoch=*-step=*.ckpt' -type f 2>/dev/null
    fi
    if [[ "${INCLUDE_FIXED_STEP_CKPTS}" == "1" || "${INCLUDE_FIXED_STEP_CKPTS}" == "true" ]]; then
      IFS=',' read -r -a fixed_steps <<<"${FIXED_STEP_CKPTS_CSV}"
      for step in "${fixed_steps[@]}"; do
        [[ -n "${step}" ]] || continue
        local ckpt="${run_root}/step_${step}.ckpt"
        [[ -f "${ckpt}" ]] && printf '%s\n' "${ckpt}"
      done
    fi
  } | awk '!seen[$0]++' | sort
}

write_line_jobs() {
  local line="$1"
  local run_root num_shards gpus_csv cache_root
  local online_args=()
  if [[ "${line}" == "A" ]]; then
    run_root="${A_RUN_ROOT}"
    num_shards="${A_NUM_SHARDS}"
    gpus_csv="${A_GPUS_CSV}"
    cache_root="${NAVTEST_CHUNK_CACHE_ROOT}"
  else
    run_root="${B_RUN_ROOT}"
    num_shards="${B_NUM_SHARDS}"
    gpus_csv="${B_GPUS_CSV}"
    cache_root="${LORA_NAVTEST_CHUNK_CACHE_ROOT:-${NAVTEST_CHUNK_CACHE_ROOT}}"
    if [[ -z "${LORA_NAVTEST_CHUNK_CACHE_ROOT}" ]]; then
      online_args=(
        --feature-source chunk_or_online
        --online-vlm-path "${B_ONLINE_VLM_PATH}"
        --online-vlm-type "${B_ONLINE_VLM_TYPE}"
        --online-vlm-lora-adapter-dir "${B_ONLINE_VLM_LORA_ADAPTER_DIR}"
      )
    fi
  fi
  local anchor_args=()
  if [[ -n "${VLM_TEXT_ANCHOR_CACHE_ROOT}" ]]; then
    [[ -f "${VLM_TEXT_ANCHOR_CACHE_ROOT}/index.jsonl" ]] || {
      echo "missing VLM text anchor cache index: ${VLM_TEXT_ANCHOR_CACHE_ROOT}/index.jsonl" >&2
      return 1
    }
    anchor_args=(--vlm-text-anchor-cache-root "${VLM_TEXT_ANCHOR_CACHE_ROOT}")
  fi

  [[ -d "${run_root}" ]] || { echo "missing run root: ${run_root}" >&2; return 1; }
  [[ -d "${cache_root}" ]] || { echo "missing navtest cache root: ${cache_root}" >&2; return 1; }
  [[ -f "${LAUNCHER}" ]] || { echo "missing stable launcher: ${LAUNCHER}" >&2; return 1; }
  if [[ "${line}" == "B" && -z "${LORA_NAVTEST_CHUNK_CACHE_ROOT}" ]]; then
    [[ -d "${B_ONLINE_VLM_PATH}" ]] || { echo "missing B online VLM path: ${B_ONLINE_VLM_PATH}" >&2; return 1; }
    [[ -d "${B_ONLINE_VLM_LORA_ADAPTER_DIR}" ]] || { echo "missing B LoRA adapter dir: ${B_ONLINE_VLM_LORA_ADAPTER_DIR}" >&2; return 1; }
  fi

  IFS=',' read -r -a gpus <<<"${gpus_csv}"
  if [[ "${num_shards}" -le 0 ]]; then
    echo "num_shards must be positive for ${line}" >&2
    return 1
  fi
  if [[ "${#gpus[@]}" -eq 0 ]]; then
    echo "no GPUs configured for ${line}" >&2
    return 1
  fi

  local jobs_tsv="${OUT_ROOT}/jobs/${line}_${TRAJECTORY_OUTPUT_KEY}_${num_shards}shard_jobs.tsv"
  local manifest_tsv="${OUT_ROOT}/jobs/${line}_${TRAJECTORY_OUTPUT_KEY}_${num_shards}shard_manifest.tsv"
  : >"${jobs_tsv}"
  : >"${manifest_tsv}"

  local ckpt_count=0
  while IFS= read -r ckpt; do
    [[ -n "${ckpt}" ]] || continue
    if [[ "${MAX_CKPTS_PER_LINE}" -gt 0 && "${ckpt_count}" -ge "${MAX_CKPTS_PER_LINE}" ]]; then
      break
    fi
    local snapshot safe run_name run_dir
    snapshot="$(snapshot_ckpt "${line}" "${ckpt}")"
    safe="$(safe_name "${snapshot}")"
    run_name="${line}_${safe}_${TRAJECTORY_OUTPUT_KEY}"
    run_dir="${OUT_ROOT}/eval/${line}/${run_name}"
    mkdir -p "${run_dir}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${line}" "${run_name}" "${snapshot}" "${run_dir}" "${num_shards}" >>"${manifest_tsv}"
    for ((shard=0; shard<num_shards; shard++)); do
      local shard_name gpu shard_dir job_name cmd
      shard_name="$(printf '%02d' "${shard}")"
      gpu="${gpus[$((((ckpt_count * num_shards) + shard) % ${#gpus[@]}))]}"
      shard_dir="${run_dir}/shard_${shard_name}"
      mkdir -p "${shard_dir}"
      job_name="${run_name}_shard_${shard_name}"
      cmd="$(printf '%q ' \
        "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py \
        --config "${CONFIG}" \
        --checkpoint "${snapshot}" \
        --chunk-cache-root "${cache_root}" \
        --chunk-name-pattern "navtest_full_chunk_*" \
        --metric-cache-dir "${METRIC_CACHE_DIR}" \
        --precision "${PRECISION}" \
        --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}" \
        --num-shards "${num_shards}" \
        --shard-index "${shard}" \
        --output-dir "${shard_dir}" \
        "${anchor_args[@]}" \
        "${online_args[@]}")"
      printf '%s\t%s\t%s\n' "${job_name}" "${gpu}" "${cmd}" >>"${jobs_tsv}"
    done
    ckpt_count=$((ckpt_count + 1))
  done < <(discover_ckpts "${run_root}")

  if [[ "${ckpt_count}" -eq 0 ]]; then
    echo "no epoch-named checkpoints found for ${line} under ${run_root}" >&2
    return 0
  fi

  {
    echo "line=${line}"
    echo "source_out_root=${SOURCE_OUT_ROOT}"
    echo "run_root=${run_root}"
    echo "out_root=${OUT_ROOT}"
    echo "config=${CONFIG}"
    echo "navtest_chunk_cache_root=${cache_root}"
    echo "metric_cache_dir=${METRIC_CACHE_DIR}"
    echo "trajectory_output_key=${TRAJECTORY_OUTPUT_KEY}"
    echo "precision=${PRECISION}"
    echo "vlm_text_anchor_cache_root=${VLM_TEXT_ANCHOR_CACHE_ROOT}"
    echo "num_shards=${num_shards}"
    echo "gpus=${gpus_csv}"
    echo "ckpt_count=${ckpt_count}"
    echo "include_topk_ckpts=${INCLUDE_TOPK_CKPTS}"
    echo "include_fixed_step_ckpts=${INCLUDE_FIXED_STEP_CKPTS}"
    echo "fixed_step_ckpts_csv=${FIXED_STEP_CKPTS_CSV}"
    echo "jobs_tsv=${jobs_tsv}"
    echo "manifest_tsv=${manifest_tsv}"
    if [[ "${line}" == "B" && -z "${LORA_NAVTEST_CHUNK_CACHE_ROOT}" ]]; then
      echo "b_feature_source=chunk_or_online"
      echo "b_online_vlm_path=${B_ONLINE_VLM_PATH}"
      echo "b_online_vlm_lora_adapter_dir=${B_ONLINE_VLM_LORA_ADAPTER_DIR}"
    fi
  } | tee -a "${OUT_ROOT}/commands.log" >&2

  printf '%s\n' "${jobs_tsv}"
}

aggregate_line() {
  local line="$1"
  local manifest_pattern="${OUT_ROOT}/jobs/${line}_${TRAJECTORY_OUTPUT_KEY}_"*"_manifest.tsv"
  SUMMARY_ROOT="${OUT_ROOT}" LINE_NAME="${line}" TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" \
  "${PYTHON_BIN}" - "${manifest_pattern}" <<'PY'
import csv
import glob
import json
import os
import sys
from pathlib import Path

summary_root = Path(os.environ["SUMMARY_ROOT"])
line_name = os.environ["LINE_NAME"]
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]
manifest_paths = [Path(p) for pattern in sys.argv[1:] for p in glob.glob(pattern)]
rows = []

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

for manifest in manifest_paths:
    with manifest.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            line, run_name, checkpoint, run_dir_raw, num_shards_raw = raw.split("\t")
            run_dir = Path(run_dir_raw)
            num_shards = int(num_shards_raw)
            shard_metrics = []
            missing = []
            for shard in range(num_shards):
                metrics_path = run_dir / f"shard_{shard:02d}" / "metrics.json"
                if metrics_path.is_file():
                    shard_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                else:
                    missing.append(str(metrics_path))
            if missing:
                rows.append({
                    "line": line,
                    "run": run_name,
                    "checkpoint": checkpoint,
                    "state": "incomplete",
                    "missing_shards": missing,
                    "metrics_path": str(run_dir / "metrics.json"),
                })
                continue
            aggregate = {
                "line": line,
                "run": run_name,
                "checkpoint": checkpoint,
                "state": "done",
                "trajectory_output_key": os.environ["TRAJECTORY_OUTPUT_KEY"],
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
            rows.append({**aggregate, "metrics_path": str(run_dir / "metrics.json")})

summary_dir = summary_root / "summary"
summary_dir.mkdir(parents=True, exist_ok=True)
(summary_dir / f"{line_name}_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
fields = [
    "line", "run", "state", "checkpoint", "trajectory_output_key", "num_shards",
    "num_samples", "num_pdm_valid", "num_pdm_missing_metric_cache", "num_pdm_failed",
    "PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC", "trajectory_l1", "metrics_path",
]
with (summary_dir / f"{line_name}_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(json.dumps({"line": line_name, "rows": len(rows), "summary": str(summary_dir / f"{line_name}_summary.csv")}, sort_keys=True))
PY
}

jobs_tsv="$(write_line_jobs "${LINE}")"
if [[ ! -s "${jobs_tsv}" ]]; then
  echo "no jobs to launch for ${LINE}"
  exit 0
fi

launcher_root="${OUT_ROOT}/launcher_${LINE}"
mkdir -p "${launcher_root}"
echo "$$" >"${OUT_ROOT}/${LINE}_driver.pid"
{
  echo "driver_pid=$$"
  echo "line=${LINE}"
  echo "launcher_root=${launcher_root}"
  echo "started_at=$(date -Is)"
} | tee -a "${OUT_ROOT}/commands.log"

"${PYTHON_BIN}" "${LAUNCHER}" \
  --jobs-tsv "${jobs_tsv}" \
  --out-root "${launcher_root}" \
  --stagger-seconds "${STAGGER_SECONDS}" \
  --poll-seconds "${POLL_SECONDS}"

aggregate_line "${LINE}" | tee -a "${OUT_ROOT}/logs/${LINE}_aggregate.log"
echo "finished ${LINE} at $(date -Is)" | tee -a "${OUT_ROOT}/commands.log"
