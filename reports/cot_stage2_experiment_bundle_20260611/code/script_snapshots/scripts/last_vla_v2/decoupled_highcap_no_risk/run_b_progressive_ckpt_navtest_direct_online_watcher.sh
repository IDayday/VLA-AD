#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_b_progressive_online_eval_${RUN_ID}}"
B_ROOT="${B_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk}"
B_PROGRESSIVE_ROOT="${B_PROGRESSIVE_ROOT:-${B_ROOT}/progressive_sft_decoupled}"
B_ADAPTER_DIR="${B_ADAPTER_DIR:-${B_ROOT}/vlm_lora_cot_alignment/adapters/vlm_lora}"
CHECKPOINTS_FILE="${CHECKPOINTS_FILE:-${B_ROOT}/checkpoints_to_eval.txt}"
CONFIG="${CONFIG:-configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_eval_flat.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
BASE_VLM_PATH="${BASE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-0,2,4,6}"
PRECISION="${PRECISION:-fp32}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
REMOTE_TRAIN_HOST="${REMOTE_TRAIN_HOST:-training-rl-zt2}"
WAIT_FOR_LOCAL_EVAL_CLEAR="${WAIT_FOR_LOCAL_EVAL_CLEAR:-1}"
LOCAL_EVAL_WAIT_PATTERN="${LOCAL_EVAL_WAIT_PATTERN:-decoupled_highcap_no_risk_a_progressive_all_ckpt_eval}"
REQUIRE_TRAIN_DONE_FOR_LATEST="${REQUIRE_TRAIN_DONE_FOR_LATEST:-1}"

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

mkdir -p "${OUT_ROOT}/eval" "${OUT_ROOT}/logs"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
SUMMARY_JSONL="${OUT_ROOT}/summary.jsonl"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"
EVALUATED_TSV="${OUT_ROOT}/evaluated_ckpts.tsv"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "run,checkpoint,checkpoint_sha256,trajectory_output_key,num_samples,num_pdm_valid,num_pdm_missing_metric_cache,num_pdm_failed,PDMS,NC,DAC,TTC,comfort,EP,DDC,trajectory_l1,online_vlm_lora_adapter_dir,metrics_path" >"${SUMMARY_CSV}"
fi
touch "${EVALUATED_TSV}"

log() {
  echo "[$(date -Is)] $*" | tee -a "${OUT_ROOT}/logs/watcher.log"
}

safe_name() {
  local ckpt="$1"
  basename "${ckpt}" .ckpt | tr '/ :' '___'
}

local_a_eval_running() {
  ps -eo cmd | grep 'eval_recogdrive_expert_pdm.py' | grep -F "${LOCAL_EVAL_WAIT_PATTERN}" | grep -v grep >/dev/null 2>&1
}

wait_for_local_eval_clear() {
  if [[ "${WAIT_FOR_LOCAL_EVAL_CLEAR}" != "1" && "${WAIT_FOR_LOCAL_EVAL_CLEAR}" != "true" ]]; then
    return 0
  fi
  while local_a_eval_running; do
    log "waiting for local A eval to clear before starting B progressive direct-online eval"
    sleep "${POLL_SECONDS}"
  done
}

remote_training_running() {
  if [[ -z "${REMOTE_TRAIN_HOST}" ]]; then
    ps -eo cmd | grep 'run_training_recogdrive.py' | grep -F "output_dir=${B_PROGRESSIVE_ROOT}" | grep -v grep >/dev/null 2>&1
    return $?
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_TRAIN_HOST}" \
    "ps -eo cmd | grep 'run_training_recogdrive.py' | grep -F 'output_dir=${B_PROGRESSIVE_ROOT}' | grep -v grep >/dev/null" >/dev/null 2>&1
}

stable_checkpoint() {
  local ckpt="$1"
  [[ -f "${ckpt}" ]] || return 1
  local size_a size_b
  size_a="$(stat -c '%s' "${ckpt}" 2>/dev/null || echo 0)"
  [[ "${size_a}" -gt 0 ]] || return 1
  sleep "${STABLE_SECONDS}"
  [[ -f "${ckpt}" ]] || return 1
  size_b="$(stat -c '%s' "${ckpt}" 2>/dev/null || echo 0)"
  [[ "${size_a}" == "${size_b}" && "${size_b}" -gt 0 ]]
}

checkpoint_sha() {
  sha256sum "$1" | awk '{print $1}'
}

sha_already_evaluated() {
  local sha="$1"
  grep -F "	${sha}	" "${EVALUATED_TSV}" >/dev/null 2>&1
}

aggregate_one() {
  local run_name="$1"
  local ckpt="$2"
  local ckpt_sha="$3"
  local run_dir="$4"
  RUN_NAME="${run_name}" \
  CKPT="${ckpt}" \
  CKPT_SHA="${ckpt_sha}" \
  RUN_DIR="${run_dir}" \
  TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  SUMMARY_JSONL="${SUMMARY_JSONL}" \
  SUMMARY_CSV="${SUMMARY_CSV}" \
  B_ADAPTER_DIR="${B_ADAPTER_DIR}" \
  "${PYTHON_BIN}" - <<'PY'
import csv
import json
import os
from pathlib import Path

run_name = os.environ["RUN_NAME"]
ckpt = os.environ["CKPT"]
ckpt_sha = os.environ["CKPT_SHA"]
run_dir = Path(os.environ["RUN_DIR"])
trajectory_key = os.environ["TRAJECTORY_OUTPUT_KEY"]
num_shards = int(os.environ["NUM_SHARDS"])
adapter_dir = os.environ["B_ADAPTER_DIR"]
metric_keys = ["PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"]

shard_metrics = []
for idx in range(num_shards):
    path = run_dir / f"shard_{idx:02d}" / "metrics.json"
    if not path.is_file():
        raise SystemExit(f"missing shard metrics: {path}")
    shard_metrics.append(json.loads(path.read_text()))

def weighted_mean(key: str, weight_key: str):
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
    "checkpoint_sha256": ckpt_sha,
    "trajectory_output_key": trajectory_key,
    "online_vlm_lora_adapter_dir": adapter_dir,
    "num_shards": num_shards,
    "num_samples": sum(int(item.get("num_samples") or 0) for item in shard_metrics),
    "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in shard_metrics),
    "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in shard_metrics),
    "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in shard_metrics),
    "trajectory_l1": weighted_mean("trajectory_l1", "num_samples"),
    "shards": [str(run_dir / f"shard_{idx:02d}") for idx in range(num_shards)],
}
for key in metric_keys:
    aggregate[key] = weighted_mean(key, "num_pdm_valid")
aggregate["pdm_score"] = aggregate["PDMS"]

(run_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
(run_dir / "report.md").write_text(
    "\n".join([
        "# B Progressive Direct-Online NAVTEST",
        "",
        f"Run: `{run_name}`",
        f"Checkpoint: `{ckpt}`",
        f"Checkpoint sha256: `{ckpt_sha}`",
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

with Path(os.environ["SUMMARY_JSONL"]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(aggregate, sort_keys=True) + "\n")

fields = [
    "run",
    "checkpoint",
    "checkpoint_sha256",
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
with Path(os.environ["SUMMARY_CSV"]).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    row = {key: aggregate.get(key) for key in fields}
    row["metrics_path"] = str(run_dir / "metrics.json")
    writer.writerow(row)
print(json.dumps(aggregate, sort_keys=True), flush=True)
PY
}

eval_one() {
  local ckpt="$1"
  local ckpt_sha="$2"
  local run_name run_dir shard pids failed shard_dir shard_log
  run_name="$(safe_name "${ckpt}")_${TRAJECTORY_OUTPUT_KEY}_online_lora"
  run_dir="${OUT_ROOT}/eval/${run_name}"
  mkdir -p "${run_dir}" "${OUT_ROOT}/logs/${run_name}"

  if [[ -f "${run_dir}/metrics.json" ]]; then
    log "skip completed aggregate ${run_name}"
    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${ckpt_sha}" "${ckpt}" "${run_dir}/metrics.json" >>"${EVALUATED_TSV}"
    return 0
  fi

  wait_for_local_eval_clear
  log "launch B progressive eval run=${run_name} ckpt=${ckpt}"
  pids=()
  failed=0
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    shard_dir="${run_dir}/shard_$(printf '%02d' "${shard}")"
    shard_log="${OUT_ROOT}/logs/${run_name}/shard_$(printf '%02d' "${shard}").log"
    mkdir -p "${shard_dir}"
    if [[ -f "${shard_dir}/metrics.json" ]]; then
      log "skip completed shard run=${run_name} shard=${shard}"
      continue
    fi
    {
      echo "[$(date -Is)] start run=${run_name} shard=${shard} gpu=${EVAL_GPUS[$shard]}"
      printf '%q ' CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$shard]}" PYTHONUNBUFFERED=1 \
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
        --online-vlm-lora-adapter-dir "${B_ADAPTER_DIR}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${shard}" \
        --output-dir "${shard_dir}"
      printf '\n'
    } | tee -a "${shard_log}" >>"${COMMANDS_LOG}"
    CUDA_VISIBLE_DEVICES="${EVAL_GPUS[$shard]}" PYTHONUNBUFFERED=1 \
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
      --online-vlm-lora-adapter-dir "${B_ADAPTER_DIR}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${shard}" \
      --output-dir "${shard_dir}" >>"${shard_log}" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=$((failed + 1))
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    log "shard eval failed for ${run_name}: failed=${failed}"
    return 1
  fi
  aggregate_one "${run_name}" "${ckpt}" "${ckpt_sha}" "${run_dir}" | tee -a "${OUT_ROOT}/logs/${run_name}/aggregate.log"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${ckpt_sha}" "${ckpt}" "${run_dir}/metrics.json" >>"${EVALUATED_TSV}"
}

cd "${PROJECT_ROOT}"
[[ -f "${B_ADAPTER_DIR}/adapter_model.bin" ]] || { echo "missing B adapter: ${B_ADAPTER_DIR}/adapter_model.bin" >&2; exit 2; }
[[ -d "${NAVTEST_CHUNK_CACHE_ROOT}" ]] || { echo "missing navtest chunk cache: ${NAVTEST_CHUNK_CACHE_ROOT}" >&2; exit 2; }
[[ -f "${CHECKPOINTS_FILE}" ]] || { echo "missing checkpoints file: ${CHECKPOINTS_FILE}" >&2; exit 2; }

log "watching B progressive checkpoints from ${CHECKPOINTS_FILE}"
log "using direct-online VLM LoRA adapter ${B_ADAPTER_DIR}; no LoRA navtest full cache will be generated"

while true; do
  launched_any=0
  while IFS= read -r ckpt; do
    [[ -z "${ckpt}" ]] && continue
    if [[ ! -f "${ckpt}" ]]; then
      continue
    fi
    if [[ "$(basename "${ckpt}")" == "latest.ckpt" && ( "${REQUIRE_TRAIN_DONE_FOR_LATEST}" == "1" || "${REQUIRE_TRAIN_DONE_FOR_LATEST}" == "true" ) ]]; then
      if remote_training_running; then
        continue
      fi
    fi
    log "candidate checkpoint found: ${ckpt}; waiting for stable size"
    if ! stable_checkpoint "${ckpt}"; then
      log "checkpoint is not stable yet: ${ckpt}"
      continue
    fi
    ckpt_sha="$(checkpoint_sha "${ckpt}")"
    if sha_already_evaluated "${ckpt_sha}"; then
      log "skip duplicate/evaluated checkpoint sha=${ckpt_sha} path=${ckpt}"
      continue
    fi
    eval_one "${ckpt}" "${ckpt_sha}"
    launched_any=1
  done <"${CHECKPOINTS_FILE}"
  if [[ "${launched_any}" -eq 0 ]]; then
    log "no new B progressive checkpoint ready"
  fi
  sleep "${POLL_SECONDS}"
done
