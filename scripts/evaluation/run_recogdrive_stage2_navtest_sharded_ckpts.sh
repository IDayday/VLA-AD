#!/usr/bin/env bash
set -Eeuo pipefail

: "${OUT_ROOT:?Set OUT_ROOT}"
: "${CHECKPOINTS:?Set CHECKPOINTS to a semicolon-separated checkpoint list}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CONFIG="${CONFIG:-/mnt/project/VLA-AD_last_vla_dev/configs/sg_fps_asmi_stage2_fs_norm_eval.yaml}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
CHUNK_NAME_PATTERN="${CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
NUM_SHARDS="${NUM_SHARDS:-7}"
GPUS_CSV="${GPUS_CSV:-1,2,3,4,5,6,7}"
PRECISION="${PRECISION:-fp32}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
INITIAL_NOISE_SEED="${INITIAL_NOISE_SEED:-}"

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
if (( ${#GPUS[@]} == 0 )); then
  echo "GPUS_CSV produced no GPUs: ${GPUS_CSV}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/state"
SUMMARY_TSV="${OUT_ROOT}/navtest_summary.tsv"
COMMANDS_LOG="${OUT_ROOT}/commands.log"
SUMMARY_LOCK="${OUT_ROOT}/state/summary.lock"
COMMANDS_LOCK="${OUT_ROOT}/state/commands.lock"
{
  flock -x 9
  if [[ ! -f "${SUMMARY_TSV}" ]]; then
    printf 'checkpoint\tcheckpoint_path\tPDMS\tNC\tDAC\tTTC\tcomfort\tEP\tDDC\ttrajectory_l1\tnum_samples\tnum_pdm_valid\tnum_pdm_failed\tnum_pdm_missing_metric_cache\teval_dir\n' >"${SUMMARY_TSV}"
  fi
} 9>"${SUMMARY_LOCK}"

safe_name() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import hashlib
import re
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
basename = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("._") or "ckpt"
path_hash = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
print(f"{basename}-{path_hash}")
PY
}

aggregate_metrics() {
  local eval_dir="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" - "${eval_dir}" "${checkpoint}" "${SUMMARY_TSV}" "${SUMMARY_LOCK}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
checkpoint = sys.argv[2]
summary_tsv = Path(sys.argv[3])
summary_lock = Path(sys.argv[4])
metrics = []
for path in sorted(eval_dir.glob("shard_*/metrics.json")):
    metrics.append(json.loads(path.read_text()))
if not metrics:
    raise SystemExit(f"no shard metrics under {eval_dir}")
initial_noise_seeds = {item.get("initial_noise_seed") for item in metrics}
if len(initial_noise_seeds) != 1:
    raise SystemExit(f"inconsistent initial_noise_seed across shards: {sorted(initial_noise_seeds, key=str)}")

def weighted_mean(key, weight_key):
    total = 0.0
    weight = 0.0
    for item in metrics:
        value = item.get(key)
        w = float(item.get(weight_key) or 0)
        if value is not None and w > 0:
            total += float(value) * w
            weight += w
    return total / weight if weight > 0 else None

payload = {
    "checkpoint": checkpoint,
    "checkpoint_name": Path(checkpoint).name,
    "eval_dir": str(eval_dir),
    "num_samples": sum(int(item.get("num_samples") or 0) for item in metrics),
    "num_pdm_valid": sum(int(item.get("num_pdm_valid") or 0) for item in metrics),
    "num_pdm_failed": sum(int(item.get("num_pdm_failed") or 0) for item in metrics),
    "num_pdm_missing_metric_cache": sum(int(item.get("num_pdm_missing_metric_cache") or 0) for item in metrics),
    "trajectory_l1": weighted_mean("trajectory_l1", "num_samples"),
    "initial_noise_seed": initial_noise_seeds.pop(),
}
for key in ("PDMS", "NC", "DAC", "TTC", "comfort", "EP", "DDC"):
    payload[key] = weighted_mean(key, "num_pdm_valid")
payload["pdm_score"] = payload["PDMS"]
(eval_dir / "aggregate_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
fields = [
    payload["checkpoint_name"], payload["checkpoint"],
    payload["PDMS"], payload["NC"], payload["DAC"], payload["TTC"], payload["comfort"], payload["EP"], payload["DDC"],
    payload["trajectory_l1"], payload["num_samples"], payload["num_pdm_valid"], payload["num_pdm_failed"],
    payload["num_pdm_missing_metric_cache"], payload["eval_dir"],
]
with summary_lock.open("a", encoding="utf-8") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    already_recorded = False
    if summary_tsv.exists():
        with summary_tsv.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                columns = line.rstrip("\n").split("\t")
                if index > 0 and len(columns) > 1 and columns[1] == checkpoint:
                    already_recorded = True
                    break
    if not already_recorded:
        with summary_tsv.open("a", encoding="utf-8") as f:
            f.write("\t".join("" if v is None else str(v) for v in fields) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

cd "${PROJECT_ROOT}"
IFS=';' read -r -a CKPTS <<<"${CHECKPOINTS}"
for checkpoint in "${CKPTS[@]}"; do
  checkpoint="${checkpoint//[$'\n\r\t ']}"
  [[ -n "${checkpoint}" ]] || continue
  if [[ ! -f "${checkpoint}" ]]; then
    echo "checkpoint not found: ${checkpoint}" >&2
    exit 2
  fi
  ckpt_name="$(safe_name "${checkpoint}")"
  eval_dir="${OUT_ROOT}/${ckpt_name}"
  mkdir -p "${eval_dir}/logs"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${eval_dir}/aggregate_metrics.json" ]]; then
    echo "skip completed ${checkpoint}"
    continue
  fi

  echo "[$(date -Is)] evaluating ${checkpoint}" | tee -a "${OUT_ROOT}/logs/run.log"
  pids=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    shard_dir="${eval_dir}/shard_$(printf '%02d' "${shard}")"
    mkdir -p "${shard_dir}"
    cmd=(
      "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
      --config "${CONFIG}"
      --checkpoint "${checkpoint}"
      --split navtest
      --chunk-cache-root "${CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${CHUNK_NAME_PATTERN}"
      --metric-cache-dir "${METRIC_CACHE_DIR}"
      --precision "${PRECISION}"
      --trajectory-output-key "${TRAJECTORY_OUTPUT_KEY}"
      --num-shards "${NUM_SHARDS}"
      --shard-index "${shard}"
      --output-dir "${shard_dir}"
    )
    if [[ -n "${INITIAL_NOISE_SEED}" ]]; then
      cmd+=(--initial-noise-seed "${INITIAL_NOISE_SEED}")
    fi
    {
      flock -x 9
      {
        printf '[%s] CUDA_VISIBLE_DEVICES=%q ' "$(date -Is)" "${gpu}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
      } >>"${COMMANDS_LOG}"
    } 9>"${COMMANDS_LOCK}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${eval_dir}/logs/shard_$(printf '%02d' "${shard}").log" 2>&1 &
    pids+=("$!")
    sleep "${STAGGER_SECONDS:-1}"
  done

  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if (( status != 0 )); then
    echo "one or more shards failed for ${checkpoint}" >&2
    exit "${status}"
  fi
  aggregate_metrics "${eval_dir}" "${checkpoint}" | tee -a "${OUT_ROOT}/logs/run.log"
done

echo "[$(date -Is)] all evals completed" | tee -a "${OUT_ROOT}/logs/run.log"
