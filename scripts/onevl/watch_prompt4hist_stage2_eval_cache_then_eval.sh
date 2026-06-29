#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT}"
TRAIN_DIR="${TRAIN_DIR:-/mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_cache_20260628T052126Z/train_full200}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/project/VLA-AD/configs/onevl_ar_answer_stage2_small.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/eval_top5}"
VAL_CACHE_ROOT="${VAL_CACHE_ROOT:-${OUT_ROOT}/cache_val6000}"
NAV_CACHE_ROOT="${NAV_CACHE_ROOT:-${OUT_ROOT}/cache_navtest}"
VAL_METRIC_CACHE="${VAL_METRIC_CACHE:-/mnt/project/onevl_navsim_exp/metric_cache_val6000_latest}"
NAV_METRIC_CACHE="${NAV_METRIC_CACHE:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
REMOTE_HOST="${REMOTE_HOST:-local}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/root/miniconda3/envs/navsim/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/VLA-AD}"
MIN_EPOCH="${MIN_EPOCH:-50}"
MIN_STEP="${MIN_STEP:-40000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
NUM_SHARDS="${NUM_SHARDS:-8}"
TOP_K="${TOP_K:-5}"

mkdir -p "${OUT_ROOT}/logs" "${EVAL_ROOT}"
WATCH_LOG="${OUT_ROOT}/logs/eval_cache_then_eval.log"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

if [[ "${REMOTE_HOST}" == "training-rl-zt3" || "${REMOTE_HOST}" == "rl-zt3" ]]; then
  echo "REMOTE_HOST=${REMOTE_HOST} is disabled for this workflow; choose another host or use local." >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}"
}

validate_eval_cache() {
  "${PYTHON_BIN}" - "${VAL_CACHE_ROOT}" "${NAV_CACHE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

val_root = Path(sys.argv[1])
nav_root = Path(sys.argv[2])
for name, root, expected in [
    ("val6000", val_root, 6000),
    ("navtest", nav_root, 12146),
]:
    summary_path = root / "aggregate_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing {name} summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    total = int(summary.get("total_records", -1))
    errors = summary.get("errors") or []
    if total != expected:
        raise SystemExit(f"{name} total_records={total}, expected={expected}")
    if errors:
        raise SystemExit(f"{name} validation errors: {errors[:5]}")
print("eval cache ready")
PY
}

log "eval watcher bootstrap started; OUT_ROOT=${OUT_ROOT}"
while true; do
  if [[ -f "${OUT_ROOT}/eval_cache.status" ]]; then
    status="$(cat "${OUT_ROOT}/eval_cache.status" || true)"
    if [[ "${status}" != "0" ]]; then
      log "eval cache failed with status=${status}; not starting evaluation watcher."
      echo "cache_failed" > "${OUT_ROOT}/eval_watch.status"
      exit 1
    fi
    if validate_eval_cache >> "${WATCH_LOG}" 2>&1; then
      break
    fi
    log "eval_cache.status=0 but validation not ready; waiting."
  fi
  sleep 60
done

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/watch_onevl_stage2_eval_top5_remote.py"
  --train-dir "${TRAIN_DIR}"
  --output-root "${EVAL_ROOT}"
  --config "${CONFIG_PATH}"
  --remote-host "${REMOTE_HOST}"
  --remote-python "${REMOTE_PYTHON}"
  --repo-root "${REPO_ROOT}"
  --min-epoch "${MIN_EPOCH}"
  --min-step "${MIN_STEP}"
  --poll-seconds "${POLL_SECONDS}"
  --stable-seconds "${STABLE_SECONDS}"
  --num-shards "${NUM_SHARDS}"
  --top-k "${TOP_K}"
  --val-cache-root "${VAL_CACHE_ROOT}"
  --val-cache-pattern "val6000_chunk_*"
  --val-metric-cache "${VAL_METRIC_CACHE}"
  --nav-cache-root "${NAV_CACHE_ROOT}"
  --nav-cache-pattern "shard_*"
  --nav-metric-cache "${NAV_METRIC_CACHE}"
)

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >> "${COMMANDS_LOG}"

ln -sfn "${EVAL_ROOT}" /mnt/project/onevl_navsim_exp/ar_answer_prompt4hist_stage2_apsd_eval_top5_latest
echo "running" > "${OUT_ROOT}/eval_watch.status"
log "starting top5 eval watcher; remote_host=${REMOTE_HOST}"
exec "${cmd[@]}" >> "${WATCH_LOG}" 2>&1
