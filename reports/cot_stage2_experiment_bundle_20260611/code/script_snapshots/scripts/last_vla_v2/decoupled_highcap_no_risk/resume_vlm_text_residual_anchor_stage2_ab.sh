#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_ab_${RUN_ID}}"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
NUM_ANCHOR_SHARDS="${NUM_ANCHOR_SHARDS:-16}"

ANCHOR_CACHE_BASE="${ANCHOR_CACHE_BASE:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk}"
TRAIN_ANCHOR_RAW_ROOT="${TRAIN_ANCHOR_RAW_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_train_${RUN_ID}_raw}"
TRAIN_ANCHOR_CACHE_ROOT="${TRAIN_ANCHOR_CACHE_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_train_${RUN_ID}}"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

shard_dir_name() {
  printf 'shard_%05d' "$1"
}

wait_for_train_anchor_shards() {
  while true; do
    local done_count=0
    local failed_count=0
    for ((shard = 0; shard < NUM_ANCHOR_SHARDS; shard++)); do
      local status_file="${TRAIN_ANCHOR_RAW_ROOT}/shards/$(shard_dir_name "${shard}")/status.txt"
      if [[ -f "${status_file}" ]]; then
        done_count=$((done_count + 1))
        if [[ "$(cat "${status_file}")" != "0" ]]; then
          failed_count=$((failed_count + 1))
        fi
      fi
    done
    echo "[$(date -Is)] train anchor shards complete: ${done_count}/${NUM_ANCHOR_SHARDS}, failed=${failed_count}"
    if [[ "${failed_count}" -gt 0 ]]; then
      return 20
    fi
    if [[ "${done_count}" -eq "${NUM_ANCHOR_SHARDS}" ]]; then
      return 0
    fi
    sleep 60
  done
}

merge_train_anchor_cache() {
  if [[ -f "${TRAIN_ANCHOR_CACHE_ROOT}/index.jsonl" && "${REUSE_EXISTING_ANCHOR_CACHE:-0}" == "1" ]]; then
    echo "Reusing merged train anchor cache: ${TRAIN_ANCHOR_CACHE_ROOT}"
    return 0
  fi
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py \
    --sharded-root "${TRAIN_ANCHOR_RAW_ROOT}" \
    --output-root "${TRAIN_ANCHOR_CACHE_ROOT}" \
    --expected-num-shards "${NUM_ANCHOR_SHARDS}" \
    --overlay-type vlm_text_anchor \
    --copy-mode hardlink \
    --strict
  "${PYTHON_BIN}" - "${TRAIN_ANCHOR_CACHE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
total = 0
parse_fail = 0
with (root / "index.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        total += 1
        if record.get("parse_ok") is False:
            parse_fail += 1
payload = {"anchor_cache_root": str(root), "total": total, "parse_fail": parse_fail}
print(json.dumps(payload, sort_keys=True))
if total == 0 or parse_fail:
    raise SystemExit(21)
PY
}

stop_remote_gpu_stress() {
  echo "[$(date -Is)] stopping remote GPU stress jobs on ${REMOTE_HOST}"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "pkill -f /tmp/gpu_stress_remote.py 2>/dev/null || true"
}

launch_training_and_eval_pipeline() {
  setsid env \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN="${PYTHON_BIN}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    RUN_ID="${RUN_ID}" \
    OUT_ROOT="${OUT_ROOT}" \
    REMOTE_HOST="${REMOTE_HOST}" \
    TRAIN_ANCHOR_RAW_ROOT="${TRAIN_ANCHOR_RAW_ROOT}" \
    TRAIN_ANCHOR_CACHE_ROOT="${TRAIN_ANCHOR_CACHE_ROOT}" \
    REUSE_EXISTING_ANCHOR_CACHE=1 \
    RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH="${RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH:-0}" \
    RUN_TOP5_EVAL_WATCHER="${RUN_TOP5_EVAL_WATCHER:-0}" \
    bash scripts/last_vla_v2/decoupled_highcap_no_risk/launch_vlm_text_residual_anchor_stage2_ab.sh \
    >"${LOG_DIR}/pipeline.resume_after_anchor.log" 2>&1 < /dev/null &
  echo "$!" >"${OUT_ROOT}/pipeline.resume_after_anchor.pid"
}

cd "${PROJECT_ROOT}"
wait_for_train_anchor_shards
merge_train_anchor_cache
stop_remote_gpu_stress
launch_training_and_eval_pipeline
echo "Resume pipeline launched after train anchor merge."
