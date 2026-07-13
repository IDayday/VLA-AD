#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
METRIC_OUT_ROOT="${METRIC_OUT_ROOT:?Set METRIC_OUT_ROOT to the metric-cache stable-launch output root.}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-stage3_rl_2b_safe_diffgrpo_online_$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${TRAIN_RUN_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-1}"
CACHE_MODE="${CACHE_MODE:-online}"

STATUS_FILE="${METRIC_OUT_ROOT}/status/metric_cache_navtrain.json"

while true; do
  if [[ ! -f "${STATUS_FILE}" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for status file: ${STATUS_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  STATE="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${STATUS_FILE}").read_text())
print(payload.get("state", "unknown"))
PY
)"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) metric_cache_state=${STATE}"
  if [[ "${STATE}" == "done" ]]; then
    "${PYTHON_BIN}" - <<PY
import sys
sys.path.insert(0, "${REPO_ROOT}")
from pathlib import Path
from navsim.common.dataloader import MetricCacheLoader
loader = MetricCacheLoader(Path("${METRIC_CACHE_DIR}"))
count = len(loader)
if count <= 0:
    raise SystemExit("metric cache completed but loader count is zero")
print(f"metric_cache_count={count}")
PY
    METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      OUT_ROOT="${TRAIN_OUT_ROOT}" \
      CACHE_MODE="${CACHE_MODE}" \
      MAX_EPOCHS="${MAX_EPOCHS:-20}" \
      LR="${LR:-1e-4}" \
      BATCH_SIZE="${BATCH_SIZE:-4}" \
      ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}" \
      GRPO_SAMPLE_TIME="${GRPO_SAMPLE_TIME:-16}" \
      BC_ANNEAL="${BC_ANNEAL:-true}" \
      BC_COEFF_START="${BC_COEFF_START:-0.10}" \
      BC_COEFF_END="${BC_COEFF_END:-0.05}" \
      BC_ANNEAL_EPOCHS="${BC_ANNEAL_EPOCHS:-5}" \
      REFERENCE_KL_COEFF="${REFERENCE_KL_COEFF:-0.02}" \
      REFERENCE_KL_CHUNK_SIZE="${REFERENCE_KL_CHUNK_SIZE:-0}" \
      GRPO_USE_GSPO_RATIO="${GRPO_USE_GSPO_RATIO:-false}" \
      GRPO_GSPO_CLIP_LOW="${GRPO_GSPO_CLIP_LOW:-0.05}" \
      GRPO_GSPO_CLIP_HIGH="${GRPO_GSPO_CLIP_HIGH:-0.05}" \
      GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL="${GRPO_BEHAVIOR_POLICY_SYNC_INTERVAL:-4}" \
      GRPO_BEHAVIOR_POLICY_SAMPLE="${GRPO_BEHAVIOR_POLICY_SAMPLE:-true}" \
      GRPO_SCHEDULER_EPOCHS="${GRPO_SCHEDULER_EPOCHS:-${MAX_EPOCHS:-20}}" \
      GRPO_SCHEDULER_WARMUP_EPOCHS="${GRPO_SCHEDULER_WARMUP_EPOCHS:-0}" \
      GRPO_SCHEDULER_MIN_LR="${GRPO_SCHEDULER_MIN_LR:-1e-5}" \
      KILL_GPU_STRESS="${KILL_GPU_STRESS}" \
      bash "${REPO_ROOT}/scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh"
    exit 0
  fi

  if [[ "${STATE}" == "failed" || "${STATE}" == "launch_failed" ]]; then
    echo "Metric cache job failed; refusing to launch Stage 3 RL." >&2
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
