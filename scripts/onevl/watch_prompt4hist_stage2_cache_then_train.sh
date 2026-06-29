#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the cache run root}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}/cache_full103k}"
TRAIN_ROOT="${TRAIN_ROOT:-${OUT_ROOT}/train_full200}"
CACHE_PID_FILE="${CACHE_PID_FILE:-${OUT_ROOT}/launcher.pid}"
POLL_SECONDS="${POLL_SECONDS:-120}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
VLA_AD_ROOT="${VLA_AD_ROOT:-/mnt/project/VLA-AD}"
CONFIG_PATH="${CONFIG_PATH:-/mnt/project/VLA-AD/configs/onevl_ar_answer_stage2_small.yaml}"
TRAIN_LATEST_LINK="${TRAIN_LATEST_LINK:-/mnt/project/onevl_navsim_exp/ar_answer_stage2_train_full200_prompt4hist_latest}"

MASTER_PORT="${MASTER_PORT:-29551}"
GLOBAL_EPOCHS="${GLOBAL_EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LR_ACTION_HEAD="${LR_ACTION_HEAD:-0.0001}"
MIN_LR="${MIN_LR:-0.000001}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
LOG_EVERY="${LOG_EVERY:-100}"
SEED="${SEED:-20260530}"

mkdir -p "${OUT_ROOT}/logs" "${TRAIN_ROOT}"
WATCH_LOG="${OUT_ROOT}/logs/watch_cache_then_train.log"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${WATCH_LOG}"
}

record_cmd() {
  {
    printf '[%s] ' "$(date -Is)"
    printf '%q ' "$@"
    printf '\n'
  } >> "${COMMANDS_LOG}"
}

validate_cache_ready() {
  "${PYTHON_BIN}" - "${CACHE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = root / "aggregate_summary.json"
if not summary_path.is_file():
    raise SystemExit(f"missing aggregate summary: {summary_path}")
summary = json.loads(summary_path.read_text())
expected = int(summary.get("expected_records", -1))
total = int(summary.get("total_records", -2))
errors = summary.get("errors") or []
if total != expected:
    raise SystemExit(f"cache count mismatch: total={total} expected={expected}")
if total <= 0:
    raise SystemExit(f"cache total_records is invalid: {total}")
if errors:
    raise SystemExit("cache validation errors: " + json.dumps(errors[:10], ensure_ascii=False))
sample_checks = summary.get("sample_checks") or []
if not sample_checks:
    raise SystemExit("cache summary has no sample_checks")
for row in sample_checks:
    if row.get("last_hidden_state") != [2800, 2560]:
        raise SystemExit(f"bad hidden shape in {row}")
    if row.get("history_trajectory") != [4, 3]:
        raise SystemExit(f"bad history shape in {row}")
    if row.get("trajectory") != [8, 3]:
        raise SystemExit(f"bad trajectory shape in {row}")
    if row.get("support_trajectories") != [3, 8, 3]:
        raise SystemExit(f"bad support_trajectories shape in {row}")
    if row.get("support_mask") != [3] or row.get("support_weights") != [3] or row.get("support_scores") != [3]:
        raise SystemExit(f"bad support field shape in {row}")
    if row.get("target_source") != "stage2_pareto_support":
        raise SystemExit(f"bad target_source in {row}")
    if row.get("stage2_target_mode") != "preserve_support":
        raise SystemExit(f"bad stage2_target_mode in {row}")
print(json.dumps({"ready": True, "total_records": total, "summary": str(summary_path)}, sort_keys=True))
PY
}

log "watcher started; OUT_ROOT=${OUT_ROOT}"
log "waiting for cache: ${CACHE_ROOT}"

while true; do
  if [[ -f "${OUT_ROOT}/train.status" ]]; then
    existing_status="$(cat "${OUT_ROOT}/train.status" || true)"
    log "train.status already exists (${existing_status}); not launching duplicate training."
    exit "${existing_status:-0}"
  fi

  if [[ -f "${OUT_ROOT}/cache.status" ]]; then
    cache_status="$(cat "${OUT_ROOT}/cache.status" || true)"
    if [[ "${cache_status}" != "0" ]]; then
      log "cache.status=${cache_status}; aborting training launch."
      echo "cache_failed" > "${OUT_ROOT}/watch.status"
      exit 1
    fi
    if validate_cache_ready >> "${WATCH_LOG}" 2>&1; then
      log "cache validation passed; launching training."
      break
    fi
    if [[ -f "${CACHE_PID_FILE}" ]]; then
      cache_pid="$(cat "${CACHE_PID_FILE}" || true)"
      if [[ -n "${cache_pid}" ]] && ! kill -0 "${cache_pid}" 2>/dev/null; then
        log "cache.status=0 but validation is not ready and cache pid ${cache_pid} has exited; aborting."
        echo "cache_validation_failed" > "${OUT_ROOT}/watch.status"
        exit 1
      fi
    fi
  fi

  if [[ -f "${CACHE_PID_FILE}" ]]; then
    cache_pid="$(cat "${CACHE_PID_FILE}" || true)"
    if [[ -n "${cache_pid}" ]] && ! kill -0 "${cache_pid}" 2>/dev/null && [[ ! -f "${OUT_ROOT}/cache.status" ]]; then
      log "cache launcher pid ${cache_pid} is gone before cache.status appeared; aborting."
      echo "cache_missing_status" > "${OUT_ROOT}/watch.status"
      exit 1
    fi
  fi
  sleep "${POLL_SECONDS}"
done

cd "${VLA_AD_ROOT}"
train_cmd=(
  torchrun
  --nproc_per_node=8
  --master_port "${MASTER_PORT}"
  "${VLA_AD_ROOT}/scripts/train_recogdrive_expert_chunked.py"
  --config "${CONFIG_PATH}"
  --chunk-cache-root "${CACHE_ROOT}"
  --chunk-name-pattern "shard_*"
  --global-epochs "${GLOBAL_EPOCHS}"
  --flat-global-dataset
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRAD_ACCUM}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --lr-scheduler official-cosine
  --lr-scheduler-epochs "${GLOBAL_EPOCHS}"
  --lr-warmup-epochs "${WARMUP_EPOCHS}"
  --min-lr "${MIN_LR}"
  --lr-action-head "${LR_ACTION_HEAD}"
  --lr-expert 0.0
  --jepa-align-weight 0.0
  --vggt-align-weight 0.0
  --precision bf16
  --final-check-precision fp32
  --save-every "${SAVE_EVERY}"
  --save-every-epoch
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
  --output-dir "${TRAIN_ROOT}"
)

record_cmd env "PYTHONPATH=${VLA_AD_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}"
ln -sfn "${TRAIN_ROOT}" "${TRAIN_LATEST_LINK}"
echo "training_running" > "${OUT_ROOT}/watch.status"
set +e
PYTHONPATH="${VLA_AD_ROOT}:${PYTHONPATH:-}" "${train_cmd[@]}" > "${OUT_ROOT}/logs/train.log" 2>&1
train_status=$?
set -e
echo "${train_status}" > "${OUT_ROOT}/train.status"
if [[ "${train_status}" -eq 0 ]]; then
  echo "complete" > "${OUT_ROOT}/watch.status"
  log "training exited successfully."
else
  echo "train_failed" > "${OUT_ROOT}/watch.status"
  log "training failed with status ${train_status}."
fi
exit "${train_status}"
