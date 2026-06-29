#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
ANSWER_RUN=${ANSWER_RUN:-/mnt/project/onevl_navsim_exp/answer_full_20260624_184133}
ANSWER_SWIFT_DIR=${ANSWER_SWIFT_DIR:-${ANSWER_RUN}/swift_output/v0-20260624-184217}
TRAIN_PID_FILE=${TRAIN_PID_FILE:-${ANSWER_RUN}/launcher.pid}
POLL_SECONDS=${POLL_SECONDS:-60}
STOP_AFTER_TRAINING=${STOP_AFTER_TRAINING:-1}
CHECKPOINT_MIN_AGE_SECONDS=${CHECKPOINT_MIN_AGE_SECONDS:-120}

mkdir -p "${ANSWER_RUN}/logs"
LOG_FILE="${ANSWER_RUN}/logs/checkpoint_validation_watcher.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "$0" "$@"
  printf '\n'
  env | sort | grep -E '^(PYTHON_BIN|ANSWER_RUN|ANSWER_SWIFT_DIR|TRAIN_PID_FILE|POLL_SECONDS|STOP_AFTER_TRAINING|CHECKPOINT_MIN_AGE_SECONDS)=' || true
} >> "${ANSWER_RUN}/commands.log"

training_running() {
  if [ ! -f "${TRAIN_PID_FILE}" ]; then
    return 1
  fi
  local pid
  pid="$(cat "${TRAIN_PID_FILE}" 2>/dev/null || true)"
  [ -n "${pid}" ] && ps -p "${pid}" >/dev/null 2>&1
}

validate_checkpoint() {
  local checkpoint_path="$1"
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint_path}")"
  local report_path="${ANSWER_RUN}/${checkpoint_name}.validation.json"
  local log_path="${ANSWER_RUN}/logs/${checkpoint_name}.validation.log"
  local tmp_report="${report_path}.tmp"
  local tmp_log="${log_path}.tmp"

  log "validating ${checkpoint_name}"
  if TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_hf_checkpoint.py" \
    --model-path "${checkpoint_path}" \
    --require-tokenizer \
    --transformers-smoke \
    --report-json "${tmp_report}" \
    > "${tmp_log}" 2>&1; then
    mv "${tmp_report}" "${report_path}"
    mv "${tmp_log}" "${log_path}"
    log "validation_ok ${checkpoint_name}"
  else
    mv "${tmp_report}" "${report_path}" 2>/dev/null || true
    mv "${tmp_log}" "${log_path}" 2>/dev/null || true
    log "validation_failed ${checkpoint_name}; see ${log_path}"
  fi
}

checkpoint_report_ok() {
  local report_path="$1"
  [ -f "${report_path}" ] || return 1
  "${PYTHON_BIN}" - "${report_path}" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if report.get("ok") is True else 1)
PY
}

checkpoint_age_seconds() {
  local checkpoint_path="$1"
  local now
  local latest
  now="$(date +%s)"
  latest="$(find "${checkpoint_path}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1)"
  if [ -z "${latest}" ]; then
    latest="$(stat -c %Y "${checkpoint_path}")"
  fi
  printf '%s\n' "$((now - latest))"
}

scan_once() {
  if [ ! -d "${ANSWER_SWIFT_DIR}" ]; then
    log "missing_answer_swift_dir ${ANSWER_SWIFT_DIR}"
    return
  fi

  mapfile -t checkpoints < <(
    find "${ANSWER_SWIFT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%p\n' | sort -V
  )
  for checkpoint_path in "${checkpoints[@]}"; do
    checkpoint_name="$(basename "${checkpoint_path}")"
    report_path="${ANSWER_RUN}/${checkpoint_name}.validation.json"
    if checkpoint_report_ok "${report_path}"; then
      continue
    fi
    age_seconds="$(checkpoint_age_seconds "${checkpoint_path}")"
    if [ "${age_seconds}" -lt "${CHECKPOINT_MIN_AGE_SECONDS}" ]; then
      log "checkpoint_too_new ${checkpoint_name} age=${age_seconds}s min_age=${CHECKPOINT_MIN_AGE_SECONDS}s"
      continue
    fi
    validate_checkpoint "${checkpoint_path}"
  done
}

log "checkpoint_validation_watcher_started answer_swift_dir=${ANSWER_SWIFT_DIR}"

while true; do
  scan_once
  if [ "${STOP_AFTER_TRAINING}" = "1" ] && ! training_running; then
    scan_once
    log "training_not_running; checkpoint_validation_watcher_done"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
