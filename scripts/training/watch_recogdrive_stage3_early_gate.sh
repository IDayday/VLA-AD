#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a Stage3 output directory.}"
THRESHOLD="${THRESHOLD:-0.88}"
MARGIN="${MARGIN:-0.005}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/early_gate_latest.json}"
LOG_FILE="${LOG_FILE:-${RUN_ROOT}/early_gate_watch.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${LOG_FILE}"
}

read_json_field() {
  local field="$1"
  python - <<PY
import json
from pathlib import Path

payload = json.loads(Path("${OUTPUT_JSON}").read_text())
value = payload.get("${field}", "")
print("" if value is None else value)
PY
}

log "starting early gate watcher run_root=${RUN_ROOT} threshold=${THRESHOLD} margin=${MARGIN} stop_on_fail=${STOP_ON_FAIL}"

while true; do
  set +e
  gate_output="$(
    python "${REPO_ROOT}/scripts/training/gate_recogdrive_stage3_early_pdms.py" \
      --run-root "${RUN_ROOT}" \
      --threshold "${THRESHOLD}" \
      --margin "${MARGIN}" \
      --output-json "${OUTPUT_JSON}" 2>&1
  )"
  rc=$?
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    log "gate command failed rc=${rc}: ${gate_output}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  action="$(read_json_field action)"
  reason="$(read_json_field reason)"
  log "action=${action} reason=${reason}"

  if [[ "${action}" == "continue" ]]; then
    log "early gate passed; exiting watcher"
    exit 0
  fi

  if [[ "${action}" == "stop" ]]; then
    if [[ "${STOP_ON_FAIL}" != "1" ]]; then
      log "early gate failed, but STOP_ON_FAIL=${STOP_ON_FAIL}; leaving training untouched"
      exit 2
    fi
    pgid="$(read_json_field train_pgid)"
    if [[ ! "${pgid}" =~ ^[0-9]+$ || "${pgid}" == "0" ]]; then
      log "early gate failed but train_pgid is invalid: ${pgid}"
      exit 3
    fi
    log "early gate failed; sending TERM to local training process group -${pgid}"
    kill -TERM "-${pgid}" || true
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
