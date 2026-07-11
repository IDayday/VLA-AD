#!/usr/bin/env bash
set -Eeuo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE1_RUN_DIR=${STAGE1_RUN_DIR:?Set STAGE1_RUN_DIR to the formal Stage1 output directory}
STAGE1_PID=${STAGE1_PID:-}
EXPECTED_STAGE1_STEPS=${EXPECTED_STAGE1_STEPS:-720}
POLL_SECONDS=${POLL_SECONDS:-30}

RUN_TAG=${RUN_TAG:-$(basename "${STAGE1_RUN_DIR}")}
CACHE_RUN_ROOT=${CACHE_RUN_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_cache_${RUN_TAG}}
STAGE2_OUTPUT_DIR=${STAGE2_OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_closest_public_${RUN_TAG}}
WATCH_DIR=${WATCH_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_watch_${RUN_TAG}}
WATCH_LOG=${WATCH_LOG:-${WATCH_DIR}/watch.log}
WATCH_STATUS=${WATCH_STATUS:-${WATCH_DIR}/status}
LOCK_PATH=${LOCK_PATH:-${WATCH_DIR}/watch.lock}

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
N_SHARDS=${N_SHARDS:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29551}
NAVSIM_PYTHON=${NAVSIM_PYTHON:-/root/miniconda3/envs/navsim/bin/python}

if [[ ! -x "${NAVSIM_PYTHON}" ]]; then
  echo "NAVSIM_PYTHON is missing or not executable: ${NAVSIM_PYTHON}" >&2
  exit 2
fi

mkdir -p "${WATCH_DIR}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "A Stage1-to-Stage2 watcher already owns ${LOCK_PATH}" >&2
  exit 75
fi

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCH_LOG}"
}

write_status() {
  printf '%s\n' "$1" > "${WATCH_STATUS}"
}

stage1_complete() {
  STAGE1_RUN_DIR="${STAGE1_RUN_DIR}" EXPECTED_STAGE1_STEPS="${EXPECTED_STAGE1_STEPS}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["STAGE1_RUN_DIR"])
expected_steps = int(os.environ["EXPECTED_STAGE1_STEPS"])
if not (root / "model.safetensors").is_file() or not (root / "config.json").is_file():
    raise SystemExit(1)
state_path = root / "trainer_state.json"
if not state_path.is_file():
    raise SystemExit(1)
state = json.loads(state_path.read_text(encoding="utf-8"))
if int(state.get("global_step", -1)) < expected_steps:
    raise SystemExit(1)
if (root / "model.safetensors").stat().st_size < 4_000_000_000:
    raise SystemExit(1)
PY
}

log "watcher started at git commit $(cd "${VLA_AD_ROOT}" && git rev-parse HEAD)"
log "Stage1=${STAGE1_RUN_DIR}; cache=${CACHE_RUN_ROOT}; Stage2=${STAGE2_OUTPUT_DIR}"
write_status waiting_for_stage1

while ! stage1_complete; do
  if [[ -n "${STAGE1_PID}" ]] && ! kill -0 "${STAGE1_PID}" 2>/dev/null; then
    log "Stage1 PID ${STAGE1_PID} exited before a complete final checkpoint appeared"
    write_status stage1_failed
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

log "Stage1 final checkpoint passed completeness checks"
write_status building_stage2_cache
set +e
env \
  VLM_PATH="${STAGE1_RUN_DIR}" \
  OUTPUT_ROOT="${CACHE_RUN_ROOT}" \
  SPLITS=train \
  EXPECTED_TRAIN_CLIPS=1000 \
  GPU_LIST="${GPU_LIST}" \
  N_SHARDS="${N_SHARDS}" \
  bash "${VLA_AD_ROOT}/scripts/bench2drive/build_recogdrive_b2d_stage2_cache.sh" \
  >> "${WATCH_DIR}/cache_build.log" 2>&1
cache_status=$?
set -e
if [[ "${cache_status}" -ne 0 ]]; then
  log "Stage2 cache build failed with status ${cache_status}"
  write_status cache_failed
  exit "${cache_status}"
fi

log "Stage2 cache build completed; validating exact 1000-clip/202656-record contract"
write_status validating_stage2_cache
"${NAVSIM_PYTHON}" "${VLA_AD_ROOT}/scripts/bench2drive/validate_recogdrive_b2d_stage2_cache.py" \
  --cache-root "${CACHE_RUN_ROOT}/train" \
  --expected-shards "${N_SHARDS}" \
  --expected-clips 1000 \
  --expected-records 202656 \
  --expected-vlm-path "${STAGE1_RUN_DIR}" \
  > "${WATCH_DIR}/cache_validation.json"

log "Cache validation passed; launching scratch Stage2 immediately"
write_status running_stage2
set +e
env \
  CHUNK_CACHE_ROOT="${CACHE_RUN_ROOT}/train" \
  EXPECTED_VLM_PATH="${STAGE1_RUN_DIR}" \
  EXPECTED_RECORDS=202656 \
  OUTPUT_DIR="${STAGE2_OUTPUT_DIR}" \
  GPU_LIST="${GPU_LIST}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_PORT="${MASTER_PORT}" \
  bash "${VLA_AD_ROOT}/scripts/bench2drive/run_recogdrive_b2d_stage2_closest_public.sh" \
  >> "${WATCH_DIR}/stage2_train.log" 2>&1
stage2_status=$?
set -e
if [[ "${stage2_status}" -eq 0 ]]; then
  log "Stage2 training completed successfully"
  write_status stage2_complete
else
  log "Stage2 training failed with status ${stage2_status}"
  write_status stage2_failed
fi
exit "${stage2_status}"
