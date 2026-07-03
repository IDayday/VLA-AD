#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
DDV2_OUT_ROOT=${DDV2_OUT_ROOT:-$(cat "${REPO_ROOT}/outputs/latest_ddv2_navtrain_candidates.txt")}
RUN_ID=${RUN_ID:-sg_fps_v3_support_ddv2_$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_ID}}
SUPPORT_ARCHIVE=${SUPPORT_ARCHIVE:-${OUT_ROOT}/support_archive}
FS_STATS=${FS_STATS:-${OUT_ROOT}/fs_norm_stats.pt}
STAGE2_OUT=${STAGE2_OUT:-${OUT_ROOT}/stage2_dpsi_fs_norm}
SHARD_COUNT=${SHARD_COUNT:-8}
POLL_SECONDS=${POLL_SECONDS:-60}
DDV2_SOURCE_NAME=${DDV2_SOURCE_NAME:-diffusiondrivev2}
EXTERNAL_CANDIDATE_ROOTS=${EXTERNAL_CANDIDATE_ROOTS:-${DDV2_SOURCE_NAME}=${DDV2_OUT_ROOT}/submissions}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids" "${SUPPORT_ARCHIVE}"

{
  echo "run_id=${RUN_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "ddv2_out_root=${DDV2_OUT_ROOT}"
  echo "support_archive=${SUPPORT_ARCHIVE}"
  echo "fs_stats=${FS_STATS}"
  echo "stage2_out=${STAGE2_OUT}"
  echo "external_candidate_roots=${EXTERNAL_CANDIDATE_ROOTS}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT_ROOT}/commands.log"

_live_pid_count() {
  local dir="$1"
  local count=0
  shopt -s nullglob
  for f in "${dir}"/*.pid; do
    local pid
    pid=$(cat "${f}" 2>/dev/null || true)
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  shopt -u nullglob
  echo "${count}"
}

echo "waiting_for_ddv2_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  live=$(_live_pid_count "${DDV2_OUT_ROOT}/pids")
  summaries=$(find "${DDV2_OUT_ROOT}/submissions" -name summary.json 2>/dev/null | wc -l)
  submissions=$(find "${DDV2_OUT_ROOT}/submissions" -name submission.pkl 2>/dev/null | wc -l)
  echo "ddv2_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) live=${live} summaries=${summaries} submissions=${submissions}"
  if [[ "${live}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" - "${DDV2_OUT_ROOT}" "${SHARD_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
bad = []
for idx in range(expected):
    shard = root / "submissions" / f"shard_{idx}"
    summary_path = shard / "summary.json"
    submission_path = shard / "submission.pkl"
    if not summary_path.exists():
        bad.append(f"missing summary: {summary_path}")
        continue
    if not submission_path.exists():
        bad.append(f"missing submission: {submission_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if int(summary.get("failure_count", 0)) != 0:
        bad.append(f"nonzero failures in {summary_path}: {summary.get('failure_count')}")
    if int(summary.get("prediction_count", 0)) <= 0:
        bad.append(f"empty predictions in {summary_path}")
if bad:
    raise SystemExit("\n".join(bad))
print("ddv2_validation=passed")
PY

echo "ddv2_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "launching_support_build_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu=$((shard % 8))
  shard_root="${OUT_ROOT}/shard_${shard}"
  mkdir -p "${shard_root}"
  {
    printf '[%s] CUDA_VISIBLE_DEVICES=%s SHARD_INDEX=%s SHARD_COUNT=%s OUT_ROOT=%q OUTPUT_PATH=%q EXTERNAL_CANDIDATE_ROOTS=%q ' \
      "$(date -Is)" "${gpu}" "${shard}" "${SHARD_COUNT}" "${shard_root}" "${SUPPORT_ARCHIVE}" "${EXTERNAL_CANDIDATE_ROOTS}"
    printf 'bash scripts/training/sg_fps/run_build_sg_fps_support.sh\n'
  } >> "${OUT_ROOT}/commands.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    SHARD_INDEX="${shard}" \
    SHARD_COUNT="${SHARD_COUNT}" \
    OUT_ROOT="${shard_root}" \
    OUTPUT_PATH="${SUPPORT_ARCHIVE}" \
    BATCH_SIZE="${SUPPORT_BUILD_BATCH_SIZE:-1}" \
    MAX_SCENES="${MAX_SCENES:-0}" \
    AWAC_PDM_SHADOW_CHECK="${AWAC_PDM_SHADOW_CHECK:-false}" \
    EXTERNAL_CANDIDATE_ROOTS="${EXTERNAL_CANDIDATE_ROOTS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_build_sg_fps_support.sh" \
    > "${OUT_ROOT}/logs/build_shard_${shard}.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/pids/build_shard_${shard}.pid"
done

while true; do
  live=$(_live_pid_count "${OUT_ROOT}/pids")
  records=$(find "${SUPPORT_ARCHIVE}" -name '*.pkl.xz' 2>/dev/null | wc -l)
  echo "support_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) live=${live} records=${records}"
  if [[ "${live}" -eq 0 ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

missing=0
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  summary="${OUT_ROOT}/shard_${shard}/awac_elite_buffer_summary.json"
  log="${OUT_ROOT}/logs/build_shard_${shard}.log"
  if [[ ! -f "${summary}" ]]; then
    echo "missing_summary=${summary}"
    missing=1
  fi
  if grep -E "Traceback|RuntimeError|ValueError|FileNotFoundError|CUDA out of memory" "${log}" >/dev/null 2>&1; then
    echo "error_in_log=${log}"
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "support_build_failed_or_incomplete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi

records=$(find "${SUPPORT_ARCHIVE}" -name '*.pkl.xz' 2>/dev/null | wc -l)
echo "support_build_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) records=${records}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/tools/build_fs_norm_stats.py" \
  --support_archive_path "${SUPPORT_ARCHIVE}" \
  --output_path "${FS_STATS}" \
  --use_robust "${USE_ROBUST:-true}" \
  --clip "${FS_NORM_CLIP:-5.0}"

mkdir -p "${STAGE2_OUT}"
echo "stage2_launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) out=${STAGE2_OUT}"
setsid env \
  SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE}" \
  OUT_ROOT="${STAGE2_OUT}" \
  CACHE_MODE="${CACHE_MODE:-offline}" \
  HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}" \
  USE_FS_NORM="${USE_FS_NORM:-true}" \
  FS_NORM_STATS_PATH="${FS_STATS}" \
  FS_NORM_USE_ROBUST="${FS_NORM_USE_ROBUST:-true}" \
  X0_AUX_WEIGHT="${X0_AUX_WEIGHT:-0.1}" \
  GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.05}" \
  SG_FPS_USE_DPSI="${SG_FPS_USE_DPSI:-true}" \
  VALIDATE_ELITE_BUFFER="${VALIDATE_ELITE_BUFFER:-false}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${REPO_ROOT}/scripts/training/sg_fps/run_train_dpsi.sh" \
  > "${OUT_ROOT}/logs/stage2_dpsi_fs_norm.log" 2>&1 < /dev/null &
echo $! > "${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid"
echo "stage2_pid=$(cat "${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid")"
