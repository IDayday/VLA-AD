#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
DDV2_OUT_ROOT=${DDV2_OUT_ROOT:-$(cat "${REPO_ROOT}/outputs/latest_ddv2_navtrain_candidates.txt")}
RUN_DRIVOR_AFTER_DDV2=${RUN_DRIVOR_AFTER_DDV2:-false}
DRIVOR_OUT_ROOT=${DRIVOR_OUT_ROOT:-}
DRIVOR_ROOT=${DRIVOR_ROOT:-/mnt/project/external/DrivoR}
DRIVOR_CHECKPOINT=${DRIVOR_CHECKPOINT:-${DRIVOR_ROOT}/weights/releases/drivor_Nav1_25epochs.pth}
DRIVOR_SHARD_COUNT=${DRIVOR_SHARD_COUNT:-${SHARD_COUNT:-8}}
DRIVOR_BATCH_SIZE=${DRIVOR_BATCH_SIZE:-64}
DRIVOR_PARTIAL_EVERY=${DRIVOR_PARTIAL_EVERY:-64}
DDV2_WAIT_MODE=${DDV2_WAIT_MODE:-final}
DDV2_MIN_READY_SHARDS=${DDV2_MIN_READY_SHARDS:-${SHARD_COUNT:-8}}
DDV2_MIN_PREDICTIONS_PER_SHARD=${DDV2_MIN_PREDICTIONS_PER_SHARD:-1}
STOP_DDV2_AFTER_READY=${STOP_DDV2_AFTER_READY:-false}
DRIVOR_WAIT_MODE=${DRIVOR_WAIT_MODE:-final}
DRIVOR_MIN_READY_SHARDS=${DRIVOR_MIN_READY_SHARDS:-${DRIVOR_SHARD_COUNT:-${SHARD_COUNT:-8}}}
DRIVOR_MIN_PREDICTIONS_PER_SHARD=${DRIVOR_MIN_PREDICTIONS_PER_SHARD:-1}
STOP_DRIVOR_AFTER_READY=${STOP_DRIVOR_AFTER_READY:-false}
RUN_ID=${RUN_ID:-sg_fps_v3_support_ddv2_$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_ID}}
SUPPORT_ARCHIVE=${SUPPORT_ARCHIVE:-${OUT_ROOT}/support_archive}
FS_STATS=${FS_STATS:-${OUT_ROOT}/fs_norm_stats.pt}
STAGE2_OUT=${STAGE2_OUT:-${OUT_ROOT}/stage2_dpsi_fs_norm}
AUTO_LAUNCH_STAGE3=${AUTO_LAUNCH_STAGE3:-false}
STAGE3_OUT=${STAGE3_OUT:-${OUT_ROOT}/stage3_feasible_pareto_grpo}
SHARD_COUNT=${SHARD_COUNT:-8}
POLL_SECONDS=${POLL_SECONDS:-60}
DDV2_SOURCE_NAME=${DDV2_SOURCE_NAME:-diffusiondrivev2}
DRIVOR_SOURCE_NAME=${DRIVOR_SOURCE_NAME:-drivor}
if [[ -z "${EXTERNAL_CANDIDATE_ROOTS+x}" ]]; then
  EXTERNAL_CANDIDATE_ROOTS="${DDV2_SOURCE_NAME}=${DDV2_OUT_ROOT}/submissions"
  EXTERNAL_CANDIDATE_ROOTS_WAS_DEFAULT=true
else
  EXTERNAL_CANDIDATE_ROOTS_WAS_DEFAULT=false
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids" "${SUPPORT_ARCHIVE}"

{
  echo "run_id=${RUN_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "ddv2_out_root=${DDV2_OUT_ROOT}"
  echo "ddv2_wait_mode=${DDV2_WAIT_MODE}"
  echo "ddv2_min_ready_shards=${DDV2_MIN_READY_SHARDS}"
  echo "ddv2_min_predictions_per_shard=${DDV2_MIN_PREDICTIONS_PER_SHARD}"
  echo "stop_ddv2_after_ready=${STOP_DDV2_AFTER_READY}"
  echo "run_drivor_after_ddv2=${RUN_DRIVOR_AFTER_DDV2}"
  echo "drivor_out_root=${DRIVOR_OUT_ROOT}"
  echo "drivor_checkpoint=${DRIVOR_CHECKPOINT}"
  echo "drivor_wait_mode=${DRIVOR_WAIT_MODE}"
  echo "drivor_min_ready_shards=${DRIVOR_MIN_READY_SHARDS}"
  echo "drivor_min_predictions_per_shard=${DRIVOR_MIN_PREDICTIONS_PER_SHARD}"
  echo "stop_drivor_after_ready=${STOP_DRIVOR_AFTER_READY}"
  echo "support_archive=${SUPPORT_ARCHIVE}"
  echo "fs_stats=${FS_STATS}"
  echo "stage2_out=${STAGE2_OUT}"
  echo "auto_launch_stage3=${AUTO_LAUNCH_STAGE3}"
  echo "stage3_out=${STAGE3_OUT}"
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

_stop_pid_dir() {
  local dir="$1"
  local label="$2"
  shopt -s nullglob
  local pids=()
  for f in "${dir}"/*.pid; do
    local pid
    pid=$(cat "${f}" 2>/dev/null || true)
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      pids+=("${pid}")
    fi
  done
  shopt -u nullglob
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "${label}_stop_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) live=0"
    return 0
  fi
  echo "${label}_stop_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) pids=${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 5
  local survivors=()
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      survivors+=("${pid}")
    fi
  done
  if [[ "${#survivors[@]}" -gt 0 ]]; then
    echo "${label}_stop_force_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) pids=${survivors[*]}"
    kill -9 "${survivors[@]}" 2>/dev/null || true
  fi
}

_candidate_ready() {
  local label="$1"
  local root="$2"
  local expected="$3"
  local mode="$4"
  local min_ready_shards="$5"
  local min_predictions_per_shard="$6"
  "${PYTHON_BIN}" - "${label}" "${root}" "${expected}" "${mode}" "${min_ready_shards}" "${min_predictions_per_shard}" <<'PY'
import json
import sys
from pathlib import Path

label = sys.argv[1]
root = Path(sys.argv[2])
expected = int(sys.argv[3])
mode = sys.argv[4]
min_ready = int(sys.argv[5])
min_predictions = int(sys.argv[6])

if mode not in {"final", "partial"}:
    raise SystemExit(f"{label}: wait mode must be final or partial, got {mode!r}")

ready = 0
prediction_total = 0
bad = []
for idx in range(expected):
    shard = root / "submissions" / f"shard_{idx}"
    if mode == "final":
        summary_path = shard / "summary.json"
        submission_path = shard / "submission.pkl"
    else:
        final_summary = shard / "summary.json"
        final_submission = shard / "submission.pkl"
        partial_summary = shard / "summary.partial.json"
        partial_submission = shard / "submission.partial.pkl"
        if final_summary.exists() and final_submission.exists():
            summary_path = final_summary
            submission_path = final_submission
        else:
            summary_path = partial_summary
            submission_path = partial_submission
    if not summary_path.exists() or not submission_path.exists():
        continue
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        bad.append(f"bad summary {summary_path}: {exc}")
        continue
    if int(summary.get("failure_count", 0)) != 0:
        bad.append(f"nonzero failures in {summary_path}: {summary.get('failure_count')}")
        continue
    count = int(summary.get("prediction_count", 0))
    prediction_total += count
    if count >= min_predictions:
        ready += 1

print(
    f"{label}_candidate_status_utc="
    f"ready_shards={ready}/{expected} prediction_total={prediction_total} "
    f"mode={mode} min_ready_shards={min_ready} min_predictions_per_shard={min_predictions} "
    f"bad_count={len(bad)}"
)
if bad:
    for item in bad[:8]:
        print(f"{label}_candidate_warning={item}")
if ready >= min_ready:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "waiting_for_ddv2_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
while true; do
  live=$(_live_pid_count "${DDV2_OUT_ROOT}/pids")
  summaries=$(find "${DDV2_OUT_ROOT}/submissions" -name summary.json 2>/dev/null | wc -l)
  submissions=$(find "${DDV2_OUT_ROOT}/submissions" -name submission.pkl 2>/dev/null | wc -l)
  partial_summaries=$(find "${DDV2_OUT_ROOT}/submissions" -name summary.partial.json 2>/dev/null | wc -l)
  partial_submissions=$(find "${DDV2_OUT_ROOT}/submissions" -name submission.partial.pkl 2>/dev/null | wc -l)
  echo "ddv2_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) live=${live} summaries=${summaries} submissions=${submissions} partial_summaries=${partial_summaries} partial_submissions=${partial_submissions}"
  if _candidate_ready "ddv2" "${DDV2_OUT_ROOT}" "${SHARD_COUNT}" "${DDV2_WAIT_MODE}" "${DDV2_MIN_READY_SHARDS}" "${DDV2_MIN_PREDICTIONS_PER_SHARD}"; then
    break
  fi
  if [[ "${live}" -eq 0 ]]; then
    echo "ddv2_no_live_processes_before_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    _candidate_ready "ddv2" "${DDV2_OUT_ROOT}" "${SHARD_COUNT}" "${DDV2_WAIT_MODE}" "${DDV2_MIN_READY_SHARDS}" "${DDV2_MIN_PREDICTIONS_PER_SHARD}"
  fi
  sleep "${POLL_SECONDS}"
done

_candidate_ready "ddv2" "${DDV2_OUT_ROOT}" "${SHARD_COUNT}" "${DDV2_WAIT_MODE}" "${DDV2_MIN_READY_SHARDS}" "${DDV2_MIN_PREDICTIONS_PER_SHARD}"

echo "ddv2_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "${STOP_DDV2_AFTER_READY}" == "true" ]]; then
  _stop_pid_dir "${DDV2_OUT_ROOT}/pids" "ddv2"
fi

if [[ "${RUN_DRIVOR_AFTER_DDV2}" == "true" ]]; then
  if [[ -z "${DRIVOR_OUT_ROOT}" ]]; then
    DRIVOR_OUT_ROOT="${REPO_ROOT}/outputs/drivor_nav1_navtrain_batched_$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  echo "launching_drivor_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) out=${DRIVOR_OUT_ROOT}"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    REPO_ROOT="${REPO_ROOT}" \
    DRIVOR_ROOT="${DRIVOR_ROOT}" \
    CHECKPOINT="${DRIVOR_CHECKPOINT}" \
    TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}" \
    DATA_ROOT="${DATA_ROOT:-/mnt/project/navsim_compat}" \
    MAPS_ROOT="${MAPS_ROOT:-/mnt/navsim/maps}" \
    OUT_ROOT="${DRIVOR_OUT_ROOT}" \
    NUM_SHARDS="${DRIVOR_SHARD_COUNT}" \
    BATCH_SIZE="${DRIVOR_BATCH_SIZE}" \
    PARTIAL_EVERY="${DRIVOR_PARTIAL_EVERY}" \
    RESUME="${DRIVOR_RESUME:-true}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_generate_drivor_navtrain_candidates.sh"
fi

if [[ -n "${DRIVOR_OUT_ROOT}" ]]; then
  echo "waiting_for_drivor_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while true; do
    live=$(_live_pid_count "${DRIVOR_OUT_ROOT}/pids")
    summaries=$(find "${DRIVOR_OUT_ROOT}/submissions" -name summary.json 2>/dev/null | wc -l)
    submissions=$(find "${DRIVOR_OUT_ROOT}/submissions" -name submission.pkl 2>/dev/null | wc -l)
    partial_summaries=$(find "${DRIVOR_OUT_ROOT}/submissions" -name summary.partial.json 2>/dev/null | wc -l)
    partial_submissions=$(find "${DRIVOR_OUT_ROOT}/submissions" -name submission.partial.pkl 2>/dev/null | wc -l)
    echo "drivor_status_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) live=${live} summaries=${summaries} submissions=${submissions} partial_summaries=${partial_summaries} partial_submissions=${partial_submissions}"
    if _candidate_ready "drivor" "${DRIVOR_OUT_ROOT}" "${DRIVOR_SHARD_COUNT}" "${DRIVOR_WAIT_MODE}" "${DRIVOR_MIN_READY_SHARDS}" "${DRIVOR_MIN_PREDICTIONS_PER_SHARD}"; then
      break
    fi
    if [[ "${live}" -eq 0 ]]; then
      echo "drivor_no_live_processes_before_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      _candidate_ready "drivor" "${DRIVOR_OUT_ROOT}" "${DRIVOR_SHARD_COUNT}" "${DRIVOR_WAIT_MODE}" "${DRIVOR_MIN_READY_SHARDS}" "${DRIVOR_MIN_PREDICTIONS_PER_SHARD}"
    fi
    sleep "${POLL_SECONDS}"
  done

  _candidate_ready "drivor" "${DRIVOR_OUT_ROOT}" "${DRIVOR_SHARD_COUNT}" "${DRIVOR_WAIT_MODE}" "${DRIVOR_MIN_READY_SHARDS}" "${DRIVOR_MIN_PREDICTIONS_PER_SHARD}"
  echo "drivor_ready_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${STOP_DRIVOR_AFTER_READY}" == "true" ]]; then
    _stop_pid_dir "${DRIVOR_OUT_ROOT}/pids" "drivor"
  fi
  if [[ "${EXTERNAL_CANDIDATE_ROOTS_WAS_DEFAULT}" == "true" ]]; then
    EXTERNAL_CANDIDATE_ROOTS="${DDV2_SOURCE_NAME}=${DDV2_OUT_ROOT}/submissions,${DRIVOR_SOURCE_NAME}=${DRIVOR_OUT_ROOT}/submissions"
    echo "external_candidate_roots=${EXTERNAL_CANDIDATE_ROOTS}" >> "${OUT_ROOT}/commands.log"
  fi
fi

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

if [[ "${AUTO_LAUNCH_STAGE3}" == "true" ]]; then
  echo "stage3_watcher_launch_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) out=${STAGE3_OUT}"
  setsid env \
    REPO_ROOT="${REPO_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    STAGE2_OUT="${STAGE2_OUT}" \
    STAGE2_PID_FILE="${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid" \
    SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE}" \
    STAGE3_OUT="${STAGE3_OUT}" \
    USE_FS_NORM="${USE_FS_NORM:-true}" \
    FS_NORM_STATS_PATH="${FS_STATS}" \
    FS_NORM_USE_ROBUST="${FS_NORM_USE_ROBUST:-true}" \
    POLL_SECONDS="${STAGE3_WATCH_POLL_SECONDS:-300}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/watch_stage2_then_train_sg_fps_grpo.sh" \
    > "${OUT_ROOT}/logs/stage3_watcher.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/pids/stage3_watcher.pid"
  echo "stage3_watcher_pid=$(cat "${OUT_ROOT}/pids/stage3_watcher.pid")"
fi
