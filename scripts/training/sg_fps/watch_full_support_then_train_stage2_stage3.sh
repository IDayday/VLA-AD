#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
FULL_SUPPORT_RUN_ROOT=${FULL_SUPPORT_RUN_ROOT:-$(cat "${REPO_ROOT}/outputs/latest_sg_fps_support_navtrain_v3_full_external_watcher.txt")}
SUPPORT_ARCHIVE_PATH=${SUPPORT_ARCHIVE_PATH:-${FULL_SUPPORT_RUN_ROOT}/support_v3}
RUN_ID=${RUN_ID:-sg_fps_stage2_stage3_after_full_external_$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_ID}}
FS_STATS=${FS_STATS:-${OUT_ROOT}/fs_norm_stats.pt}
STAGE2_OUT=${STAGE2_OUT:-${OUT_ROOT}/stage2_dpsi_fs_norm}
STAGE3_OUT=${STAGE3_OUT:-${OUT_ROOT}/stage3_feasible_pareto_grpo}
POLL_SECONDS=${POLL_SECONDS:-300}
MIN_SUPPORT_RECORDS=${MIN_SUPPORT_RECORDS:-80000}
MIN_EXTERNAL_SCENE_RATIO=${MIN_EXTERNAL_SCENE_RATIO:-0.80}
MIN_SELECTED_VALID_RATIO=${MIN_SELECTED_VALID_RATIO:-0.85}
AUTO_LAUNCH_STAGE3=${AUTO_LAUNCH_STAGE3:-true}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"
echo "${OUT_ROOT}" > "${REPO_ROOT}/outputs/latest_sg_fps_stage2_stage3_watcher.txt"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/watcher.log"
}

validate_audit() {
  local audit_json="$1"
  "${PYTHON_BIN}" - "$audit_json" "$MIN_SUPPORT_RECORDS" "$MIN_EXTERNAL_SCENE_RATIO" "$MIN_SELECTED_VALID_RATIO" <<'PY'
import json
import sys

path = sys.argv[1]
min_records = int(float(sys.argv[2]))
min_external = float(sys.argv[3])
min_selected_valid = float(sys.argv[4])
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
record_count = int(data.get("record_count", 0))
external_ratio = float(data.get("external_candidate_scene_ratio", 0.0))
selected_valid = data.get("selected_valid_ratio", {})
if isinstance(selected_valid, dict):
    selected_valid_mean = float(selected_valid.get("mean", 0.0))
else:
    selected_valid_mean = float(selected_valid)
failures = []
if record_count < min_records:
    failures.append(f"record_count {record_count} < {min_records}")
if external_ratio < min_external:
    failures.append(f"external_candidate_scene_ratio {external_ratio:.6f} < {min_external:.6f}")
if selected_valid_mean < min_selected_valid:
    failures.append(f"selected_valid_ratio.mean {selected_valid_mean:.6f} < {min_selected_valid:.6f}")
print(
    json.dumps(
        {
            "record_count": record_count,
            "external_candidate_scene_ratio": external_ratio,
            "selected_valid_ratio_mean": selected_valid_mean,
            "failures": failures,
        },
        sort_keys=True,
    )
)
if failures:
    raise SystemExit(2)
PY
}

wait_full_support() {
  local audit_json="${FULL_SUPPORT_RUN_ROOT}/sg_fps_support_audit_after_reselect.json"
  log "waiting for full-external support audit: ${audit_json}"
  while [[ ! -f "${audit_json}" ]]; do
    local records=0
    records=$(find "${SUPPORT_ARCHIVE_PATH}" -maxdepth 1 -type f -name '*.pkl.xz' 2>/dev/null | wc -l || true)
    log "full_support_wait records=${records}"
    sleep "${POLL_SECONDS}"
  done
  log "validating support audit"
  validate_audit "${audit_json}" | tee "${OUT_ROOT}/support_audit_gate.json"
  log "support audit gate passed"
}

build_fs_stats() {
  log "building FS-Norm stats: ${FS_STATS}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/tools/build_fs_norm_stats.py" \
    --support_archive_path "${SUPPORT_ARCHIVE_PATH}" \
    --output_path "${FS_STATS}" \
    --use_robust true \
    --clip "${FS_NORM_CLIP:-5.0}" \
    > "${OUT_ROOT}/logs/build_fs_norm_stats.log" 2>&1
  log "FS-Norm stats ready"
}

launch_stage2() {
  mkdir -p "${STAGE2_OUT}/logs"
  log "launching Stage2 DPSI: ${STAGE2_OUT}"
  {
    printf '[%s] ' "$(date -Is)"
    printf '%q ' env \
      SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}" \
      OUT_ROOT="${STAGE2_OUT}" \
      CACHE_MODE=offline \
      HIDDEN_CACHE_DIR=/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b \
      USE_FS_NORM=true \
      FS_NORM_STATS_PATH="${FS_STATS}" \
      FS_NORM_USE_ROBUST=true \
      X0_AUX_WEIGHT="${X0_AUX_WEIGHT:-0.1}" \
      GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.05}" \
      SG_FPS_USE_DPSI=true \
      VALIDATE_ELITE_BUFFER=false \
      RUN_STAGE3=1 \
      DRY_RUN=0 \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash "${REPO_ROOT}/scripts/training/sg_fps/run_train_dpsi.sh"
    printf '\n'
  } >> "${OUT_ROOT}/commands.log"
  setsid env \
    SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}" \
    OUT_ROOT="${STAGE2_OUT}" \
    CACHE_MODE=offline \
    HIDDEN_CACHE_DIR=/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b \
    USE_FS_NORM=true \
    FS_NORM_STATS_PATH="${FS_STATS}" \
    FS_NORM_USE_ROBUST=true \
    X0_AUX_WEIGHT="${X0_AUX_WEIGHT:-0.1}" \
    GEO_AUX_WEIGHT="${GEO_AUX_WEIGHT:-0.05}" \
    SG_FPS_USE_DPSI=true \
    VALIDATE_ELITE_BUFFER=false \
    RUN_STAGE3=1 \
    DRY_RUN=0 \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/run_train_dpsi.sh" \
    > "${STAGE2_OUT}/logs/stage2_dpsi_fs_norm.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid"
  log "stage2_pid=$(cat "${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid")"
}

launch_stage3_watcher() {
  if [[ "${AUTO_LAUNCH_STAGE3}" != "true" ]]; then
    log "AUTO_LAUNCH_STAGE3=false; not starting Stage3 watcher"
    return
  fi
  log "launching Stage3 watcher: ${STAGE3_OUT}"
  setsid env \
    REPO_ROOT="${REPO_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    STAGE2_OUT="${STAGE2_OUT}" \
    STAGE2_PID_FILE="${OUT_ROOT}/pids/stage2_dpsi_fs_norm.pid" \
    SUPPORT_ARCHIVE_PATH="${SUPPORT_ARCHIVE_PATH}" \
    STAGE3_OUT="${STAGE3_OUT}" \
    USE_FS_NORM=true \
    FS_NORM_STATS_PATH="${FS_STATS}" \
    FS_NORM_USE_ROBUST=true \
    POLL_SECONDS="${POLL_SECONDS}" \
    bash "${REPO_ROOT}/scripts/training/sg_fps/watch_stage2_then_train_sg_fps_grpo.sh" \
    > "${OUT_ROOT}/logs/stage3_watcher.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/pids/stage3_watcher.pid"
  log "stage3_watcher_pid=$(cat "${OUT_ROOT}/pids/stage3_watcher.pid")"
}

{
  echo "run_id=${RUN_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "full_support_run_root=${FULL_SUPPORT_RUN_ROOT}"
  echo "support_archive_path=${SUPPORT_ARCHIVE_PATH}"
  echo "fs_stats=${FS_STATS}"
  echo "stage2_out=${STAGE2_OUT}"
  echo "stage3_out=${STAGE3_OUT}"
  echo "min_support_records=${MIN_SUPPORT_RECORDS}"
  echo "min_external_scene_ratio=${MIN_EXTERNAL_SCENE_RATIO}"
  echo "min_selected_valid_ratio=${MIN_SELECTED_VALID_RATIO}"
  echo "auto_launch_stage3=${AUTO_LAUNCH_STAGE3}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT_ROOT}/commands.log"

wait_full_support
build_fs_stats
launch_stage2
launch_stage3_watcher
