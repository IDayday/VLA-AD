#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
EXTERNAL_RUN_ROOT=${EXTERNAL_RUN_ROOT:-$(cat "${REPO_ROOT}/outputs/latest_sg_fps_external_navtrain_candidates.txt")}
DDV2_OUT_ROOT=${DDV2_OUT_ROOT:-${EXTERNAL_RUN_ROOT}/ddv2_candidates}
DRIVOR_OUT_ROOT=${DRIVOR_OUT_ROOT:-${EXTERNAL_RUN_ROOT}/drivor_candidates}
RUN_ID=${RUN_ID:-sg_fps_support_navtrain_v3_full_external_$(date -u +%Y%m%dT%H%M%SZ)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_ID}}
SUPPORT_ARCHIVE=${SUPPORT_ARCHIVE:-${OUT_ROOT}/support_v3}
NUM_SHARDS=${NUM_SHARDS:-8}
POLL_SECONDS=${POLL_SECONDS:-300}
CACHE_PATH=${CACHE_PATH:-/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b}
METRIC_CACHE_PATH=${METRIC_CACHE_PATH:-/mnt/project/VLA-AD/cache/metric_cache_train_full}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids"
echo "${OUT_ROOT}" > "${REPO_ROOT}/outputs/latest_sg_fps_support_navtrain_v3_full_external_pending.txt"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${OUT_ROOT}/logs/watcher.log"
}

live_pid_count() {
  local pid_dir="$1"
  local count=0
  local pid_file pid
  shopt -s nullglob
  for pid_file in "${pid_dir}"/*.pid; do
    pid=$(cat "${pid_file}" 2>/dev/null || true)
    if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
      count=$((count + 1))
    fi
  done
  shopt -u nullglob
  echo "${count}"
}

count_files() {
  local root="$1"
  local pattern="$2"
  if [[ ! -d "${root}" ]]; then
    echo 0
    return
  fi
  find "${root}" -name "${pattern}" 2>/dev/null | wc -l
}

wait_external_complete() {
  log "waiting for full external candidates"
  while true; do
    local ddv2_summaries drivor_summaries ddv2_live drivor_live
    ddv2_summaries=$(count_files "${DDV2_OUT_ROOT}/submissions" "summary.json")
    drivor_summaries=$(count_files "${DRIVOR_OUT_ROOT}/submissions" "summary.json")
    ddv2_live=$(live_pid_count "${DDV2_OUT_ROOT}/pids")
    drivor_live=$(live_pid_count "${DRIVOR_OUT_ROOT}/pids")
    log "external_status ddv2=${ddv2_summaries}/${NUM_SHARDS} live=${ddv2_live} driveor=${drivor_summaries}/${NUM_SHARDS} live=${drivor_live}"
    if [[ "${ddv2_summaries}" -eq "${NUM_SHARDS}" && "${drivor_summaries}" -eq "${NUM_SHARDS}" ]]; then
      break
    fi
    if [[ "${ddv2_live}" -eq 0 && "${drivor_live}" -eq 0 && ( "${ddv2_summaries}" -gt 0 || "${drivor_summaries}" -gt 0 ) ]]; then
      if [[ "${ddv2_summaries}" -ne "${NUM_SHARDS}" || "${drivor_summaries}" -ne "${NUM_SHARDS}" ]]; then
        log "external candidates stopped before full completion"
      fi
    fi
    sleep "${POLL_SECONDS}"
  done
}

launch_support_build() {
  local external_roots="ddv2=${DDV2_OUT_ROOT}/submissions,driveor=${DRIVOR_OUT_ROOT}/submissions"
  log "launching full-external support build: ${SUPPORT_ARCHIVE}"
  log "external_candidate_roots=${external_roots}"
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu shard_root
    gpu=$((shard % 8))
    shard_root="${OUT_ROOT}/shard_${shard}"
    mkdir -p "${shard_root}"
    {
      printf '[%s] CUDA_VISIBLE_DEVICES=%s SHARD_INDEX=%s SHARD_COUNT=%s OUT_ROOT=%q OUTPUT_PATH=%q EXTERNAL_CANDIDATE_ROOTS=%q ' \
        "$(date -Is)" "${gpu}" "${shard}" "${NUM_SHARDS}" "${shard_root}" "${SUPPORT_ARCHIVE}" "${external_roots}"
      printf 'bash scripts/training/sg_fps/run_build_sg_fps_support.sh\n'
    } >> "${OUT_ROOT}/commands.log"
    setsid env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      SHARD_INDEX="${shard}" \
      SHARD_COUNT="${NUM_SHARDS}" \
      BUILD_LOG_SPLIT=train \
      CACHE_PATH="${CACHE_PATH}" \
      METRIC_CACHE_PATH="${METRIC_CACHE_PATH}" \
      OUT_ROOT="${shard_root}" \
      OUTPUT_PATH="${SUPPORT_ARCHIVE}" \
      BATCH_SIZE=1 \
      MAX_SCENES=0 \
      ELITE_TOP_M=32 \
      SG_FPS_SUPPORT_TOP_M=12 \
      SG_FPS_EXPAND_EXTERNAL_CANDIDATES=true \
      SG_FPS_EXTERNAL_EXPANSION_MAX_PER_SCENE=32 \
      AWAC_PDM_SHADOW_CHECK=false \
      EXTERNAL_CANDIDATE_ROOTS="${external_roots}" \
      ONLINE_POLICY_SAMPLES=0 \
      ONLINE_USE_CURRENT_POLICY=false \
      ONLINE_USE_OLD_POLICY=true \
      ONLINE_USE_GT=true \
      PERTURB_GT=true \
      PERTURB_IL=true \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash "${REPO_ROOT}/scripts/training/sg_fps/run_build_sg_fps_support.sh" \
      > "${OUT_ROOT}/logs/build_shard_${shard}.log" 2>&1 < /dev/null &
    echo $! > "${OUT_ROOT}/pids/build_shard_${shard}.pid"
  done
}

wait_support_build() {
  while true; do
    local live records
    live=$(live_pid_count "${OUT_ROOT}/pids")
    records=$(find "${SUPPORT_ARCHIVE}" -maxdepth 1 -type f -name '*.pkl.xz' 2>/dev/null | wc -l)
    log "support_build_status live=${live} records=${records}"
    if [[ "${live}" -eq 0 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

run_postprocess() {
  log "running support reselect"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/tools/reselect_sg_fps_support_archive.py" \
    --support-archive-path "${SUPPORT_ARCHIVE}" \
    --output-json "${OUT_ROOT}/sg_fps_support_reselect.json" \
    --support-top-m 12 \
    --ddc-min-absolute 0.95 \
    --ddc-drop-tolerance 0.01 \
    --ddc-gate-mode "${SG_FPS_DDC_GATE_MODE:-ref_relative}" \
    --relax-ddc-when-ref-below-min \
    --feas-cost-max 0.10 \
    --feas-gate-mode "${SG_FPS_FEAS_GATE_MODE:-relax_ref_above_max}" \
    --feas-cost-tolerance "${SG_FPS_FEAS_COST_TOLERANCE:-0.03}" \
    --comfort-min 0.95 \
    --comfort-gate-mode "${SG_FPS_COMFORT_GATE_MODE:-ref_relative}" \
    --comfort-drop-tolerance "${SG_FPS_COMFORT_DROP_TOLERANCE:-0.05}" \
    --selection-strategy quality_pareto \
    --enable-train-quality-gate \
    --enable-semantic-gate \
    --min-non-gt-pdms "${SG_FPS_MIN_NON_GT_PDMS:-0.90}" \
    --reward-gate-mode "${SG_FPS_REWARD_GATE_MODE:-absolute_or_gt_improver}" \
    --min-non-gt-improver-pdms "${SG_FPS_MIN_NON_GT_IMPROVER_PDMS:-0.70}" \
    --gt-improver-margin "${SG_FPS_GT_IMPROVER_MARGIN:-0.05}" \
    --gt-improver-ref-max-pdms "${SG_FPS_GT_IMPROVER_REF_MAX_PDMS:-0.90}" \
    --max-first-xy-error-m "${SG_FPS_MAX_FIRST_XY_ERROR_M:-1.0}" \
    --max-first-heading-error-rad "${SG_FPS_MAX_FIRST_HEADING_ERROR_RAD:-0.8}" \
    --max-xy-turn-rad "${SG_FPS_MAX_XY_TURN_RAD:-1.2}" \
    --max-early-xy-turn-rad "${SG_FPS_MAX_EARLY_XY_TURN_RAD:-1.0}" \
    --max-step-m "${SG_FPS_MAX_STEP_M:-12.0}" \
    --gt-keep-min-reward "${SG_FPS_GT_KEEP_MIN_REWARD:-0.85}" \
    --gt-replace-margin "${SG_FPS_GT_REPLACE_MARGIN:-0.05}" \
    --no-include-il-anchor \
    > "${OUT_ROOT}/logs/reselect_support.log" 2>&1
  log "running support audit"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/tools/audit_sg_fps_support_archive.py" \
    --support-archive-path "${SUPPORT_ARCHIVE}" \
    --output-json "${OUT_ROOT}/sg_fps_support_audit_after_reselect.json" \
    --output-md "${OUT_ROOT}/sg_fps_support_audit_after_reselect.md" \
    --min-records 1 \
    --selected-min-reward "${SG_FPS_MIN_NON_GT_PDMS:-0.90}" \
    --reward-gate-mode "${SG_FPS_REWARD_GATE_MODE:-absolute_or_gt_improver}" \
    --min-non-gt-improver-pdms "${SG_FPS_MIN_NON_GT_IMPROVER_PDMS:-0.70}" \
    --gt-improver-margin "${SG_FPS_GT_IMPROVER_MARGIN:-0.05}" \
    --gt-improver-ref-max-pdms "${SG_FPS_GT_IMPROVER_REF_MAX_PDMS:-0.90}" \
    --min-best-valid-above-gt-ratio 0.0 \
    --min-selected-source-diversity-ge2-ratio 0.0 \
    --min-external-scene-ratio 0.80 \
    --max-fallback-tag-ratio 1.0 \
    > "${OUT_ROOT}/logs/audit_after_reselect.log" 2>&1 || true
  echo "${OUT_ROOT}" > "${REPO_ROOT}/outputs/latest_sg_fps_support_navtrain_v3_full_external.txt"
  log "full-external support build complete"
}

{
  echo "run_id=${RUN_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "support_archive=${SUPPORT_ARCHIVE}"
  echo "external_run_root=${EXTERNAL_RUN_ROOT}"
  echo "ddv2_out_root=${DDV2_OUT_ROOT}"
  echo "drivor_out_root=${DRIVOR_OUT_ROOT}"
  echo "cache_path=${CACHE_PATH}"
  echo "metric_cache_path=${METRIC_CACHE_PATH}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT_ROOT}/commands.log"

wait_external_complete
launch_support_build
wait_support_build
run_postprocess
