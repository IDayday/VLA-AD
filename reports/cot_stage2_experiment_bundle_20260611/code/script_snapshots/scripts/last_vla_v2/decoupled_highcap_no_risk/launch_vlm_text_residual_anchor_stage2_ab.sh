#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_ab_${RUN_ID}}"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LOCAL_MASTER_PORT="${LOCAL_MASTER_PORT:-29721}"
REMOTE_MASTER_PORT="${REMOTE_MASTER_PORT:-29722}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

BASE_VLM_PATH="${BASE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
A_CACHE="${A_CACHE:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks}"
A_COT="${A_COT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/A/serverA_frozen_vlm_decoupled_highcap_no_risk/cot_alignment/latest.ckpt}"
B_CACHE="${B_CACHE:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/lora_regenerated_train_hidden_cache}"
B_COT="${B_COT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/last_vla_cot_adapter.pt}"
B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH:-${BASE_VLM_PATH}}"
B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora}"

TRAIN_NAVSIM_LOG_PATH="${TRAIN_NAVSIM_LOG_PATH:-/mnt/navsim/trainval_navsim_logs/trainval}"
TRAIN_SENSOR_BLOBS_PATH="${TRAIN_SENSOR_BLOBS_PATH:-/mnt/navsim/trainval_sensor_blobs/trainval}"
TRAIN_SCENE_FILTER_YAML="${TRAIN_SCENE_FILTER_YAML:-navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml}"
NAVTEST_NAVSIM_LOG_PATH="${NAVTEST_NAVSIM_LOG_PATH:-/mnt/navsim/test_navsim_logs/test}"
NAVTEST_SENSOR_BLOBS_PATH="${NAVTEST_SENSOR_BLOBS_PATH:-/mnt/navsim/test_sensor_blobs/test}"
NAVTEST_SCENE_FILTER_YAML="${NAVTEST_SCENE_FILTER_YAML:-navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"

ANCHOR_CACHE_BASE="${ANCHOR_CACHE_BASE:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk}"
TRAIN_ANCHOR_RAW_ROOT="${TRAIN_ANCHOR_RAW_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_train_${RUN_ID}_raw}"
TRAIN_ANCHOR_CACHE_ROOT="${TRAIN_ANCHOR_CACHE_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_train_${RUN_ID}}"
NAVTEST_ANCHOR_RAW_ROOT="${NAVTEST_ANCHOR_RAW_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_navtest_${RUN_ID}_raw}"
NAVTEST_ANCHOR_CACHE_ROOT="${NAVTEST_ANCHOR_CACHE_ROOT:-${ANCHOR_CACHE_BASE}/vlm_text_anchor_2bbase_navtest_${RUN_ID}}"

NUM_ANCHOR_SHARDS="${NUM_ANCHOR_SHARDS:-16}"
LOCAL_ANCHOR_SHARDS="${LOCAL_ANCHOR_SHARDS:-0,1,2,3,4,5,6,7}"
REMOTE_ANCHOR_SHARDS="${REMOTE_ANCHOR_SHARDS:-8,9,10,11,12,13,14,15}"
ANCHOR_PRECISION="${ANCHOR_PRECISION:-bf16}"
ANCHOR_CAM_TYPE="${ANCHOR_CAM_TYPE:-single}"
ANCHOR_PROMPT_TYPE="${ANCHOR_PROMPT_TYPE:-base}"
ANCHOR_MAX_NEW_TOKENS="${ANCHOR_MAX_NEW_TOKENS:-512}"
ANCHOR_SAVE_EVERY="${ANCHOR_SAVE_EVERY:-100}"
ANCHOR_PROGRESS_EVERY="${ANCHOR_PROGRESS_EVERY:-25}"
RUN_TRAIN_AFTER_ANCHOR="${RUN_TRAIN_AFTER_ANCHOR:-1}"
RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH="${RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH:-1}"
RUN_TOP5_EVAL_WATCHER="${RUN_TOP5_EVAL_WATCHER:-1}"

export NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"

A_OUT="${OUT_ROOT}/A_frozen_vlm_stage2_progressive"
B_OUT="${OUT_ROOT}/B_lora_stage2_progressive"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${A_OUT}" "${B_OUT}" "${LOG_DIR}"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

split_csv() {
  tr ',' ' ' <<<"$1"
}

shard_dir_name() {
  printf 'shard_%05d' "$1"
}

gpu_for_shard() {
  local shard="$1"
  echo $(( shard % 8 ))
}

write_ckpt_list() {
  local run_dir="$1"
  cat >"${run_dir}/checkpoints_to_eval.txt" <<EOF
${run_dir}/step_00050000.ckpt
${run_dir}/step_00060000.ckpt
${run_dir}/step_00080000.ckpt
${run_dir}/step_00100000.ckpt
${run_dir}/step_00120000.ckpt
${run_dir}/step_00140000.ckpt
${run_dir}/step_00160000.ckpt
${run_dir}/latest.ckpt
EOF
}

launch_anchor_shard_local() {
  local split_name="$1"
  local raw_root="$2"
  local shard="$3"
  local gpu
  gpu="$(gpu_for_shard "${shard}")"
  local shard_dir="${raw_root}/shards/$(shard_dir_name "${shard}")"
  mkdir -p "${shard_dir}"
  local log_file="${LOG_DIR}/${split_name}_anchor_shard_$(printf '%05d' "${shard}").log"
  local status_file="${shard_dir}/status.txt"
  local cmd=(
    "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
    --model-path "${BASE_VLM_PATH}"
    --navsim-log-path "${TRAIN_NAVSIM_LOG_PATH}"
    --sensor-blobs-path "${TRAIN_SENSOR_BLOBS_PATH}"
    --scene-filter-yaml "${TRAIN_SCENE_FILTER_YAML}"
    --base-chunk-root "${A_CACHE}"
    --output-dir "${shard_dir}"
    --num-shards "${NUM_ANCHOR_SHARDS}"
    --shard-index "${shard}"
    --precision "${ANCHOR_PRECISION}"
    --cam-type "${ANCHOR_CAM_TYPE}"
    --prompt-type "${ANCHOR_PROMPT_TYPE}"
    --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
    --save-every "${ANCHOR_SAVE_EVERY}"
    --progress-every "${ANCHOR_PROGRESS_EVERY}"
    --allow-tolerant-parse
  )
  if [[ "${split_name}" == "navtest" ]]; then
    cmd=(
      "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
      --model-path "${BASE_VLM_PATH}"
      --navsim-log-path "${NAVTEST_NAVSIM_LOG_PATH}"
      --sensor-blobs-path "${NAVTEST_SENSOR_BLOBS_PATH}"
      --scene-filter-yaml "${NAVTEST_SCENE_FILTER_YAML}"
      --base-chunk-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
      --output-dir "${shard_dir}"
      --num-shards "${NUM_ANCHOR_SHARDS}"
      --shard-index "${shard}"
      --precision "${ANCHOR_PRECISION}"
      --cam-type "${ANCHOR_CAM_TYPE}"
      --prompt-type "${ANCHOR_PROMPT_TYPE}"
      --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
      --save-every "${ANCHOR_SAVE_EVERY}"
      --progress-every "${ANCHOR_PROGRESS_EVERY}"
      --allow-tolerant-parse
    )
  fi
  {
    printf '[%s] local %s anchor shard=%s gpu=%s ' "$(date -Is)" "${split_name}" "${shard}" "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${COMMANDS_LOG}"
  (
    cd "${PROJECT_ROOT}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${cmd[@]}" >"${log_file}" 2>&1
    echo "$?" >"${status_file}"
  ) &
}

launch_anchor_shard_remote() {
  local split_name="$1"
  local raw_root="$2"
  local shard="$3"
  local gpu
  gpu="$(gpu_for_shard "${shard}")"
  local shard_dir="${raw_root}/shards/$(shard_dir_name "${shard}")"
  mkdir -p "${shard_dir}"
  local log_file="${LOG_DIR}/${split_name}_anchor_shard_$(printf '%05d' "${shard}").remote.log"
  local status_file="${shard_dir}/status.txt"
  local cmd=(
    "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
    --model-path "${BASE_VLM_PATH}"
    --navsim-log-path "${TRAIN_NAVSIM_LOG_PATH}"
    --sensor-blobs-path "${TRAIN_SENSOR_BLOBS_PATH}"
    --scene-filter-yaml "${TRAIN_SCENE_FILTER_YAML}"
    --base-chunk-root "${A_CACHE}"
    --output-dir "${shard_dir}"
    --num-shards "${NUM_ANCHOR_SHARDS}"
    --shard-index "${shard}"
    --precision "${ANCHOR_PRECISION}"
    --cam-type "${ANCHOR_CAM_TYPE}"
    --prompt-type "${ANCHOR_PROMPT_TYPE}"
    --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
    --save-every "${ANCHOR_SAVE_EVERY}"
    --progress-every "${ANCHOR_PROGRESS_EVERY}"
    --allow-tolerant-parse
  )
  if [[ "${split_name}" == "navtest" ]]; then
    cmd=(
      "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
      --model-path "${BASE_VLM_PATH}"
      --navsim-log-path "${NAVTEST_NAVSIM_LOG_PATH}"
      --sensor-blobs-path "${NAVTEST_SENSOR_BLOBS_PATH}"
      --scene-filter-yaml "${NAVTEST_SCENE_FILTER_YAML}"
      --base-chunk-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
      --output-dir "${shard_dir}"
      --num-shards "${NUM_ANCHOR_SHARDS}"
      --shard-index "${shard}"
      --precision "${ANCHOR_PRECISION}"
      --cam-type "${ANCHOR_CAM_TYPE}"
      --prompt-type "${ANCHOR_PROMPT_TYPE}"
      --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
      --save-every "${ANCHOR_SAVE_EVERY}"
      --progress-every "${ANCHOR_PROGRESS_EVERY}"
      --allow-tolerant-parse
    )
  fi
  {
    printf '[%s] remote %s anchor shard=%s gpu=%s ' "$(date -Is)" "${split_name}" "${shard}" "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >>"${COMMANDS_LOG}"
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && mkdir -p %q && setsid -f bash -lc %q > %q 2>&1 < /dev/null' \
      "${PROJECT_ROOT}" \
      "${shard_dir}" \
      "$(printf 'echo $$ > %q; set +e; CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${shard_dir}/launcher.pid" "${gpu}")$(printf '%q ' "${cmd[@]}")$(printf '; echo "$?" > %q' "${status_file}")" \
      "${log_file}"
  )
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "${remote_cmd}"
}

launch_anchor_shard_remote_legacy() {
  local split_name="$1"
  local raw_root="$2"
  local shard="$3"
  local gpu
  gpu="$(gpu_for_shard "${shard}")"
  local shard_dir="${raw_root}/shards/$(shard_dir_name "${shard}")"
  mkdir -p "${shard_dir}"
  local log_file="${LOG_DIR}/${split_name}_anchor_shard_$(printf '%05d' "${shard}").remote.log"
  local status_file="${shard_dir}/status.txt"
  local cmd=(
    "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
    --model-path "${BASE_VLM_PATH}"
    --navsim-log-path "${TRAIN_NAVSIM_LOG_PATH}"
    --sensor-blobs-path "${TRAIN_SENSOR_BLOBS_PATH}"
    --scene-filter-yaml "${TRAIN_SCENE_FILTER_YAML}"
    --base-chunk-root "${A_CACHE}"
    --output-dir "${shard_dir}"
    --num-shards "${NUM_ANCHOR_SHARDS}"
    --shard-index "${shard}"
    --precision "${ANCHOR_PRECISION}"
    --cam-type "${ANCHOR_CAM_TYPE}"
    --prompt-type "${ANCHOR_PROMPT_TYPE}"
    --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
    --save-every "${ANCHOR_SAVE_EVERY}"
    --progress-every "${ANCHOR_PROGRESS_EVERY}"
    --allow-tolerant-parse
  )
  if [[ "${split_name}" == "navtest" ]]; then
    cmd=(
      "${PYTHON_BIN}" scripts/build_vlm_text_traj_anchor_cache.py
      --model-path "${BASE_VLM_PATH}"
      --navsim-log-path "${NAVTEST_NAVSIM_LOG_PATH}"
      --sensor-blobs-path "${NAVTEST_SENSOR_BLOBS_PATH}"
      --scene-filter-yaml "${NAVTEST_SCENE_FILTER_YAML}"
      --base-chunk-root "${NAVTEST_CHUNK_CACHE_ROOT}"
      --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN}"
      --output-dir "${shard_dir}"
      --num-shards "${NUM_ANCHOR_SHARDS}"
      --shard-index "${shard}"
      --precision "${ANCHOR_PRECISION}"
      --cam-type "${ANCHOR_CAM_TYPE}"
      --prompt-type "${ANCHOR_PROMPT_TYPE}"
      --max-new-tokens "${ANCHOR_MAX_NEW_TOKENS}"
      --save-every "${ANCHOR_SAVE_EVERY}"
      --progress-every "${ANCHOR_PROGRESS_EVERY}"
      --allow-tolerant-parse
    )
  fi
  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && mkdir -p %q && (set +e; CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${PROJECT_ROOT}" "${shard_dir}" "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf ' > %q 2>&1; echo "$?" > %q) &' "${log_file}" "${status_file}"
  )
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "${remote_cmd}"
}

wait_anchor_split() {
  local split_name="$1"
  local raw_root="$2"
  local failures=0
  while true; do
    local done_count=0
    local total=0
    for shard in $(split_csv "${LOCAL_ANCHOR_SHARDS}") $(split_csv "${REMOTE_ANCHOR_SHARDS}"); do
      total=$((total + 1))
      [[ -f "${raw_root}/shards/$(shard_dir_name "${shard}")/status.txt" ]] && done_count=$((done_count + 1))
    done
    echo "[$(date -Is)] ${split_name} anchor shards complete: ${done_count}/${total}" | tee -a "${LOG_DIR}/${split_name}_anchor_wait.log"
    [[ "${done_count}" -eq "${total}" ]] && break
    sleep 60
  done
  for shard in $(split_csv "${LOCAL_ANCHOR_SHARDS}") $(split_csv "${REMOTE_ANCHOR_SHARDS}"); do
    local status
    status="$(cat "${raw_root}/shards/$(shard_dir_name "${shard}")/status.txt")"
    if [[ "${status}" != "0" ]]; then
      echo "${split_name} anchor shard ${shard} failed with status ${status}" >&2
      failures=$((failures + 1))
    fi
  done
  [[ "${failures}" -eq 0 ]] || exit 20
}

merge_anchor_split() {
  local split_name="$1"
  local raw_root="$2"
  local cache_root="$3"
  "${PYTHON_BIN}" scripts/merge_last_vla_overlay_shards.py \
    --sharded-root "${raw_root}" \
    --output-root "${cache_root}" \
    --expected-num-shards "${NUM_ANCHOR_SHARDS}" \
    --overlay-type vlm_text_anchor \
    --copy-mode hardlink \
    --strict | tee "${LOG_DIR}/${split_name}_anchor_merge.log"
  "${PYTHON_BIN}" - "${cache_root}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
total = 0
parse_fail = 0
with (root / "index.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        total += 1
        if rec.get("parse_ok") is False:
            parse_fail += 1
print(json.dumps({"anchor_cache_root": str(root), "total": total, "parse_fail": parse_fail}, sort_keys=True))
if total == 0 or parse_fail:
    raise SystemExit(21)
PY
}

build_anchor_split() {
  local split_name="$1"
  local raw_root="$2"
  local cache_root="$3"
  if [[ -f "${cache_root}/index.jsonl" && "${REUSE_EXISTING_ANCHOR_CACHE:-0}" == "1" ]]; then
    echo "Reusing existing ${split_name} anchor cache: ${cache_root}"
    return 0
  fi
  rm -f "${raw_root}"/shards/shard_*/status.txt 2>/dev/null || true
  mkdir -p "${raw_root}/shards"
  for shard in $(split_csv "${LOCAL_ANCHOR_SHARDS}"); do
    launch_anchor_shard_local "${split_name}" "${raw_root}" "${shard}"
  done
  for shard in $(split_csv "${REMOTE_ANCHOR_SHARDS}"); do
    launch_anchor_shard_remote "${split_name}" "${raw_root}" "${shard}"
  done
  wait_anchor_split "${split_name}" "${raw_root}"
  merge_anchor_split "${split_name}" "${raw_root}" "${cache_root}"
}

launch_training_pair() {
  write_ckpt_list "${A_OUT}"
  write_ckpt_list "${B_OUT}"
  local common_overrides=(
    agent.last_vla_condition_mode=decoupled_cot_residual
    agent.last_vla_cot_bottleneck_mode=false
    agent.last_vla_raw_vlm_context_to_dit=true
    agent.last_vla_use_residual_diffusion=true
    agent.last_vla_residual_anchor_source=vlm_text_traj
    agent.last_vla_require_residual_anchor=true
    "agent.last_vla_residual_anchor_cache_dir=${TRAIN_ANCHOR_CACHE_ROOT}"
    agent.last_vla_residual_alpha_start=1.0
    agent.last_vla_residual_alpha_end=1.0
    agent.last_vla_residual_alpha_warmup_epochs=0
    agent.last_vla_use_risk_head=false
    agent.num_jepa_tokens=128
    agent.use_vggt=false
    agent.num_dynamic_tokens=128
    agent.num_vggt_tokens=128
    agent.num_geometry_tokens=192
    agent.num_risk_tokens=0
    agent.last_vla_cot_num_tokens=192
    agent.last_vla_cot_num_steps=5
    agent.last_vla_require_full_geometry=true
    agent.last_vla_allow_patch_geometry_fallback=false
    agent.last_vla_geometry_teacher_dim=512
    agent.last_vla_geometry_grid_rows=12
    agent.last_vla_geometry_grid_cols=16
  )

  local cmd_a=(
    "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port "${LOCAL_MASTER_PORT}"
    navsim/planning/script/run_training_recogdrive.py
    +experiment=last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual
    "train_test_split=${TRAIN_TEST_SPLIT}"
    "cache_path=${A_CACHE}"
    use_cache_without_dataset=true
    force_cache_computation=false
    "output_dir=${A_OUT}"
    "agent.last_vla_adapter_checkpoint=${A_COT}"
    "${common_overrides[@]}"
    trainer.params.devices="${NPROC_PER_NODE}"
    trainer.params.strategy=ddp_find_unused_parameters_true
  )
  local cmd_b=(
    "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port "${REMOTE_MASTER_PORT}"
    navsim/planning/script/run_training_recogdrive.py
    +experiment=last_vla_decoupled_progressive_highcap_no_risk_vlm_text_residual
    "train_test_split=${TRAIN_TEST_SPLIT}"
    "cache_path=${B_CACHE}"
    use_cache_without_dataset=true
    force_cache_computation=false
    "output_dir=${B_OUT}"
    "agent.last_vla_adapter_checkpoint=${B_COT}"
    "${common_overrides[@]}"
    trainer.params.devices="${NPROC_PER_NODE}"
    trainer.params.strategy=ddp_find_unused_parameters_true
  )
  {
    echo "training_anchor_cache=${TRAIN_ANCHOR_CACHE_ROOT}"
    echo "residual_diffusion=true"
    echo "residual_anchor_source=2b_base_direct_text_pt"
    printf 'A_CMD='; printf '%q ' "${cmd_a[@]}"; printf '\n'
    printf 'B_CMD='; printf '%q ' "${cmd_b[@]}"; printf '\n'
  } >>"${COMMANDS_LOG}"

  [[ -d "${A_CACHE}" ]] || { echo "missing A_CACHE: ${A_CACHE}" >&2; exit 2; }
  [[ -f "${A_COT}" ]] || { echo "missing A_COT: ${A_COT}" >&2; exit 2; }
  [[ -d "${B_CACHE}" ]] || { echo "missing B_CACHE: ${B_CACHE}" >&2; exit 2; }
  [[ -f "${B_COT}" ]] || { echo "missing B_COT: ${B_COT}" >&2; exit 2; }
  [[ -f "${TRAIN_ANCHOR_CACHE_ROOT}/index.jsonl" ]] || { echo "missing train anchor cache: ${TRAIN_ANCHOR_CACHE_ROOT}" >&2; exit 2; }

  (
    cd "${PROJECT_ROOT}"
    exec setsid env PYTHONUNBUFFERED=1 "${cmd_a[@]}"
  ) >"${LOG_DIR}/A_stage2_local.log" 2>&1 < /dev/null &
  echo "$!" >"${OUT_ROOT}/A_stage2_local.pid"

  local remote_cmd
  remote_cmd=$(
    printf 'cd %q && exec setsid env PYTHONUNBUFFERED=1 ' "${PROJECT_ROOT}"
    printf '%q ' "${cmd_b[@]}"
    printf ' > %q 2>&1 < /dev/null & echo $! > %q' "${LOG_DIR}/B_stage2_remote.log" "${OUT_ROOT}/B_stage2_remote.pid"
  )
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "${remote_cmd}"
}

launch_eval_watcher() {
  if [[ "${RUN_TOP5_EVAL_WATCHER}" != "1" && "${RUN_TOP5_EVAL_WATCHER}" != "true" ]]; then
    return 0
  fi
  [[ -f "${NAVTEST_ANCHOR_CACHE_ROOT}/index.jsonl" ]] || {
    echo "Skipping eval watcher until navtest anchor cache exists: ${NAVTEST_ANCHOR_CACHE_ROOT}" >&2
    return 0
  }
  (
    cd "${PROJECT_ROOT}"
    exec setsid env \
      OUT_ROOT="${OUT_ROOT}" \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      REMOTE_HOST="${REMOTE_HOST}" \
      NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
      NAVTEST_CHUNK_NAME_PATTERN="${NAVTEST_CHUNK_NAME_PATTERN}" \
      METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      CONFIG="configs/last_vla_v2/decoupled_highcap_no_risk/progressive_sft_vlm_text_residual_eval_flat.yaml" \
      VLM_TEXT_ANCHOR_CACHE_ROOT="${NAVTEST_ANCHOR_CACHE_ROOT}" \
      B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH}" \
      B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/watch_ab_top5_eval_after_training.sh
  ) >"${LOG_DIR}/ab_top5_eval_watcher.outer.log" 2>&1 < /dev/null &
  echo "$!" >"${OUT_ROOT}/ab_top5_eval_watcher.pid"
}

{
  echo "run_id=${RUN_ID}"
  echo "project_root=${PROJECT_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "remote_host=${REMOTE_HOST}"
  echo "git_head=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
  echo "base_vlm_path=${BASE_VLM_PATH}"
  echo "A_CACHE=${A_CACHE}"
  echo "B_CACHE=${B_CACHE}"
  echo "train_anchor_raw_root=${TRAIN_ANCHOR_RAW_ROOT}"
  echo "train_anchor_cache_root=${TRAIN_ANCHOR_CACHE_ROOT}"
  echo "navtest_anchor_raw_root=${NAVTEST_ANCHOR_RAW_ROOT}"
  echo "navtest_anchor_cache_root=${NAVTEST_ANCHOR_CACHE_ROOT}"
} >>"${COMMANDS_LOG}"

build_anchor_split train "${TRAIN_ANCHOR_RAW_ROOT}" "${TRAIN_ANCHOR_CACHE_ROOT}"

if [[ "${RUN_TRAIN_AFTER_ANCHOR}" == "1" || "${RUN_TRAIN_AFTER_ANCHOR}" == "true" ]]; then
  launch_training_pair
  ln -sfn "${OUT_ROOT}" /mnt/project/VLA-AD/outputs/last_vla_v2/vlm_text_residual_anchor_stage2_ab_latest
fi

if [[ "${RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH}" == "1" || "${RUN_NAVTEST_ANCHOR_AFTER_TRAIN_LAUNCH}" == "true" ]]; then
  build_anchor_split navtest "${NAVTEST_ANCHOR_RAW_ROOT}" "${NAVTEST_ANCHOR_CACHE_ROOT}"
  launch_eval_watcher
fi

echo "Launched residual-anchor A/B stage2 pipeline."
echo "OUT_ROOT=${OUT_ROOT}"
echo "TRAIN_ANCHOR_CACHE_ROOT=${TRAIN_ANCHOR_CACHE_ROOT}"
echo "NAVTEST_ANCHOR_CACHE_ROOT=${NAVTEST_ANCHOR_CACHE_ROOT}"
