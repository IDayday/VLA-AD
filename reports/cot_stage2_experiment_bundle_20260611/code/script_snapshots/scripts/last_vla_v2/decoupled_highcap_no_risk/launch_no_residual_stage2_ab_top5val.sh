#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-/mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_${RUN_ID}}"
REMOTE_HOST="${REMOTE_HOST:-training-rl-zt2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LOCAL_MASTER_PORT="${LOCAL_MASTER_PORT:-29681}"
REMOTE_MASTER_PORT="${REMOTE_MASTER_PORT:-29682}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"

A_CACHE="${A_CACHE:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks}"
A_COT="${A_COT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/A/serverA_frozen_vlm_decoupled_highcap_no_risk/cot_alignment/latest.ckpt}"
B_CACHE="${B_CACHE:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/lora_regenerated_train_hidden_cache}"
B_COT="${B_COT:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/last_vla_cot_adapter.pt}"
B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH:-/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B}"
B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR:-/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_ab_cacheenv_setsid_20260607T022706Z/B/serverB_vlm_lora_decoupled_highcap_no_risk/vlm_lora_cot_alignment/adapters/vlm_lora}"
NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"

export NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/mnt/project/VLA-AD}"

A_OUT="${OUT_ROOT}/A_frozen_vlm_stage2_progressive"
B_OUT="${OUT_ROOT}/B_lora_stage2_progressive"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${A_OUT}" "${B_OUT}" "${LOG_DIR}"
COMMANDS_LOG="${OUT_ROOT}/commands.log"

common_overrides=(
  agent.last_vla_condition_mode=decoupled_cot_residual
  agent.last_vla_cot_bottleneck_mode=false
  agent.last_vla_raw_vlm_context_to_dit=true
  agent.last_vla_use_residual_diffusion=false
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

cmd_a=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port "${LOCAL_MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_decoupled_progressive_highcap_no_risk
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

cmd_b=(
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --master_port "${REMOTE_MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=last_vla_decoupled_progressive_highcap_no_risk
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

write_ckpt_list "${A_OUT}"
write_ckpt_list "${B_OUT}"

{
  echo "run_id=${RUN_ID}"
  echo "project_root=${PROJECT_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "remote_host=${REMOTE_HOST}"
  echo "git_head=$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
  echo "residual_diffusion=false"
  echo "validation=enabled_for_lightning_top5"
  echo "A_CACHE=${A_CACHE}"
  echo "A_COT=${A_COT}"
  echo "B_CACHE=${B_CACHE}"
  echo "B_COT=${B_COT}"
  printf 'A_CMD='; printf '%q ' "${cmd_a[@]}"; printf '\n'
  printf 'B_CMD='; printf '%q ' "${cmd_b[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${RUN_TRAIN:-1}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi

[[ -d "${A_CACHE}" ]] || { echo "missing A_CACHE: ${A_CACHE}" >&2; exit 2; }
[[ -f "${A_COT}" ]] || { echo "missing A_COT: ${A_COT}" >&2; exit 2; }
[[ -d "${B_CACHE}" ]] || { echo "missing B_CACHE: ${B_CACHE}" >&2; exit 2; }
[[ -f "${B_COT}" ]] || { echo "missing B_COT: ${B_COT}" >&2; exit 2; }

(
  cd "${PROJECT_ROOT}"
  exec setsid env PYTHONUNBUFFERED=1 "${cmd_a[@]}"
) >"${LOG_DIR}/A_stage2_local.log" 2>&1 < /dev/null &
echo "$!" >"${OUT_ROOT}/A_stage2_local.pid"

remote_cmd=$(
  printf 'cd %q && exec setsid env PYTHONUNBUFFERED=1 ' "${PROJECT_ROOT}"
  printf '%q ' "${cmd_b[@]}"
  printf ' > %q 2>&1 < /dev/null & echo $! > %q' "${LOG_DIR}/B_stage2_remote.log" "${OUT_ROOT}/B_stage2_remote.pid"
)
ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" "${remote_cmd}"

if [[ "${RUN_TOP5_EVAL_WATCHER:-1}" == "1" || "${RUN_TOP5_EVAL_WATCHER:-1}" == "true" ]]; then
  (
    cd "${PROJECT_ROOT}"
    exec setsid env \
      OUT_ROOT="${OUT_ROOT}" \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      REMOTE_HOST="${REMOTE_HOST}" \
      NAVTEST_CHUNK_CACHE_ROOT="${NAVTEST_CHUNK_CACHE_ROOT}" \
      METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      B_ONLINE_VLM_PATH="${B_ONLINE_VLM_PATH}" \
      B_ONLINE_VLM_LORA_ADAPTER_DIR="${B_ONLINE_VLM_LORA_ADAPTER_DIR}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      bash scripts/last_vla_v2/decoupled_highcap_no_risk/watch_ab_top5_eval_after_training.sh
  ) >"${LOG_DIR}/ab_top5_eval_watcher.outer.log" 2>&1 < /dev/null &
  echo "$!" >"${OUT_ROOT}/ab_top5_eval_watcher.pid"
fi

ln -sfn "${OUT_ROOT}" /mnt/project/VLA-AD/outputs/last_vla_v2/no_residual_stage2_ab_top5val_latest

echo "Launched A/B no-residual stage2 with validation/top-5 enabled."
echo "OUT_ROOT=${OUT_ROOT}"
echo "Monitor: ${PYTHON_BIN} scripts/monitor_last_vla_stage2_ab_progress.py --out-root ${OUT_ROOT} --watch 30"
