#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_rl_2b_safe_diffgrpo_online_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
STABLE_LAUNCHER="${STABLE_LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CACHE_MODE="${CACHE_MODE:-online}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-1}"

mkdir -p "${OUT_ROOT}"

JOB_CMD="cd $(printf '%q' "${REPO_ROOT}") && "
JOB_CMD+="OUT_ROOT=$(printf '%q' "${OUT_ROOT}/train") "
JOB_CMD+="CACHE_MODE=$(printf '%q' "${CACHE_MODE}") "
JOB_CMD+="KILL_GPU_STRESS=$(printf '%q' "${KILL_GPU_STRESS}") "
if [[ -n "${METRIC_CACHE_DIR:-}" ]]; then
  JOB_CMD+="METRIC_CACHE_DIR=$(printf '%q' "${METRIC_CACHE_DIR}") "
fi
if [[ -n "${HIDDEN_CACHE_DIR:-}" ]]; then
  JOB_CMD+="HIDDEN_CACHE_DIR=$(printf '%q' "${HIDDEN_CACHE_DIR}") "
fi
if [[ -n "${IL_CHECKPOINT:-}" ]]; then
  JOB_CMD+="IL_CHECKPOINT=$(printf '%q' "${IL_CHECKPOINT}") "
fi
if [[ -n "${VLM_PATH:-}" ]]; then
  JOB_CMD+="VLM_PATH=$(printf '%q' "${VLM_PATH}") "
fi
if [[ -n "${MAX_SCENES:-}" ]]; then
  JOB_CMD+="MAX_SCENES=$(printf '%q' "${MAX_SCENES}") "
fi
if [[ -n "${MAX_EPOCHS:-}" ]]; then
  JOB_CMD+="MAX_EPOCHS=$(printf '%q' "${MAX_EPOCHS}") "
fi
if [[ -n "${LR:-}" ]]; then
  JOB_CMD+="LR=$(printf '%q' "${LR}") "
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  JOB_CMD+="BATCH_SIZE=$(printf '%q' "${BATCH_SIZE}") "
fi
if [[ -n "${ACCUMULATE_GRAD_BATCHES:-}" ]]; then
  JOB_CMD+="ACCUMULATE_GRAD_BATCHES=$(printf '%q' "${ACCUMULATE_GRAD_BATCHES}") "
fi
if [[ -n "${GRPO_SAMPLE_TIME:-}" ]]; then
  JOB_CMD+="GRPO_SAMPLE_TIME=$(printf '%q' "${GRPO_SAMPLE_TIME}") "
fi
if [[ -n "${BC_ANNEAL:-}" ]]; then
  JOB_CMD+="BC_ANNEAL=$(printf '%q' "${BC_ANNEAL}") "
fi
if [[ -n "${BC_COEFF_START:-}" ]]; then
  JOB_CMD+="BC_COEFF_START=$(printf '%q' "${BC_COEFF_START}") "
fi
if [[ -n "${BC_COEFF_END:-}" ]]; then
  JOB_CMD+="BC_COEFF_END=$(printf '%q' "${BC_COEFF_END}") "
fi
if [[ -n "${BC_ANNEAL_EPOCHS:-}" ]]; then
  JOB_CMD+="BC_ANNEAL_EPOCHS=$(printf '%q' "${BC_ANNEAL_EPOCHS}") "
fi
if [[ -n "${GRPO_SCHEDULER_EPOCHS:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_EPOCHS=$(printf '%q' "${GRPO_SCHEDULER_EPOCHS}") "
fi
if [[ -n "${GRPO_SCHEDULER_WARMUP_EPOCHS:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_WARMUP_EPOCHS=$(printf '%q' "${GRPO_SCHEDULER_WARMUP_EPOCHS}") "
fi
if [[ -n "${GRPO_SCHEDULER_MIN_LR:-}" ]]; then
  JOB_CMD+="GRPO_SCHEDULER_MIN_LR=$(printf '%q' "${GRPO_SCHEDULER_MIN_LR}") "
fi
JOB_CMD+="PYTHON_BIN=$(printf '%q' "${PYTHON_BIN}") "
JOB_CMD+="bash $(printf '%q' "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_rl_2b_local.sh")"

{
  printf 'name\tgpu\tcommand\n'
  printf 'stage3_rl_2b\t%s\t%s\n' "${GPU_LIST}" "${JOB_CMD}"
} > "${OUT_ROOT}/jobs.tsv"

setsid "${PYTHON_BIN}" "${STABLE_LAUNCHER}" \
  --jobs-tsv "${OUT_ROOT}/jobs.tsv" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds 15 \
  > "${OUT_ROOT}/launcher.log" 2>&1 < /dev/null &

echo "$!" > "${OUT_ROOT}/launcher.pid"
echo "launched ${OUT_ROOT}"
