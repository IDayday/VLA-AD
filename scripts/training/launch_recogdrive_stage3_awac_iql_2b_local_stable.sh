#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_awac_iql_2b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
STABLE_LAUNCHER="${STABLE_LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
CACHE_MODE="${CACHE_MODE:-offline}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-1}"
DRY_RUN="${DRY_RUN:-1}"

mkdir -p "${OUT_ROOT}"

JOB_CMD="cd $(printf '%q' "${REPO_ROOT}") && "
JOB_CMD+="OUT_ROOT=$(printf '%q' "${OUT_ROOT}/train") "
JOB_CMD+="CACHE_MODE=$(printf '%q' "${CACHE_MODE}") "
JOB_CMD+="KILL_GPU_STRESS=$(printf '%q' "${KILL_GPU_STRESS}") "
JOB_CMD+="DRY_RUN=$(printf '%q' "${DRY_RUN}") "
for name in \
  METRIC_CACHE_DIR HIDDEN_CACHE_DIR ELITE_BUFFER_DIR IL_CHECKPOINT VLM_PATH \
  MAX_SCENES MAX_EPOCHS LR BATCH_SIZE ACCUMULATE_GRAD_BATCHES \
  SCHEDULER_WARMUP_EPOCHS SCHEDULER_MIN_LR ONLINE_AWAC_CANDIDATES \
  AWAC_ELITE_TOP_M AWAC_BC_LOSS_WEIGHT AWAC_GRPO_LOSS_WEIGHT \
  AWAC_BASELINE_MODE AWAC_ADVANTAGE_TEMPERATURE AWAC_WEIGHT_MAX \
  AWAC_REQUIRE_NC AWAC_REQUIRE_DAC AWAC_REQUIRE_DDC_GUARD \
  AWAC_DDC_MIN_ABSOLUTE AWAC_ONLINE_POLICY_SAMPLES \
  AWAC_STRICT_REWARD_SUBMETRICS AWAC_MISSING_SUBMETRIC_POLICY \
  AWAC_REQUIRE_BUFFER_VALID_MASK AWAC_ALLOW_V1_BUFFER_RECOMPUTE_VALID_MASK \
  AWAC_SELECT_VALID_TOPK_ONLY AWAC_TRAIN_INVALID_FALLBACK_CANDIDATES \
  AWAC_FALLBACK_INVALID_CANDIDATE_WEIGHT AWAC_USE_FINAL_HEADING_GUARD \
  VALIDATE_ELITE_BUFFER; do
  if [[ -n "${!name:-}" ]]; then
    JOB_CMD+="${name}=$(printf '%q' "${!name}") "
  fi
done
JOB_CMD+="PYTHON_BIN=$(printf '%q' "${PYTHON_BIN}") "
JOB_CMD+="bash $(printf '%q' "${REPO_ROOT}/scripts/training/run_recogdrive_stage3_awac_iql_2b_local.sh")"

{
  printf 'name\tgpu\tcommand\n'
  printf 'stage3_awac_iql_2b\t%s\t%s\n' "${GPU_LIST}" "${JOB_CMD}"
} > "${OUT_ROOT}/jobs.tsv"

if [[ "${RUN_STAGE3:-0}" != "1" && "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "Stage3 AWAC/IQL stable launch is dry-run only. jobs.tsv written to ${OUT_ROOT}/jobs.tsv"
  echo "Set RUN_STAGE3=1 or RUN_TRAIN=1 to launch the stable GPU job."
  exit 0
fi

setsid "${PYTHON_BIN}" "${STABLE_LAUNCHER}" \
  --jobs-tsv "${OUT_ROOT}/jobs.tsv" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds 15 \
  > "${OUT_ROOT}/launcher.log" 2>&1 < /dev/null &

echo "$!" > "${OUT_ROOT}/launcher.pid"
echo "launched ${OUT_ROOT}"
