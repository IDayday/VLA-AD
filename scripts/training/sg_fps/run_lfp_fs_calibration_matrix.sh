#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MATRIX_OUT_ROOT="${MATRIX_OUT_ROOT:-${PROJECT_ROOT}/outputs/lfp_fs_calibration_$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILES_CSV="${LFP_FS_PROFILES:-baseline,cov_c2,cov_c4,gate_only}"
IFS=',' read -r -a profiles <<< "${PROFILES_CSV}"

mkdir -p "${MATRIX_OUT_ROOT}/logs"
printf '%s\n' "${PROFILES_CSV}" > "${MATRIX_OUT_ROOT}/profiles.csv"
git -C "${PROJECT_ROOT}" rev-parse HEAD > "${MATRIX_OUT_ROOT}/git_commit.txt"
git -C "${PROJECT_ROOT}" status --short > "${MATRIX_OUT_ROOT}/git_status.txt"
sha256sum \
  "${PROJECT_ROOT}/navsim/agents/recogdrive/stage3_policy_geometry.py" \
  "${PROJECT_ROOT}/navsim/agents/recogdrive/stage3_optimization.py" \
  "${PROJECT_ROOT}/navsim/agents/recogdrive/stage3_lfp_grpo.py" \
  "${PROJECT_ROOT}/navsim/agents/recogdrive/recogdrive_diffusion_planner.py" \
  > "${MATRIX_OUT_ROOT}/code_sha256.txt"

for profile in "${profiles[@]}"; do
  run_name="lfp_fs_calibration_${profile}"
  run_root="${MATRIX_OUT_ROOT}/${profile}"
  mkdir -p "${run_root}/logs"
  env \
    LFP_FS_PROFILE="${profile}" \
    RUN_NAME="${run_name}" \
    OUT_ROOT="${run_root}" \
    MAX_STEPS=1 \
    MAX_EPOCHS=1 \
    GRPO_SCHEDULER_WARMUP_EPOCHS=0 \
    LFP_CURRICULUM_ENABLED=false \
    CHECKPOINT_EVERY_N_TRAIN_STEPS=1 \
    LOG_EVERY_N_STEPS=1 \
    bash "${SCRIPT_DIR}/run_train_lfp_grpo_fs_probe.sh" \
    > "${run_root}/logs/train.log" 2>&1
  date -Is > "${run_root}/completed"
done

"${PYTHON_BIN:-python}" "${PROJECT_ROOT}/scripts/evaluation/summarize_lfp_fs_calibration.py" \
  --matrix-root "${MATRIX_OUT_ROOT}" \
  > "${MATRIX_OUT_ROOT}/logs/summary.log" 2>&1
date -Is > "${MATRIX_OUT_ROOT}/completed"
