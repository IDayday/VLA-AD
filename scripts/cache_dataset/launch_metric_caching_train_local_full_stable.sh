#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-metric_cache_navtrain_full_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}"
STABLE_LAUNCHER="${STABLE_LAUNCHER:-/mnt/project/skill/stable-gpu-job-launch/scripts/stable_gpu_launcher.py}"

mkdir -p "${OUT_ROOT}"

JOB_CMD="cd $(printf '%q' "${REPO_ROOT}") && "
JOB_CMD+="RUN_ROOT=$(printf '%q' "${OUT_ROOT}/metric_cache_job") "
JOB_CMD+="ARTIFACT_ROOT=$(printf '%q' "${ARTIFACT_ROOT}") "
JOB_CMD+="METRIC_CACHE_DIR=$(printf '%q' "${METRIC_CACHE_DIR}") "
if [[ -n "${NAVSIM_DATA_ROOT:-}" ]]; then
  JOB_CMD+="NAVSIM_DATA_ROOT=$(printf '%q' "${NAVSIM_DATA_ROOT}") "
fi
if [[ -n "${MAX_SCENES:-}" ]]; then
  JOB_CMD+="MAX_SCENES=$(printf '%q' "${MAX_SCENES}") "
fi
JOB_CMD+="PYTHON_BIN=$(printf '%q' "${PYTHON_BIN}") "
JOB_CMD+="bash $(printf '%q' "${REPO_ROOT}/scripts/cache_dataset/run_metric_caching_train_local_full.sh")"

{
  printf 'name\tgpu\tcommand\n'
  printf 'metric_cache_navtrain\t-\t%s\n' "${JOB_CMD}"
} > "${OUT_ROOT}/jobs.tsv"

setsid "${PYTHON_BIN}" "${STABLE_LAUNCHER}" \
  --jobs-tsv "${OUT_ROOT}/jobs.tsv" \
  --out-root "${OUT_ROOT}" \
  --stagger-seconds 8 \
  > "${OUT_ROOT}/launcher.log" 2>&1 < /dev/null &

echo "$!" > "${OUT_ROOT}/launcher.pid"
echo "launched ${OUT_ROOT}"
