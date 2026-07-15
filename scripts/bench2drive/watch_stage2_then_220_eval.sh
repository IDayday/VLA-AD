#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
STAGE2_OUT=${STAGE2_OUT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_closest_public_bench2drive_recogdrive_stage1_official_multiview_20260711T102501Z}
VLM_PATH=${VLM_PATH:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage1_official_multiview_20260711T102501Z}
EVAL_ROOT=${EVAL_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_closed_loop}
BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:-/mnt/data/Bench2Drive}
CARLA_ROOT=${CARLA_ROOT:-/mnt/project/CARLA_0.9.15_nonroot}
TASK_NUM=${TASK_NUM:-16}

echo "[$(date -u +%FT%TZ)] waiting for Stage2 training to finish"
while pgrep -f '[t]rain_recogdrive_expert_chunked.py' >/dev/null; do sleep 30; done

CKPT=${PLANNER_CHECKPOINT:-${STAGE2_OUT}/latest.ckpt}
if [[ ! -f "${CKPT}" ]]; then
  CKPT=$(find "${STAGE2_OUT}" -maxdepth 1 -type f -name 'step_*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
fi
test -n "${CKPT}" && test -f "${CKPT}"
test -x "${CARLA_ROOT}/CarlaUE4.sh"
echo "[$(date -u +%FT%TZ)] CARLA root-safe launcher verified: ${CARLA_ROOT}/CarlaUE4.sh"

RUN_ID=${RUN_ID:-local_final_220route_optimized${TASK_NUM}_$(date -u +%Y%m%dT%H%M%SZ)}
OUT="${EVAL_ROOT}/${RUN_ID}"
mkdir -p "${OUT}"
echo "[$(date -u +%FT%TZ)] launching final checkpoint=${CKPT} out=${OUT}"

exec env \
  VLA_AD_ROOT="${VLA_AD_ROOT}" \
  BENCH2DRIVE_ROOT="${BENCH2DRIVE_ROOT}" \
  CARLA_ROOT="${CARLA_ROOT}" \
  PLANNER_CHECKPOINT="${CKPT}" VLM_PATH="${VLM_PATH}" \
  TASK_NUM="${TASK_NUM}" RUN_ID="${RUN_ID}" OUT="${OUT}" \
  bash "${VLA_AD_ROOT}/scripts/bench2drive/launch_local_b2d220_eval.sh"
