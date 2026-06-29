#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/mnt/project/onevl_navsim_exp/ar_answer_stage2_train_full200_local86_latest}"
REPO_ROOT="${REPO_ROOT:-/mnt/project/VLA-AD}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/navsim/bin/torchrun}"
MASTER_PORT="${MASTER_PORT:-29747}"
RESUME_STEP="${RESUME_STEP:-30000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABLE_SECONDS="${STABLE_SECONDS:-60}"

RUN_DIR="$(readlink -f "${RUN_DIR}")"
CKPT="${RUN_DIR}/step_$(printf '%08d' "${RESUME_STEP}").ckpt"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
SWITCH_LOG="${LOG_DIR}/switch_epoch_step_ckpts.log"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${SWITCH_LOG}"
}

is_stable() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local now mtime size1 size2
  now=$(date +%s)
  mtime=$(stat -c %Y "${path}")
  if (( now - mtime < STABLE_SECONDS )); then
    return 1
  fi
  size1=$(stat -c %s "${path}")
  sleep 1
  size2=$(stat -c %s "${path}")
  [[ "${size1}" == "${size2}" ]]
}

log "waiting for stable checkpoint: ${CKPT}"
until is_stable "${CKPT}"; do
  sleep "${POLL_SECONDS}"
done
log "checkpoint is stable"

OLD_LAUNCHER=""
if [[ -f "${RUN_DIR}/launcher.pid" ]]; then
  OLD_LAUNCHER="$(cat "${RUN_DIR}/launcher.pid" || true)"
fi
if [[ -n "${OLD_LAUNCHER}" ]] && ps -p "${OLD_LAUNCHER}" >/dev/null 2>&1; then
  log "stopping old training process group: ${OLD_LAUNCHER}"
  kill -TERM -- "-${OLD_LAUNCHER}" >/dev/null 2>&1 || kill -TERM "${OLD_LAUNCHER}" >/dev/null 2>&1 || true
else
  log "old launcher pid not alive; continuing"
fi

for _ in $(seq 1 60); do
  if ! pgrep -f "scripts/train_recogdrive_expert_chunked.py.*${RUN_DIR}" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if pgrep -f "scripts/train_recogdrive_expert_chunked.py.*${RUN_DIR}" >/dev/null 2>&1; then
  log "old training processes still alive after TERM; sending KILL"
  pkill -KILL -f "scripts/train_recogdrive_expert_chunked.py.*${RUN_DIR}" || true
  pkill -KILL -f "torchrun.*${RUN_DIR}" || true
fi

cd "${REPO_ROOT}"
cmd=(
  "${TORCHRUN_BIN}"
  --nproc_per_node 8
  --master_port "${MASTER_PORT}"
  scripts/train_recogdrive_expert_chunked.py
  --config /mnt/project/VLA-AD/configs/onevl_ar_answer_stage2_small.yaml
  --resume-from "${CKPT}"
  --resume-mode full
  --chunk-cache-root /mnt/project/onevl_navsim_exp/ar_answer_stage2_cache_full103k_latest
  --chunk-name-pattern shard_*
  --flat-global-dataset
  --output-dir "${RUN_DIR}"
  --global-epochs 200
  --batch-size 16
  --gradient-accumulation-steps 1
  --num-workers 4
  --prefetch-factor 4
  --lr-action-head 1e-4
  --lr-expert 0.0
  --jepa-align-weight 0.0
  --vggt-align-weight 0.0
  --precision bf16
  --lr-scheduler official-cosine
  --lr-scheduler-epochs 200
  --lr-warmup-epochs 3
  --min-lr 1e-6
  --log-every 100
  --save-every 10000
  --save-every-epoch
  --final-check-precision fp32
)

{
  printf '[%s] ' "$(date -Is)"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} >> "${RUN_DIR}/commands.log"

log "starting resumed training with epoch and step checkpoints"
set +e
"${cmd[@]}" > "${LOG_DIR}/train_resume_epoch_step_ckpts.log" 2>&1
status=$?
set -e
log "resumed training exited with status ${status}"
exit "${status}"
