#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:?Set TRAIN_OUT_ROOT to the Stage 3 training stable-launch output root.}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-stage3_safe_diffgrpo_eval_after_train_$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${EVAL_RUN_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-600}"

STATUS_FILE="${TRAIN_OUT_ROOT}/status/stage3_rl_2b.json"

mkdir -p "${EVAL_OUT_ROOT}"

while true; do
  if [[ ! -f "${STATUS_FILE}" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for training status file: ${STATUS_FILE}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  STATE="$("${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path("${STATUS_FILE}").read_text())
print(payload.get("state", "unknown"))
PY
)"

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) training_state=${STATE}"
  if [[ "${STATE}" == "done" ]]; then
    CHECKPOINT="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
root = Path("${TRAIN_OUT_ROOT}") / "train"
checkpoints = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)
if not checkpoints:
    raise SystemExit(f"No checkpoint found under {root}")
print(checkpoints[-1])
PY
)"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) launching eval checkpoint=${CHECKPOINT}"
    CHECKPOINT="${CHECKPOINT}" \
      OUT_ROOT="${EVAL_OUT_ROOT}" \
      bash "${REPO_ROOT}/scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu.sh"
    exit 0
  fi

  if [[ "${STATE}" == "failed" || "${STATE}" == "launch_failed" ]]; then
    echo "Stage 3 training failed; refusing to launch evaluation." >&2
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
