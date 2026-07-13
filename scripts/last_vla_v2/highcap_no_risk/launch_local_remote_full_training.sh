#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/last_vla_highcap_no_risk}"
READINESS_REPORT="${READINESS_REPORT:-${OUT_ROOT}/readiness/final_readiness_report.md}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
MASTER_PORT_LOCAL="${MASTER_PORT_LOCAL:-29531}"
MASTER_PORT_REMOTE="${MASTER_PORT_REMOTE:-29541}"
LOCAL_RUN_NAME="${LOCAL_RUN_NAME:-serverA_frozen_highcap_no_risk}"
LORA_PRESET="${LORA_PRESET:-attention_mlp}"
LORA_SCOPE="${LORA_SCOPE:-llm}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_USE_RSLORA="${LORA_USE_RSLORA:-true}"
LORA_USE_DORA="${LORA_USE_DORA:-false}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-}"
LORA_HIDDEN_ANCHOR_EVERY_N_STEPS="${LORA_HIDDEN_ANCHOR_EVERY_N_STEPS:-4}"
REPORT_JSON="${OUT_ROOT}/reports/last_vla_v2_highcap_training_launch_report.json"
mkdir -p "${OUT_ROOT}/logs" "$(dirname "${REPORT_JSON}")"

if [[ ! -f "${READINESS_REPORT}" ]] || ! grep -q "Status: READY" "${READINESS_REPORT}"; then
  echo "Final readiness report is missing or not READY: ${READINESS_REPORT}" >&2
  exit 2
fi

local_cmd=(
  env RUN_TRAIN=1
  FULL_GEOMETRY_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-}"
  A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-}"
  OUT_ROOT="${OUT_ROOT}/train_local"
  MASTER_PORT="${MASTER_PORT_LOCAL}"
  PYTHON_BIN="${PYTHON_BIN}"
  scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh
)

remote_env=(
  RUN_TRAIN=1
  FULL_GEOMETRY_CHUNK_ROOT="${REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-}"
  A0_INIT_CHECKPOINT="${REMOTE_A0_INIT_CHECKPOINT:-}"
  VLM_PATH="${REMOTE_VLM_PATH:-}"
  NAVSIM_LOG_PATH="${REMOTE_NAVSIM_LOG_PATH:-}"
  SENSOR_BLOBS_PATH="${REMOTE_SENSOR_BLOBS_PATH:-}"
  OUT_ROOT="${REMOTE_OUT_ROOT:-}/train_remote"
  MASTER_PORT="${MASTER_PORT_REMOTE}"
  PYTHON_BIN="${REMOTE_PYTHON_BIN:-${PYTHON_BIN}}"
  LORA_PRESET="${LORA_PRESET}"
  LORA_SCOPE="${LORA_SCOPE}"
  LORA_R="${LORA_R}"
  LORA_ALPHA="${LORA_ALPHA}"
  LORA_DROPOUT="${LORA_DROPOUT}"
  LORA_USE_RSLORA="${LORA_USE_RSLORA}"
  LORA_USE_DORA="${LORA_USE_DORA}"
  LORA_TARGET_MODULES="${LORA_TARGET_MODULES}"
  LORA_HIDDEN_ANCHOR_EVERY_N_STEPS="${LORA_HIDDEN_ANCHOR_EVERY_N_STEPS}"
)

remote_cmd_text="cd ${REMOTE_PROJECT_ROOT:-} && git fetch origin && git checkout feature/recogdrive-last-vla-v2 && git pull --ff-only origin feature/recogdrive-last-vla-v2 && env"
for item in "${remote_env[@]}"; do remote_cmd_text+=" $(printf '%q' "${item}")"; done
remote_cmd_text+=" scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh"
local_cmd_text="$(printf '%q ' "${local_cmd[@]}")"

{
  echo "# Last-VLA v2 High-cap Manual Launch Commands"
  echo
  echo "## Local Server A"
  printf '```bash\n'
  printf '%q ' "${local_cmd[@]}"; printf '\n'
  printf '```\n\n'
  echo "## Remote Server B"
  printf '```bash\nssh %q %q\n```\n' "${REMOTE_HOST:-REMOTE_HOST}" "${remote_cmd_text}"
} >"${OUT_ROOT}/reports/last_vla_v2_highcap_manual_launch_commands.md"

if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; wrote manual commands to ${OUT_ROOT}/reports/last_vla_v2_highcap_manual_launch_commands.md"
  exit 0
fi

for name in FULL_HIGHCAP_TRAIN_CHUNK_ROOT A0_INIT_CHECKPOINT REMOTE_HOST REMOTE_PROJECT_ROOT REMOTE_OUT_ROOT REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT REMOTE_A0_INIT_CHECKPOINT REMOTE_VLM_PATH REMOTE_NAVSIM_LOG_PATH REMOTE_SENSOR_BLOBS_PATH; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required launch variable: ${name}" >&2
    exit 2
  fi
done
[[ -d "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]] || { echo "Missing local train cache: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "Missing local A0 checkpoint: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }

local_commit="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
remote_commit="$(
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "cd '${REMOTE_PROJECT_ROOT}' && git fetch origin >/dev/null && git checkout feature/recogdrive-last-vla-v2 >/dev/null && git pull --ff-only origin feature/recogdrive-last-vla-v2 >/dev/null && git rev-parse HEAD"
)"
if [[ "${remote_commit}" != "${local_commit}" && "${ALLOW_REMOTE_COMMIT_MISMATCH:-0}" != "1" ]]; then
  echo "Remote commit ${remote_commit} differs from local ${local_commit}; set ALLOW_REMOTE_COMMIT_MISMATCH=1 to override." >&2
  exit 2
fi

ssh -o BatchMode=yes -o ConnectTimeout=8 "${REMOTE_HOST}" \
  "test -d '${REMOTE_FULL_HIGHCAP_TRAIN_CHUNK_ROOT}' && test -e '${REMOTE_A0_INIT_CHECKPOINT}' && test -e '${REMOTE_VLM_PATH}' && test -d '${REMOTE_NAVSIM_LOG_PATH}' && test -d '${REMOTE_SENSOR_BLOBS_PATH}'"

local_log="${OUT_ROOT}/logs/${LOCAL_RUN_NAME}.log"
nohup "${local_cmd[@]}" >"${local_log}" 2>&1 < /dev/null &
local_pid=$!

remote_log="${REMOTE_OUT_ROOT}/logs/serverB_lora_highcap_no_risk.remote_launch.log"
ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_OUT_ROOT}/logs' && cd '${REMOTE_PROJECT_ROOT}' && nohup bash -lc $(printf '%q' "${remote_cmd_text}") >'${remote_log}' 2>&1 < /dev/null & echo \$!" >"${OUT_ROOT}/logs/remote_pid.txt"
remote_pid="$(cat "${OUT_ROOT}/logs/remote_pid.txt" | tail -n 1)"

python - "$REPORT_JSON" "$local_cmd_text" "$remote_cmd_text" <<PY
import json, sys, time
payload = {
  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "git_commit": "${local_commit}",
  "readiness_report": "${READINESS_REPORT}",
  "local_command": sys.argv[2],
  "local_pid": "${local_pid}",
  "local_log": "${local_log}",
  "remote_host": "${REMOTE_HOST}",
  "remote_command": sys.argv[3],
  "remote_pid": "${remote_pid}",
  "remote_log": "${remote_log}",
  "training_launched": True,
  "full_eval_launched": False,
}
with open("${REPORT_JSON}", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\\n")
PY

cat "${REPORT_JSON}"
