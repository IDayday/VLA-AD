#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-/mnt/project/VLA-AD/outputs}"
CURRENT_RUN="${CURRENT_RUN:-stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z}"
SESSION_NAME="${SESSION_NAME:-stage3_supervisor_dryrun}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
ALLOW_LAUNCH="${ALLOW_LAUNCH:-0}"

SUMMARY_TSV="${SUMMARY_TSV:-${OUTPUTS_ROOT}/stage3_runs_summary_latest.tsv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUTS_ROOT}/stage3_runs_summary_latest.json}"
REPORT_MD="${REPORT_MD:-${OUTPUTS_ROOT}/stage3_algorithm_status_latest.md}"
REPORT_JSON="${REPORT_JSON:-${OUTPUTS_ROOT}/stage3_algorithm_status_latest.json}"
DECISION_JSON="${DECISION_JSON:-${OUTPUTS_ROOT}/stage3_next_action_latest.json}"
COMMAND_FILE="${COMMAND_FILE:-${OUTPUTS_ROOT}/stage3_next_action_command.sh}"
STATUS_JSON="${STATUS_JSON:-${OUTPUTS_ROOT}/stage3_supervisor_status_latest.json}"
LOG_FILE="${LOG_FILE:-${OUTPUTS_ROOT}/stage3_supervisor.log}"
STDOUT_LOG="${STDOUT_LOG:-${OUTPUTS_ROOT}/stage3_supervisor.stdout.log}"
TMUX_FILE="${TMUX_FILE:-${OUTPUTS_ROOT}/stage3_supervisor.tmux}"

BASELINE_PDMS="${BASELINE_PDMS:-0.9061843202874436}"
ORIGINAL_STAGE3_PDMS="${ORIGINAL_STAGE3_PDMS:-0.9055}"
CONTINUE_MARGIN="${CONTINUE_MARGIN:-0.003}"
SWITCH_MARGIN="${SWITCH_MARGIN:-0.015}"
MIN_SAFE_RATIO="${MIN_SAFE_RATIO:-0.88}"
MIN_EVAL_ROWS="${MIN_EVAL_ROWS:-1}"
LAUNCH_RUNNING_POLICY="${LAUNCH_RUNNING_POLICY:-wait_until_finished}"
ACTIVE_EVENT_AGE_SEC="${ACTIVE_EVENT_AGE_SEC:-1800}"
MAX_RUNS="${MAX_RUNS:-100}"
PRINT_LIMIT="${PRINT_LIMIT:-12}"

mkdir -p "${OUTPUTS_ROOT}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for the persistent supervisor launcher." >&2
  echo "Run one foreground refresh instead:" >&2
  printf '  %q %q --once\n' "${PYTHON_BIN}" "${REPO_ROOT}/scripts/training/supervise_recogdrive_stage3_experiment.py" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Stage3 supervisor already running: ${SESSION_NAME}"
  tmux list-sessions | grep "${SESSION_NAME}" || true
  exit 0
fi

allow_arg=()
if [[ "${ALLOW_LAUNCH}" == "1" || "${ALLOW_LAUNCH,,}" == "true" ]]; then
  allow_arg=(--allow-launch)
fi

supervisor_args=(
  "scripts/training/supervise_recogdrive_stage3_experiment.py"
  "--workdir" "${REPO_ROOT}"
  "--outputs-root" "${OUTPUTS_ROOT}"
  "--current-run" "${CURRENT_RUN}"
  "--summary-tsv" "${SUMMARY_TSV}"
  "--summary-json" "${SUMMARY_JSON}"
  "--report-md" "${REPORT_MD}"
  "--report-json" "${REPORT_JSON}"
  "--decision-json" "${DECISION_JSON}"
  "--command-file" "${COMMAND_FILE}"
  "--status-json" "${STATUS_JSON}"
  "--log-file" "${LOG_FILE}"
  "--baseline-pdms" "${BASELINE_PDMS}"
  "--original-stage3-pdms" "${ORIGINAL_STAGE3_PDMS}"
  "--continue-margin" "${CONTINUE_MARGIN}"
  "--switch-margin" "${SWITCH_MARGIN}"
  "--min-safe-ratio" "${MIN_SAFE_RATIO}"
  "--min-eval-rows" "${MIN_EVAL_ROWS}"
  "--launch-running-policy" "${LAUNCH_RUNNING_POLICY}"
  "--active-event-age-sec" "${ACTIVE_EVENT_AGE_SEC}"
  "--max-runs" "${MAX_RUNS}"
  "--print-limit" "${PRINT_LIMIT}"
  "--interval-sec" "${INTERVAL_SEC}"
)
if [[ "${#allow_arg[@]}" -gt 0 ]]; then
  supervisor_args+=("--allow-launch")
fi

printf -v supervisor_cmd '%q ' "${PYTHON_BIN}" "${supervisor_args[@]}"
tmux_command="cd $(printf '%q' "${REPO_ROOT}") && exec ${supervisor_cmd}>> $(printf '%q' "${STDOUT_LOG}") 2>&1"
tmux new-session -d -s "${SESSION_NAME}" "${tmux_command}"
echo "${SESSION_NAME}" > "${TMUX_FILE}"

{
  echo "repo_root=${REPO_ROOT}"
  echo "python_bin=${PYTHON_BIN}"
  echo "outputs_root=${OUTPUTS_ROOT}"
  echo "current_run=${CURRENT_RUN}"
  echo "session_name=${SESSION_NAME}"
  echo "interval_sec=${INTERVAL_SEC}"
  echo "allow_launch=${ALLOW_LAUNCH}"
  echo "summary_tsv=${SUMMARY_TSV}"
  echo "report_md=${REPORT_MD}"
  echo "decision_json=${DECISION_JSON}"
  echo "command_file=${COMMAND_FILE}"
  echo "status_json=${STATUS_JSON}"
  echo "log_file=${LOG_FILE}"
  echo "stdout_log=${STDOUT_LOG}"
  echo "baseline_pdms=${BASELINE_PDMS}"
  echo "original_stage3_pdms=${ORIGINAL_STAGE3_PDMS}"
  echo "continue_margin=${CONTINUE_MARGIN}"
  echo "switch_margin=${SWITCH_MARGIN}"
  echo "min_safe_ratio=${MIN_SAFE_RATIO}"
  echo "min_eval_rows=${MIN_EVAL_ROWS}"
  echo "launch_running_policy=${LAUNCH_RUNNING_POLICY}"
} > "${OUTPUTS_ROOT}/stage3_supervisor_launch_config.txt"

echo "Started Stage3 supervisor: ${SESSION_NAME}"
echo "allow_launch=${ALLOW_LAUNCH}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Status: ${STATUS_JSON}"
