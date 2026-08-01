#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: supplementary/launch/run_p0_ffpgrpo.sh [OPTIONS]

Dry-run or launch the P0 FF-PGRPO credit-assignment ablation after restoring
the audited implementation. No training is started by default.

Options:
  --variant NAME   scalar_grpo | hard_feasibility_gate |
                   coherent_reference_guard | pareto_positive_credit_gate |
                   full_ffpgrpo (default)
  --epochs N       10 for the paper protocol (default), or 20 only to reproduce
                   the separately labelled archived configuration.
  --execute        Request execution. Also requires ALLOW_TRAIN=1.
  --command CMD    Explicit restored-tree command; alternatively set
                   AMPT_FFPGRPO_COMMAND.
  -h, --help       Show this help.

Expected resources: 8 x A800; wall-clock time and storage were not measured.
The 658-scene supplement is locked to 367 versus 440 (net +73) and explicitly
does not substitute results from the later 91.45 checkpoint.
EOF
}

execute=0
variant=full_ffpgrpo
epochs=10
command_override="${AMPT_FFPGRPO_COMMAND:-}"
while (($#)); do
  case "$1" in
    --execute) execute=1; shift ;;
    --variant)
      [[ $# -ge 2 ]] || { echo "error: --variant requires a value" >&2; exit 2; }
      variant="$2"; shift 2 ;;
    --epochs)
      [[ $# -ge 2 ]] || { echo "error: --epochs requires a value" >&2; exit 2; }
      epochs="$2"; shift 2 ;;
    --command)
      [[ $# -ge 2 ]] || { echo "error: --command requires a value" >&2; exit 2; }
      command_override="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$variant" in
  scalar_grpo|hard_feasibility_gate|coherent_reference_guard|pareto_positive_credit_gate|full_ffpgrpo) ;;
  *) echo "error: unsupported FF-PGRPO variant: $variant" >&2; exit 2 ;;
esac
case "$epochs" in
  10|20) ;;
  *) echo "error: --epochs must be 10 (paper) or 20 (separately labelled archive)" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
protocol="$repo_root/supplementary/configs/reproduction_protocol.yaml"
planner="$repo_root/navsim/agents/recogdrive/recogdrive_diffusion_planner.py"

echo "AMPT P0 / FF-PGRPO"
echo "  mode: $([[ $execute -eq 1 ]] && echo execute-requested || echo dry-run)"
echo "  variant: $variant"
echo "  epochs: $epochs $([[ $epochs == 10 ]] && echo '(paper-reported)' || echo '(archived, not paper-equivalent)')"
echo "  group size: 16; batch/GPU=2; accumulation=4; GPUs=8; effective batch=64"
echo "  LR=1e-4; BC=0.10->0.05; reference KL=0.02 (archived evidence)"
echo "  hard-658 protocol: Scalar GRPO=367, paper AMPT=440, net recovery=73; exclude final 91.45 branch"
echo "  expected resources: 8 x A800; wall-clock=not measured; storage=not measured"
if [[ -n "$command_override" ]]; then
  printf '  command: %s\n' "$command_override"
else
  echo "  command: <set AMPT_FFPGRPO_COMMAND after restoring the audited implementation>"
fi

[[ -f "$protocol" ]] || { echo "error: missing protocol: $protocol" >&2; exit 3; }
implementation_present=0
if [[ -f "$planner" ]] && grep -Eq '_compute_(feasible|core)_pareto_advantages' "$planner"; then
  implementation_present=1
fi
if [[ $execute -eq 0 ]]; then
  [[ $implementation_present -eq 1 ]] || echo "  preflight note: historical FF-PGRPO credit-assignment functions are absent in this checkout"
  echo "DRY-RUN ONLY: no process was started."
  exit 0
fi

[[ "${ALLOW_TRAIN:-0}" == "1" ]] || {
  echo "error: execution refused; set ALLOW_TRAIN=1 in addition to --execute" >&2
  exit 4
}
command -v python3 >/dev/null || { echo "error: python3 is unavailable" >&2; exit 3; }
[[ $implementation_present -eq 1 ]] || {
  echo "error: audited FF-PGRPO functions are absent from navsim/agents/recogdrive/recogdrive_diffusion_planner.py" >&2
  echo "restore the audited implementation commit/worktree before executing" >&2
  exit 3
}
[[ -n "$command_override" ]] || {
  echo "error: set AMPT_FFPGRPO_COMMAND (or --command) to the restored-tree command" >&2
  exit 3
}
for required in AMPT_CHECKPOINT_ROOT AMPT_CACHE_ROOT AMPT_OUTPUT_ROOT; do
  [[ -n "${!required:-}" ]] || { echo "error: required environment variable is unset: $required" >&2; exit 3; }
done

export AMPT_EXPERIMENT_VARIANT="$variant"
export AMPT_MAX_EPOCHS="$epochs"
echo "EXECUTING user-supplied restored-tree command for variant=$variant, epochs=$epochs"
exec bash -lc "$command_override"
