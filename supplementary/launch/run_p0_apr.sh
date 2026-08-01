#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: supplementary/launch/run_p0_apr.sh [OPTIONS]

Dry-run or launch a step-matched P0 APR control after restoring the audited
teacher-pool builder and retention implementation. No training is started by
default.

Options:
  --variant NAME   extra_training_only | one_shot_distillation |
                   repeated_fixed_teacher_set | dynamic_teachers |
                   no_interpolation | no_post_interpolation_evaluation |
                   no_policy_retention | full_apr (default)
  --round N        APR round index, 1..3 (default: 1).
  --execute        Request execution. Also requires ALLOW_TRAIN=1.
  --command CMD    Explicit restored-tree command; alternatively set
                   AMPT_APR_COMMAND.
  -h, --help       Show this help.

Audited settings: LR=5e-7, minimum LR=2.5e-7, 3 epochs, 4 GPUs, batch/GPU=8,
global batch=32, BF16, advantage temperature=0.05, weight clip=5, and policy
retention weight=1. Expected wall-clock time and storage were not measured.
EOF
}

execute=0
variant=full_apr
round=1
command_override="${AMPT_APR_COMMAND:-}"
while (($#)); do
  case "$1" in
    --execute) execute=1; shift ;;
    --variant)
      [[ $# -ge 2 ]] || { echo "error: --variant requires a value" >&2; exit 2; }
      variant="$2"; shift 2 ;;
    --round)
      [[ $# -ge 2 ]] || { echo "error: --round requires a value" >&2; exit 2; }
      round="$2"; shift 2 ;;
    --command)
      [[ $# -ge 2 ]] || { echo "error: --command requires a value" >&2; exit 2; }
      command_override="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$variant" in
  extra_training_only|one_shot_distillation|repeated_fixed_teacher_set|dynamic_teachers|no_interpolation|no_post_interpolation_evaluation|no_policy_retention|full_apr) ;;
  *) echo "error: unsupported APR variant: $variant" >&2; exit 2 ;;
esac
case "$round" in 1|2|3) ;; *) echo "error: --round must be 1, 2, or 3" >&2; exit 2 ;; esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
protocol="$repo_root/supplementary/configs/reproduction_protocol.yaml"
teacher_builder="$repo_root/scripts/stage3/build_pdms_external_submission_teacher_reference.py"
planner="$repo_root/navsim/agents/recogdrive/recogdrive_diffusion_planner.py"

echo "AMPT P0 / APR"
echo "  mode: $([[ $execute -eq 1 ]] && echo execute-requested || echo dry-run)"
echo "  variant: $variant"
echo "  round: $round of 3"
echo "  audited optimization: lr=5e-7, min_lr=2.5e-7, epochs=3, GPUs=4, batch/GPU=8, global=32, precision=bf16-mixed"
echo "  audited policy weighting: temperature=0.05, clip=5, retention=1, prior=0.02, jerk=0.005"
echo "  expected resources: 4 x A800; wall-clock=not measured; storage=not measured"
if [[ -n "$command_override" ]]; then
  printf '  command: %s\n' "$command_override"
else
  echo "  command: <set AMPT_APR_COMMAND after restoring the audited implementation>"
fi

[[ -f "$protocol" ]] || { echo "error: missing protocol: $protocol" >&2; exit 3; }
implementation_present=0
if [[ -f "$teacher_builder" ]] && [[ -f "$planner" ]] && grep -q 'offline_rl_awac' "$planner"; then
  implementation_present=1
fi
if [[ $execute -eq 0 ]]; then
  [[ $implementation_present -eq 1 ]] || echo "  preflight note: historical APR teacher builder/retention implementation is absent in this checkout"
  echo "DRY-RUN ONLY: no process was started."
  exit 0
fi

[[ "${ALLOW_TRAIN:-0}" == "1" ]] || {
  echo "error: execution refused; set ALLOW_TRAIN=1 in addition to --execute" >&2
  exit 4
}
command -v python3 >/dev/null || { echo "error: python3 is unavailable" >&2; exit 3; }
[[ $implementation_present -eq 1 ]] || {
  echo "error: audited APR teacher builder and policy-retention implementation are absent from this checkout" >&2
  echo "restore the audited implementation commit/worktree before executing" >&2
  exit 3
}
[[ -n "$command_override" ]] || {
  echo "error: set AMPT_APR_COMMAND (or --command) to the restored-tree command" >&2
  exit 3
}
for required in AMPT_CHECKPOINT_ROOT AMPT_CACHE_ROOT AMPT_OUTPUT_ROOT; do
  [[ -n "${!required:-}" ]] || { echo "error: required environment variable is unset: $required" >&2; exit 3; }
done

export AMPT_EXPERIMENT_VARIANT="$variant"
export AMPT_APR_ROUND="$round"
echo "EXECUTING user-supplied restored-tree command for variant=$variant, round=$round"
exec bash -lc "$command_override"
