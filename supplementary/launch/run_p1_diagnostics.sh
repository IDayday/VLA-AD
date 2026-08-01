#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: supplementary/launch/run_p1_diagnostics.sh [OPTIONS]

Dry-run or launch one proposed P1 diagnostic suite after restoring the audited
AMPT experiment drivers. No training is started by default.

Options:
  --suite NAME     compatibility_metrics | pcmts_sensitivity |
                   group_dynamics | coherent_reference | apr_teacher_sources
                   (default: compatibility_metrics)
  --execute        Request execution. Also requires ALLOW_TRAIN=1.
  --command CMD    Explicit restored-tree command; alternatively set
                   AMPT_P1_COMMAND.
  -h, --help       Show this help.

Resource envelope: PC-MTS/FF-PGRPO diagnostics use up to 8 x A800 and APR
teacher-source diagnostics use up to 4 x A800. Aggregate wall-clock time and
storage are not measured. Each sweep point must retain the proposed three-seed
schedule and write to a separate output directory.
EOF
}

execute=0
suite=compatibility_metrics
command_override="${AMPT_P1_COMMAND:-}"
while (($#)); do
  case "$1" in
    --execute) execute=1; shift ;;
    --suite)
      [[ $# -ge 2 ]] || { echo "error: --suite requires a value" >&2; exit 2; }
      suite="$2"; shift 2 ;;
    --command)
      [[ $# -ge 2 ]] || { echo "error: --command requires a value" >&2; exit 2; }
      command_override="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$suite" in
  compatibility_metrics|pcmts_sensitivity|group_dynamics|coherent_reference|apr_teacher_sources) ;;
  *) echo "error: unsupported P1 suite: $suite" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
protocol="$repo_root/supplementary/configs/reproduction_protocol.yaml"
pcmts_marker="$repo_root/navsim/agents/recogdrive/curriculum/builder.py"
apr_marker="$repo_root/scripts/stage3/build_pdms_external_submission_teacher_reference.py"
planner="$repo_root/navsim/agents/recogdrive/recogdrive_diffusion_planner.py"

case "$suite" in
  compatibility_metrics)
    grid="candidate-to-GT ADE/FDE; deterministic-policy distance; minimum-bank distance; average-KNN distance; calibrated KNN"
    resources="up to 8 x A800" ;;
  pcmts_sensitivity)
    grid="bank={8,16,32}; K={1,3,5}; q={0.90,0.95,0.975}; perturb={4,8,16}; pass={0.625,0.75,0.875}"
    resources="up to 8 x A800" ;;
  group_dynamics)
    grid="Scalar GRPO vs FF-PGRPO; log group/feasibility/advantage diagnostics every 25 steps"
    resources="up to 8 x A800" ;;
  coherent_reference)
    grid="GT; current policy; highest-score feasible; nearest safe; none; component-wise envelope"
    resources="up to 8 x A800" ;;
  apr_teacher_sources)
    grid="leave one source out: historical; GRPO seeds; structured expansion; safety/progress/structure specialists"
    resources="up to 4 x A800" ;;
esac

echo "AMPT P1 diagnostic"
echo "  mode: $([[ $execute -eq 1 ]] && echo execute-requested || echo dry-run)"
echo "  suite: $suite"
echo "  grid: $grid"
echo "  seeds: 20260726,20260727,20260728 (proposed, unrun)"
echo "  expected resources: $resources; aggregate wall-clock=not measured; storage=not measured"
if [[ -n "$command_override" ]]; then
  printf '  command: %s\n' "$command_override"
else
  echo "  command: <set AMPT_P1_COMMAND after restoring the audited implementation>"
fi

[[ -f "$protocol" ]] || { echo "error: missing protocol: $protocol" >&2; exit 3; }
implementation_present=0
case "$suite" in
  compatibility_metrics|pcmts_sensitivity)
    [[ -f "$pcmts_marker" ]] && implementation_present=1 ;;
  group_dynamics|coherent_reference)
    if [[ -f "$planner" ]] && grep -Eq '_compute_(feasible|core)_pareto_advantages' "$planner"; then implementation_present=1; fi ;;
  apr_teacher_sources)
    [[ -f "$apr_marker" ]] && implementation_present=1 ;;
esac
if [[ $execute -eq 0 ]]; then
  [[ $implementation_present -eq 1 ]] || echo "  preflight note: implementation required by this suite is absent in the current checkout"
  echo "DRY-RUN ONLY: no process was started."
  exit 0
fi

[[ "${ALLOW_TRAIN:-0}" == "1" ]] || {
  echo "error: execution refused; set ALLOW_TRAIN=1 in addition to --execute" >&2
  exit 4
}
command -v python3 >/dev/null || { echo "error: python3 is unavailable" >&2; exit 3; }
[[ $implementation_present -eq 1 ]] || {
  echo "error: audited implementation for suite '$suite' is absent from this checkout" >&2
  echo "restore the audited implementation commit/worktree before executing" >&2
  exit 3
}
[[ -n "$command_override" ]] || {
  echo "error: set AMPT_P1_COMMAND (or --command) to the restored-tree command" >&2
  exit 3
}
for required in AMPT_CHECKPOINT_ROOT AMPT_CACHE_ROOT AMPT_OUTPUT_ROOT; do
  [[ -n "${!required:-}" ]] || { echo "error: required environment variable is unset: $required" >&2; exit 3; }
done

export AMPT_DIAGNOSTIC_SUITE="$suite"
echo "EXECUTING user-supplied restored-tree command for suite=$suite"
exec bash -lc "$command_override"
