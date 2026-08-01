#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: supplementary/launch/run_p0_pcmts.sh [OPTIONS]

Dry-run and, only after restoring the audited PC-MTS implementation, launch the
P0 PC-MTS matched-control study. No training is started by default.

Options:
  --variant NAME   score_only | score_plus_pareto |
                   score_pareto_compatibility | full_pcmts (default)
  --execute        Request execution. Also requires ALLOW_TRAIN=1.
  --command CMD    Explicit restored-tree training command. Equivalent to
                   setting AMPT_PCMTS_COMMAND.
  -h, --help       Show this help.

Execution example (after restoring the audited implementation):
  ALLOW_TRAIN=1 AMPT_CHECKPOINT_ROOT=... AMPT_CACHE_ROOT=... \
    AMPT_OUTPUT_ROOT=... AMPT_PCMTS_COMMAND='torchrun ...' \
    supplementary/launch/run_p0_pcmts.sh --variant full_pcmts --execute

Expected resources: 8 x A800 (paper-reported infrastructure); wall-clock time
and storage were not measured. See configs/reproduction_protocol.yaml.
EOF
}

execute=0
variant=full_pcmts
command_override="${AMPT_PCMTS_COMMAND:-}"
while (($#)); do
  case "$1" in
    --execute) execute=1; shift ;;
    --variant)
      [[ $# -ge 2 ]] || { echo "error: --variant requires a value" >&2; exit 2; }
      variant="$2"; shift 2 ;;
    --command)
      [[ $# -ge 2 ]] || { echo "error: --command requires a value" >&2; exit 2; }
      command_override="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$variant" in
  score_only|score_plus_pareto|score_pareto_compatibility|full_pcmts) ;;
  *) echo "error: unsupported PC-MTS variant: $variant" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
protocol="$repo_root/supplementary/configs/reproduction_protocol.yaml"
historical_marker="$repo_root/navsim/agents/recogdrive/curriculum/builder.py"

echo "AMPT P0 / PC-MTS"
echo "  mode: $([[ $execute -eq 1 ]] && echo execute-requested || echo dry-run)"
echo "  variant: $variant"
echo "  protocol: supplementary/configs/reproduction_protocol.yaml"
echo "  proposed controls: bank=16, K=3, quantile=0.95, perturbations=8, pass=6/8"
echo "  seeds: 20260726,20260727,20260728 (proposed, unrun)"
echo "  expected resources: 8 x A800; wall-clock=not measured; storage=not measured"
if [[ -n "$command_override" ]]; then
  printf '  command: %s\n' "$command_override"
else
  echo "  command: <set AMPT_PCMTS_COMMAND after restoring the audited implementation>"
fi

[[ -f "$protocol" ]] || { echo "error: missing protocol: $protocol" >&2; exit 3; }
if [[ $execute -eq 0 ]]; then
  [[ -f "$historical_marker" ]] || echo "  preflight note: historical PC-MTS builder is absent in this checkout (expected dry-run condition)"
  echo "DRY-RUN ONLY: no process was started."
  exit 0
fi

[[ "${ALLOW_TRAIN:-0}" == "1" ]] || {
  echo "error: execution refused; set ALLOW_TRAIN=1 in addition to --execute" >&2
  exit 4
}
command -v python3 >/dev/null || { echo "error: python3 is unavailable" >&2; exit 3; }
[[ -f "$historical_marker" ]] || {
  echo "error: audited PC-MTS code is absent: navsim/agents/recogdrive/curriculum/builder.py" >&2
  echo "restore the audited implementation commit/worktree; this launcher will not substitute unrelated current-tree code" >&2
  exit 3
}
[[ -n "$command_override" ]] || {
  echo "error: set AMPT_PCMTS_COMMAND (or --command) to the restored-tree command" >&2
  exit 3
}
for required in AMPT_CHECKPOINT_ROOT AMPT_CACHE_ROOT AMPT_OUTPUT_ROOT; do
  [[ -n "${!required:-}" ]] || { echo "error: required environment variable is unset: $required" >&2; exit 3; }
done

export AMPT_EXPERIMENT_VARIANT="$variant"
echo "EXECUTING user-supplied restored-tree command for variant=$variant"
exec bash -lc "$command_override"
