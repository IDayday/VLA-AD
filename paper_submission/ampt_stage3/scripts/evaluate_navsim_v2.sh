#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: evaluate_navsim_v2.sh [--execute]
Required: PREDICTIONS_DIR, NAVSIM_V2_ROOT, V2_METRIC_CACHE, OUTPUT_DIR.
PREDICTIONS_DIR must contain predictions.json from the single-trajectory export.
EOF
}
execute=0
case "${1:-}" in
  --execute) execute=1 ;;
  --help|-h) usage; exit 0 ;;
  "") ;;
  *) usage >&2; exit 2 ;;
esac
release_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="${RECOGDRIVE_ROOT:-$(cd -- "$release_root/../.." && pwd)}"
predictions="${PREDICTIONS_DIR:-/path/to/predictions}"
navsim_v2="${NAVSIM_V2_ROOT:-/path/to/official_navsim_v2}"
metric_cache="${V2_METRIC_CACHE:-/path/to/navsim_v2_metric_cache}"
output="${OUTPUT_DIR:-$repo_root/outputs/ampt_navsim_v2_eval}"
command=(
  python -m ampt_stage3.score_navsim_v2
  --predictions-dir "$predictions"
  --navsim-root "$navsim_v2"
  --metric-cache-path "$metric_cache"
  --output-dir "$output"
  --split navtest
  --workers "${WORKERS:-32}"
)
printf 'PYTHONPATH=%q ' "$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
printf '%q ' "${command[@]}"; printf '\n'
[[ "$execute" == 1 ]] || exit 0
[[ -e "$predictions" ]] || { echo "Missing PREDICTIONS_DIR: $predictions" >&2; exit 2; }
[[ -d "$navsim_v2/navsim" ]] || { echo "Invalid NAVSIM_V2_ROOT: $navsim_v2" >&2; exit 2; }
[[ -d "$metric_cache" ]] || { echo "Missing V2_METRIC_CACHE: $metric_cache" >&2; exit 2; }
export PYTHONPATH="$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${command[@]}"
