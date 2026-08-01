#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_reproduction.sh [--execute]
Required: V1_CHECKPOINT, V2_ARTIFACT, V1_RESULTS_CSV, V2_SUMMARY_JSON,
OUTPUT_FILE. Optional: EXPECTED_V1_SHA256, EXPECTED_V2_SHA256.
The receipt contains hashes and metrics, never local input paths.
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
v1_checkpoint="${V1_CHECKPOINT:-/data/evaluation_policy_v1.ckpt}"
v2_artifact="${V2_ARTIFACT:-/data/evaluation_artifact_v2}"
v1="${V1_RESULTS_CSV:-/data/results/navsim_v1.csv}"
v2="${V2_SUMMARY_JSON:-/data/results/navsim_v2_summary.json}"
output="${OUTPUT_FILE:-$repo_root/outputs/ampt_reproduction_receipt.json}"
command=(
  python -m ampt_stage3.verify_results
  --v1-checkpoint "$v1_checkpoint"
  --v2-artifact "$v2_artifact"
  --expected-v1-sha256 "${EXPECTED_V1_SHA256:-}"
  --expected-v2-sha256 "${EXPECTED_V2_SHA256:-}"
  --v1-csv "$v1"
  --v2-summary "$v2"
  --v1-protocol "$release_root/configs/eval_navsim_v1.yaml"
  --v2-protocol "$release_root/configs/eval_navsim_v2.yaml"
  --output "$output"
)
printf 'PYTHONPATH=%q ' "$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
printf '%q ' "${command[@]}"; printf '\n'
[[ "$execute" == 1 ]] || exit 0
for path in "$v1_checkpoint" "$v2_artifact" "$v1" "$v2"; do
  [[ -e "$path" ]] || { echo "Missing required artifact." >&2; exit 2; }
done
export PYTHONPATH="$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${command[@]}"
