#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: export_single_trajectory_predictions.sh [--execute]
Required: CHECKPOINT, VLM_CHECKPOINT, OUTPUT_DIR, OPENSCENE_DATA_ROOT,
NAVSIM_EXP_ROOT, NUPLAN_MAPS_ROOT. Optional SPLIT=navtest.
Produces submission.pkl and predictions.json.
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
checkpoint="${CHECKPOINT:-/data/evaluation_policy_v2.ckpt}"
vlm="${VLM_CHECKPOINT:-/path/to/InternVL3-2B}"
output="${OUTPUT_DIR:-$repo_root/outputs/ampt_predictions}"
split="${SPLIT:-navtest}"
seed="${EVAL_SEED:-20260726}"
inference=(
  python "$repo_root/navsim/planning/script/run_create_submission_pickle.py"
  "train_test_split=$split"
  agent=recogdrive_agent
  agent._target_=ampt_stage3.eval_agent.AMPTReproductionAgent
  "agent.checkpoint_path=$checkpoint"
  "agent.vlm_path=$vlm"
  agent.grpo=false
  agent.allow_random_init=false
  agent.cache_hidden_state=false
  agent.vlm_type=internvl
  agent.dit_type=small
  agent.vlm_size=small
  agent.sampling_method=ddim
  +agent.vlm_feature_dim=1536
  "+agent.evaluation_seed=$seed"
  "output_dir=$output"
  team_name=anonymous authors=anonymous email=anonymous institution=anonymous country=anonymous
)
printf '%q ' "${inference[@]}"; printf '\n'
printf 'python -m ampt_stage3.submission_to_manifest --submission %q --output-json %q\n' \
  "$output/submission.pkl" "$output/predictions.json"
[[ "$execute" == 1 ]] || exit 0
for name in OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT NUPLAN_MAPS_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "Set $name." >&2; exit 2; }
done
[[ -f "$checkpoint" ]] || { echo "Missing CHECKPOINT: $checkpoint" >&2; exit 2; }
[[ -d "$vlm" ]] || { echo "Missing VLM_CHECKPOINT: $vlm" >&2; exit 2; }
mkdir -p "$output"
export PYTHONPATH="$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m ampt_stage3.validate_checkpoint \
  --checkpoint "$checkpoint" \
  --expected-sha256 "${EXPECTED_CHECKPOINT_SHA256:-}" \
  --output "$output/anonymous_checkpoint_receipt.json"
"${inference[@]}"
python -m ampt_stage3.submission_to_manifest \
  --submission "$output/submission.pkl" \
  --output-json "$output/predictions.json"
