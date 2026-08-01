#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: evaluate_navsim_v1.sh [--execute]
Required: CHECKPOINT, VLM_CHECKPOINT, METRIC_CACHE, OUTPUT_DIR,
OPENSCENE_DATA_ROOT, NAVSIM_EXP_ROOT, NUPLAN_MAPS_ROOT.
The command emits one trajectory per scene and performs no reranking.
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
checkpoint="${CHECKPOINT:-/data/evaluation_policy_v1.ckpt}"
vlm="${VLM_CHECKPOINT:-/path/to/InternVL3-2B}"
metric_cache="${METRIC_CACHE:-/path/to/navtest_metric_cache}"
output="${OUTPUT_DIR:-$repo_root/outputs/ampt_navsim_v1_eval}"
seed="${EVAL_SEED:-20260726}"
command=(
  torchrun --standalone --nproc_per_node="${GPUS:-8}"
  "$repo_root/navsim/planning/script/run_pdm_score_recogdrive.py"
  train_test_split=navtest
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
  "metric_cache_path=$metric_cache"
  "output_dir=$output"
  experiment_name=anonymous_ampt_navsim_v1
)
printf '%q ' "${command[@]}"; printf '\n'
[[ "$execute" == 1 ]] || exit 0
for name in OPENSCENE_DATA_ROOT NAVSIM_EXP_ROOT NUPLAN_MAPS_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "Set $name." >&2; exit 2; }
done
[[ -f "$checkpoint" ]] || { echo "Missing CHECKPOINT: $checkpoint" >&2; exit 2; }
[[ -d "$vlm" ]] || { echo "Missing VLM_CHECKPOINT: $vlm" >&2; exit 2; }
[[ -d "$metric_cache" ]] || { echo "Missing METRIC_CACHE: $metric_cache" >&2; exit 2; }
mkdir -p "$output"
export PYTHONPATH="$release_root/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
python -m ampt_stage3.validate_checkpoint \
  --checkpoint "$checkpoint" \
  --expected-sha256 "${EXPECTED_CHECKPOINT_SHA256:-}" \
  --output "$output/anonymous_checkpoint_receipt.json"
exec "${command[@]}"
