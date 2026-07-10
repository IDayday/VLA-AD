#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_stage2_init_probe_${RUN_ID}}

GPU_LIST=${GPU_LIST:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MAX_SAMPLES=${MAX_SAMPLES:-512}
NUM_OPTIMIZER_STEPS=${NUM_OPTIMIZER_STEPS:-50}
GLOBAL_EPOCHS=${GLOBAL_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
NUM_WORKERS=${NUM_WORKERS:-2}
SAVE_EVERY=${SAVE_EVERY:-1000}
MODE=${MODE:-both}

mkdir -p "${OUTPUT_ROOT}"

run_probe() {
  local name="$1"
  local random_init="$2"
  local out_dir="${OUTPUT_ROOT}/${name}"

  echo "[probe] starting ${name}: RANDOM_INIT_POLICY=${random_init}, output=${out_dir}"
  GPU_LIST="${GPU_LIST}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  RANDOM_INIT_POLICY="${random_init}" \
  OUTPUT_DIR="${out_dir}" \
  MAX_SAMPLES="${MAX_SAMPLES}" \
  NUM_OPTIMIZER_STEPS="${NUM_OPTIMIZER_STEPS}" \
  GLOBAL_EPOCHS="${GLOBAL_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  SAVE_EVERY="${SAVE_EVERY}" \
    bash "${VLA_AD_ROOT}/scripts/bench2drive/run_recogdrive_b2d_stage2_il_official.sh"
}

case "${MODE}" in
  scratch)
    run_probe "scratch_stage1_vlm_random_policy" "1"
    ;;
  il)
    run_probe "official_2b_il_init" "0"
    ;;
  both)
    run_probe "scratch_stage1_vlm_random_policy" "1"
    run_probe "official_2b_il_init" "0"
    ;;
  *)
    echo "Unsupported MODE=${MODE}; use scratch, il, or both." >&2
    exit 2
    ;;
esac

cat > "${OUTPUT_ROOT}/README.md" <<EOF
# Bench2Drive Stage2 Init Probe

- scratch_stage1_vlm_random_policy: uses cached ReCogDrive-VLM hidden states, with policy/action-head initialized from config.
- official_2b_il_init: starts from the released ReCogDrive-2B-IL policy checkpoint.

Compare each run's \`checkpoint_load.json\`, \`summary.json\`, and early training loss. If scratch converges much slower but catches up after enough steps/epochs, the IL checkpoint is mainly a warm start. If scratch remains far below after matched training budget, the current B2D result depends heavily on the released Stage2 planner prior.
EOF
