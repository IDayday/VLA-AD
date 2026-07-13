#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${OUT_ROOT:-/tmp/last_vla_highcap_eval_prep}"
SERVER_A_ROOT="${SERVER_A_ROOT:-${OUT_ROOT}/train_local/serverA_frozen_highcap_no_risk}"
SERVER_B_ROOT="${SERVER_B_ROOT:-${OUT_ROOT}/train_remote/serverB_lora_highcap_no_risk}"
NAVTEST_HIGHCAP_CHUNK_ROOT="${NAVTEST_HIGHCAP_CHUNK_ROOT:-}"
VLM_PATH="${VLM_PATH:-}"
PDM_METRIC_CACHE_DIR="${PDM_METRIC_CACHE_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PREP_DIR="${OUT_ROOT}/eval_prep"
mkdir -p "${PREP_DIR}"

B1_ADAPTER_DIR="${SERVER_B_ROOT}/vlm_lora_cot_alignment/adapters/vlm_lora"
B_NAVTEST_CACHE="${PREP_DIR}/serverB_lora_navtest_hidden_cache"

cat >"${PREP_DIR}/build_serverB_lora_navtest_cache.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
${PYTHON_BIN} scripts/build_recogdrive_hidden_cache_with_lora.py \\
  --base-chunk-root "${NAVTEST_HIGHCAP_CHUNK_ROOT}" \\
  --output-chunk-root "${B_NAVTEST_CACHE}" \\
  --vlm-path "${VLM_PATH}" \\
  --vlm-type "${VLM_TYPE:-internvl}" \\
  --vlm-lora-adapter-dir "${B1_ADAPTER_DIR}" \\
  --precision fp32 \\
  --cache-variant highcap_no_risk_lora_navtest
EOF
chmod +x "${PREP_DIR}/build_serverB_lora_navtest_cache.sh"

CHECKPOINTS=(
  step_00050000.ckpt
  step_00060000.ckpt
  step_00080000.ckpt
  step_00100000.ckpt
  step_00120000.ckpt
  latest.ckpt
)
{
  echo "# Server A checkpoint sweep commands"
  for ckpt in "${CHECKPOINTS[@]}"; do
    echo "CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml CHECKPOINT=${SERVER_A_ROOT}/progressive_bottleneck/${ckpt} CACHE_PATH=${NAVTEST_HIGHCAP_CHUNK_ROOT} PDM_METRIC_CACHE_DIR=${PDM_METRIC_CACHE_DIR} scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh"
  done
  echo
  echo "# Server B checkpoint sweep commands; use LoRA-regenerated navtest cache"
  for ckpt in "${CHECKPOINTS[@]}"; do
    echo "CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml CHECKPOINT=${SERVER_B_ROOT}/progressive_bottleneck/${ckpt} CACHE_PATH=${B_NAVTEST_CACHE} PDM_METRIC_CACHE_DIR=${PDM_METRIC_CACHE_DIR} scripts/last_vla_v2/highcap_no_risk/eval_checkpoint_sweep_highcap.sh"
  done
} >"${PREP_DIR}/checkpoint_sweep_commands.txt"

for mode in normal zero_all_cot zero_geometry_cot zero_dynamic_cot zero_ego_cot zero_action_refine_cot zero_coarse_prior drop_vlm_summary; do
  cat >"${PREP_DIR}/corruption_${mode}.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
CORRUPTION_MODE="${mode}" \\
CONFIG=configs/last_vla_v2/last_vla_progressive_bottleneck_eval.yaml \\
CACHE_PATH="${B_NAVTEST_CACHE}" \\
PDM_METRIC_CACHE_DIR="${PDM_METRIC_CACHE_DIR}" \\
scripts/last_vla_v2/highcap_no_risk/eval_cot_corruption_highcap.sh
EOF
  chmod +x "${PREP_DIR}/corruption_${mode}.sh"
done

cat >"${PREP_DIR}/README.md" <<EOF
# Last-VLA v2 High-cap Eval Prep

- Full eval was not run by this script.
- Server A eval cache: ${NAVTEST_HIGHCAP_CHUNK_ROOT}
- Server B eval cache must be regenerated with: ${PREP_DIR}/build_serverB_lora_navtest_cache.sh
- Server B must not be evaluated on the original hidden cache.
- Baseline: A0-official-aligned PDMS=0.864891
EOF

echo "Wrote eval preparation files to ${PREP_DIR}"
