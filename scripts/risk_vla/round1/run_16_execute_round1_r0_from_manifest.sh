#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
SPLIT="${SPLIT:-navval}"
DEVICE="${DEVICE:-cpu}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
MANIFEST_YAML="${MANIFEST_YAML:-configs/risk_vla/round1_input_manifest.yaml}"
R0_DIR="${R0_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
LABEL_DIR="${LABEL_DIR:-${ROUND_DIR}/risk_labels}"
MANIFEST_JSON="${ROUND_DIR}/input_manifest_resolved.json"
MANIFEST_MD="${ROUND_DIR}/input_manifest_resolved.md"

mkdir -p "${ROUND_DIR}" "${LABEL_DIR}" "${R0_DIR}"

echo "[run_16] manifest: ${MANIFEST_YAML}"
python scripts/risk_vla/validate_round1_input_manifest.py \
  --manifest-yaml "${MANIFEST_YAML}" \
  --output-json "${MANIFEST_JSON}" \
  --output-md "${MANIFEST_MD}"

readarray -t TRAINVAL_ROWS < <(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
for name, entry in sorted(data.get("trainval_pdm", {}).items()):
    if entry.get("csv"):
        print(f"{name}\t{entry.get('csv')}\t{entry.get('split') or name}")
PY
)

SOURCE_CACHE_DIR="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
print(data.get("cache", {}).get("source_chunk_cache_dir") or "")
PY
)"
OVERLAY_OUTPUT_DIR="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
print(data.get("cache", {}).get("overlay_output_dir") or "${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay")
PY
)"
CHECKPOINT_PATH="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
print(data.get("checkpoints", {}).get("base_or_bit_checkpoint") or "")
PY
)"
A0_CSV="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
print(data.get("found_inputs", {}).get("a0_pdm_csv") or "")
PY
)"
B3_CSV="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${MANIFEST_JSON}").read_text())
print(data.get("found_inputs", {}).get("bit_pdm_csv") or "")
PY
)"

echo "[run_16] source cache: ${SOURCE_CACHE_DIR}"
echo "[run_16] overlay: ${OVERLAY_OUTPUT_DIR}"
echo "[run_16] checkpoint: ${CHECKPOINT_PATH}"
echo "[run_16] train/val PDM rows: ${#TRAINVAL_ROWS[@]}"
echo "[run_16] R0 diagnostic config is enforced: use_risk_vla=true, strategy scales=0, oracle=false."

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[run_16] EXECUTE=0; no labels, overlay, training, export, or aggregation will be run."
  echo "[run_16] Sequence if executed:"
  echo "  1. build train/val risk labels with split metadata"
  echo "  2. run leakage guard"
  echo "  3. copy/inject labeled overlay without mutating source cache"
  echo "  4. run small R0 risk-head diagnostic only"
  echo "  5. resolve artifacts, export ${SPLIT} predictions, aggregate diagnostics"
  echo "  6. run GO/NO-GO, build first-result tables and evidence report"
  exit 0
fi

if [[ ${#TRAINVAL_ROWS[@]} -eq 0 ]]; then
  echo "[run_16] no trainval_pdm entries available; cannot run R0." >&2
  exit 1
fi

LABEL_FILES=()
for row in "${TRAINVAL_ROWS[@]}"; do
  IFS=$'\t' read -r name csv split <<<"${row}"
  jsonl="${LABEL_DIR}/${name}_risk_labels.jsonl"
  csv_out="${LABEL_DIR}/${name}_risk_label_summary.csv"
  md_out="${LABEL_DIR}/${name}_risk_label_report.md"
  python scripts/risk_vla/build_risk_labels_from_pdm.py \
    --input-csv "${csv}" \
    --output-jsonl "${jsonl}" \
    --output-csv "${csv_out}" \
    --output-md "${md_out}" \
    --schema both \
    --split "${split}"
  LABEL_FILES+=("${jsonl}")
done

cat "${LABEL_FILES[@]}" >"${LABEL_DIR}/risk_labels.jsonl"
python scripts/risk_vla/check_label_leakage.py \
  --labels-jsonl "${LABEL_DIR}/risk_labels.jsonl" \
  --for-training \
  --output-md "${LABEL_DIR}/leakage_check.md"

python scripts/risk_vla/create_labeled_chunk_cache_overlay.py \
  --source-cache-dir "${SOURCE_CACHE_DIR}" \
  --output-cache-dir "${OVERLAY_OUTPUT_DIR}" \
  --labels-jsonl "${LABEL_DIR}/risk_labels.jsonl" \
  --mode copy \
  --max-samples "${MAX_SAMPLES}" \
  --write \
  --overwrite

EXECUTE=1 \
ROUND_DIR="${ROUND_DIR}" \
R0_DIR="${R0_DIR}" \
CACHE_PATH="${OVERLAY_OUTPUT_DIR}" \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
MAX_EPOCHS="${MAX_EPOCHS}" \
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES}" \
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
bash scripts/risk_vla/round1/run_10_execute_r0_diagnostic_small.sh

python scripts/risk_vla/resolve_r0_diagnostic_artifacts.py \
  --r0-output-dir "${R0_DIR}" \
  --output-json "${R0_DIR}/r0_artifacts.json" \
  --output-md "${R0_DIR}/r0_artifacts.md"

EXECUTE=1 \
ROUND_DIR="${ROUND_DIR}" \
R0_DIR="${R0_DIR}" \
CACHE_PATH="${OVERLAY_OUTPUT_DIR}" \
SPLIT="${SPLIT}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
DEVICE="${DEVICE}" \
bash scripts/risk_vla/round1/run_11_export_r0_predictions_small.sh

EXECUTE=1 \
ROUND_DIR="${ROUND_DIR}" \
R0_DIR="${R0_DIR}" \
SPLIT="${SPLIT}" \
LABELS_JSONL="${LABEL_DIR}/risk_labels.jsonl" \
bash scripts/risk_vla/round1/run_12_aggregate_r0_diagnostics.sh

DIAG_DIR="${R0_DIR}/diagnostics_${SPLIT}"
cp "${DIAG_DIR}/risk_vla_diagnostics.md" "${R0_DIR}/risk_vla_diagnostics.md"
cp "${DIAG_DIR}/risk_prediction_metrics.csv" "${R0_DIR}/risk_prediction_metrics.csv"
cp "${DIAG_DIR}/strategy_activation_by_subset.csv" "${R0_DIR}/strategy_activation_by_subset.csv"

python scripts/risk_vla/round1/check_round1_go_no_go.py \
  --risk-metrics-csv "${DIAG_DIR}/risk_prediction_metrics.csv" \
  --strategy-activation-csv "${DIAG_DIR}/strategy_activation_by_subset.csv" \
  --output-md "${R0_DIR}/go_no_go_report.md"
cp "${R0_DIR}/go_no_go_report.md" "${ROUND_DIR}/go_no_go_report.md"

TRANSITION_CSV=""
if [[ -n "${A0_CSV}" && -n "${B3_CSV}" ]]; then
  TRANSITION_CSV="${ROUND_DIR}/matched_pdm/a0_b3_transition.csv"
  python scripts/risk_vla/build_risk_transition_matrix.py \
    --base-csv "${A0_CSV}" \
    --method-csv "${B3_CSV}" \
    --method-name B3_direct_bit \
    --output-csv "${TRANSITION_CSV}" \
    --output-md "${ROUND_DIR}/matched_pdm/a0_b3_transition.md"
fi

FIRST_RESULT_ARGS=(
  python scripts/risk_vla/build_round1_first_result_tables.py
  --risk-metrics-csv "${DIAG_DIR}/risk_prediction_metrics.csv"
  --strategy-activation-csv "${DIAG_DIR}/strategy_activation_by_subset.csv"
  --go-no-go-md "${ROUND_DIR}/go_no_go_report.md"
  --output-dir "${ROUND_DIR}/first_result_tables"
)
if [[ -n "${TRANSITION_CSV}" ]]; then
  FIRST_RESULT_ARGS+=(--matched-analysis-csv "${TRANSITION_CSV}")
fi
"${FIRST_RESULT_ARGS[@]}"

python scripts/risk_vla/build_round1_evidence_report.py \
  --input-manifest-md "${MANIFEST_MD}" \
  --r0-diagnostics-md "${DIAG_DIR}/risk_vla_diagnostics.md" \
  --go-no-go-md "${ROUND_DIR}/go_no_go_report.md" \
  --first-result-summary-md "${ROUND_DIR}/first_result_tables/round1_first_result_summary.md" \
  --output-md "${ROUND_DIR}/ROUND1_EVIDENCE_REPORT.md"

echo "[run_16] R0 evidence loop complete. R1/R2 were not run."
