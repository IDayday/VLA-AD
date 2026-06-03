#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
BIT_CACHE_ROOT="${BIT_CACHE_ROOT:-/mnt/project/bit_drive_left_tail/cache}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
R0_DIR="${R0_DIR:-${ROUND_DIR}/R0_risk_head_diagnostic}"
CACHE_PATH="${CACHE_PATH:-${BIT_CACHE_ROOT}/risk_vla_round1_labeled_overlay}"
SPLIT="${SPLIT:-navval}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cpu}"
OUTPUT_CSV="${OUTPUT_CSV:-${R0_DIR}/predictions_${SPLIT}.csv}"
ARTIFACT_JSON="${R0_DIR}/r0_artifacts.json"
ARTIFACT_MD="${R0_DIR}/r0_artifacts.md"

echo "[run_11] output csv: ${OUTPUT_CSV}"

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[run_11] EXECUTE=0; would resolve R0 checkpoint and export predictions."
  exit 0
fi

python scripts/risk_vla/resolve_r0_diagnostic_artifacts.py --r0-output-dir "${R0_DIR}" --output-json "${ARTIFACT_JSON}" --output-md "${ARTIFACT_MD}"
CHECKPOINT="${CHECKPOINT:-$(python - <<PY
import json
from pathlib import Path
data=json.loads(Path("${ARTIFACT_JSON}").read_text())
print(data.get("best_checkpoint") or data.get("latest_checkpoint") or "")
PY
)}"

CMD=(python scripts/risk_vla/export_risk_vla_predictions.py --split "${SPLIT}" --output-csv "${OUTPUT_CSV}" --max-samples "${MAX_SAMPLES}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --device "${DEVICE}")
if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}" --cache-path "${CACHE_PATH}")
else
  echo "[run_11] no checkpoint found; running synthetic smoke export."
  CMD+=(--synthetic-smoke)
fi
printf '[run_11] command: '; printf '%q ' "${CMD[@]}"; echo
"${CMD[@]}"
