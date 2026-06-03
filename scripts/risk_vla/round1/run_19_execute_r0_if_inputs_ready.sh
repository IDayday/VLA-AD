#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
MANIFEST_YAML="${MANIFEST_YAML:-${ROUND_DIR}/round1_input_manifest.local.yaml}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
MAX_SAMPLES="${MAX_SAMPLES:-1024}"
SPLIT="${SPLIT:-navval}"

echo "[run_19] manifest: ${MANIFEST_YAML}"
echo "[run_19] EXECUTE=${EXECUTE}"

if [[ ! -f "${MANIFEST_YAML}" ]]; then
  echo "[run_19] manifest missing; run run_18_materialize_or_register_pdm_inputs.sh first." >&2
  exit 1
fi

python scripts/risk_vla/validate_round1_input_manifest.py \
  --input-yaml "${MANIFEST_YAML}" \
  --output-json "${ROUND_DIR}/input_manifest_resolved.json" \
  --output-md "${ROUND_DIR}/input_manifest_resolved.md"

python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${ROUND_DIR}/input_manifest_resolved.json").read_text())
trainval = data.get("trainval_pdm", {})
missing = [name for name in ("train", "val") if not trainval.get(name, {}).get("csv")]
if missing:
    raise SystemExit(f"Missing train/val PDM entries required for R0: {missing}")
for name, entry in trainval.items():
    split = str(entry.get("split", "")).lower()
    if any(marker in split for marker in ("test", "navtest", "challenge", "eval-only")):
        raise SystemExit(f"Blocked train-time split for {name}: {split}")
PY

if [[ "${EXECUTE}" != "1" ]]; then
  echo "[run_19] EXECUTE=0; would run small R0 diagnostic only."
  echo "[run_19] enforced: use_risk_vla=true, strategy scales=0, oracle=false, risk loss > 0."
  exit 0
fi

EXECUTE=1 \
MAX_EPOCHS="${MAX_EPOCHS}" \
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES}" \
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES}" \
MAX_SAMPLES="${MAX_SAMPLES}" \
SPLIT="${SPLIT}" \
MANIFEST_YAML="${MANIFEST_YAML}" \
ROUND_DIR="${ROUND_DIR}" \
bash scripts/risk_vla/round1/run_16_execute_round1_r0_from_manifest.sh
