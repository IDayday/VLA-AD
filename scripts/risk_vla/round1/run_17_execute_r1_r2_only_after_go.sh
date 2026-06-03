#!/usr/bin/env bash
set -euo pipefail

EXECUTE="${EXECUTE:-0}"
BIT_EXP_ROOT="${BIT_EXP_ROOT:-/mnt/project/bit_drive_left_tail/experiments/risk_vla}"
ROUND_DIR="${ROUND_DIR:-${BIT_EXP_ROOT}/round1}"
GO_NO_GO_REPORT="${GO_NO_GO_REPORT:-${ROUND_DIR}/go_no_go_report.md}"

decision="$(GO_NO_GO_REPORT="${GO_NO_GO_REPORT}" python - <<'PY'
import json
import os
import re
from pathlib import Path
path = Path(os.environ["GO_NO_GO_REPORT"])
if not path.is_file():
    print("NO_GO_FIX_EXPORT")
elif path.suffix.lower() == ".json":
    data = json.loads(path.read_text())
    print(data.get("decision", "NO_GO_FIX_EXPORT"))
else:
    text = path.read_text()
    match = re.search(r"Decision:\s*`?([A-Z0-9_]+)`?", text)
    print(match.group(1) if match else "NO_GO_FIX_EXPORT")
PY
)"

echo "[run_17] GO/NO-GO report: ${GO_NO_GO_REPORT}"
echo "[run_17] decision: ${decision}"

case "${decision}" in
  GO_R1_R2)
    echo "[run_17] R1 and R2 are permitted if EXECUTE=1."
    echo "ORACLE ROUTER IS ANALYSIS-ONLY. DO NOT USE TEST LABELS FOR TRAINING CLAIMS."
    EXECUTE="${EXECUTE}" bash scripts/risk_vla/round1/run_13_execute_r1_oracle_router_small.sh
    EXECUTE="${EXECUTE}" bash scripts/risk_vla/round1/run_14_execute_r2_predicted_router_small.sh
    ;;
  GO_R1_ONLY)
    echo "[run_17] R1 oracle-router analysis is permitted; R2 predicted-router is blocked."
    echo "ORACLE ROUTER IS ANALYSIS-ONLY. DO NOT USE TEST LABELS FOR TRAINING CLAIMS."
    EXECUTE="${EXECUTE}" bash scripts/risk_vla/round1/run_13_execute_r1_oracle_router_small.sh
    echo "[run_17] R2 blocked by GO_R1_ONLY."
    ;;
  NO_GO_FIX_LABELS|NO_GO_FIX_EXPORT)
    echo "[run_17] R2 blocked by ${decision}. R1/R2 execution not launched."
    ;;
  *)
    echo "[run_17] Unknown decision ${decision}; blocking R2."
    exit 1
    ;;
esac
