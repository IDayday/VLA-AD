#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_HIGHCAP_TRAIN_CHUNK_ROOT OUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PREFLIGHT_DIR="${OUT_ROOT}/decoupled_highcap_no_risk_preflight"
mkdir -p "${PREFLIGHT_DIR}/logs"
COMMANDS_LOG="${PREFLIGHT_DIR}/commands.log"
MANIFEST="${PREFLIGHT_DIR}/train_manifest.json"
READINESS="${PREFLIGHT_DIR}/readiness.md"

cmd_manifest=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output "${MANIFEST}"
  --strict-full-geometry
  --strict-no-risk
  --min-full-geometry-coverage "${MIN_FULL_GEOMETRY_COVERAGE:-0.99}"
  --expected-jepa-tokens 128
  --expected-geometry-tokens 192
  --geometry-teacher-dim 512
)
cmd_tests=(
  "${PYTHON_BIN}" -m pytest -q
  tests/test_last_vla_decoupled_config_compose.py
  tests/test_last_vla_decoupled_cot_shapes.py
  tests/test_dit_decoupled_cot_branch_zero_init.py
)

command_string() {
  printf '%q ' "$@"
}

write_not_ready() {
  local failed_command="$1"
  local log_path="$2"
  local exit_code="$3"
  local blockers="$4"
  cat >"${READINESS}" <<EOF
# Last-VLA v2 Decoupled HighCap NoRisk Preflight

Status: NOT READY

- Design: ReCogDrive-LaST-v2 Decoupled HighCap NoRisk
- A0-official-aligned baseline PDMS: \`0.864891\`
- Cache path: \`${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}\`
- Manifest path: \`${MANIFEST}\`
- Failed command: \`${failed_command}\`
- Log path: \`${log_path}\`
- Blocker list: ${blockers}
- Exit code: \`${exit_code}\`
- Training launched: no
- Full eval launched: no
EOF
}

run_checked() {
  local log_path="$1"
  shift
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  return "${status}"
}

{
  date -Is
  printf '%q ' "${cmd_manifest[@]}"; printf '\n'
  printf '%q ' "${cmd_tests[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

if [[ "${RUN_PREFLIGHT:-0}" != "1" ]]; then
  write_not_ready "dry-run: $(command_string "${cmd_manifest[@]}")" "${COMMANDS_LOG}" "0" "RUN_PREFLIGHT is not 1; commands were written but not executed."
  echo "RUN_PREFLIGHT is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi

manifest_log="${PREFLIGHT_DIR}/logs/manifest.log"
tests_log="${PREFLIGHT_DIR}/logs/tests.log"

set +e
run_checked "${manifest_log}" "${cmd_manifest[@]}"
code=$?
set -e
if (( code != 0 )); then
  write_not_ready "$(command_string "${cmd_manifest[@]}")" "${manifest_log}" "${code}" "cache manifest audit failed."
  exit "${code}"
fi
set +e
run_checked "${tests_log}" "${cmd_tests[@]}"
code=$?
set -e
if (( code != 0 )); then
  write_not_ready "$(command_string "${cmd_tests[@]}")" "${tests_log}" "${code}" "preflight pytest failed."
  exit "${code}"
fi

set +e
"${PYTHON_BIN}" - "${MANIFEST}" "${READINESS}" "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
readiness_path = Path(sys.argv[2])
cache_path = sys.argv[3]
report = json.loads(manifest_path.read_text(encoding="utf-8"))
mode_dist = report.get("vggt_geometry_mode_distribution") or {}
risk = report.get("risk_label_coverage") or {}
risk_count = sum(int(item.get("count", 0)) for item in risk.values() if isinstance(item, dict))
full_geometry_coverage = float(report.get("full_geometry_coverage", 0.0))
patch_fallback_count = int(mode_dist.get("patch_fallback", 0))
ready = bool(report.get("ready_for_highcap_no_risk_strict"))
status = "READY" if ready else "NOT READY"
blockers = []
if not ready:
    blockers.append("ready_for_highcap_no_risk_strict=false")
if patch_fallback_count:
    blockers.append(f"patch_fallback_count={patch_fallback_count}")
if risk_count:
    blockers.append(f"risk_label_count={risk_count}")
if report.get("num_errors", 0):
    blockers.append(f"num_errors={report.get('num_errors')}")
readiness_path.write_text(
    "\n".join(
        [
            "# Last-VLA v2 Decoupled HighCap NoRisk Preflight",
            "",
            f"Status: {status}",
            "",
            "- Design: ReCogDrive-LaST-v2 Decoupled HighCap NoRisk",
            "- A0-official-aligned baseline PDMS: `0.864891`",
            f"- Cache path: `{cache_path}`",
            f"- Manifest path: `{manifest_path}`",
            "- Expected JEPA shape: `[128,1024]`",
            "- Expected VGGT context shape: `[128,2048]`",
            "- Expected geometry shape: `[192,512]`",
            f"- Full geometry coverage: `{full_geometry_coverage:.6f}`",
            f"- Patch fallback count: `{patch_fallback_count}`",
            f"- No risk confirmed: `{risk_count == 0}`",
            "- Tests passed: `yes`",
            f"- Blocker list: {', '.join(blockers) if blockers else 'none'}",
            "- Training launched: no",
            "- Full eval launched: no",
            "",
        ]
    ),
    encoding="utf-8",
)
sys.exit(0 if ready else 5)
PY
code=$?
set -e
if (( code != 0 )); then
  write_not_ready "readiness manifest parse" "${MANIFEST}" "${code}" "manifest passed execution but readiness criteria were not satisfied."
  exit "${code}"
fi

echo "Preflight READY. Report written to ${READINESS}"
