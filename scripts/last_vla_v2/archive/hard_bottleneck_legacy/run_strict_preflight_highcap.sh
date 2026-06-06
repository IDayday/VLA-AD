#!/usr/bin/env bash
set -Eeuo pipefail

required=(FULL_GEOMETRY_CHUNK_ROOT OUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PREFLIGHT_DIR="${OUT_ROOT}/highcap_no_risk_preflight"
mkdir -p "${PREFLIGHT_DIR}/logs"
COMMANDS_LOG="${PREFLIGHT_DIR}/commands.log"
MANIFEST="${PREFLIGHT_DIR}/highcap_no_risk_manifest.json"

cmd_manifest=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${FULL_GEOMETRY_CHUNK_ROOT}"
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
  tests/test_last_vla_highcap_no_risk_shapes.py
  tests/test_last_vla_highcap_no_risk_cache_schema.py
  tests/test_last_vla_highcap_config_compose.py
)

{
  date -Is
  printf '%q ' "${cmd_manifest[@]}"; printf '\n'
  printf '%q ' "${cmd_tests[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

cat >"${PREFLIGHT_DIR}/readiness.md" <<EOF
# Last-VLA v2 High-cap No-risk Preflight

- A0-official-aligned baseline PDMS: \`0.864891\`
- Full geometry cache: \`${FULL_GEOMETRY_CHUNK_ROOT}\`
- Manifest: \`${MANIFEST}\`
- Training launched: no
- Full eval launched: no
EOF

if [[ "${RUN_PREFLIGHT:-0}" != "1" ]]; then
  echo "RUN_PREFLIGHT is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
  exit 0
fi
"${cmd_manifest[@]}" 2>&1 | tee "${PREFLIGHT_DIR}/logs/manifest.log"
"${cmd_tests[@]}" 2>&1 | tee "${PREFLIGHT_DIR}/logs/tests.log"
