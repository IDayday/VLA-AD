#!/usr/bin/env bash
set -Eeuo pipefail

required=(TRAIN_CHUNK_CACHE_ROOT FULL_GEOMETRY_CHUNK_ROOT A0_INIT_CHECKPOINT OUT_ROOT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done
[[ -d "${TRAIN_CHUNK_CACHE_ROOT}" ]] || { echo "TRAIN_CHUNK_CACHE_ROOT missing: ${TRAIN_CHUNK_CACHE_ROOT}" >&2; exit 2; }
[[ -d "${FULL_GEOMETRY_CHUNK_ROOT}" ]] || { echo "FULL_GEOMETRY_CHUNK_ROOT missing: ${FULL_GEOMETRY_CHUNK_ROOT}" >&2; exit 2; }
[[ -e "${A0_INIT_CHECKPOINT}" ]] || { echo "A0_INIT_CHECKPOINT missing: ${A0_INIT_CHECKPOINT}" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PREFLIGHT_DIR="${OUT_ROOT}/preflight"
mkdir -p "${PREFLIGHT_DIR}/logs"
COMMANDS_LOG="${PREFLIGHT_DIR}/commands.log"
MANIFEST="${PREFLIGHT_DIR}/full_geometry_manifest.json"

cmd_manifest=(
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py
  --cache-root "${FULL_GEOMETRY_CHUNK_ROOT}"
  --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}"
  --output "${MANIFEST}"
  --strict-full-geometry
  --min-full-geometry-coverage "${MIN_FULL_GEOMETRY_COVERAGE:-0.99}"
  --geometry-teacher-dim "${GEOMETRY_TEACHER_DIM:-512}"
)

cmd_tests=(
  "${PYTHON_BIN}" -m pytest -q
  tests/test_last_vla_config_compose.py
  tests/test_last_vla_no_future_leakage.py
  tests/test_no_future_leakage.py
)

{
  date -Is
  printf '%q ' "${cmd_manifest[@]}"; printf '\n'
  printf '%q ' "${cmd_tests[@]}"; printf '\n'
} >>"${COMMANDS_LOG}"

"${cmd_manifest[@]}" 2>&1 | tee "${PREFLIGHT_DIR}/logs/manifest.log"
"${cmd_tests[@]}" 2>&1 | tee "${PREFLIGHT_DIR}/logs/tests.log"

cat >"${PREFLIGHT_DIR}/readiness.md" <<EOF
# Last-VLA v2 Round2 Strict Preflight

- A0 checkpoint: \`${A0_INIT_CHECKPOINT}\`
- A0-official baseline PDMS: \`0.864891\`
- Full geometry cache: \`${FULL_GEOMETRY_CHUNK_ROOT}\`
- Manifest: \`${MANIFEST}\`
- Training launched: no
- Full eval launched: no
EOF

echo "Strict preflight complete: ${PREFLIGHT_DIR}/readiness.md"
