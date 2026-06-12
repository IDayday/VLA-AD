#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
REPORT_DIR="${REPORT_DIR:-reports/two_expert_slot}"
mkdir -p "${REPORT_DIR}"

cmd=("${PYTHON_BIN}" scripts/audit_two_expert_teacher_cache.py)
if [[ -n "${TEACHER_CACHE_ROOT:-}" ]]; then cmd+=(--cache-root "${TEACHER_CACHE_ROOT}"); fi
if [[ -n "${JEPA_CACHE_ROOT:-}" ]]; then cmd+=(--jepa-cache-root "${JEPA_CACHE_ROOT}"); fi
if [[ -n "${VGGT_CACHE_ROOT:-}" ]]; then cmd+=(--vggt-cache-root "${VGGT_CACHE_ROOT}"); fi
if [[ "${STRICT_TEACHER:-0}" == "1" ]]; then cmd+=(--strict); fi
if [[ -n "${EXPECTED_TOTAL:-}" ]]; then cmd+=(--expected-total "${EXPECTED_TOTAL}"); fi
cmd+=(--min-coverage "${MIN_COVERAGE:-0.99}" --json-out "${REPORT_DIR}/two_expert_teacher_cache_audit.json")

printf '%q ' "${cmd[@]}" >"${REPORT_DIR}/audit_teacher_cache.command"
printf '\n' >>"${REPORT_DIR}/audit_teacher_cache.command"
"${cmd[@]}"
