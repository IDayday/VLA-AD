#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OPD_PYTHON="${OPD_PYTHON:-/root/miniconda3/envs/navsim/bin/python}"
case "${1:-check}" in
  check)
    "$OPD_PYTHON" -m pytest tests/opd -q --junitxml=/mnt/project/VLA-AD/outputs/recogdrive_opd_20261020/tests/cpu.xml
    "$OPD_PYTHON" scripts/opd/final_audit.py tests
    ;;
  formal) "$OPD_PYTHON" scripts/opd/suite.py formal ;;
  test) "$OPD_PYTHON" scripts/opd/suite.py test ;;
  e6) "$OPD_PYTHON" scripts/opd/suite.py e6 ;;
  report)
    "$OPD_PYTHON" scripts/opd/analyze.py --phase baseline --split dev
    "$OPD_PYTHON" scripts/opd/analyze.py --split dev
    "$OPD_PYTHON" scripts/opd/diagnostics.py chains
    "$OPD_PYTHON" scripts/opd/figures.py --split dev --cases
    "$OPD_PYTHON" scripts/opd/report.py
    "$OPD_PYTHON" scripts/opd/final_audit.py accounting
    ;;
  *) echo 'Usage: run_suite.sh [check|formal|test|e6|report]' >&2; exit 2 ;;
esac
