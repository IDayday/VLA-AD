#!/usr/bin/env bash
set -euo pipefail

BENCH2DRIVE_ROOT=${BENCH2DRIVE_ROOT:?Set BENCH2DRIVE_ROOT to the official Bench2Drive repo path}
ROUTE_JSON_DIR=${ROUTE_JSON_DIR:-${BENCH2DRIVE_ROOT}/recogdrive_b2d_only_traj}
METRIC_DIR=${METRIC_DIR:-${SAVE_PATH:-${PWD}/outputs/bench2drive_recogdrive_closed_loop/bench2drive220}}
MERGED_JSON=${MERGED_JSON:-${ROUTE_JSON_DIR%/}/merged.json}

cd "${BENCH2DRIVE_ROOT}"
python tools/merge_route_json.py -f "${ROUTE_JSON_DIR}"
python tools/ability_benchmark.py -r "${MERGED_JSON}"
python tools/efficiency_smoothness_benchmark.py -f "${MERGED_JSON}" -m "${METRIC_DIR}"
