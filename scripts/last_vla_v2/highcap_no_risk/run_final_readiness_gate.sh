#!/usr/bin/env bash
set -Eeuo pipefail

OUT_ROOT="${OUT_ROOT:-/tmp/last_vla_highcap_readiness}"
READINESS_DIR="${OUT_ROOT}/readiness"
REPORT="${READINESS_DIR}/final_readiness_report.md"
MANIFEST="${READINESS_DIR}/train_manifest.json"
mkdir -p "${READINESS_DIR}/logs"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
BLOCKERS=()

add_blocker() {
  BLOCKERS+=("$1")
}

if [[ -z "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-}" ]]; then
  add_blocker "FULL_HIGHCAP_TRAIN_CHUNK_ROOT is not set."
elif [[ ! -d "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]]; then
  add_blocker "FULL_HIGHCAP_TRAIN_CHUNK_ROOT does not exist: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}"
fi

if [[ -z "${A0_INIT_CHECKPOINT:-}" ]]; then
  add_blocker "A0_INIT_CHECKPOINT is not set."
elif [[ ! -e "${A0_INIT_CHECKPOINT}" ]]; then
  add_blocker "A0_INIT_CHECKPOINT does not exist: ${A0_INIT_CHECKPOINT}"
fi

OBSOLETE_PRESENT=()
for path in \
  configs/last_vla_v2/last_vla_cot_alignment_geometry_lite.yaml \
  configs/last_vla_v2/last_vla_teacher_traj_sft.yaml \
  configs/last_vla_v2/last_vla_teacher_traj_sft_eval.yaml \
  navsim/planning/script/config/experiment/last_vla_teacher_traj_sft.yaml \
  scripts/last_vla_v2/round2/serverA_frozen_vlm_full_sft.sh \
  scripts/last_vla_v2/round2/serverB_vlm_lora_full_sft.sh \
  scripts/last_vla_v2/run_teacher_traj_sft_8gpu.sh; do
  if [[ -e "${path}" ]]; then OBSOLETE_PRESENT+=("${path}"); fi
done
if (( ${#OBSOLETE_PRESENT[@]} > 0 )); then
  add_blocker "Obsolete Last-VLA production configs/scripts still exist: ${OBSOLETE_PRESENT[*]}"
fi

if grep -RE "num_jepa_tokens:[[:space:]]*12([[:space:]]|$)|last_vla_cot_num_tokens:[[:space:]]*32([[:space:]]|$)|last_vla_vlm_summary_tokens:[[:space:]]*4([[:space:]]|$)|last_vla_allow_patch_geometry_fallback:[[:space:]]*true([[:space:]]|$)" \
  configs/last_vla_v2 --exclude-dir=archive >/dev/null 2>&1; then
  add_blocker "Production configs still contain minimal or patch_fallback values."
fi

AUDIT_STATUS="not_run"
if [[ -n "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-}" && -d "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" ]]; then
  set +e
  "${PYTHON_BIN}" scripts/audit_last_vla_cache_manifest.py \
    --cache-root "${FULL_HIGHCAP_TRAIN_CHUNK_ROOT}" \
    --chunk-name-pattern "${TRAIN_CHUNK_NAME_PATTERN:-train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*}" \
    --strict-full-geometry \
    --min-full-geometry-coverage 0.99 \
    --expected-jepa-tokens 128 \
    --expected-geometry-tokens 192 \
    --geometry-teacher-dim 512 \
    --strict-no-risk \
    --output "${MANIFEST}" >"${READINESS_DIR}/logs/audit.log" 2>&1
  status=$?
  set -e
  if [[ "${status}" == "0" ]]; then
    AUDIT_STATUS="pass"
  else
    AUDIT_STATUS="fail"
    add_blocker "Strict high-cap cache audit failed. See ${READINESS_DIR}/logs/audit.log"
  fi
fi

TEST_STATUS="not_run"
set +e
"${PYTHON_BIN}" -m pytest -q \
  tests/test_last_vla_highcap_config_compose.py \
  tests/test_last_vla_lora_scope_filter.py \
  tests/test_last_vla_lora_target_resolution.py \
  tests/test_last_vla_lora_hidden_anchor_loss.py >"${READINESS_DIR}/logs/tests.log" 2>&1
test_status=$?
set -e
if [[ "${test_status}" == "0" ]]; then
  TEST_STATUS="pass"
else
  TEST_STATUS="fail"
  add_blocker "Readiness pytest subset failed. See ${READINESS_DIR}/logs/tests.log"
fi

DRYRUN_STATUS="not_run"
set +e
LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS=1 RUN_TRAIN=0 \
FULL_GEOMETRY_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-/tmp/fake_highcap_cache}" \
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-/tmp/fake_a0.ckpt}" \
OUT_ROOT="${READINESS_DIR}/dryrun" MASTER_PORT="${MASTER_PORT_LOCAL:-29531}" \
scripts/last_vla_v2/highcap_no_risk/serverA_frozen_vlm_highcap_no_risk.sh >"${READINESS_DIR}/logs/serverA_dryrun.log" 2>&1
a_status=$?
LAST_VLA_DRY_RUN_ALLOW_MISSING_PATHS=1 RUN_TRAIN=0 \
FULL_GEOMETRY_CHUNK_ROOT="${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-/tmp/fake_highcap_cache}" \
A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT:-/tmp/fake_a0.ckpt}" \
VLM_PATH="${VLM_PATH:-/tmp/fake_vlm}" \
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-/tmp/fake_logs}" \
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-/tmp/fake_blobs}" \
OUT_ROOT="${READINESS_DIR}/dryrun" MASTER_PORT="${MASTER_PORT_REMOTE:-29541}" \
scripts/last_vla_v2/highcap_no_risk/serverB_vlm_lora_highcap_no_risk.sh >"${READINESS_DIR}/logs/serverB_dryrun.log" 2>&1
b_status=$?
set -e
if [[ "${a_status}" == "0" && "${b_status}" == "0" ]]; then
  DRYRUN_STATUS="pass"
else
  DRYRUN_STATUS="fail"
  add_blocker "Server A/B high-cap dry-run failed. See ${READINESS_DIR}/logs/server*_dryrun.log"
fi

READY="READY"
if (( ${#BLOCKERS[@]} > 0 )); then READY="NOT READY"; fi

{
  echo "# Last-VLA v2 High-cap No-risk Final Readiness"
  echo
  echo "- Status: ${READY}"
  echo "- Baseline: A0-official-aligned PDMS=0.864891"
  echo "- A0 checkpoint: ${A0_INIT_CHECKPOINT:-missing}"
  echo "- Train cache: ${FULL_HIGHCAP_TRAIN_CHUNK_ROOT:-missing}"
  echo "- Cache audit: ${AUDIT_STATUS}"
  echo "- Readiness tests: ${TEST_STATUS}"
  echo "- Launcher dry-run: ${DRYRUN_STATUS}"
  echo "- Training launched: no"
  echo "- Full eval launched: no"
  echo
  echo "## Blockers"
  if (( ${#BLOCKERS[@]} == 0 )); then
    echo "- none"
  else
    for item in "${BLOCKERS[@]}"; do echo "- ${item}"; done
  fi
} >"${REPORT}"

cat "${REPORT}"
if [[ "${READY}" != "READY" ]]; then exit 2; fi
