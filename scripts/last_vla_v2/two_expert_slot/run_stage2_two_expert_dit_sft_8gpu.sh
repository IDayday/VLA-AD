#!/usr/bin/env bash
set -Eeuo pipefail

required=(TRAIN_CHUNK_CACHE_ROOT OUTPUT_DIR MASTER_PORT)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-$(dirname "${PYTHON_BIN}")/torchrun}"
mkdir -p "${OUTPUT_DIR}/logs"
COMMANDS_LOG="${OUTPUT_DIR}/commands.log"
TRAIN_LOG="${OUTPUT_DIR}/logs/two_expert_stage2_dit_sft.train.log"

cmd=(
  "${TORCHRUN_BIN}" --nproc_per_node=8 --master_port "${MASTER_PORT}"
  navsim/planning/script/run_training_recogdrive.py
  +experiment=two_expert_slot_stage2_dit_sft
  "cache_path=${TRAIN_CHUNK_CACHE_ROOT}"
  use_cache_without_dataset=true
  force_cache_computation=false
  "train_test_split=${TRAIN_TEST_SPLIT:-navtrain}"
  "output_dir=${OUTPUT_DIR}"
  seed="${SEED:-0}"
  trainer.params.devices=8
  trainer.params.strategy=ddp_find_unused_parameters_true
)
if [[ -n "${A0_INIT_CHECKPOINT:-}" ]]; then cmd+=(agent.checkpoint_path="${A0_INIT_CHECKPOINT}"); fi

printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
if [[ "${RUN_TRAIN:-0}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; dry-run only. Command written to ${COMMANDS_LOG}"
  exit 0
fi
if [[ "${ALLOW_SKIP_READINESS_GATE:-0}" != "1" ]]; then
  if [[ -z "${READINESS_GATE_JSON:-}" ]]; then
    echo "READINESS_GATE_JSON is required for Stage2 full training unless ALLOW_SKIP_READINESS_GATE=1." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); ok=d.get("status")=="READY" and d.get("ok") is True; sys.exit(0 if ok else 1)' "${READINESS_GATE_JSON}" || {
    echo "Readiness gate is not READY: ${READINESS_GATE_JSON}" >&2
    exit 2
  }
  if [[ "${ALLOW_LOW_COVERAGE:-0}" != "1" ]]; then
    "${PYTHON_BIN}" -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); ok=d.get("coverage_ok") is True and d.get("all_teacher_coverage") is not None; sys.exit(0 if ok else 1)' "${READINESS_GATE_JSON}" || {
      echo "Readiness gate JSON lacks passing coverage. Run preflight_two_expert_coverage.py and rebuild the gate, or set ALLOW_LOW_COVERAGE=1 for explicit dev-only override." >&2
      exit 2
    }
  fi
fi
"${cmd[@]}" >"${TRAIN_LOG}" 2>&1
