#!/usr/bin/env bash
set -Eeuo pipefail

required=(
  HIDDEN_CACHE_ROOT
  HIDDEN_CACHE_LOG_ROOT
  BASE_CHUNK_ROOT
  VLM_PATH
  READINESS_GATE_JSON
  A0_INIT_CHECKPOINT
  STAGE2_OUTPUT_ROOT
  MASTER_PORT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
EXPECTED_SAMPLES="${EXPECTED_SAMPLES:-103036}"
POLL_SECONDS="${POLL_SECONDS:-600}"
AUDIT_MAX_RECORDS="${AUDIT_MAX_RECORDS:-2048}"
SPLIT="${SPLIT:-navtrain}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
STAGE2_LAUNCHER="${STAGE2_LAUNCHER:-scripts/last_vla_v2/two_expert_slot/run_stage2_two_expert_dit_sft_8gpu.sh}"

cd "${PROJECT_ROOT}"

echo "wait_hidden_cache_then_run_stage2 started at $(date -u)"
echo "hidden_cache_root=${HIDDEN_CACHE_ROOT}"
echo "hidden_cache_log_root=${HIDDEN_CACHE_LOG_ROOT}"
echo "stage2_output_root=${STAGE2_OUTPUT_ROOT}"

wait_for_cache_shards() {
  while true; do
    local running=0
    for pid_file in "${HIDDEN_CACHE_LOG_ROOT}"/shard_*.pid; do
      [[ -e "${pid_file}" ]] || continue
      local pid
      pid="$(cat "${pid_file}")"
      if ps -p "${pid}" >/dev/null 2>&1; then
        running=$((running + 1))
      fi
    done
    echo "$(date -u) cache_shards_running=${running}"
    if [[ "${running}" -eq 0 ]]; then
      break
    fi
    sleep "${POLL_SECONDS}"
  done
}

check_shards() {
  local total=0
  for shard_dir in "${HIDDEN_CACHE_ROOT}"/shards/shard_*; do
    [[ -d "${shard_dir}" ]] || continue
    if [[ ! -f "${shard_dir}/metadata.json" || ! -f "${shard_dir}/index.jsonl" ]]; then
      echo "Shard missing metadata or index: ${shard_dir}" >&2
      exit 3
    fi
    local lines
    lines="$(wc -l < "${shard_dir}/index.jsonl")"
    total=$((total + lines))
    echo "shard=$(basename "${shard_dir}") samples=${lines}"
  done
  echo "hidden_cache_shard_total=${total}"
  if [[ "${EXPECTED_SAMPLES}" != "0" && "${total}" -ne "${EXPECTED_SAMPLES}" ]]; then
    echo "Expected ${EXPECTED_SAMPLES} hidden-cache samples, got ${total}." >&2
    exit 4
  fi
  if rg -n "Traceback|RuntimeError|CUDA out of memory|Killed" "${HIDDEN_CACHE_LOG_ROOT}"/shard_*.log; then
    echo "Found errors in hidden-cache shard logs." >&2
    exit 5
  fi
}

merge_cache() {
  RUN_CACHE=1 "${PYTHON_BIN}" scripts/last_vla_v2/two_expert_slot/build_two_expert_hidden_cache.py \
    --base-chunk-root "${BASE_CHUNK_ROOT}" \
    --output-root "${HIDDEN_CACHE_ROOT}" \
    --vlm-path "${VLM_PATH}" \
    --merge \
    --overwrite
}

audit_cache() {
  "${PYTHON_BIN}" scripts/audit_two_expert_hidden_cache.py \
    "${HIDDEN_CACHE_ROOT}" \
    --split "${SPLIT}" \
    --max-records "${AUDIT_MAX_RECORDS}" \
    --json-out "${HIDDEN_CACHE_ROOT}/audit_${AUDIT_MAX_RECORDS}.json"
  "${PYTHON_BIN}" - "${HIDDEN_CACHE_ROOT}/audit_${AUDIT_MAX_RECORDS}.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if not payload.get("ok"):
    raise SystemExit(f"Hidden-cache audit failed: {payload}")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

run_stage2() {
  mkdir -p "${STAGE2_OUTPUT_ROOT}/logs"
  TRAIN_CHUNK_CACHE_ROOT="${HIDDEN_CACHE_ROOT}" \
  OUTPUT_DIR="${STAGE2_OUTPUT_ROOT}" \
  MASTER_PORT="${MASTER_PORT}" \
  READINESS_GATE_JSON="${READINESS_GATE_JSON}" \
  A0_INIT_CHECKPOINT="${A0_INIT_CHECKPOINT}" \
  TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT}" \
  RUN_TRAIN=1 \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  bash "${STAGE2_LAUNCHER}"
}

wait_for_cache_shards
check_shards
merge_cache
audit_cache
run_stage2

