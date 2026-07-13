#!/usr/bin/env bash
set -Eeuo pipefail

required=(OUT_ROOT NAVTEST_CHUNK_CACHE_ROOT METRIC_CACHE_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
CONFIG="${CONFIG:-configs/last_vla_v2/two_expert_slot/stage2_dit_sft.yaml}"
EVAL_ROOT="${EVAL_ROOT:-${OUT_ROOT}/two_expert_slot_eval}"
COMMANDS_LOG="${EVAL_ROOT}/commands.log"
mkdir -p "${EVAL_ROOT}/logs"

collect_ckpts() {
  if [[ -n "${CHECKPOINT_LIST:-}" && -f "${CHECKPOINT_LIST}" ]]; then
    grep -v '^[[:space:]]*$' "${CHECKPOINT_LIST}"
  else
    find "${OUT_ROOT}" -path '*/checkpoints/*.ckpt' -type f | sort
  fi
}

while IFS= read -r ckpt; do
  [[ -z "${ckpt}" ]] && continue
  name="$(basename "${ckpt}" .ckpt)"
  out_dir="${EVAL_ROOT}/${name}"
  mkdir -p "${out_dir}"
  cmd=(
    "${PYTHON_BIN}" scripts/eval_recogdrive_expert_pdm.py
    --config "${CONFIG}"
    --checkpoint "${ckpt}"
    --chunk-cache-root "${NAVTEST_CHUNK_CACHE_ROOT}"
    --chunk-name-pattern "${NAVTEST_CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
    --metric-cache-dir "${METRIC_CACHE_DIR}"
    --precision "${PRECISION:-fp32}"
    --output-dir "${out_dir}"
  )
  printf '%q ' "${cmd[@]}" >>"${COMMANDS_LOG}"; printf '\n' >>"${COMMANDS_LOG}"
  if [[ "${RUN_EVAL:-0}" == "1" ]]; then
    "${cmd[@]}" >"${out_dir}/eval.log" 2>&1
  fi
done < <(collect_ckpts)

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  echo "RUN_EVAL is not 1; dry-run only. Commands written to ${COMMANDS_LOG}"
fi
