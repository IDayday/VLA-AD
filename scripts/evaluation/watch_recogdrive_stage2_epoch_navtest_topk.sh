#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the training output directory containing epoch_*.ckpt}"
: "${OUT_ROOT:?Set OUT_ROOT to the evaluation output directory}"

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/project/VLA-AD_last_vla_dev}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
EPOCH_MIN="${EPOCH_MIN:-100}"
EPOCH_MAX="${EPOCH_MAX:-200}"
TOP_K="${TOP_K:-3}"
POLL_SECONDS="${POLL_SECONDS:-900}"
STABLE_SECONDS="${STABLE_SECONDS:-180}"
NUM_SHARDS="${NUM_SHARDS:-2}"
GPUS_CSV="${GPUS_CSV:-1,2}"
PRECISION="${PRECISION:-fp32}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/sg_fps_asmi_stage2_fs_norm_eval.yaml}"
CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT:-/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1}"
CHUNK_NAME_PATTERN="${CHUNK_NAME_PATTERN:-navtest_full_chunk_*}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1}"
TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY:-pred_traj}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/state" "${OUT_ROOT}/top${TOP_K}_ckpts"
WATCH_LOG="${OUT_ROOT}/logs/watcher.log"
STATE_FILE="${OUT_ROOT}/state/evaluated_ckpts.txt"
SUMMARY_TSV="${OUT_ROOT}/navtest_summary.tsv"
TOP_TSV="${OUT_ROOT}/top${TOP_K}_summary.tsv"
touch "${STATE_FILE}"

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date -Is)" "${msg}" | tee -a "${WATCH_LOG}"
}

epoch_from_path() {
  local name
  name="$(basename "$1")"
  name="${name#epoch_}"
  name="${name%.ckpt}"
  [[ "${name}" =~ ^[0-9]+$ ]] || return 1
  printf '%d\n' "$((10#${name}))"
}

is_stable_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local age size1 size2
  age=$(( $(date +%s) - $(stat -c %Y "${path}") ))
  (( age >= STABLE_SECONDS )) || return 1
  size1="$(stat -c %s "${path}")"
  sleep 2
  size2="$(stat -c %s "${path}")"
  [[ "${size1}" == "${size2}" && "${size1}" -gt 0 ]]
}

refresh_topk() {
  [[ -f "${SUMMARY_TSV}" ]] || return 0
  "${PYTHON_BIN}" - "${SUMMARY_TSV}" "${OUT_ROOT}/top${TOP_K}_ckpts" "${TOP_TSV}" "${TOP_K}" <<'PY'
import csv
import os
import shutil
import sys
from pathlib import Path

summary = Path(sys.argv[1])
top_dir = Path(sys.argv[2])
top_tsv = Path(sys.argv[3])
top_k = int(sys.argv[4])
rows = []
with summary.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        try:
            row["_pdms"] = float(row.get("PDMS") or "nan")
        except ValueError:
            continue
        ckpt = Path(row.get("checkpoint_path") or "")
        if ckpt.is_file():
            rows.append(row)
rows.sort(key=lambda row: row["_pdms"], reverse=True)
top = rows[:top_k]
top_dir.mkdir(parents=True, exist_ok=True)

active_names = set()
for rank, row in enumerate(top, 1):
    ckpt = Path(row["checkpoint_path"])
    dst = top_dir / f"rank{rank}_{row['checkpoint']}"
    active_names.add(dst.name)
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        try:
            os.link(ckpt, tmp)
        except OSError:
            shutil.copy2(ckpt, tmp)
        tmp.replace(dst)

for old in top_dir.glob("rank*_*.ckpt"):
    if old.name not in active_names:
        old.unlink()

with top_tsv.open("w", encoding="utf-8", newline="") as f:
    fieldnames = [
        "rank",
        "checkpoint",
        "checkpoint_path",
        "PDMS",
        "NC",
        "DAC",
        "TTC",
        "comfort",
        "EP",
        "DDC",
        "trajectory_l1",
        "num_samples",
        "num_pdm_valid",
        "eval_dir",
        "backup_path",
    ]
    writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
    writer.writeheader()
    for rank, row in enumerate(top, 1):
        out = {key: row.get(key, "") for key in fieldnames}
        out["rank"] = str(rank)
        out["backup_path"] = str(top_dir / f"rank{rank}_{row['checkpoint']}")
        writer.writerow(out)
PY
}

log "watcher started run_root=${RUN_ROOT} out_root=${OUT_ROOT} epoch_min=${EPOCH_MIN} epoch_max=${EPOCH_MAX} gpus=${GPUS_CSV} num_shards=${NUM_SHARDS} config=${CONFIG}"

while true; do
  mapfile -t candidates < <(
    find "${RUN_ROOT}" -maxdepth 1 -type f -name 'epoch_*.ckpt' -printf '%f\t%p\n' 2>/dev/null \
      | sort -V \
      | while IFS=$'\t' read -r _ path; do
          epoch="$(epoch_from_path "${path}" || true)"
          [[ -n "${epoch:-}" ]] || continue
          (( epoch >= EPOCH_MIN && epoch <= EPOCH_MAX )) || continue
          grep -Fxq "${path}" "${STATE_FILE}" && continue
          printf '%s\n' "${path}"
        done
  )

  if (( ${#candidates[@]} == 0 )); then
    refresh_topk
    log "no pending stable candidates; sleeping ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
    continue
  fi

  for ckpt in "${candidates[@]}"; do
    if ! is_stable_file "${ckpt}"; then
      log "skip unstable ckpt=${ckpt}"
      continue
    fi
    epoch="$(epoch_from_path "${ckpt}")"
    log "evaluating epoch=${epoch} ckpt=${ckpt}"
    set +e
    OUT_ROOT="${OUT_ROOT}" \
      CHECKPOINTS="${ckpt}" \
      PROJECT_ROOT="${PROJECT_ROOT}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      NUM_SHARDS="${NUM_SHARDS}" \
      GPUS_CSV="${GPUS_CSV}" \
      PRECISION="${PRECISION}" \
      SKIP_COMPLETED="${SKIP_COMPLETED}" \
      CONFIG="${CONFIG}" \
      CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT}" \
      CHUNK_NAME_PATTERN="${CHUNK_NAME_PATTERN}" \
      METRIC_CACHE_DIR="${METRIC_CACHE_DIR}" \
      TRAJECTORY_OUTPUT_KEY="${TRAJECTORY_OUTPUT_KEY}" \
      bash "${PROJECT_ROOT}/scripts/evaluation/run_recogdrive_stage2_navtest_sharded_ckpts.sh" \
      >> "${WATCH_LOG}" 2>&1
    status=$?
    set -e
    if (( status == 0 )); then
      echo "${ckpt}" >> "${STATE_FILE}"
      refresh_topk
      log "completed epoch=${epoch} ckpt=${ckpt}"
    else
      log "evaluation failed status=${status} epoch=${epoch} ckpt=${ckpt}; will retry later"
    fi
  done
done
