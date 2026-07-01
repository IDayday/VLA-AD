#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"

STAGE="${STAGE:-stage2}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the PSI ${STAGE} run root.}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${RUN_ROOT}/checkpoints/raw}"
EVAL_SPLIT="${EVAL_SPLIT:-val6000}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluation/run_psi_stage2_pdm_eval_8gpu_exact_pool.sh}"
POLL_SECONDS="${POLL_SECONDS:-300}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
EXIT_ONCE="${EXIT_ONCE:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
CHECKPOINT_LIST_TSV="${CHECKPOINT_LIST_TSV:-}"
CHECKPOINT_LIST_COLUMN="${CHECKPOINT_LIST_COLUMN:-object_path}"
CHECKPOINT_LIST_TOP_K="${CHECKPOINT_LIST_TOP_K:-0}"
CHECKPOINT_LIST_MIN_ROWS="${CHECKPOINT_LIST_MIN_ROWS:-0}"
CHECKPOINT_MIN_EPOCH="${CHECKPOINT_MIN_EPOCH:-0}"
CHECKPOINT_MAX_EPOCH="${CHECKPOINT_MAX_EPOCH:-0}"
CHECKPOINT_EPOCH_ONLY="${CHECKPOINT_EPOCH_ONLY:-0}"
CHECKPOINT_STEP_ONLY="${CHECKPOINT_STEP_ONLY:-0}"
EVAL_MIN_CHECKPOINT_STEP="${EVAL_MIN_CHECKPOINT_STEP:-0}"
EVAL_CHECKPOINT_STEP_INTERVAL="${EVAL_CHECKPOINT_STEP_INTERVAL:-0}"
CHECKPOINT_ORDER="${CHECKPOINT_ORDER:-asc}"
ARCHIVE_ALL_BEFORE_EVAL="${ARCHIVE_ALL_BEFORE_EVAL:-1}"
MAX_EVALS_PER_POLL="${MAX_EVALS_PER_POLL:-0}"
BACKUP_RANKED_CHECKPOINTS="${BACKUP_RANKED_CHECKPOINTS:-0}"
BACKUP_TOP_K="${BACKUP_TOP_K:-}"
BACKUP_ROOT="${BACKUP_ROOT:-${RUN_ROOT}/checkpoint_backups}"
RANK_MIN_EPOCH="${RANK_MIN_EPOCH:-${CHECKPOINT_MIN_EPOCH}}"
RANK_MAX_EPOCH="${RANK_MAX_EPOCH:-${CHECKPOINT_MAX_EPOCH}}"
WAIT_FOR_STATUS_TSV="${WAIT_FOR_STATUS_TSV:-}"
WAIT_FOR_STATUS_EXPECTED="${WAIT_FOR_STATUS_EXPECTED:-}"
WAIT_FOR_STATUS_LABEL="${WAIT_FOR_STATUS_LABEL:-precondition}"
WAIT_FOR_STATUS_FAIL_ON_FAILED="${WAIT_FOR_STATUS_FAIL_ON_FAILED:-1}"
WAIT_FOR_STATUS_MIN_EPOCH="${WAIT_FOR_STATUS_MIN_EPOCH:-${CHECKPOINT_MIN_EPOCH}}"
WAIT_FOR_STATUS_MAX_EPOCH="${WAIT_FOR_STATUS_MAX_EPOCH:-${CHECKPOINT_MAX_EPOCH}}"
EVAL_HOST_ID="${EVAL_HOST_ID:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo local)}"
WATCH_LOCK_SCOPE="${WATCH_LOCK_SCOPE:-global}"
EVAL_RESOURCE_LOCK_SCOPE="${EVAL_RESOURCE_LOCK_SCOPE:-global}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-1}"
GPU_MAX_MEM_USED_MB="${GPU_MAX_MEM_USED_MB:-2000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS:-2}"
ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND:-process}"
ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD:-spawn}"
ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE:-$((ASYNC_PDM_WORKERS * 2))}"
ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY:-100}"
ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE:-0}"
ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE:-1}"
PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER:-exact_pool}"
FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR:-}"
MAX_SCENES="${MAX_SCENES:-0}"
DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-3600}"
EVAL_MASTER_PORT_MIN="${EVAL_MASTER_PORT_MIN:-63700}"
EVAL_MASTER_PORT_MAX="${EVAL_MASTER_PORT_MAX:-64500}"

case "${EVAL_SPLIT}" in
  val6000|navtest) ;;
  *)
    echo "EVAL_SPLIT must be val6000 or navtest, got: ${EVAL_SPLIT}" >&2
    exit 2
    ;;
esac

case "${CHECKPOINT_ORDER}" in
  asc|desc) ;;
  *)
    echo "CHECKPOINT_ORDER must be asc or desc, got: ${CHECKPOINT_ORDER}" >&2
    exit 2
    ;;
esac

case "${WATCH_LOCK_SCOPE}" in
  global|host) ;;
  *)
    echo "WATCH_LOCK_SCOPE must be global or host, got: ${WATCH_LOCK_SCOPE}" >&2
    exit 2
    ;;
esac

case "${EVAL_RESOURCE_LOCK_SCOPE}" in
  global|host|none) ;;
  *)
    echo "EVAL_RESOURCE_LOCK_SCOPE must be global, host, or none, got: ${EVAL_RESOURCE_LOCK_SCOPE}" >&2
    exit 2
    ;;
esac

mkdir -p \
  "${RUN_ROOT}/checkpoint_store/objects" \
  "${RUN_ROOT}/rankings/${EVAL_SPLIT}/history" \
  "${RUN_ROOT}/eval/${EVAL_SPLIT}" \
  "${RUN_ROOT}/state" \
  "${RUN_ROOT}/logs"

WATCH_LOG="${RUN_ROOT}/logs/watch_${STAGE}_eval_${EVAL_SPLIT}.log"

if [[ "${WATCH_LOCK_SCOPE}" == "host" ]]; then
  WATCH_LOCK="${RUN_ROOT}/state/watch_${STAGE}_eval_${EVAL_SPLIT}.${EVAL_HOST_ID}.lock"
else
  WATCH_LOCK="${RUN_ROOT}/state/watch_${STAGE}_eval_${EVAL_SPLIT}.lock"
fi

exec 9>"${WATCH_LOCK}"
if ! flock -n 9; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) another ${STAGE} eval watcher holds ${EVAL_SPLIT} lock for ${RUN_ROOT} lock=${WATCH_LOCK}" | tee -a "${WATCH_LOG}"
  exit 0
fi

SUMMARY_TSV="${RUN_ROOT}/eval/${EVAL_SPLIT}/summary.tsv"
STATUS_TSV="${RUN_ROOT}/eval/${EVAL_SPLIT}/checkpoint_eval_status.tsv"
GLOBAL_EVAL_LOCK="${RUN_ROOT}/state/${STAGE}_global_eval.lock"
POSTPROCESS_LOCK="${RUN_ROOT}/state/${STAGE}_${EVAL_SPLIT}_postprocess.lock"

if [[ ! -f "${STATUS_TSV}" ]]; then
  printf 'timestamp\tcheckpoint_id\tsha256\tstate\tcheckpoint\tobject_path\teval_dir\trc\n' > "${STATUS_TSV}"
fi

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "${WATCH_LOG}"
}

eval_resource_lock_path() {
  case "${EVAL_RESOURCE_LOCK_SCOPE}" in
    global)
      printf '%s\n' "${GLOBAL_EVAL_LOCK}"
      ;;
    host)
      printf '%s\n' "${RUN_ROOT}/state/${STAGE}_${EVAL_HOST_ID}_eval.lock"
      ;;
    none)
      printf '\n'
      ;;
  esac
}

status_gate_target_count() {
  if [[ -n "${WAIT_FOR_STATUS_EXPECTED}" ]]; then
    printf '%s\n' "${WAIT_FOR_STATUS_EXPECTED}"
    return
  fi
  if [[ -d "${CHECKPOINT_ROOT}" ]]; then
    "${PYTHON_BIN}" - "${CHECKPOINT_ROOT}" "${WAIT_FOR_STATUS_MIN_EPOCH}" "${WAIT_FOR_STATUS_MAX_EPOCH}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_epoch = int(sys.argv[2])
max_epoch = int(sys.argv[3])

def infer_epoch(path: Path):
    for text in (path.name, str(path)):
        for pattern in (r"(?:^|[/_.-])epoch[_=-]?(\d+)(?:\D|$)", r"(?:^|[/_.-])ep[_=-]?(\d+)(?:\D|$)"):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

count = 0
for path in root.rglob("*.ckpt"):
    epoch = infer_epoch(path)
    if min_epoch > 0 and (epoch is None or epoch < min_epoch):
        continue
    if max_epoch > 0 and (epoch is None or epoch > max_epoch):
        continue
    count += 1
print(count)
PY
  else
    printf '0\n'
  fi
}

status_gate_ready() {
  [[ -n "${WAIT_FOR_STATUS_TSV}" ]] || return 0
  local target done failed started
  target="$(status_gate_target_count)"
  if [[ ! "${target}" =~ ^[0-9]+$ ]]; then
    log "invalid WAIT_FOR_STATUS_EXPECTED=${target}"
    exit 2
  fi
  if [[ ! -f "${WAIT_FOR_STATUS_TSV}" ]]; then
    log "waiting for ${WAIT_FOR_STATUS_LABEL}: missing status file ${WAIT_FOR_STATUS_TSV}"
    return 1
  fi
  IFS=$'\t' read -r done failed started < <(
    "${PYTHON_BIN}" - "${WAIT_FOR_STATUS_TSV}" "${WAIT_FOR_STATUS_MIN_EPOCH}" "${WAIT_FOR_STATUS_MAX_EPOCH}" <<'PY'
import csv
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_epoch = int(sys.argv[2])
max_epoch = int(sys.argv[3])

def infer_epoch(*texts):
    for text in texts:
        if not text:
            continue
        for pattern in (r"(?:^|[/_.-])epoch[_=-]?(\d+)(?:\D|$)", r"(?:^|[/_.-])ep[_=-]?(\d+)(?:\D|$)"):
            match = re.search(pattern, str(text), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

latest = {}
with path.open("r", encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        sha = row.get("sha256", "")
        state = row.get("state", "")
        if not sha or not state:
            continue
        epoch = infer_epoch(
            row.get("checkpoint_id"),
            row.get("checkpoint"),
            row.get("object_path"),
            row.get("eval_dir"),
        )
        if min_epoch > 0 and (epoch is None or epoch < min_epoch):
            continue
        if max_epoch > 0 and (epoch is None or epoch > max_epoch):
            continue
        latest[sha] = state

done = sum(1 for state in latest.values() if state == "done")
failed = sum(1 for state in latest.values() if state == "failed")
started = sum(1 for state in latest.values() if state == "started")
print(f"{done}\t{failed}\t{started}")
PY
  )
  done="${done:-0}"
  failed="${failed:-0}"
  started="${started:-0}"
  if [[ "${WAIT_FOR_STATUS_FAIL_ON_FAILED}" == "1" && "${failed}" -gt 0 ]]; then
    log "${WAIT_FOR_STATUS_LABEL} has failed checkpoint evals failed=${failed}; refusing ${EVAL_SPLIT} eval"
    exit 3
  fi
  if [[ "${target}" -gt 0 && "${done}" -ge "${target}" ]]; then
    return 0
  fi
  log "waiting for ${WAIT_FOR_STATUS_LABEL}: done=${done}/${target} started=${started} failed=${failed}"
  return 1
}

checkpoint_source_ready() {
  if [[ -n "${CHECKPOINT_LIST_TSV}" ]]; then
    if [[ ! -f "${CHECKPOINT_LIST_TSV}" ]]; then
      log "waiting for checkpoint list: ${CHECKPOINT_LIST_TSV}"
      return 1
    fi
    local count
    count="$("${PYTHON_BIN}" - "${CHECKPOINT_LIST_TSV}" "${CHECKPOINT_LIST_COLUMN}" "${CHECKPOINT_LIST_TOP_K}" "${CHECKPOINT_MIN_EPOCH}" "${CHECKPOINT_MAX_EPOCH}" <<'PY'
import csv
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
column = sys.argv[2]
top_k = int(sys.argv[3])
min_epoch = int(sys.argv[4])
max_epoch = int(sys.argv[5])

def infer_epoch(*texts):
    for text in texts:
        if not text:
            continue
        for pattern in (r"(?:^|[/_.-])epoch[_=-]?(\d+)(?:\D|$)", r"(?:^|[/_.-])ep[_=-]?(\d+)(?:\D|$)"):
            match = re.search(pattern, str(text), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

with path.open("r", encoding="utf-8", newline="") as fh:
    rows = []
    for row in csv.DictReader(fh, delimiter="\t"):
        if not row.get(column):
            continue
        epoch = infer_epoch(
            row.get("epoch"),
            row.get("checkpoint_id"),
            row.get("source_path"),
            row.get("checkpoint"),
            row.get("eval_dir"),
            row.get(column),
        )
        if min_epoch > 0 and (epoch is None or epoch < min_epoch):
            continue
        if max_epoch > 0 and (epoch is None or epoch > max_epoch):
            continue
        rows.append(row)
if top_k > 0:
    rows = rows[:top_k]
print(len(rows))
PY
)"
    if [[ "${CHECKPOINT_LIST_MIN_ROWS}" -gt 0 && "${count}" -lt "${CHECKPOINT_LIST_MIN_ROWS}" ]]; then
      log "waiting for checkpoint list rows ${count}/${CHECKPOINT_LIST_MIN_ROWS}: ${CHECKPOINT_LIST_TSV}"
      return 1
    fi
    return 0
  fi
  [[ -d "${CHECKPOINT_ROOT}" ]]
}

iter_checkpoint_paths() {
  if [[ -n "${CHECKPOINT_LIST_TSV}" ]]; then
    "${PYTHON_BIN}" - "${CHECKPOINT_LIST_TSV}" "${CHECKPOINT_LIST_COLUMN}" "${CHECKPOINT_LIST_TOP_K}" "${CHECKPOINT_MIN_EPOCH}" "${CHECKPOINT_MAX_EPOCH}" <<'PY'
import csv
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
column = sys.argv[2]
top_k = int(sys.argv[3])
min_epoch = int(sys.argv[4])
max_epoch = int(sys.argv[5])

def infer_epoch(*texts):
    for text in texts:
        if not text:
            continue
        for pattern in (r"(?:^|[/_.-])epoch[_=-]?(\d+)(?:\D|$)", r"(?:^|[/_.-])ep[_=-]?(\d+)(?:\D|$)"):
            match = re.search(pattern, str(text), flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
    return None

with path.open("r", encoding="utf-8", newline="") as fh:
    rows = []
    for row in csv.DictReader(fh, delimiter="\t"):
        if not row.get(column):
            continue
        epoch = infer_epoch(
            row.get("epoch"),
            row.get("checkpoint_id"),
            row.get("source_path"),
            row.get("checkpoint"),
            row.get("eval_dir"),
            row.get(column),
        )
        if min_epoch > 0 and (epoch is None or epoch < min_epoch):
            continue
        if max_epoch > 0 and (epoch is None or epoch > max_epoch):
            continue
        rows.append(row)
if top_k > 0:
    rows = rows[:top_k]
for row in rows:
    print(row[column])
PY
  else
    "${PYTHON_BIN}" - "${CHECKPOINT_ROOT}" "${CHECKPOINT_MIN_EPOCH}" "${CHECKPOINT_MAX_EPOCH}" "${CHECKPOINT_ORDER}" "${CHECKPOINT_EPOCH_ONLY}" "${CHECKPOINT_STEP_ONLY}" "${EVAL_MIN_CHECKPOINT_STEP}" "${EVAL_CHECKPOINT_STEP_INTERVAL}" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_epoch = int(sys.argv[2])
max_epoch = int(sys.argv[3])
order = sys.argv[4]
epoch_only = sys.argv[5] == "1"
step_only = sys.argv[6] == "1"
min_step = int(sys.argv[7])
step_interval = int(sys.argv[8])

def infer_epoch(path: Path):
    # Only inspect the checkpoint filename. Run directories may include tokens like
    # epoch011 even when the checkpoint itself is a step checkpoint.
    text = path.name
    for pattern in (r"(?:^|[/_.-])epoch[_=-]?(\d+)(?:\D|$)", r"(?:^|[/_.-])ep[_=-]?(\d+)(?:\D|$)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def infer_step(path: Path):
    text = path.name
    for pattern in (r"(?:^|[/_.-])step[-_=]+(?:step[-_=]+)?(\d+)(?:\D|$)", r"(?:^|[/_.-])step[_=-]?(\d+)(?:\D|$)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None

def is_epoch_checkpoint(path: Path):
    return infer_epoch(path) is not None

def is_step_checkpoint(path: Path):
    name = path.name.lower()
    return infer_step(path) is not None and not is_epoch_checkpoint(path) and name.startswith("step")

items = []
for path in root.rglob("*.ckpt"):
    epoch = infer_epoch(path)
    step = infer_step(path)
    if epoch_only and epoch is None:
        continue
    if step_only and not is_step_checkpoint(path):
        continue
    if min_epoch > 0 and (epoch is None or epoch < min_epoch):
        continue
    if max_epoch > 0 and (epoch is None or epoch > max_epoch):
        continue
    if min_step > 0 and (step is None or step < min_step):
        continue
    if step_interval > 0 and (step is None or step % step_interval != 0):
        continue
    items.append((step if step is not None else -1, epoch if epoch is not None else -1, str(path)))

items.sort(key=lambda item: (item[0], item[1], item[2]), reverse=(order == "desc"))
for _, _, path in items:
    print(path)
PY
  fi
}

checkpoint_id() {
  basename "$1" .ckpt | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/[=]+/_/g'
}

marker_path() {
  local sha="$1"
  local state="$2"
  echo "${RUN_ROOT}/state/${STAGE}_${EVAL_SPLIT}_${sha}.${state}"
}

checkpoint_stable() {
  local ckpt="$1"
  [[ -s "${ckpt}" ]] || return 1
  local age
  age=$(( $(date +%s) - $(stat -c '%Y' "${ckpt}") ))
  [[ "${age}" -ge "${STABLE_SECONDS}" ]]
}

gpu_ready() {
  if [[ "${WAIT_FOR_FREE_GPUS}" != "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - <<PY
import subprocess
import sys

gpu_list = {item.strip() for item in "${GPU_LIST}".split(",") if item.strip()}
max_mem = int("${GPU_MAX_MEM_USED_MB}")
max_util = int("${GPU_MAX_UTIL}")
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
busy = []
for line in out.strip().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        continue
    idx, mem, util = parts
    if idx not in gpu_list:
        continue
    if int(mem) > max_mem or int(util) > max_util:
        busy.append(f"{idx}:mem={mem}MB,util={util}%")
if busy:
    print("busy_gpus=" + ";".join(busy))
    sys.exit(1)
print("gpus_ready")
PY
}

append_status() {
  local checkpoint_id="$1"
  local sha="$2"
  local state="$3"
  local checkpoint="$4"
  local object_path="$5"
  local eval_dir="$6"
  local rc="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${checkpoint_id}" \
    "${sha}" \
    "${state}" \
    "${checkpoint}" \
    "${object_path}" \
    "${eval_dir}" \
    "${rc}" >> "${STATUS_TSV}"
}

archive_checkpoint() {
  local ckpt="$1"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/archive_checkpoint_immutable.py" \
    --run-root "${RUN_ROOT}" \
    --checkpoint "${ckpt}" \
    --stage "${STAGE}" \
    --stable-seconds 0
}

lookup_archived_checkpoint() {
  local ckpt="$1"
  "${PYTHON_BIN}" - "${RUN_ROOT}" "${ckpt}" <<'PY'
import csv
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
inventory_path = run_root / "checkpoint_store" / "inventory.tsv"
if not inventory_path.is_file():
    sys.exit(0)

with inventory_path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        object_path = row.get("object_path", "")
        sha256 = row.get("sha256", "")
        if checkpoint not in {row.get("source_path"), object_path} or not sha256 or not object_path:
            continue
        if Path(object_path).is_file():
            print(f"{sha256}\t{object_path}")
            break
PY
}

archive_checkpoint_if_needed() {
  local ckpt="$1"
  checkpoint_stable "${ckpt}" || return 0
  local archived_info archive_out sha
  archived_info="$(lookup_archived_checkpoint "${ckpt}")"
  [[ -n "${archived_info}" ]] && return 0
  archive_out="$(archive_checkpoint "${ckpt}")"
  log "${archive_out}"
  sha="$(printf '%s\n' "${archive_out}" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p' | tail -1)"
  if [[ -z "${sha}" ]]; then
    log "failed to parse archive sha for ${ckpt}"
  fi
}

process_checkpoint() {
  local ckpt="$1"
  checkpoint_stable "${ckpt}" || return 0

  local archive_out archived_info sha object_path id eval_dir done_marker failed_marker running_marker claim_lock resource_lock
  archived_info="$(lookup_archived_checkpoint "${ckpt}")"
  if [[ -n "${archived_info}" ]]; then
    IFS=$'\t' read -r sha object_path <<< "${archived_info}"
  else
    archive_out="$(archive_checkpoint "${ckpt}")"
    log "${archive_out}"
    sha="$(printf '%s\n' "${archive_out}" | sed -n 's/.*sha256=\([0-9a-f]*\).*/\1/p' | tail -1)"
    if [[ -z "${sha}" ]]; then
      log "failed to parse archive sha for ${ckpt}"
      return 0
    fi
    object_path="${RUN_ROOT}/checkpoint_store/objects/${sha}.ckpt"
  fi
  id="$(checkpoint_id "${ckpt}")"
  eval_dir="${RUN_ROOT}/eval/${EVAL_SPLIT}/${id}_${sha:0:12}"
  done_marker="$(marker_path "${sha}" done)"
  failed_marker="$(marker_path "${sha}" failed)"
  running_marker="$(marker_path "${sha}" running)"

  claim_lock="${RUN_ROOT}/state/${STAGE}_${EVAL_SPLIT}_${sha}.claim.lock"
  exec 7>"${claim_lock}"
  if ! flock -n 7; then
    exec 7>&- || true
    return 0
  fi

  if [[ -f "${done_marker}" ]]; then
    flock -u 7 || true
    exec 7>&- || true
    return 0
  fi
  if [[ -f "${failed_marker}" && "${RETRY_FAILED}" != "1" ]]; then
    flock -u 7 || true
    exec 7>&- || true
    return 0
  fi
  if [[ -f "${running_marker}" ]]; then
    flock -u 7 || true
    exec 7>&- || true
    return 0
  fi

  resource_lock="$(eval_resource_lock_path)"
  if [[ -n "${resource_lock}" ]]; then
    exec 8>"${resource_lock}"
    if ! flock -n 8; then
      log "eval resource lock busy scope=${EVAL_RESOURCE_LOCK_SCOPE} host=${EVAL_HOST_ID}; defer split=${EVAL_SPLIT} checkpoint=${ckpt}"
      exec 8>&- || true
      flock -u 7 || true
      exec 7>&- || true
      return 20
    fi
  fi
  if ! gpu_ready >> "${WATCH_LOG}" 2>&1; then
    log "GPUs busy; defer split=${EVAL_SPLIT} checkpoint=${ckpt}"
    if [[ -n "${resource_lock}" ]]; then
      flock -u 8 || true
      exec 8>&- || true
    fi
    flock -u 7 || true
    exec 7>&- || true
    return 21
  fi

  rm -f "${failed_marker}"
  touch "${running_marker}"
  append_status "${id}" "${sha}" "started" "${ckpt}" "${object_path}" "${eval_dir}" "-"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/update_checkpoint_inventory_eval_metrics.py" \
    --run-root "${RUN_ROOT}" \
    --split "${EVAL_SPLIT}" \
    --sha256 "${sha}" \
    --state started \
    >> "${WATCH_LOG}" 2>&1 || true
  mkdir -p "${eval_dir}"
  log "starting eval split=${EVAL_SPLIT} id=${id} sha=${sha} out=${eval_dir}"
  flock -u 7 || true
  exec 7>&- || true

  local master_port rc
  master_port="$("${PYTHON_BIN}" - <<PY
import random
print(random.randint(int("${EVAL_MASTER_PORT_MIN}"), int("${EVAL_MASTER_PORT_MAX}")))
PY
)"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    CHECKPOINT="${object_path}" \
    EVAL_SPLIT="${EVAL_SPLIT}" \
    OUT_ROOT="${eval_dir}" \
    GPUS_PER_NODE="${GPUS_PER_NODE}" \
    MASTER_PORT="${master_port}" \
    ASYNC_PDM_WORKERS="${ASYNC_PDM_WORKERS}" \
    ASYNC_PDM_BACKEND="${ASYNC_PDM_BACKEND}" \
    ASYNC_PDM_PROCESS_START_METHOD="${ASYNC_PDM_PROCESS_START_METHOD}" \
    ASYNC_PDM_QUEUE_SIZE="${ASYNC_PDM_QUEUE_SIZE}" \
    ASYNC_PDM_PROGRESS_EVERY="${ASYNC_PDM_PROGRESS_EVERY}" \
    ASYNC_PDM_PROFILE="${ASYNC_PDM_PROFILE}" \
    ASYNC_PDM_TASK_CHUNK_SIZE="${ASYNC_PDM_TASK_CHUNK_SIZE}" \
    PDM_EVAL_RUNNER="${PDM_EVAL_RUNNER}" \
    FAST_METRIC_CACHE_DIR="${FAST_METRIC_CACHE_DIR}" \
    MAX_SCENES="${MAX_SCENES}" \
    DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS}" \
    bash "${EVAL_SCRIPT}" > "${eval_dir}/eval.log" 2>&1
  rc=$?
  set -e

  exec 6>"${POSTPROCESS_LOCK}"
  flock 6
  rm -f "${running_marker}"
  if [[ "${rc}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluation/summarize_recogdrive_eval_submetrics.py" \
      --eval-dir "${eval_dir}" \
      --checkpoint-id "${sha}" \
      --summary-tsv "${SUMMARY_TSV}" >> "${eval_dir}/eval.log" 2>&1
	    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/rank_eval_checkpoints.py" \
	      --run-root "${RUN_ROOT}" \
	      --split "${EVAL_SPLIT}" \
	      --eval-summary "${SUMMARY_TSV}" \
	      --min-epoch "${RANK_MIN_EPOCH}" \
	      --max-epoch "${RANK_MAX_EPOCH}" >> "${eval_dir}/eval.log" 2>&1
	    if [[ "${BACKUP_RANKED_CHECKPOINTS}" == "1" ]]; then
	      local backup_top_k
	      backup_top_k="${BACKUP_TOP_K}"
	      if [[ -z "${backup_top_k}" ]]; then
	        if [[ "${EVAL_SPLIT}" == "navtest" ]]; then
	          backup_top_k=3
	        else
	          backup_top_k=5
	        fi
	      fi
	      "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/backup_ranked_checkpoints.py" \
	        --run-root "${RUN_ROOT}" \
	        --split "${EVAL_SPLIT}" \
	        --top-k "${backup_top_k}" \
	        --backup-root "${BACKUP_ROOT}" >> "${eval_dir}/eval.log" 2>&1 || \
	        log "ranked checkpoint backup not ready split=${EVAL_SPLIT} top_k=${backup_top_k}; see ${eval_dir}/eval.log"
	    fi
	    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/update_checkpoint_inventory_eval_metrics.py" \
	      --run-root "${RUN_ROOT}" \
      --split "${EVAL_SPLIT}" \
      --sha256 "${sha}" \
      --state done \
      --summary-tsv "${SUMMARY_TSV}" >> "${eval_dir}/eval.log" 2>&1
    touch "${done_marker}"
    append_status "${id}" "${sha}" "done" "${ckpt}" "${object_path}" "${eval_dir}" "${rc}"
    log "eval done split=${EVAL_SPLIT} id=${id} sha=${sha}"
  else
    touch "${failed_marker}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/checkpoints/update_checkpoint_inventory_eval_metrics.py" \
      --run-root "${RUN_ROOT}" \
      --split "${EVAL_SPLIT}" \
      --sha256 "${sha}" \
      --state failed \
      >> "${WATCH_LOG}" 2>&1 || true
    append_status "${id}" "${sha}" "failed" "${ckpt}" "${object_path}" "${eval_dir}" "${rc}"
    log "eval failed split=${EVAL_SPLIT} id=${id} rc=${rc}; see ${eval_dir}/eval.log"
  fi
  flock -u 6 || true
  exec 6>&- || true
  if [[ -n "${resource_lock}" ]]; then
    flock -u 8 || true
    exec 8>&- || true
  fi
  set +e
  return 30
}

log "watcher start stage=${STAGE} split=${EVAL_SPLIT} run_root=${RUN_ROOT} checkpoint_root=${CHECKPOINT_ROOT} checkpoint_min_epoch=${CHECKPOINT_MIN_EPOCH} checkpoint_max_epoch=${CHECKPOINT_MAX_EPOCH} checkpoint_epoch_only=${CHECKPOINT_EPOCH_ONLY} checkpoint_step_only=${CHECKPOINT_STEP_ONLY} eval_min_checkpoint_step=${EVAL_MIN_CHECKPOINT_STEP} eval_checkpoint_step_interval=${EVAL_CHECKPOINT_STEP_INTERVAL} checkpoint_order=${CHECKPOINT_ORDER} rank_min_epoch=${RANK_MIN_EPOCH} rank_max_epoch=${RANK_MAX_EPOCH} archive_all_before_eval=${ARCHIVE_ALL_BEFORE_EVAL} max_evals_per_poll=${MAX_EVALS_PER_POLL} host=${EVAL_HOST_ID} watch_lock_scope=${WATCH_LOCK_SCOPE} eval_resource_lock_scope=${EVAL_RESOURCE_LOCK_SCOPE}"
if [[ -n "${CHECKPOINT_LIST_TSV}" ]]; then
  log "watcher checkpoint list split=${EVAL_SPLIT} list=${CHECKPOINT_LIST_TSV} column=${CHECKPOINT_LIST_COLUMN} top_k=${CHECKPOINT_LIST_TOP_K} min_rows=${CHECKPOINT_LIST_MIN_ROWS}"
fi

while true; do
  if ! status_gate_ready; then
    :
  elif checkpoint_source_ready; then
    if [[ "${ARCHIVE_ALL_BEFORE_EVAL}" == "1" ]]; then
      while IFS= read -r ckpt; do
        [[ -n "${ckpt}" ]] || continue
        archive_checkpoint_if_needed "${ckpt}"
      done < <(iter_checkpoint_paths)
    fi
    evals_this_poll=0
    while IFS= read -r ckpt; do
      [[ -n "${ckpt}" ]] || continue
      set +e
      process_checkpoint "${ckpt}"
      rc=$?
      set -e
      if [[ "${rc}" -eq 20 || "${rc}" -eq 21 ]]; then
        log "defer remaining checkpoints this poll split=${EVAL_SPLIT} rc=${rc}"
        break
      fi
      if [[ "${rc}" -eq 30 ]]; then
        evals_this_poll=$((evals_this_poll + 1))
        if [[ "${MAX_EVALS_PER_POLL}" -gt 0 && "${evals_this_poll}" -ge "${MAX_EVALS_PER_POLL}" ]]; then
          log "max evals per poll reached split=${EVAL_SPLIT} count=${evals_this_poll}; refresh checkpoint list next poll"
          break
        fi
      fi
    done < <(iter_checkpoint_paths)
  else
    if [[ -n "${CHECKPOINT_LIST_TSV}" ]]; then
      log "waiting for checkpoint list: ${CHECKPOINT_LIST_TSV}"
    else
      log "waiting for checkpoint root: ${CHECKPOINT_ROOT}"
    fi
  fi
  if [[ "${EXIT_ONCE}" == "1" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}" 9>&- 8>&-
done
