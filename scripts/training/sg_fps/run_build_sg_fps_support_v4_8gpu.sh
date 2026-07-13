#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"

: "${OUTPUT_PATH:?set OUTPUT_PATH to the shared SG-FPS v4 archive directory}"
: "${POLICY_CHECKPOINT:?set POLICY_CHECKPOINT to the Stage2 frontier checkpoint}"
: "${FS_NORM_STATS_PATH:?set FS_NORM_STATS_PATH to the checkpoint FS-Norm statistics}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
SHARD_COUNT="${SHARD_COUNT:-${#GPU_ARRAY[@]}}"
if [[ "${SHARD_COUNT}" -ne "${#GPU_ARRAY[@]}" ]]; then
  echo "SHARD_COUNT=${SHARD_COUNT} must equal the number of GPU_IDS (${#GPU_ARRAY[@]})." >&2
  exit 2
fi

MAX_SCENES="${MAX_SCENES:-0}"
EXPECTED_TOKEN_SOURCE="${EXPECTED_TOKEN_SOURCE:-}"
BUILD_LOG_SPLIT="${BUILD_LOG_SPLIT:-train_val}"
if [[ "${MAX_SCENES}" -eq 0 && -z "${EXPECTED_TOKEN_SOURCE}" ]]; then
  echo "A full build requires EXPECTED_TOKEN_SOURCE for exact token coverage validation." >&2
  exit 2
fi

EXPECTED_SCENE_COUNT="${EXPECTED_SCENE_COUNT:-0}"
if [[ "${MAX_SCENES}" -eq 0 && "${EXPECTED_SCENE_COUNT}" -eq 0 ]]; then
  if [[ -d "${EXPECTED_TOKEN_SOURCE}" ]]; then
    EXPECTED_SCENE_COUNT="$(find "${EXPECTED_TOKEN_SOURCE}" -maxdepth 1 -type f -name '*.pkl.xz' -printf '.' | wc -c)"
  else
    EXPECTED_SCENE_COUNT="$(
      EXPECTED_TOKEN_SOURCE="${EXPECTED_TOKEN_SOURCE}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["EXPECTED_TOKEN_SOURCE"])
text = path.read_text(encoding="utf-8").strip()
try:
    payload = json.loads(text)
except json.JSONDecodeError:
    payload = None
if isinstance(payload, dict):
    items = payload.get("tokens", payload.get("scene_tokens", payload.get("items", [])))
elif isinstance(payload, list):
    items = payload
else:
    items = [line.split()[0] for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
tokens = {
    str(item.get("token", item.get("scene_token")) if isinstance(item, dict) else item).strip()
    for item in items
}
tokens.discard("")
print(len(tokens))
PY
    )"
  fi
fi
if [[ "${MAX_SCENES}" -eq 0 && "${EXPECTED_SCENE_COUNT}" -le 0 ]]; then
  echo "Could not derive a positive EXPECTED_SCENE_COUNT from ${EXPECTED_TOKEN_SOURCE}." >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-$(dirname "${OUTPUT_PATH}")/sg_fps_v4_build}"
RESUME="${RESUME:-true}"
VALIDATOR_WORKERS="${VALIDATOR_WORKERS:-16}"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_PATH}" "${RUN_ROOT}/logs" "${RUN_ROOT}/shards"

if [[ "${RESUME}" != "true" ]] && find "${OUTPUT_PATH}" -maxdepth 1 -name '*.pkl.xz' -print -quit | grep -q .; then
  echo "OUTPUT_PATH already contains records and RESUME is not true: ${OUTPUT_PATH}" >&2
  exit 2
fi

POLICY_CHECKPOINT_SHA256="${POLICY_CHECKPOINT_SHA256:-$(sha256sum "${POLICY_CHECKPOINT}" | awk '{print $1}')}"
FS_NORM_STATS_SHA256="${FS_NORM_STATS_SHA256:-$(sha256sum "${FS_NORM_STATS_PATH}" | awk '{print $1}')}"
export POLICY_CHECKPOINT_SHA256 FS_NORM_STATS_SHA256

{
  printf '[%s] git_head=%s\n' "$(date -Is)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  printf 'output=%q run_root=%q gpu_ids=%q shards=%q max_scenes=%q resume=%q split=%q expected_scene_count=%q\n' \
    "${OUTPUT_PATH}" "${RUN_ROOT}" "${GPU_IDS}" "${SHARD_COUNT}" "${MAX_SCENES}" "${RESUME}" \
    "${BUILD_LOG_SPLIT}" "${EXPECTED_SCENE_COUNT}"
  printf 'policy=%q policy_sha256=%s fs_stats=%q fs_sha256=%s expected_tokens=%q\n' \
    "${POLICY_CHECKPOINT}" "${POLICY_CHECKPOINT_SHA256}" \
    "${FS_NORM_STATS_PATH}" "${FS_NORM_STATS_SHA256}" "${EXPECTED_TOKEN_SOURCE}"
  printf 'external_candidate_roots=%q\n' "${EXTERNAL_CANDIDATE_ROOTS:-}"
} >> "${RUN_ROOT}/commands.log"

declare -a PIDS=()
declare -a SHARDS=()

terminate_children() {
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for ((shard = 0; shard < SHARD_COUNT; shard++)); do
  gpu="${GPU_ARRAY[shard]}"
  shard_root="${RUN_ROOT}/shards/shard_$(printf '%02d' "${shard}")"
  mkdir -p "${shard_root}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OUT_ROOT="${shard_root}" \
    SHARD_INDEX="${shard}" \
    SHARD_COUNT="${SHARD_COUNT}" \
    MAX_SCENES="${MAX_SCENES}" \
    BUILD_LOG_SPLIT="${BUILD_LOG_SPLIT}" \
    EXPECTED_SCENE_COUNT="${EXPECTED_SCENE_COUNT}" \
    SKIP_EXISTING_RECORDS="${RESUME}" \
    PREFILTER_EXISTING_RECORDS=true \
    VALIDATE_EXISTING_RECORDS=true \
    POLICY_CHECKPOINT_SHA256="${POLICY_CHECKPOINT_SHA256}" \
    FS_NORM_STATS_SHA256="${FS_NORM_STATS_SHA256}" \
    bash scripts/training/sg_fps/run_build_sg_fps_support_v4.sh \
    > "${RUN_ROOT}/logs/build_shard_$(printf '%02d' "${shard}").log" 2>&1 < /dev/null &
  PIDS+=("$!")
  SHARDS+=("${shard}")
done

failed=0
: > "${RUN_ROOT}/shard_status.tsv"
for index in "${!PIDS[@]}"; do
  pid="${PIDS[index]}"
  shard="${SHARDS[index]}"
  if wait "${pid}"; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s\t%s\n' "${shard}" "${status}" >> "${RUN_ROOT}/shard_status.tsv"
done

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more SG-FPS v4 build shards failed; inspect ${RUN_ROOT}/logs." >&2
  exit 1
fi

validator=(
  "${PYTHON_BIN}" scripts/tools/validate_sg_fps_v4_archive.py
  --archive-path "${OUTPUT_PATH}"
  --workers "${VALIDATOR_WORKERS}"
  --output-json "${RUN_ROOT}/validation.json"
  --output-md "${RUN_ROOT}/validation.md"
)
if [[ -n "${EXPECTED_TOKEN_SOURCE}" ]]; then
  validator+=(--expected-token-source "${EXPECTED_TOKEN_SOURCE}")
fi
(
  cd "${REPO_ROOT}"
  "${validator[@]}"
) > "${RUN_ROOT}/logs/validation.log" 2>&1

echo "SG-FPS v4 build and validation completed: ${OUTPUT_PATH}"
