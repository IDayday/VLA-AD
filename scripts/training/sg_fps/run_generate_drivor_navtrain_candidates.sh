#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}
REPO_ROOT=${REPO_ROOT:-/mnt/project/VLA-AD_last_vla_dev}
DRIVOR_ROOT=${DRIVOR_ROOT:-/mnt/project/external/DrivoR}
CHECKPOINT=${CHECKPOINT:-${DRIVOR_ROOT}/weights/releases/drivor_Nav1_25epochs.pth}
AGENT_NAME=${AGENT_NAME:-drivoR}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
DATA_ROOT=${DATA_ROOT:-/mnt/navsim}
MAPS_ROOT=${MAPS_ROOT:-/mnt/navsim/maps}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/drivor_nav1_navtrain_batched_$(date -u +%Y%m%dT%H%M%SZ)}
NUM_SHARDS=${NUM_SHARDS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PARTIAL_EVERY=${PARTIAL_EVERY:-64}
RESUME=${RESUME:-true}
MAX_SCENES_PER_SHARD=${MAX_SCENES_PER_SHARD:-}

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/pids" "${OUT_ROOT}/log_names" "${OUT_ROOT}/submissions"

export REPO_ROOT DRIVOR_ROOT TRAIN_TEST_SPLIT OUT_ROOT NUM_SHARDS
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

drivor_root = Path(os.environ["DRIVOR_ROOT"])
split = os.environ["TRAIN_TEST_SPLIT"]
out_root = Path(os.environ["OUT_ROOT"])
num_shards = int(os.environ["NUM_SHARDS"])
cfg_path = drivor_root / "navsim" / "planning" / "script" / "config" / "common" / "train_test_split" / "scene_filter" / f"{split}.yaml"
cfg = OmegaConf.load(cfg_path)
log_names = list(cfg.log_names)
shards = [log_names[i::num_shards] for i in range(num_shards)]
for idx, shard in enumerate(shards):
    with open(out_root / "log_names" / f"shard_{idx}.json", "w", encoding="utf-8") as f:
        json.dump(shard, f, indent=2)
with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
    json.dump({"train_test_split": split, "log_count": len(log_names), "num_shards": num_shards}, f, indent=2)
PY

: > "${OUT_ROOT}/commands.log"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu=$((shard % 8))
  shard_out="${OUT_ROOT}/submissions/shard_${shard}"
  mkdir -p "${shard_out}"
  cmd=(
    "${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/tools/generate_drivor_submission.py"
    --drivor-root "${DRIVOR_ROOT}"
    --checkpoint "${CHECKPOINT}"
    --agent-name "${AGENT_NAME}"
    --train-test-split "${TRAIN_TEST_SPLIT}"
    --data-root "${DATA_ROOT}"
    --maps-root "${MAPS_ROOT}"
    --output-dir "${shard_out}"
    --experiment-name "drivor_${TRAIN_TEST_SPLIT}_shard_${shard}"
    --log-names-json "${OUT_ROOT}/log_names/shard_${shard}.json"
    --batch-size "${BATCH_SIZE}"
    --partial-every "${PARTIAL_EVERY}"
    --device cuda
  )
  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "${MAX_SCENES_PER_SHARD}" ]]; then
    cmd+=(--max-scenes "${MAX_SCENES_PER_SHARD}")
  fi
  {
    printf '[%s] CUDA_VISIBLE_DEVICES=%s ' "$(date -Is)" "${gpu}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  } >> "${OUT_ROOT}/commands.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OPENSCENE_DATA_ROOT="${DATA_ROOT}" \
    NUPLAN_MAPS_ROOT="${MAPS_ROOT}" \
    PYTHONPATH="${DRIVOR_ROOT}:${PYTHONPATH:-}" \
    "${cmd[@]}" > "${OUT_ROOT}/logs/drivor_shard_${shard}.log" 2>&1 < /dev/null &
  echo $! > "${OUT_ROOT}/pids/drivor_shard_${shard}.pid"
done

echo "${OUT_ROOT}" > "${REPO_ROOT}/outputs/latest_drivor_navtrain_candidates.txt"
echo "Launched ${NUM_SHARDS} DrivoR candidate shards under ${OUT_ROOT}"
