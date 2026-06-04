#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

CHECKPOINT="${TMP_DIR}/dummy.ckpt"
TRAIN_CACHE="${TMP_DIR}/train_cache"
TEACHER_CACHE="${TMP_DIR}/teacher_cache"
OUT_ROOT="${TMP_DIR}/out"
touch "${CHECKPOINT}"
mkdir -p "${TRAIN_CACHE}" "${TEACHER_CACHE}/samples" "${OUT_ROOT}"

python - "${TEACHER_CACHE}" <<'PY'
from pathlib import Path
import json
import sys
import torch

root = Path(sys.argv[1])
sample = root / "samples" / "sample_000000.pt"
torch.save(
    {
        "sample_token": "sample_000000",
        "scene_token": "scene_000000",
        "teacher_trajectory": torch.zeros(8, 3),
        "teacher_score": torch.tensor(1.0),
        "gt_score": torch.tensor(0.5),
        "oracle_best_of_k_score": torch.tensor(1.0),
        "candidate_count": torch.tensor(1),
    },
    sample,
)
(root / "index.jsonl").write_text(
    json.dumps({"sample_token": "sample_000000", "scene_token": "scene_000000", "path": "samples/sample_000000.pt"}) + "\n",
    encoding="utf-8",
)
PY

(
  cd "${ROOT_DIR}"
  LAST_VLA_DRY_RUN=1 \
  PYTHON_BIN="$(command -v python)" \
  TORCHRUN_BIN="/bin/false" \
  TRAIN_CHUNK_CACHE_ROOT="${TRAIN_CACHE}" \
  OUTPUT_DIR="${OUT_ROOT}/progressive" \
  MASTER_PORT=29991 \
  COT_ALIGNMENT_CHECKPOINT="${CHECKPOINT}" \
  scripts/last_vla_v2/run_progressive_bottleneck_8gpu.sh

  LAST_VLA_DRY_RUN=1 \
  PYTHON_BIN="$(command -v python)" \
  TORCHRUN_BIN="/bin/false" \
  TEACHER_TRAJ_CHUNK_ROOT="${TEACHER_CACHE}" \
  OUTPUT_DIR="${OUT_ROOT}/teacher" \
  MASTER_PORT=29992 \
  PROGRESSIVE_CHECKPOINT="${CHECKPOINT}" \
  scripts/last_vla_v2/run_teacher_traj_sft_8gpu.sh
)

for log in "${OUT_ROOT}/progressive/commands.log" "${OUT_ROOT}/teacher/commands.log"; do
  grep -q 'agent.last_vla_adapter_checkpoint=' "${log}"
  if grep -q '++agent.last_vla_adapter_checkpoint' "${log}"; then
    echo "Bad Hydra override found in ${log}" >&2
    exit 1
  fi
done

echo "Launcher dry-run override check passed."
