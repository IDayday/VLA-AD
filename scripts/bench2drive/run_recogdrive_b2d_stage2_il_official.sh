#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}

CONFIG=${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_stage2_official_2b.yaml}
CHUNK_CACHE_ROOT=${CHUNK_CACHE_ROOT:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_vlm_cache_full_v1}
CHUNK_NAME_PATTERN=${CHUNK_NAME_PATTERN:-shard_*}
INIT_POLICY_CHECKPOINT=${INIT_POLICY_CHECKPOINT:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}
RANDOM_INIT_POLICY=${RANDOM_INIT_POLICY:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_il_official_2b_${RUN_ID}}

# ReCogDrive reports Stage II diffusion planner training with large-batch
# IL. With 8 GPUs, 32 per GPU and grad-accum 2 gives effective batch 512.
GLOBAL_EPOCHS=${GLOBAL_EPOCHS:-200}
BATCH_SIZE=${BATCH_SIZE:-32}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PRECISION=${PRECISION:-bf16}
LR_ACTION_HEAD=${LR_ACTION_HEAD:-1e-4}
LR_SCHEDULER=${LR_SCHEDULER:-official-cosine}
LR_SCHEDULER_EPOCHS=${LR_SCHEDULER_EPOCHS:-200}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-3}
MIN_LR=${MIN_LR:-1e-6}
SAVE_EVERY=${SAVE_EVERY:-1000}
SAVE_EVERY_EPOCH=${SAVE_EVERY_EPOCH:-0}
LOG_EVERY=${LOG_EVERY:-20}
SEED=${SEED:-20260709}

GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python - <<'PY'
import os
gpus = [g for g in os.environ.get("GPU_LIST", "0").split(",") if g.strip()]
print(max(1, len(gpus)))
PY
)}
MASTER_PORT=${MASTER_PORT:-29521}

RESUME_FROM=${RESUME_FROM:-}
RESUME_MODE=${RESUME_MODE:-full}
MAX_SAMPLES=${MAX_SAMPLES:-}
NUM_OPTIMIZER_STEPS=${NUM_OPTIMIZER_STEPS:-}

if [[ "${RANDOM_INIT_POLICY}" == "1" || "${RANDOM_INIT_POLICY}" == "true" || "${RANDOM_INIT_POLICY}" == "TRUE" ]]; then
  INIT_POLICY_CHECKPOINT=""
fi

export CHUNK_CACHE_ROOT CHUNK_NAME_PATTERN OUTPUT_DIR

cd "${VLA_AD_ROOT}"
mkdir -p "${OUTPUT_DIR}"

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CHUNK_CACHE_ROOT"])
pattern = os.environ["CHUNK_NAME_PATTERN"]
chunks = sorted(root.glob(pattern))
if not chunks:
    raise SystemExit(f"No chunk cache shards matched {root / pattern}")
total = 0
summary = []
for chunk in chunks:
    meta_path = chunk / "metadata.json"
    index_path = chunk / "index.jsonl"
    if not meta_path.is_file():
        raise SystemExit(f"Missing metadata: {meta_path}")
    if not index_path.is_file():
        raise SystemExit(f"Missing index: {index_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("is_dummy"):
        raise SystemExit(f"Refusing dummy cache shard: {chunk}")
    if not meta.get("contains_vlm_hidden"):
        raise SystemExit(f"Shard lacks VLM hidden states: {chunk}")
    count = int(meta.get("num_records") or sum(1 for _ in index_path.open("r", encoding="utf-8")))
    total += count
    summary.append({"name": chunk.name, "num_records": count, "hidden_source": meta.get("hidden_source")})
if total <= 0:
    raise SystemExit("Chunk cache contains zero records")
Path(os.environ["OUTPUT_DIR"], "cache_summary.json").write_text(
    json.dumps({"total_records": total, "shards": summary}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"Validated {len(chunks)} cache shards, {total} total records.")
PY

cat > "${OUTPUT_DIR}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
vla_ad_root=${VLA_AD_ROOT}
config=${CONFIG}
chunk_cache_root=${CHUNK_CACHE_ROOT}
chunk_name_pattern=${CHUNK_NAME_PATTERN}
init_policy_checkpoint=${INIT_POLICY_CHECKPOINT}
random_init_policy=${RANDOM_INIT_POLICY}
output_dir=${OUTPUT_DIR}
global_epochs=${GLOBAL_EPOCHS}
batch_size=${BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
num_workers=${NUM_WORKERS}
prefetch_factor=${PREFETCH_FACTOR}
precision=${PRECISION}
lr_action_head=${LR_ACTION_HEAD}
lr_scheduler=${LR_SCHEDULER}
lr_scheduler_epochs=${LR_SCHEDULER_EPOCHS}
lr_warmup_epochs=${LR_WARMUP_EPOCHS}
min_lr=${MIN_LR}
save_every=${SAVE_EVERY}
save_every_epoch=${SAVE_EVERY_EPOCH}
log_every=${LOG_EVERY}
seed=${SEED}
gpu_list=${GPU_LIST}
nproc_per_node=${NPROC_PER_NODE}
master_port=${MASTER_PORT}
resume_from=${RESUME_FROM}
resume_mode=${RESUME_MODE}
max_samples=${MAX_SAMPLES}
num_optimizer_steps=${NUM_OPTIMIZER_STEPS}
EOF

args=(
  --config "${CONFIG}"
  --chunk-cache-root "${CHUNK_CACHE_ROOT}"
  --chunk-name-pattern "${CHUNK_NAME_PATTERN}"
  --output-dir "${OUTPUT_DIR}"
  --global-epochs "${GLOBAL_EPOCHS}"
  --flat-global-dataset
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --precision "${PRECISION}"
  --lr-action-head "${LR_ACTION_HEAD}"
  --lr-scheduler "${LR_SCHEDULER}"
  --lr-scheduler-epochs "${LR_SCHEDULER_EPOCHS}"
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS}"
  --min-lr "${MIN_LR}"
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --seed "${SEED}"
)

if [[ -n "${INIT_POLICY_CHECKPOINT}" ]]; then
  args+=(--init-policy-checkpoint "${INIT_POLICY_CHECKPOINT}")
fi

if [[ "${SAVE_EVERY_EPOCH}" == "1" || "${SAVE_EVERY_EPOCH}" == "true" || "${SAVE_EVERY_EPOCH}" == "TRUE" ]]; then
  args+=(--save-every-epoch)
fi

if [[ -n "${RESUME_FROM}" ]]; then
  args+=(--resume-from "${RESUME_FROM}" --resume-mode "${RESUME_MODE}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${NUM_OPTIMIZER_STEPS}" ]]; then
  args+=(--num-optimizer-steps "${NUM_OPTIMIZER_STEPS}")
fi

if (( NPROC_PER_NODE > 1 )); then
  cmd=(torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" scripts/train_recogdrive_expert_chunked.py "${args[@]}")
else
  cmd=(python scripts/train_recogdrive_expert_chunked.py "${args[@]}")
fi

printf '%q ' "${cmd[@]}" > "${OUTPUT_DIR}/launch_command.txt"
printf '\n' >> "${OUTPUT_DIR}/launch_command.txt"

exec env CUDA_VISIBLE_DEVICES="${GPU_LIST}" CHUNK_CACHE_ROOT="${CHUNK_CACHE_ROOT}" CHUNK_NAME_PATTERN="${CHUNK_NAME_PATTERN}" OUTPUT_DIR="${OUTPUT_DIR}" \
  "${CONDA_BIN}" run --no-capture-output -n "${NAVSIM_ENV}" "${cmd[@]}"
