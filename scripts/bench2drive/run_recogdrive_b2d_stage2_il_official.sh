#!/usr/bin/env bash
set -euo pipefail

VLA_AD_ROOT=${VLA_AD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
FORMAL_CLOSEST_PUBLIC=${FORMAL_CLOSEST_PUBLIC:-0}
ALLOW_RETIRED_CUSTOM_B2D_PIPELINE=${ALLOW_RETIRED_CUSTOM_B2D_PIPELINE:-0}
if [[ "${FORMAL_CLOSEST_PUBLIC}" != "1" && "${ALLOW_RETIRED_CUSTOM_B2D_PIPELINE}" != "1" ]]; then
  cat >&2 <<'EOF'
This launcher uses the retired custom front-only/4-history/8-waypoint Stage2
cache contract. It is not admissible for the ReCogDrive Bench2Drive
reproduction. See docs/bench2drive_recogdrive_reproduction_gate.md. Set
ALLOW_RETIRED_CUSTOM_B2D_PIPELINE=1 only for an explicitly labeled diagnostic.
EOF
  exit 64
fi

CONDA_BIN=${CONDA_BIN:-/root/miniconda3/bin/conda}
NAVSIM_ENV=${NAVSIM_ENV:-navsim}
NAVSIM_PYTHON=${NAVSIM_PYTHON:-/root/miniconda3/envs/${NAVSIM_ENV}/bin/python}

if [[ "${FORMAL_CLOSEST_PUBLIC}" == "1" ]]; then
  CONFIG=${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_stage2_closest_public_2b.yaml}
  EXPECTED_CONTRACT_ID=${EXPECTED_CONTRACT_ID:-recogdrive_b2d_closest_public_multiview_10hz_6x0p5s_v1}
  EXPECTED_RECORDS=${EXPECTED_RECORDS:-202656}
else
  CONFIG=${CONFIG:-${VLA_AD_ROOT}/configs/bench2drive_recogdrive_stage2_official_2b.yaml}
  EXPECTED_CONTRACT_ID=${EXPECTED_CONTRACT_ID:-}
  EXPECTED_RECORDS=${EXPECTED_RECORDS:-}
fi
CHUNK_CACHE_ROOT=${CHUNK_CACHE_ROOT:-}
CHUNK_NAME_PATTERN=${CHUNK_NAME_PATTERN:-shard_*}
INIT_POLICY_CHECKPOINT=${INIT_POLICY_CHECKPOINT:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}
RANDOM_INIT_POLICY=${RANDOM_INIT_POLICY:-1}
BASE_VLM_PATH=${BASE_VLM_PATH:-${VLA_AD_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}
EXPECTED_VLM_PATH=${EXPECTED_VLM_PATH:-}
ALLOW_BASE_VLM_CACHE=${ALLOW_BASE_VLM_CACHE:-0}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-${VLA_AD_ROOT}/outputs/bench2drive_recogdrive_stage2_closest_public_2b_${RUN_ID}}
LAUNCH_LOCK_PATH=${LAUNCH_LOCK_PATH:-${OUTPUT_DIR}.launch.lock}

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

if [[ -z "${CHUNK_CACHE_ROOT}" ]]; then
  echo "Set CHUNK_CACHE_ROOT to the train/ directory produced by build_recogdrive_b2d_stage2_cache.sh" >&2
  exit 2
fi

export CHUNK_CACHE_ROOT CHUNK_NAME_PATTERN OUTPUT_DIR BASE_VLM_PATH EXPECTED_VLM_PATH ALLOW_BASE_VLM_CACHE CONFIG
export EXPECTED_CONTRACT_ID EXPECTED_RECORDS
export FORMAL_CLOSEST_PUBLIC
export RANDOM_INIT_POLICY GLOBAL_EPOCHS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NPROC_PER_NODE
export PRECISION LR_ACTION_HEAD LR_SCHEDULER LR_SCHEDULER_EPOCHS LR_WARMUP_EPOCHS MIN_LR

cd "${VLA_AD_ROOT}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${LAUNCH_LOCK_PATH}")"
exec 9>"${LAUNCH_LOCK_PATH}"
if ! flock -n 9; then
  echo "Another Stage2 launcher already owns ${LAUNCH_LOCK_PATH}; refusing a duplicate run." >&2
  exit 75
fi
GIT_COMMIT=$(git rev-parse HEAD)

"${NAVSIM_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import yaml

config_path = Path(os.environ["CONFIG"])
if not config_path.is_file():
    raise SystemExit(f"Stage2 config does not exist: {config_path}")
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
if os.environ.get("FORMAL_CLOSEST_PUBLIC") == "1":
    expected_contract = os.environ["EXPECTED_CONTRACT_ID"]
    required = {
        "contract_id": expected_contract,
        "action_horizon": 6,
        "use_expert_features": False,
        "use_jepa": False,
        "use_vggt": False,
        "use_teacher_context_tokens": False,
        "use_student_latent_adapters": False,
        "use_alignment_loss": False,
        "use_action_aware_aux": False,
        "use_bit_drive": False,
        "use_risk_vla": False,
        "use_last_rd": False,
        "grpo": False,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise SystemExit(f"Formal Stage2 config mismatch: {mismatches}")
    effective_batch = (
        int(os.environ["BATCH_SIZE"])
        * int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
        * int(os.environ["NPROC_PER_NODE"])
    )
    runtime_checks = {
        "random_init_policy": os.environ["RANDOM_INIT_POLICY"].lower() in {"1", "true"},
        "global_epochs": int(os.environ["GLOBAL_EPOCHS"]) == 200,
        "effective_global_batch": effective_batch == 512,
        "precision": os.environ["PRECISION"] == "bf16",
        "learning_rate": float(os.environ["LR_ACTION_HEAD"]) == 1e-4,
        "scheduler": os.environ["LR_SCHEDULER"] == "official-cosine",
        "scheduler_epochs": int(os.environ["LR_SCHEDULER_EPOCHS"]) == 200,
        "warmup_epochs": int(os.environ["LR_WARMUP_EPOCHS"]) == 3,
        "minimum_lr": float(os.environ["MIN_LR"]) == 1e-6,
    }
    failed_runtime = {key: value for key, value in runtime_checks.items() if not value}
    if failed_runtime:
        raise SystemExit(
            f"Formal Stage2 runtime hyperparameters changed: {failed_runtime}; "
            f"effective_batch={effective_batch}"
        )

root = Path(os.environ["CHUNK_CACHE_ROOT"])
pattern = os.environ["CHUNK_NAME_PATTERN"]
chunks = sorted(path for path in root.glob(pattern) if path.is_dir())
if not chunks:
    raise SystemExit(f"No chunk cache shards matched {root / pattern}")
total = 0
summary = []
vlm_paths = set()
contract_ids = set()
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
    if meta.get("hidden_source") != "recogdrive_vlm":
        raise SystemExit(f"Shard was not generated by a ReCogDrive VLM: {chunk}")
    if meta.get("system_prompt_profile") != "bench2drive":
        raise SystemExit(f"Shard does not use the Bench2Drive prompt contract: {chunk}")
    vlm_path = meta.get("recogdrive_vlm_path")
    if not vlm_path:
        raise SystemExit(f"Shard does not record its source VLM: {chunk}")
    vlm_paths.add(str(Path(vlm_path).resolve()))
    contract_ids.add(str(meta.get("contract_id") or ""))
    count = int(meta.get("num_records") or sum(1 for _ in index_path.open("r", encoding="utf-8")))
    total += count
    summary.append({"name": chunk.name, "num_records": count, "hidden_source": meta.get("hidden_source")})
if total <= 0:
    raise SystemExit("Chunk cache contains zero records")
if len(vlm_paths) != 1:
    raise SystemExit(f"Cache shards came from multiple VLM checkpoints: {sorted(vlm_paths)}")
cache_vlm_path = next(iter(vlm_paths))
expected_contract = os.environ.get("EXPECTED_CONTRACT_ID", "").strip()
if expected_contract and contract_ids != {expected_contract}:
    raise SystemExit(f"Cache contract mismatch: found {sorted(contract_ids)}, expected {expected_contract}")
expected_records = os.environ.get("EXPECTED_RECORDS", "").strip()
if expected_records and total != int(expected_records):
    raise SystemExit(f"Cache contains {total} records, expected {expected_records}")
base_vlm_path = str(Path(os.environ["BASE_VLM_PATH"]).resolve())
allow_base = os.environ.get("ALLOW_BASE_VLM_CACHE", "0").lower() in {"1", "true", "yes"}
if cache_vlm_path == base_vlm_path and not allow_base:
    raise SystemExit(
        "Refusing hidden states generated from the pre-Stage1 base VLM. "
        "Rebuild the cache from the completed Bench2Drive Stage1 checkpoint."
    )
expected = os.environ.get("EXPECTED_VLM_PATH", "").strip()
if expected and cache_vlm_path != str(Path(expected).resolve()):
    raise SystemExit(f"Cache VLM {cache_vlm_path} does not match EXPECTED_VLM_PATH={expected}")
Path(os.environ["OUTPUT_DIR"], "cache_summary.json").write_text(
    json.dumps({"total_records": total, "source_vlm_path": cache_vlm_path, "shards": summary}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"Validated {len(chunks)} cache shards, {total} total records.")
PY

cat > "${OUTPUT_DIR}/launch_env.txt" <<EOF
date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=${GIT_COMMIT}
vla_ad_root=${VLA_AD_ROOT}
config=${CONFIG}
chunk_cache_root=${CHUNK_CACHE_ROOT}
chunk_name_pattern=${CHUNK_NAME_PATTERN}
init_policy_checkpoint=${INIT_POLICY_CHECKPOINT}
random_init_policy=${RANDOM_INIT_POLICY}
base_vlm_path=${BASE_VLM_PATH}
expected_vlm_path=${EXPECTED_VLM_PATH}
allow_base_vlm_cache=${ALLOW_BASE_VLM_CACHE}
formal_closest_public=${FORMAL_CLOSEST_PUBLIC}
expected_contract_id=${EXPECTED_CONTRACT_ID}
expected_records=${EXPECTED_RECORDS}
output_dir=${OUTPUT_DIR}
launch_lock_path=${LAUNCH_LOCK_PATH}
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
