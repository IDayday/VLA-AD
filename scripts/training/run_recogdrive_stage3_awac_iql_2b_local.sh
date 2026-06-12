#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/navsim/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/navsim/bin/torchrun}"
NAVSIM_DATA_ROOT="${NAVSIM_DATA_ROOT:-/mnt/navsim}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/project/VLA-AD}"
RUN_NAME="${RUN_NAME:-stage3_awac_iql_2b_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-${ARTIFACT_ROOT}/outputs/${RUN_NAME}}"

IL_CHECKPOINT="${IL_CHECKPOINT:-${RECOGDRIVE_IL_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt}}"
VLM_PATH="${VLM_PATH:-${RECOGDRIVE_VLM_PATH:-${ARTIFACT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B}}"
METRIC_CACHE_DIR="${METRIC_CACHE_DIR:-${RECOGDRIVE_METRIC_CACHE_DIR:-${ARTIFACT_ROOT}/cache/metric_cache_train_full}}"
HIDDEN_CACHE_DIR="${HIDDEN_CACHE_DIR:-${RECOGDRIVE_HIDDEN_CACHE_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_agent_cache_dir_train_2b}}"
ELITE_BUFFER_DIR="${ELITE_BUFFER_DIR:-${ARTIFACT_ROOT}/cache/recogdrive_stage3_awac_elite_buffer_train}"
NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_DATA_ROOT}/trainval_navsim_logs/trainval}"
SENSOR_BLOBS_PATH="${SENSOR_BLOBS_PATH:-${NAVSIM_DATA_ROOT}/trainval_sensor_blobs/trainval}"

CACHE_MODE="${CACHE_MODE:-offline}"  # online or offline hidden-state features
ONLINE_AWAC_CANDIDATES="${ONLINE_AWAC_CANDIDATES:-false}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NODES="${NODES:-1}"
NODE_RANK="${NODE_RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-63669}}"
KILL_GPU_STRESS="${KILL_GPU_STRESS:-0}"
DRY_RUN="${DRY_RUN:-0}"

STAGE3_LR="${LR:-1e-4}"
STAGE3_MAX_EPOCHS="${MAX_EPOCHS:-20}"
STAGE3_BATCH_SIZE="${BATCH_SIZE:-8}"
STAGE3_ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
STAGE3_SCHEDULER_WARMUP_EPOCHS="${SCHEDULER_WARMUP_EPOCHS:-0}"
STAGE3_SCHEDULER_MIN_LR="${SCHEDULER_MIN_LR:-1e-5}"
STAGE3_TRAIN_TEST_SPLIT="navtrain"
STAGE3_EXPERIMENT_NAME="training_recogdrive_agent"

AWAC_ELITE_TOP_M="${AWAC_ELITE_TOP_M:-8}"
AWAC_BC_LOSS_WEIGHT="${AWAC_BC_LOSS_WEIGHT:-0.05}"
AWAC_GRPO_LOSS_WEIGHT="${AWAC_GRPO_LOSS_WEIGHT:-0.0}"
AWAC_BASELINE_MODE="${AWAC_BASELINE_MODE:-max_gt_il}"
AWAC_ADVANTAGE_TEMPERATURE="${AWAC_ADVANTAGE_TEMPERATURE:-0.03}"
AWAC_WEIGHT_MAX="${AWAC_WEIGHT_MAX:-20.0}"
AWAC_REQUIRE_NC="${AWAC_REQUIRE_NC:-true}"
AWAC_REQUIRE_DAC="${AWAC_REQUIRE_DAC:-true}"
AWAC_REQUIRE_DDC_GUARD="${AWAC_REQUIRE_DDC_GUARD:-true}"
AWAC_DDC_MIN_ABSOLUTE="${AWAC_DDC_MIN_ABSOLUTE:-0.99}"
AWAC_ONLINE_POLICY_SAMPLES="${AWAC_ONLINE_POLICY_SAMPLES:-8}"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${OUT_ROOT}/hydra}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_DATA_ROOT}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "${CACHE_MODE}" != "online" && "${CACHE_MODE}" != "offline" ]]; then
  echo "CACHE_MODE must be online or offline, got: ${CACHE_MODE}" >&2
  exit 2
fi
if [[ "${ONLINE_AWAC_CANDIDATES}" != "true" && "${ONLINE_AWAC_CANDIDATES}" != "false" ]]; then
  echo "ONLINE_AWAC_CANDIDATES must be true or false, got: ${ONLINE_AWAC_CANDIDATES}" >&2
  exit 2
fi
if [[ "${ONLINE_AWAC_CANDIDATES}" == "false" ]]; then
  if [[ ! -d "${ELITE_BUFFER_DIR}" ]] || ! find "${ELITE_BUFFER_DIR}" -name '*.pkl.xz' -print -quit | grep -q .; then
    echo "ELITE_BUFFER_DIR must contain *.pkl.xz records unless ONLINE_AWAC_CANDIDATES=true: ${ELITE_BUFFER_DIR}" >&2
    exit 2
  fi
fi
if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Python or torchrun is not executable: ${PYTHON_BIN}, ${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -f "${IL_CHECKPOINT}" ]]; then
  echo "Stage2 IL checkpoint does not exist: ${IL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${METRIC_CACHE_DIR}" ]]; then
  echo "Training metric cache directory does not exist: ${METRIC_CACHE_DIR}" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}"

HYDRA_ARGS=(
  "agent=recogdrive_agent"
  "agent.stage3_objective=awac_iql"
  "agent.grpo=False"
  "agent.lr=${STAGE3_LR}"
  "agent.scheduler_epochs=${STAGE3_MAX_EPOCHS}"
  "agent.scheduler_warmup_epochs=${STAGE3_SCHEDULER_WARMUP_EPOCHS}"
  "agent.scheduler_min_lr=${STAGE3_SCHEDULER_MIN_LR}"
  "agent.vlm_path=${VLM_PATH}"
  "agent.cam_type=single"
  "agent.cache_mode=False"
  "agent.vlm_type=internvl"
  "agent.checkpoint_path=${IL_CHECKPOINT}"
  "agent.dit_type=small"
  "agent.vlm_size=small"
  "agent.sampling_method=ddim"
  "agent.metric_cache_path=${METRIC_CACHE_DIR}"
  "agent.reference_policy_checkpoint=${IL_CHECKPOINT}"
  "agent.offline_rl_enabled=True"
  "agent.offline_rl_elite_buffer_path=${ELITE_BUFFER_DIR}"
  "agent.offline_rl_build_candidates_online=${ONLINE_AWAC_CANDIDATES}"
  "agent.offline_rl_online_policy_samples=${AWAC_ONLINE_POLICY_SAMPLES}"
  "agent.offline_rl_awac_loss_weight=1.0"
  "agent.offline_rl_bc_loss_weight=${AWAC_BC_LOSS_WEIGHT}"
  "agent.offline_rl_grpo_loss_weight=${AWAC_GRPO_LOSS_WEIGHT}"
  "agent.offline_rl_baseline_mode=${AWAC_BASELINE_MODE}"
  "agent.offline_rl_advantage_temperature=${AWAC_ADVANTAGE_TEMPERATURE}"
  "agent.offline_rl_weight_max=${AWAC_WEIGHT_MAX}"
  "agent.offline_rl_elite_top_m=${AWAC_ELITE_TOP_M}"
  "agent.offline_rl_require_nc=${AWAC_REQUIRE_NC}"
  "agent.offline_rl_require_dac=${AWAC_REQUIRE_DAC}"
  "agent.offline_rl_require_ddc_guard=${AWAC_REQUIRE_DDC_GUARD}"
  "agent.offline_rl_ddc_min_absolute=${AWAC_DDC_MIN_ABSOLUTE}"
  "trainer.params.max_epochs=${STAGE3_MAX_EPOCHS}"
  "trainer.params.num_nodes=${NODES}"
  "trainer.params.devices=${GPUS_PER_NODE}"
  "trainer.params.strategy=ddp_find_unused_parameters_true"
  "trainer.params.accumulate_grad_batches=${STAGE3_ACCUMULATE_GRAD_BATCHES}"
  "dataloader.params.batch_size=${STAGE3_BATCH_SIZE}"
  "experiment_name=${STAGE3_EXPERIMENT_NAME}"
  "train_test_split=${STAGE3_TRAIN_TEST_SPLIT}"
  "force_cache_computation=False"
)

if [[ "${CACHE_MODE}" == "offline" ]]; then
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=True"
    "cache_path=${HIDDEN_CACHE_DIR}"
    "use_cache_without_dataset=True"
  )
else
  HYDRA_ARGS+=(
    "agent.cache_hidden_state=False"
    "cache_path=null"
    "use_cache_without_dataset=False"
    "navsim_log_path=${NAVSIM_LOG_PATH}"
    "sensor_blobs_path=${SENSOR_BLOBS_PATH}"
  )
fi

if [[ -n "${MAX_SCENES:-}" ]]; then
  HYDRA_ARGS+=("train_test_split.scene_filter.max_scenes=${MAX_SCENES}")
fi

CMD=(
  "${TORCHRUN_BIN}"
  "--nnodes=${NODES}"
  "--node_rank=${NODE_RANK}"
  "--master_addr=${MASTER_ADDR}"
  "--nproc_per_node=${GPUS_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  "${REPO_ROOT}/navsim/planning/script/run_training_recogdrive_rl.py"
  "${HYDRA_ARGS[@]}"
)

{
  echo "repo_root=${REPO_ROOT}"
  echo "out_root=${OUT_ROOT}"
  echo "stage3_objective=awac_iql"
  echo "lr=${STAGE3_LR}"
  echo "max_epochs=${STAGE3_MAX_EPOCHS}"
  echo "scheduler_min_lr=${STAGE3_SCHEDULER_MIN_LR}"
  echo "batch_size=${STAGE3_BATCH_SIZE}"
  echo "elite_buffer_dir=${ELITE_BUFFER_DIR}"
  echo "online_awac_candidates=${ONLINE_AWAC_CANDIDATES}"
  echo "metric_cache_dir=${METRIC_CACHE_DIR}"
  printf 'command='
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "${OUT_ROOT}/resolved_command.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${OUT_ROOT}/resolved_command.txt"
  exit 0
fi

if [[ "${KILL_GPU_STRESS}" == "1" ]]; then
  pkill -f '^python /mnt/project/gpu_stress.py$' || true
  sleep 5
fi

exec "${CMD[@]}"
