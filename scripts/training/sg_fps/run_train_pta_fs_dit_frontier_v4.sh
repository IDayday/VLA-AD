#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

: "${SUPPORT_ARCHIVE_PATH:?set SUPPORT_ARCHIVE_PATH to an SG-FPS v4 archive}"

# Fine-tuning must retain the initialization checkpoint's FS-Norm coordinate
# system even though the teacher archive changes. Set true only for a fresh
# model whose stats were built from this exact archive.
export FS_NORM_REQUIRE_ARCHIVE_MATCH="${FS_NORM_REQUIRE_ARCHIVE_MATCH:-false}"

# One GT anchor and one uniformly sampled explicit behavior mode per scene.
# Mode exposure rises with the existing epoch ramp, while paired residual mass
# is capped so a hard support cannot dominate the anchor gradient. The local
# v4 reachability gate makes 0.45 safe enough to avoid the mode-starvation
# feedback observed with the legacy 0.25 cap.
export DPSI_TARGET_DISTRIBUTION=frontier_v4
export DPSI_BETA_MAX="${DPSI_BETA_MAX:-0.25}"
export DPSI_BETA_WARMUP_EPOCHS="${DPSI_BETA_WARMUP_EPOCHS:-40}"
export DPSI_TARGET_SAMPLE_M=2
export DPSI_TARGET_SAMPLE_M_AFTER_WARMUP=2
export DPSI_PAIR_TARGET_RANDOMNESS=true
export DPSI_RESIDUAL_BUDGET_ENABLED=true
export DPSI_NON_GT_RESIDUAL_MASS_CAP="${DPSI_NON_GT_RESIDUAL_MASS_CAP:-0.45}"
export DPSI_FRONTIER_CURRICULUM_ENABLED="${DPSI_FRONTIER_CURRICULUM_ENABLED:-true}"
export DPSI_FRONTIER_DIFFICULTY_START="${DPSI_FRONTIER_DIFFICULTY_START:-0.45}"
export DPSI_FRONTIER_DIFFICULTY_END="${DPSI_FRONTIER_DIFFICULTY_END:-1.0}"
export DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS="${DPSI_FRONTIER_DIFFICULTY_WARMUP_EPOCHS:-40}"
export DPSI_USE_REWARD_MARGIN_WEIGHT=false
export DPSI_USE_SOURCE_WEIGHT=false
export DPSI_SCENE_NORMALIZE_WEIGHTS=true

# Per-timestep x0 auxiliaries improved GT precision but measurably contracted
# the sampled diffusion distribution. Frontier-v4 trains the epsilon objective
# directly; the auxiliary implementation remains available for ablation.
export TRAJECTORY_AUX_WEIGHT="${TRAJECTORY_AUX_WEIGHT:-0.0}"
export FEASIBILITY_AUX_WEIGHT="${FEASIBILITY_AUX_WEIGHT:-0.0}"

exec "${REPO_ROOT}/scripts/training/sg_fps/run_train_pta_fs_dit.sh"
