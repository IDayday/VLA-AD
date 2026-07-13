# Safe DiffGRPO++ Base: elite142 two-expert Stage3 plan

Date: 2026-06-18

## Objective

Run the first Safe DiffGRPO++ base experiment from the current best elite Stage2 checkpoint, without ablations.

The goal is to preserve the current repo's Safe DiffGRPO safety/reward infrastructure while integrating the transferable RL settings from the PDMS91 Reinforce++ recipe. The PDMS91 feature-side changes are intentionally excluded.

## Starting checkpoint

Primary Stage2 checkpoint:

```text
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_dual_val6000_navtest_watch_20260617T020404Z/navtest_elite_full103k_top3/checkpoint_snapshots/epoch_142-step_115115.ckpt/epoch=142-step=115115.ckpt
```

Availability check on 2026-06-18:

```text
The report path above is currently missing on disk.
The original training checkpoint epoch=142-step=115115.ckpt is also not present under the retained stage2_train checkpoint directory.
Do not silently substitute a later checkpoint for the base++ run.
```

Currently retained nearby/later checkpoints include:

```text
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T230916Z_jsonfix/stage2_train/lightning_logs/version_0/checkpoints/epoch=196-step=158585.ckpt
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T230916Z_jsonfix/stage2_train/lightning_logs/version_0/checkpoints/epoch=199-step=161000.ckpt
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T230916Z_jsonfix/stage2_train/step_00160000.ckpt
```

These are not the selected elite142 starting point.

Known full navtest result:

```text
PDMS = 0.8728629594201617
traj_l1 = 0.35099197744252153
num_pdm_valid = 12138
```

This checkpoint uses the two-expert prefuse architecture:

```text
use_two_expert_slots=true
two_expert_condition_mode=prefuse_cross_attention
two_expert_dit_condition_mode=prefuse_cross_attention
dit_dropout=0.05
```

Training data/env:

```text
NAVSIM_DATA_ROOT=/mnt/navsim
METRIC_CACHE_DIR=/mnt/project/VLA-AD/cache/metric_cache_train_full
HIDDEN_CACHE_DIR=/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z
```

The hidden cache is an indexed `.pt` shard cache under `shards/shard_*/index.jsonl`, not the old `internvl_feature.gz` cache layout.

## Base++ changes

### 1. Safe RPP advantage mode

Added `grpo_advantage_mode` with two modes:

```text
safe_zscore  # old default
safe_rpp     # base++ mode
```

`safe_rpp` computes:

```text
safe-shaped reward
-> group de-centering per scene
-> batch normalization over B*G
-> existing asymmetric safe/unsafe advantage transform
-> quantile clip
-> existing denoising-step discount in the policy loss
```

The old behavior remains default through `safe_zscore`.

Base++ default:

```text
GRPO_ADVANTAGE_MODE=safe_rpp
GRPO_NORMALIZE_ADVANTAGE_BATCH=false
GRPO_ADVANTAGE_CLIP_ABS=5.0
```

The extra batch normalization flag is kept off because `safe_rpp` already performs the Reinforce++ batch normalization inside the advantage builder.

### 2. Explicit PDMS91 diffusion-RL stability knobs

These settings are now exposed through agent/Hydra/launcher instead of relying only on Python defaults:

```text
GRPO_SAMPLE_TIME=16
GRPO_GAMMA_DENOISING=0.6
GRPO_DENOISED_CLIP_VALUE=1.0
GRPO_EVAL_RANDN_CLIP_VALUE=1.0
GRPO_RANDN_CLIP_VALUE=5.0
GRPO_FINAL_ACTION_CLIP_VALUE=1.0
GRPO_EVAL_MIN_SAMPLING_DENOISING_STD=0.0001
GRPO_MIN_SAMPLING_DENOISING_STD=0.04
GRPO_MIN_LOGPROB_DENOISING_STD=0.1
GRPO_CLIP_ADVANTAGE_LOWER_QUANTILE=0.0
GRPO_CLIP_ADVANTAGE_UPPER_QUANTILE=1.0
```

### 3. Keep current exact KL and BC anneal

Do not copy PDMS91's older approximate KL. Keep the current repo's exact transition KL.

Base++ default:

```text
REFERENCE_KL_COEFF=0.02
REFERENCE_KL_CHUNK_SIZE=16
BC_ANNEAL=true
BC_COEFF_START=0.10
BC_COEFF_END=0.05
BC_ANNEAL_EPOCHS=5
```

### 4. Safety mode for the base run

No hard TTC/DDC gate in the base run.

Base++ uses soft safety shaping:

```text
GRPO_REWARD_MODE=safe_diffgrpo
GRPO_SAFETY_ADVANTAGE_MODE=soft_penalty
GRPO_HARD_GATE_TTC=false
GRPO_HARD_GATE_DDC=false
GRPO_TTC_SAFE_THRESHOLD=0.95
GRPO_DDC_SAFE_THRESHOLD=0.99
GRPO_SOFT_SAFETY_PENALTY_WEIGHT=0.50
GRPO_SOFT_SAFETY_PENALTY_CLIP=0.50
```

Rationale: previous hard TTC/DDC gate attempts risk suppressing EP/DAC. Base++ should first test the stable RPP advantage plus soft safety path.

### 5. No ablation features in base++

Disabled for this first run:

```text
GRPO_USE_GSPO_RATIO=false
OFFLINE_RL_ENABLED=false
GRPO_BUFFER_GUIDANCE_ENABLED=false
GRPO_BUFFER_REWARD_BONUS_WEIGHT=0.0
GRPO_BUFFER_DISTILL_LOSS_WEIGHT=0.0
GRPO_BUFFER_PREFERENCE_DPO_LOSS_WEIGHT=0.0
GRPO_SELF_IMITATION_LOSS_WEIGHT=0.0
```

Core-Pareto, buffer distill, self-imitation, Buffer-DPO, GSPO, and hard-gate variants are deferred.

## Code touch points

Implemented files:

```text
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
navsim/agents/recogdrive/recogdrive_agent.py
navsim/planning/script/config/common/agent/recogdrive_agent.yaml
navsim/planning/script/run_training_recogdrive.py
navsim/planning/script/run_training_recogdrive_rl.py
scripts/training/run_recogdrive_stage3_rl_2b_local.sh
scripts/training/launch_recogdrive_stage3_rl_2b_local_stable.sh
scripts/training/launch_recogdrive_stage3_safe_diffgrpo_pp_base_elite142_2b_local.sh
```

Key compatibility fixes:

- RL training can now read indexed `.pt` chunk cache in `cache_path` through `IndexedPtCacheDataset`.
- RL collate now stacks `two_expert_h_dyn` and `two_expert_h_geo`.
- The lower stage3 launcher accepts indexed chunk-cache roots during offline cache preflight.
- The non-RL chunk dataset also recognizes `root/shards/shard_*/index.jsonl`.

## Launch commands

Dry-run config check:

```bash
USE_STABLE_LAUNCHER=0 DRY_RUN=1 \
  scripts/training/launch_recogdrive_stage3_safe_diffgrpo_pp_base_elite142_2b_local.sh
```

One-batch smoke, if GPUs are available:

```bash
USE_STABLE_LAUNCHER=0 \
DRY_RUN=0 \
MAX_SCENES=64 \
LIMIT_TRAIN_BATCHES=1 \
LIMIT_VAL_BATCHES=0 \
CHECKPOINT_EVERY_N_TRAIN_STEPS=0 \
START_EARLY_GATE_WATCHER=0 \
RUN_NAME=stage3_safe_diffgrpo_pp_base_elite142_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
  scripts/training/launch_recogdrive_stage3_safe_diffgrpo_pp_base_elite142_2b_local.sh
```

Full base++ run:

```bash
scripts/training/launch_recogdrive_stage3_safe_diffgrpo_pp_base_elite142_2b_local.sh
```

Default full-run shape:

```text
8 GPUs
BATCH_SIZE=2
ACCUMULATE_GRAD_BATCHES=4
effective optimizer batch = 64 scenes/update
MAX_EPOCHS=20
CHECKPOINT_EVERY_N_TRAIN_STEPS=300
CHECKPOINT_EVERY_N_EPOCHS=1
```

## Evaluation protocol

Early diagnostics:

```text
step300 / step600 / step900 / epoch0
```

Early checkpoints are diagnostic only. Do not use them as final evidence against long-horizon references.

Long-horizon references:

```text
Historical Safe DiffGRPO: epoch_12-step_17290, PDMS 0.9061843202874436
PDMS91 Reinforce++ reference: epoch=9-step=13300, about 0.910 PDMS
```

Base++ success criteria:

- Smoke must load the elite142 checkpoint with no missing two-expert architecture failure.
- Training logs must show `safe_rpp_advantage_enabled=1`.
- Early PDMS should clear the existing GRPO-family sanity band and avoid NC/TTC/DDC collapse.
- Final comparison should be made only after comparable training length.

## Current status

Base++ code and launcher are prepared. Static Python/shell syntax checks passed.

Additional dry-run validation:

```text
The launcher/Hydra parameter chain was dry-run successfully with an existing later checkpoint
epoch=199-step=161000.ckpt as a file-existence surrogate.
This was only a command-generation check; it is not the approved base++ starting checkpoint.
Resolved command confirmed:
- grpo_advantage_mode=safe_rpp
- grpo_safety_advantage_mode=soft_penalty
- grpo_use_gspo_ratio=false
- use_two_expert_slots=true
- two_expert_condition_mode=prefuse_cross_attention
- cache_path points to the indexed two-expert hidden cache
```

Current blocker:

```text
epoch=142-step=115115.ckpt is not present at the report snapshot path or the original stage2 train checkpoint path.
```

Next required step:

1. Restore or locate the exact elite142 checkpoint.
2. Re-run the dry-run command.
3. Run the one-batch smoke.
4. Start the full base++ training if smoke passes.
