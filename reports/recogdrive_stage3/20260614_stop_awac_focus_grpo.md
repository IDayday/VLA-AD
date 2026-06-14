# Stop AWAC/IQL Attempts And Focus Stage3 GRPO

Date: 2026-06-14 UTC

## Decision

Stop active AWAC/IQL Stage3 training/evaluation attempts and move Stage3 improvement effort back to the GRPO/GSPO path.

## Empirical Basis

The AWAC/IQL family did not approach the strongest GRPO result observed so far:

- Historical best Safe DiffGRPO: `stage3_safe_diffgrpo_ckpt_stream_eval_live_20260610T030355Z`, `epoch_12-step_17290`, PDMS `0.9061843202874436`.
- Original local Stage3 reference reported by the user: 10 epoch PDMS `90.55` / `0.9055`.
- AWAC/IQL variants were materially lower:
  - `stage3_awac_lownoise_dpo3gpu_20260613T163215Z`: best checked PDMS about `0.7803`.
  - `stage3_awac_gapdpo_lowvalid_2gpu_20260613T155703Z`: best checked PDMS about `0.7845`.
  - `stage3_awac_gapdpo_hybrid4gpu_v2_20260613T155350Z`: best checked PDMS about `0.7982`.
  - `stage3_awac_blend035_lownoise_dpo8gpu_20260613T164458Z`: best checked PDMS about `0.8334`.
  - `stage3_awac_iql_pref_rank_ttc_20260613T112709Z`: best checked PDMS about `0.8507`.
  - `stage3_awac_iql_bc015_20260613T0648Z`: best checked PDMS about `0.8601`.

The elite buffer is still useful as a source of high-reward trajectory knowledge, but the current AWAC/IQL objective has not transferred that knowledge into a strong diffusion policy. The next experiments should keep GRPO as the policy-improvement backbone and use the buffer only as a small auxiliary guidance signal.

## Actions Taken

Stopped AWAC/IQL training and evaluation watcher processes matching AWAC run paths or `stage3_objective=awac_iql` on:

- local machine
- `training-vla-zt2`
- `training-rl-zt3`

Kept running:

- Main GRPO run: `stage3_grpo_refkl_s16_lr2e4_b4acc2_chunk32_fast_8gpu_20260613T213145Z`.
- Main GRPO checkpoint evaluation watchers on zt2 and zt3.
- Local queued GRPO/GSPO buffer-guided run.

Started an additional zt2 GRPO/GSPO buffer-guided run on free GPUs 0-3:

- Run: `stage3_grpo_buffer_guided_gspo_s16_lr2e4_b2acc8_zt2_4gpu_20260614T012106Z`.
- GPUs: `training-vla-zt2` GPU `0,1,2,3`.
- Effective batch scale: `batch_size=2`, `accumulate_grad_batches=8`, `grpo_sample_time=16`.
- LR schedule: `2e-4` initial, 20 epochs, scheduler minimum LR `1e-5`.
- Auxiliary buffer/self-imitation guidance:
  - `grpo_buffer_reward_bonus_weight=0.03`
  - `grpo_buffer_distill_loss_weight=0.03`
  - `grpo_self_imitation_loss_weight=0.02`
  - low-noise distillation/self-imitation timesteps
- Checkpoint watchers are attached with free-GPU gates.

## Next Admission Rule

Further GRPO changes should only be made when they have:

- a lesson from the past runs,
- a literature or implementation basis,
- a testable hypothesis tied to PDMS/submetrics,
- a risk guard for NC/DAC/TTC/DDC,
- and a concrete validation plan.
