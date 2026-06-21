# ReCogDrive Stage3 Core-Pareto GRPO v2 Results Summary

Date: 2026-06-21 UTC

## Run

| Field | Value |
|---|---|
| Run name | `stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z` |
| Run root | `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z` |
| Branch | `feature/recogdrive-last-vla-v2` |
| Objective | GRPO primary, Core-Pareto advantage |
| Offline/AWAC/DPO/buffer distill/self-imitation | disabled |
| Training split/cache | navtrain/train metric cache only |
| Navtest usage | evaluation only |

## Configuration

| Item | Value |
|---|---|
| Initial LR | `1e-4` |
| Scheduler | 20 epochs, min LR `1e-5`, no zero final LR |
| Max epochs | `20` |
| GPUs | local 8 GPU DDP |
| Batch | per-GPU `2`, accumulate `4`, effective scene batch `64` |
| GRPO samples per scene | `16` |
| BC | anneal `0.10 -> 0.05` over 5 epochs |
| Reference KL | `0.02` |
| Core | `(5*EP + 5*TTC + 2*Comfort) / 12` |
| Feasibility | NC and DAC hard constraints |
| DDC | guard only, no positive reward |
| EP | reference-relative floor |
| TTC | optimized through Core plus soft EP/TTC tradeoff |
| Pareto | EP/TTC/Comfort Pareto front bonus, dominated positive cap |

## Reference Baselines

| Baseline | Checkpoint / note | PDMS |
|---|---|---:|
| Original ReCogDrive Stage3 local historical run | `epoch9-step13300` | `0.905500` |
| Safe DiffGRPO best local reference | recorded reference | `0.906184` |

## Latest Evaluation State

As of the latest readout, completed navtest eval reaches `step-step_24600`.
Training is still running toward the configured 20 epochs.

| Rank | Checkpoint | PDMS | Core | NC | DAC | TTC | EP | Comfort | DDC | Delta vs original | Delta vs Safe DiffGRPO |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `step-step_21600` | `0.910274` | `0.923721` | `0.984388` | `0.980475` | `0.958148` | `0.858781` | `1.000000` | `0.977344` | `+0.004774` | `+0.004090` |
| 2 | `step-step_24300` | `0.908567` | `0.923048` | `0.981710` | `0.978827` | `0.953946` | `0.861370` | `1.000000` | `0.975284` | `+0.003067` | `+0.002383` |
| 3 | `step-step_20400` | `0.907306` | `0.922248` | `0.982287` | `0.978085` | `0.952546` | `0.860850` | `1.000000` | `0.974749` | `+0.001806` | `+0.001122` |
| 4 | `step-step_24000` | `0.907301` | `0.922169` | `0.979733` | `0.979156` | `0.946861` | `0.866346` | `1.000000` | `0.971412` | `+0.001801` | `+0.001117` |
| 5 | `step-step_23100` | `0.907023` | `0.922344` | `0.980475` | `0.978168` | `0.951475` | `0.862151` | `1.000000` | `0.974749` | `+0.001523` | `+0.000839` |
| 6 | `step-step_18600` | `0.906764` | `0.923797` | `0.978168` | `0.975614` | `0.946037` | `0.871076` | `1.000000` | `0.975161` | `+0.001264` | `+0.000580` |
| 7 | `step-step_18900` | `0.906721` | `0.921577` | `0.985088` | `0.977426` | `0.960537` | `0.851248` | `1.000000` | `0.978580` | `+0.001221` | `+0.000537` |
| 8 | `step-step_17400` | `0.906594` | `0.922474` | `0.980310` | `0.977261` | `0.948591` | `0.865379` | `0.999918` | `0.967952` | `+0.001094` | `+0.000410` |

Latest non-top checkpoint:

| Checkpoint | PDMS | Core | NC | DAC | TTC | EP | Comfort | DDC | Delta vs original | Delta vs Safe DiffGRPO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `step-step_24600` | `0.903886` | `0.921205` | `0.979651` | `0.974543` | `0.948344` | `0.862548` | `1.000000` | `0.974131` | `-0.001614` | `-0.002298` |

## Interpretation

- Current best is `step-step_21600` with PDMS `0.910274`.
- The run has multiple checkpoints above both comparison references:
  `17400`, `18600`, `18900`, `20400`, `21600`, `23100`, `24000`, and `24300`.
- `step-step_21600` is the strongest point because NC, DAC, TTC, DDC, and Core
  are all high at the same time. It is not merely a high-Core point.
- `step-step_18600` has the highest Core among the top group, but lower
  `NC*DAC`, so its final PDMS is below `step-step_21600`. This confirms that
  Core is the main continuous term but not sufficient by itself.
- `step-step_24300` becoming the second-best checkpoint shows that the late-run
  improvement is not a single isolated spike.
- `step-step_24600` regressed below both comparison references, so the late
  phase still has visible checkpoint volatility.
- The remaining risk is checkpoint volatility: `21900-23700` included several
  regressions before the late recovery at `24000/24300`, and `24600` regressed
  again.

## SOTA Backup

The current best checkpoint has been copied to an independent backup location.

| Field | Value |
|---|---|
| Backed-up checkpoint | `step-step=21600.ckpt` |
| Backup path | `/mnt/project/VLA-AD/checkpoints/recogdrive/stage3_sota_backups/core_pareto_grpo_v2_step21600_pdms0.910274_20260621/recogdrive_stage3_core_pareto_grpo_v2_step21600_pdms0.910274.ckpt` |
| Latest pointer | `/mnt/project/VLA-AD/checkpoints/recogdrive/stage3_sota_backups/latest_core_pareto_sota` |
| Manifest | `/mnt/project/VLA-AD/checkpoints/recogdrive/stage3_sota_backups/core_pareto_grpo_v2_step21600_pdms0.910274_20260621/manifest.md` |
| Checksum file | `/mnt/project/VLA-AD/checkpoints/recogdrive/stage3_sota_backups/core_pareto_grpo_v2_step21600_pdms0.910274_20260621/sha256sums.txt` |
| SHA256 | `b8f747710e70e264084e1a7f9c2dd9f09761cdff063ec2080332acfe2ff042d5` |

## Current Evidence

The Core-Pareto GRPO v2 direction is currently the strongest Stage3 candidate
among the local attempts because it:

- keeps the proven GRPO training spine;
- disables buffer/AWAC/DPO/self-imitation confounders for E1;
- directly aligns the advantage target with the PDMS formula;
- preserves EP better than hard safety-heavy variants;
- produces several checkpoints above original Stage3 and Safe DiffGRPO.

Continue the active run to the full 20 epochs and keep evaluating new step
checkpoints. The final model selection should report both the best checkpoint
and the final-epoch checkpoint, with navtest used only as evaluation.

## Related Documents

- Detailed experiment log:
  `reports/recogdrive_stage3/core_pareto_grpo_v2_experiment_log.md`
- Planned attempt:
  `reports/recogdrive_stage3_core_pareto_grpo_v2_plan_20260616.md`
- Earlier summary:
  `reports/recogdrive_stage3_core_pareto_grpo_v2_summary_20260616.md`
- Stage3 ledger:
  `reports/recogdrive_stage3/stage3_algorithm_attempts_and_reference_protocol.md`
