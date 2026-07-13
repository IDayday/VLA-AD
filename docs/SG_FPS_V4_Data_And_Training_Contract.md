# SG-FPS v4 Data and Training Contract

## 1. Why v3 cannot be reused as the Stage2 teacher

The final v3 archive contains useful trajectories, but its selected set is not a coherent teacher
distribution. The actual construction chain was:

1. generate GT plus 32 fixed structured perturbations;
2. score them and retain roughly top-12 by `PDMS - 0.02 * GT_distance - 0.005 * jerk`;
3. append raw DDV2/DriveOR candidates and mechanically expanded external candidates;
4. apply the v3 Pareto/quota selector;
5. use selector quota tags as Stage2 target weights.

This creates three causal errors:

- internal proposals pass through two selectors while external proposals pass through one;
- `best_pdms`, `vector_pareto`, and `diversity_max` describe selector provenance, not behavior
  probability;
- fixed target probability does not limit gradient influence when non-GT targets have much larger
  denoising residuals.

The full 103,288-scene audit measured 10.279 selected rows per scene but only 1.977 coarse SNSAD
modes. Non-GT targets had mean ADE 1.570 m and 26.91% exceeded 2 m. In controlled Stage2 probes,
about 14.8% non-GT target mass produced about 37.7% residual mass, with non-GT loss roughly 8.7
times GT loss.

These facts rule out FS-Norm, Planning Token Adapter, insufficient training time, or simultaneous
M-target averaging as the primary cause. The teacher distribution and its optimization semantics
were inconsistent.

## 2. Required v4 data

Each scene is represented by a compact set, not a fixed-size quota:

```text
S_i = {GT anchor} union {up to three locally learnable behavior modes}
```

Every non-GT mode must satisfy all of the following:

1. official evaluator NC/DAC and reference-relative DDC/comfort/feasibility guards;
2. `ADE(candidate, GT) <= 1.5 m` and `FDE(candidate, GT) <= 4.0 m`;
3. reward no lower than `reward(GT) - 0.05`;
4. membership in the epsilon-Pareto front of `[EP, TTC, comfort]`, including valid GT in the
   comparison, with epsilon `0.01`;
5. pairwise SNSAD distance at least `0.25` from GT and every previously selected mode;
6. no DDV2/DriveOR failure expansion or trust-region repair derivative;
7. finite SNSAD distance no greater than `0.50` from one of eight stochastic rollouts of the
   Stage2 initialization policy.

Safety is a hard constraint. It is not traded against progress or used as a soft target weight.
Scenes with no valid alternative remain GT-only. Natural differences in per-scene mode capacity are
therefore preserved instead of being hidden by a fixed support count.

The archive stores all raw candidates and explicit semantics:

```text
version = 4
support_indices
mode_ids
teacher_eligible_mask
source_conditioned_mask
gt_relative_ade
gt_relative_fde
policy_reachability_snsad
pareto_front_mask
build_metadata
```

`mode_id`, not source or selector tag, is the unit of target probability. Exact duplicate
trajectories cannot re-enter the selected set through fingerprint aliasing. Policy rollouts define
the local learning frontier but receive no source quota or privileged training weight.

After the hard gates and epsilon-Pareto test, the selector does not sort the front by scalar PDMS.
It greedily covers the scene-normalized `[EP, TTC, comfort]` objective space, with SNSAD trajectory
novelty as the secondary key and capped scalar reward only as a tie breaker. The archive records
`pareto_objective_novelty` and the immutable provenance value:

```text
mode_selection_order = scene_normalized_objective_fps_then_snsad
```

This prevents a nominally multi-objective archive from collapsing back to one scalar optimum at
the final top-k operation. If all evaluator components are saturated, objective novelty is zero and
the selector retains geometrically distinct equal-quality behavior rather than inventing metric
spread.

## 3. Raw construction is mandatory

`mode_pareto_v4` must see the complete internal pool before any scalar top-k:

```text
GT + 32 structured counterfactuals + 8 current-Stage2 rollouts
   + raw scene-conditioned external proposals
```

External mechanical expansion is disabled. A v4-shaped archive obtained by reselecting v3 is only a
shadow diagnostic because the removed internal candidates cannot be recovered. Both the validator
and the Stage2 loader reject such an archive for production training.

The initialization checkpoint, its FS-Norm statistics, SHA256 fingerprints, policy sample count,
seed, and reachability threshold are part of every record's provenance. Missing current-policy
samples or provenance is a hard build/loader error. This makes the data an adjacent frontier that
can be reached by Stage2 training and subsequently explored on-policy in Stage3, rather than a bag
of globally good but potentially unreachable trajectories.

The 512-scene raw reconstruction provides direct causal evidence:

| Quantity | strict v3 shadow | raw v4 |
|---|---:|---:|
| internal candidates per scene | 12.06 | 33.00 |
| selected trajectories per scene | 2.203 | 3.010 |
| multi-mode scene ratio | 83.59% | 95.51% |
| non-GT ADE mean | about 0.97 m | 0.684 m |
| non-GT reward delta mean | +0.0187 | +0.0117 |

Among raw-v4 selected non-GT modes, 54.32% were absent from the v3 candidate pool, and 50.59% of
scenes recovered at least one selected mode removed by the hidden top-12. These recovered modes are
mainly lateral, progress, and timing counterfactuals, not external mechanical expansions.

Artifacts:

```text
outputs/sg_fps_support_v4_shadow_strict_20260713
outputs/sg_fps_support_v4_raw_smoke_20260713
```

The subsequent 128-scene policy-reachable reconstruction used eight A5 rollouts per scene and a
maximum nearest-policy SNSAD of `0.50`. It passed the strict validator with zero errors. The final
objective-covering selector retained 3.094 trajectories per scene, a 96.09% multi-mode scene ratio,
and 2.094 non-GT modes per scene. Non-GT nearest-policy SNSAD was 0.359 mean / 0.477 p90 / 0.499 max;
only the hard bound, not source identity, grants eligibility. The selected set is stored at:

```text
outputs/sg_fps_support_v4_reachable_smoke_20260713/support_v4_paretofps_reach050
```

## 4. Stage2 target distribution

For scene `i`, let `K_i` be its number of non-GT modes and `beta_i` the scheduled mode mass:

```text
q_i(GT) = 1 - beta_i
q_i(mode k) = beta_i / K_i
```

`beta_i=0` for GT-only scenes. Otherwise it is independent of `K_i`, reward margin, source count,
or selector tag. The default final mass is `0.25`, ramped over 40 epochs.

Each optimizer step uses GT and one uniformly sampled non-GT mode. The sampled mode receives the
full `beta_i` mass for that step, which is an unbiased estimator of the uniform mode objective.
GT/mode pairs share diffusion timestep and noise. A detached per-scene residual proxy then rescales
non-GT mass so its contribution does not exceed 45% of total residual mass, while preserving each
scene's total loss mass. The previous 25% cap reduced a nominal 24% non-GT target probability to
only about 8% effective weight and created mode starvation; the policy became more precise but
narrower. With reachable v4 modes, the 45% cap yields about 16% effective non-GT weight without
allowing hard targets to dominate.

The frontier-v4 launcher disables trajectory and feasibility x0 auxiliaries by default. Those
per-timestep regression terms remain available for ablation, but on the controlled 128-scene probe
they acted as mean-seeking regularizers and contracted sampled dispersion. The epsilon objective,
FS representation, and output feasibility bounds remain unchanged.

This separates four concepts that v3 mixed together:

- mode existence: archive/evaluator contract;
- mode probability: uniform per explicit mode;
- exposure schedule: epoch beta ramp;
- current optimization influence: residual budget.

FS-Norm is a model coordinate system, not a statistic that may be silently replaced whenever the
teacher archive changes. Fine-tuning A5 on v4 must retain A5's exact FS statistics; a fresh model may
use statistics rebuilt from the final v4 archive. An archive-path mismatch is therefore a warning by
default and becomes a hard error only with `FS_NORM_REQUIRE_ARCHIVE_MATCH=true`.

The implementation entry point is:

```text
scripts/training/sg_fps/run_train_pta_fs_dit_frontier_v4.sh
```

## 5. Promotion gates

Static validation is necessary but insufficient. A full 103k rebuild is admitted to Stage2 only if:

1. `validate_sg_fps_v4_archive.py` reports zero contract errors;
2. every record has `raw_internal_candidates=true` and `expand_external_candidates=false`;
3. GT inclusion, safety, trust-region, Pareto, and mode-separation violation rates are zero;
4. every selected non-GT mode is within the declared current-policy reachability radius;
5. checkpoint and FS-statistics SHA256 provenance is complete;
6. at least 25% of scenes retain a valid non-GT mode;
7. no single generator's row count is used as probability mass.

Stage2 is promoted beyond a short probe only if:

1. GT loss does not regress materially from the A5 start;
2. fixed-noise, fixed-timestep non-GT loss and non-GT/GT loss ratio decrease across checkpoints;
3. residual budget is finite, anchored, and respects its cap;
4. held-out NC/TTC do not regress outside paired confidence bounds;
5. support-aligned precision, normalized recall, and mode coverage improve jointly rather than width
   increasing through off-support drift.

The fixed-noise gate must replay the same token order, selected mode, timestep, diffusion noise,
condition-dropout mask, and FS-Norm coordinate system for every compared checkpoint. Raw training
step averages are useful for finite-value monitoring but cannot replace this paired check, because
the GT and non-GT loss ratio has high variance across timesteps.

A Stage2 checkpoint is admitted to Stage3 only if on-policy rollouts show:

1. high NC/DAC/DDC feasible ratio;
2. nonzero within-scene EP/quality spread;
3. both positive and negative valid advantages;
4. nonzero bidirectional Pareto advantage energy;
5. a short Stage3 probe improves held-out reward without collapsing support-aligned coverage.

Static archive quality is not evidence of final NAVTEST improvement. Conversely, PDMS alone is not
evidence that the intended multimodal prior was learned.

## 6. Commands

Build raw v4 data:

```bash
OUTPUT_PATH=/path/to/support_v4 \
POLICY_CHECKPOINT=/path/to/stage2_initialization.ckpt \
FS_NORM_STATS_PATH=/path/to/that_checkpoint_fs_norm_stats_v2.npz \
EXTERNAL_CANDIDATE_ROOTS='ddv2=/path/to/ddv2 driveor=/path/to/driveor' \
bash scripts/training/sg_fps/run_build_sg_fps_support_v4.sh
```

Validate before training:

```bash
python scripts/tools/validate_sg_fps_v4_archive.py \
  --archive-path /path/to/support_v4 \
  --workers 24 \
  --output-json /path/to/validation.json \
  --output-md /path/to/validation.md
```

Train only after validation passes:

```bash
SUPPORT_ARCHIVE_PATH=/path/to/support_v4 \
FS_NORM_STATS_PATH=/path/to/initialization_checkpoint_fs_norm_stats_v2.npz \
bash scripts/training/sg_fps/run_train_pta_fs_dit_frontier_v4.sh
```

## 7. What the Stage2 dataset must optimize for

The desired dataset is not the widest set of valid trajectories. It must satisfy five properties at
the same time:

1. **Correct:** safety and evaluator semantics are hard constraints, not soft probabilities.
2. **Distinct:** rows represent separated trajectory modes instead of repeated perturbation aliases.
3. **Reachable:** each new mode lies near the current policy distribution, permitting gradual
   expansion rather than one-step imitation of an external policy.
4. **Balanced:** each scene has one unit of loss mass, and each explicit mode has a declared
   probability independent of generator frequency or scene candidate count.
5. **Useful for RL:** the learned rollout group must preserve feasible EP/quality trade-offs and
   reward spread, so Stage3 receives both positive and negative on-policy credit instead of an
   all-identical or all-unsafe group.

These properties define the Stage2-to-Stage3 interface. Static candidate count, absolute rollout
width, or archive PDMS alone cannot promote a dataset.

## 8. Dataset construction as a bounded frontier process

One global archive built from arbitrary high-scoring policies is not the intended dataset. The
production process is a small number of explicit frontier rounds:

1. freeze policy `pi_r`, its FS-Norm stats, and all evaluator versions;
2. build the raw candidate library from GT, structured counterfactuals, current-policy rollouts,
   and only scene-conditioned external proposals;
3. apply coherent hard safety and quality gates to every source;
4. select a compact GT-anchored, policy-reachable Pareto cover;
5. run a paired fixed-noise Stage2 probe and support-aligned rollout evaluation;
6. promote `pi_(r+1)` only when non-GT fit and normalized mode coverage improve without GT/safety
   regression;
7. rebuild at most one adjacent frontier from the promoted policy before the full Stage2 run.

The policy checkpoint does not label its own samples as correct. It defines reachability only;
official metrics and the same Pareto contract still decide correctness. GT remains in every scene,
and held-out scenes are never used to tune reachability thresholds. These constraints avoid a
self-training loop that merely confirms the current policy's errors.

The frontier must be bounded on both sides. Modes closer than the SNSAD separation threshold do not
add behavior capacity; modes farther than the policy reachability threshold create a hard-imitation
gradient that dominates their nominal probability. The interval between those thresholds is the
actual Stage2 expansion band. Stage3 then optimizes on-policy inside and just beyond this learned
band; it is not expected to repair a disconnected Stage2 teacher distribution.

## 9. Controlled learnability and rollout evidence

All values below use the same 128 scenes, A5 epoch155 initialization, A5 FS-Norm statistics, global
batch 32, learning rate `1e-5`, and 100 optimizer steps. A separate 16-batch replay verified that
token order, selected mode, beta, condition dropout, timestep/noise statistics, and reward were
paired between A5 and the candidate checkpoint.

The final objective-covering archive is learnable: versus A5, paired total loss fell by `0.00356`
on average (15/16 batches), non-GT denoising loss fell by `0.00617` (14/16), and trajectory
reconstruction loss improved in all 16 batches. This rules out a disconnected teacher set on the
tested subset.

The rollout result identifies the training-side correction:

| 100-step condition vs A5 | precision AUC | recall AUC | F1 AUC | dispersion ratio | SNSAD | mean GT ADE |
|---|---:|---:|---:|---:|---:|---:|
| residual cap 0.25 + x0 aux | +0.00657 | +0.00151 | +0.00320 | -0.04795 | -0.00304 | -0.02790 m |
| residual cap 0.45 + x0 aux | +0.00390 | +0.00498 | +0.00454 | -0.00487 | +0.00134 | -0.01836 m |
| residual cap 0.45, no x0 aux | +0.00297 | +0.00722 | +0.00571 | -0.00218 | +0.00321 | -0.01386 m |

For the final row, recall, F1, SNSAD, and GT ADE improvements have paired bootstrap 95% intervals
excluding zero; dispersion change is not significant. Compared directly with cap 0.45 plus
auxiliary, removing auxiliary improves recall by `0.00224`, kernel mode coverage by `0.00054`, and
SNSAD by `0.00187`, while accepting a small precision/GT-ADE trade-off. This is the current v4
default because it improves support fit and usable policy width together rather than optimizing
precision alone.

Artifacts:

```text
outputs/stage2_frontier_v4_paretofps128_fixed_replay_v2_20260713
outputs/stage2_frontier_v4_gradient_budget_ablation_20260713
outputs/stage2_frontier_v4_paretofps128_diversity_20260713/aggregate_budget_ablation
```
