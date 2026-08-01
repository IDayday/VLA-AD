# Main-paper reading notes

## Source, coverage, and provenance

- Audited source: `docs/AuthorKit27 (7).pdf`.
- The PDF has 9 pages. Every page was text-extracted and visually checked from a rendered page image; equations, figures, and Tables 1--5 were checked against the rendered pages rather than relying only on extraction order.
- PDF page numbers in this note are 1-indexed physical PDF pages.
- The manuscript is anonymous and titled **“Aligning Multi-Trajectory Supervision with Policy Optimization for VLA Driving.”**
- This note records only what the PDF states. It does not infer missing thresholds, seeds, run counts, candidate counts, optimizer settings, or evaluator versions. Table cells and explicit prose claims are also normalized in `paper_claims.csv`.

## Paper structure by page

| PDF page | Content |
|---:|---|
| 1 | Title, abstract, introduction, Figure 1 (motivation and IL-to-GRPO reversal). |
| 2 | Introduction continued, three contributions, start of Related Work. |
| 3 | Related Work continued, Figure 2 (AMPT overview), Preliminaries, diffusion SFT Eq. (1). |
| 4 | GRPO Eqs. (2)--(3), Method overview, PC-MTS, FF-PGRPO, credit assignment Eq. (4). |
| 5 | FF-PGRPO continued, APR, Experimental Setup, implementation details, compared variants, start of main results. |
| 6 | Figure 3 qualitative examples, NAVSIM v1 Table 1, trajectory-set Table 2, NAVSIM v1/v2 discussion. |
| 7 | NAVSIM v2 Table 3, stage ablation Table 4, APR rounds Table 5, failure recovery, conclusion and stated limitations. |
| 8--9 | References (32 entries). |

## Central thesis and claimed causal chain

The paper argues that multi-trajectory supervision must be judged by the rollout distribution it induces for downstream optimization, not by the isolated evaluator score of each target or by the imitation checkpoint alone. The central empirical observation is the reversal in Table 2: Score and Pareto supervision improve the IL checkpoint but deteriorate after the same scalar GRPO, whereas GT and PC-MTS improve after GRPO. AMPT is presented as a three-stage response:

1. **PC-MTS** filters trajectory supervision for quality, component-wise Pareto optimality, compatibility with a frozen GT-only policy, and local feasibility, producing `pi_initial`.
2. **FF-PGRPO** changes rollout credit so feasibility and coherent-reference guards precede Pareto-positive credit; a fully infeasible group receives no positive rollout and may receive weak recovery supervision from a safe reference.
3. **APR** rebuilds a policy-relative teacher set after each round, validates interpolated local targets, distills verified improvement with retention, and removes or activates teachers as the policy changes.

The final policy predicts **one trajectory**. The evaluator, candidate pools, specialist branches, group sampling, and candidate ranking are described as training-only; the text explicitly says there is no test-time scorer or candidate reranking.

## Detailed method contract stated in the PDF

### Preliminaries and architecture

- Input context `o`: multi-view camera observations, navigation command, and ego-vehicle state (p. 3).
- Output: a continuous future trajectory `tau = {w_t}_{t=1}^H` (p. 3).
- Architecture: a VLM encodes multimodal context and a conditional diffusion planner models `pi_theta(tau | o)` (p. 3).
- The supervised diffusion target is formed as `x_t = sqrt(alpha_bar_t) tau* + sqrt(1-alpha_bar_t) epsilon`; Eq. (1) minimizes expected squared noise-prediction error (p. 3).
- Training logged trajectories with Eq. (1) produces the frozen GT-only policy `pi_GT`; the same denoising objective is later used for accepted candidates and distilled local targets (pp. 3--4).
- GRPO samples a group of `G` trajectories for one scene and uses aggregate planning rewards. Eq. (2) is a group-average advantage-weighted trajectory log-probability objective plus `beta L_reg`; the trajectory log-probability accumulates denoising-transition log-probabilities. `L_reg` is described as BC or KL regularization (p. 4).
- Eq. (3) defines the standard group-relative advantage as `(R_i - mu_R)/(sigma_R + epsilon)` (p. 4).

### Stage 1: PC-MTS

The PDF states the following selection order (p. 4):

1. Require basic safety/compliance and a sufficient aggregate score.
2. Keep candidates on the scene-wise component Pareto front.
3. For NAVSIM v1, compare EP, TTC, and comfort on the Pareto front; NC and DAC must be valid and DDC is protected.
4. Sample multiple rollouts from frozen `pi_GT`; measure a candidate by average waypoint distance to its `K` nearest policy rollouts.
5. Calibrate the acceptance threshold from held-out rollouts of the same policy, separately per navigation command.
6. Reject insufficient local feasibility by creating a small number of temporally coherent nearby trajectories using normal variation observed in `pi_GT` rollouts and requiring most to remain feasible.
7. Always retain the logged trajectory.
8. At fine-tuning, each scene contributes one target sampled from its logged trajectory and accepted candidates, independent of candidate-pool size.
9. Fine-tune a copy of `pi_GT` with the same diffusion objective to obtain `pi_initial`.

Not specified in the PDF: candidate-source names/counts, aggregate threshold, rollout-bank size, `K`, distance normalization, held-out/calibration size or quantile, local perturbation count/correlation construction, the numerical “most feasible” threshold, GT/candidate sampling probabilities, or the empty-candidate fallback beyond retaining the logged target.

### Metric hierarchy

- NAVSIM v1: NC and DAC determine hard feasibility; DDC is an additional compliance guard; EP, TTC, and comfort are compared inside the feasible region (p. 4).
- NAVSIM v2: the paper says it follows the same principle while adding DDC and TLC to feasibility/compliance checks, then compares continuous planning metrics (p. 4). Table 3 further states that NC, DAC, DDC, and TLC are multiplicative compliance metrics, while TTC, EP, LK, HC, and EC form the weighted EPDMS component (p. 7).
- The paper does not provide the exact NAVSIM v1/v2 score formula, thresholds, scorer commit/version, or numeric metric guards.

### Stage 2: FF-PGRPO

- It starts at `pi_initial` and samples `G` trajectories per scene (p. 4); implementation details later fix this to 16 rollouts per scene (p. 5).
- A coherent scene reference is the highest-scoring feasible trajectory among the logged trajectory, retained candidates, and the single-trajectory prediction of `pi_initial`. All reference metrics therefore come from one executable trajectory, not a component-wise envelope (p. 4).
- On NAVSIM v1 a rollout must satisfy NC and DAC; DDC must pass an absolute or reference-relative guard; a reference-relative EP floor prevents improved TTC merely by slowing down. EP, TTC, and comfort define the admissible Pareto front after these checks (p. 4).
- Group quality is based on PDMS. Mixed groups additionally penalize insufficient EP and joint EP/TTC regression relative to the reference. All-feasible groups normalize only the continuous part of PDMS because binary feasibility terms are already satisfied (p. 4).
- Eq. (4), for a group with at least one feasible rollout, assigns `max(A_hat_i, 0)` to a rollout in the admissible Pareto front, `min(A_hat_i, 0)` to a feasible but non-Pareto rollout, and `-lambda_unsafe + min(A_hat_i, 0)` to an infeasible rollout (pp. 4--5).
- A feasible rollout that fails reference guards cannot receive positive credit. Aggregate quality controls update strength, while feasibility, reference performance, and Pareto membership gate positive credit (p. 5).
- For an all-infeasible group, no rollout is positive; non-positive credit depends on violation severity, weak diffusion supervision from a safe reference is added when available, and the group is downweighted (p. 5).
- Updates are regularized toward `pi_initial` with BC or KL terms (p. 5).

Not specified: exact coherent-reference fallback if none of its sources is feasible, DDC/EP guard values, mixed-group penalty formula, continuous-PDMS construction, `lambda_unsafe`, violation-severity formula, all-infeasible group weight, safe-reference recovery weight, advantage clipping, BC/KL weights or annealing, optimizer details, and how trajectory advantage is shared across diffusion denoising transitions beyond accumulated log-probability.

### Stage 3: APR

- The pool contains trajectories from other policy seeds, structured trajectory expanders, and policies specialized for safety, progress, and trajectory regularity (p. 5). Figure 2 depicts sources as Current, GRPO, `pi_initial`, and History.
- The current policy's single-trajectory prediction is the per-scene baseline. A teacher must be feasible, exceed the baseline aggregate score by a margin, and preserve protected metrics. Preference favors larger improvement, smaller behavioral distance, and lower curvature/jerk (p. 5).
- The chosen teacher is interpolated with the current prediction over a short descending list of ratios. Every interpolated trajectory is re-evaluated, and the largest local target that remains feasible, improves aggregate score, preserves protected metrics, and stays compatible with current-policy rollouts is retained (p. 5).
- Accepted targets are weighted by verified score gains and trained with the same diffusion objective. Training retains current predictions and adds distillation or KL control; scenes with no valid teacher contribute only retention (p. 5).
- Safety-, progress-, and structure-oriented branches may be trained separately and combined with a constrained, anchor-aligned task-vector merge (p. 5).
- After each round the validated checkpoint becomes current, predictions and rollout bank are regenerated, and the teacher set is rebuilt. Obsolete teachers leave and previously unsuitable candidates may activate (p. 5).
- Three APR rounds are reported (pp. 5 and 7).

Not specified: teacher-source counts/proportions, score margin, protected-metric tolerances, distance/curvature/jerk formulas, interpolation ratios, weighting temperature/clip, loss weights, retention/teacher sampling mix, per-round steps/epochs, teacher promotion/expiry thresholds, stop criterion, or any task-vector mask/sparsity/norm/anchor parameters.

## Experimental protocol stated in the PDF

- Benchmarks: NAVSIM v1 PDMS and NAVSIM v2 EPDMS (p. 5).
- NAVSIM v1 components stated in setup: NC, DAC, TTC, C, and EP (p. 5).
- NAVSIM v2 additionally includes DDC, TLC, LK, HC, and EC (p. 5).
- Backbone/planner: InternVL3-2B with the ReCogDrive diffusion planner (p. 5).
- GT-only and multi-trajectory IL variants: **200 epochs** (p. 5).
- GRPO: **16 rollouts per scene** and **10 epochs** (p. 5).
- APR: **3 rounds** (p. 5).
- Hardware: **8 NVIDIA A800 GPUs** (p. 5).
- Controlled comparisons are claimed to use the same initialization, trajectory representation, optimizer, training budget, and evaluation protocol (p. 5).
- In the Score/Pareto/PC-MTS comparison, the same IL objective is followed by the same original scalar GRPO. Score uses aggregate-score selection; Pareto adds component-wise Pareto filtering; PC-MTS additionally removes candidates incompatible with frozen `pi_GT` (p. 5).
- The PDF does not state dataset split identifiers, scorer version, evaluation seed, training seeds, number of independent runs, mean/std or confidence intervals, optimizer name, learning rate, batch/accumulation, precision, warmup, weight decay, clipping, diffusion steps, wall-clock time, storage, offline scoring cost, or whether each baseline is reported or locally reproduced.

## Figures

### Figure 1 (p. 1)

- Panel (a) contrasts conventional score/Pareto/geometric-diversity selection with policy-compatible selection. The conceptual claim is that individually strong candidates may lie far from the frozen policy's rollout support.
- Panel (b) shows representative trajectories and the same four IL/GRPO pairs as Table 2: GT `86.4 -> 90.3`, Score `87.2 -> 85.8`, Pareto `87.5 -> 86.7`, and “Ours”/PC-MTS `86.9 -> 91.1`.
- No scene identifier, compatibility distance, candidate count, or statistical uncertainty is shown.

### Figure 2 (p. 3)

- Four panels depict the driving VLA pipeline, PC-MTS, FF-PGRPO, and APR.
- FF-PGRPO's legend distinguishes sampled, unsafe, feasible-but-dominated, and feasible Pareto-front trajectories, with safe-reference recovery for unsafe outcomes.
- APR depicts iterative local interpolation/validation, teacher auditing, weighted distillation/retention, and a multi-source pool.

### Figure 3 (p. 6)

- Two qualitative scenes: command “Turn left” and command “Turn right,” shown in front-camera and bird's-eye views.
- Trajectories shown: AMPT, ReCogDrive, DiffusionDriveV2, and GT.
- The figure contains no scene IDs, component scores, failure labels, or quantitative conclusion, so it cannot by itself support per-scene recovery or safety attribution.

## Exact experimental results

### Table 1: NAVSIM v1, single-trajectory inference (p. 6)

All metrics are higher-is-better.

| Method | NC | DAC | TTC | C | EP | PDMS |
|---|---:|---:|---:|---:|---:|---:|
| LAW | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| FSDrive | 98.2 | 93.8 | 93.3 | 99.9 | 80.1 | 85.1 |
| Hydra-MDP | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| DiffusionDrive | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| PWM | 98.6 | 95.9 | 95.4 | 100 | 81.8 | 88.1 |
| AutoVLA | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveVLA-W0 | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| GoalFlow | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| Curious-VLA | 98.4 | 96.9 | 97.9 | 98.1 | 88.5 | 90.3 |
| AdathinkDrive | 98.4 | 97.8 | 95.2 | 100 | 84.4 | 90.3 |
| ReCogDrive | 97.9 | 97.3 | 94.9 | 100 | 87.3 | 90.8 |
| DiffusionDriveV2 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| AMPT | 98.5 | 98.0 | 96.0 | 100 | 86.7 | 91.4 |

The prose calls AMPT best among the listed methods and 0.2 PDMS above DiffusionDriveV2. Relative to that row, the component differences are consistent with the prose: NC `+0.2`, DAC `+0.1`, TTC `+1.2`, C `+0.1`, and EP `-0.8`. The PDF does not document protocol matching or uncertainty for these baseline rows.

### Table 2: trajectory-set construction (p. 6)

| Data | IL PDMS | scalar-GRPO PDMS | Delta |
|---|---:|---:|---:|
| GT | 86.4 | 90.3 | +3.9 |
| Score | 87.2 | 85.8 | -1.4 |
| Pareto | 87.5 | 86.7 | -0.8 |
| PC-MTS | 86.9 | 91.1 | +4.2 |

The arithmetic deltas are correct at the displayed precision. No candidate-count/source matching data, seeds, variance, or fixed-budget learning curve is provided. Note that the no-stage row in Table 4 is 86.5 rather than Table 2's GT IL value of 86.4; the PDF does not explain whether this is rounding, a different run, or a different checkpoint/protocol.

### Table 3: NAVSIM v2, single-trajectory inference (p. 7)

All metrics are higher-is-better.

| Method | NC | DAC | DDC | TLC | TTC | EP | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TransFuser | 97.7 | 92.8 | 98.3 | 99.9 | 92.8 | 79.2 | 67.6 | 100 | 95.3 | 77.8 |
| ReCogDrive | 98.3 | 95.2 | 99.5 | 99.8 | 97.5 | 87.1 | 96.6 | 98.3 | 86.5 | 83.6 |
| Hydra-MDP++ | 98.8 | 97.8 | 99.1 | 100 | 95.3 | 84.0 | 70.1 | 100 | 96.8 | 84.1 |
| DiffusionDrive | 98.2 | 95.9 | 99.4 | 99.8 | 97.3 | 87.5 | 96.8 | 98.3 | 87.7 | 84.5 |
| Curious-VLA | 98.4 | 96.9 | 99.2 | 99.8 | 97.9 | 88.5 | 96.9 | 98.1 | 81.5 | 85.3 |
| DiffusionDriveV2 | 97.7 | 96.6 | 99.2 | 99.8 | 97.2 | 88.9 | 96.0 | 97.8 | 91.0 | 85.5 |
| DriveVLA-W0 | 98.5 | 99.1 | 98.0 | 99.7 | 98.1 | 86.4 | 93.2 | 97.9 | 58.9 | 86.1 |
| Drive-JEPA | 98.4 | 98.6 | 99.1 | 99.8 | 97.8 | 88.4 | 97.6 | 97.9 | 84.8 | 87.8 |
| AMPT | 98.4 | 97.7 | 99.0 | 99.7 | 97.7 | 89.4 | 91.7 | 97.1 | 87.7 | 89.1 |

The prose correctly gives the AMPT--Drive-JEPA aggregate gap as `+1.3` EPDMS and associates it with EP (`+1.0`) and EC (`+2.9`). AMPT is lower on DAC, DDC, TLC, TTC, LK, and HC and tied on NC; the prose qualifies these compliance/safety metrics as close to saturated rather than claiming component-wise dominance.

### Table 4: three-stage ablation on NAVSIM v1 (p. 7)

| PC-MTS | FF-PGRPO | APR | NC | DAC | TTC | C | EP | PDMS |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| no | no | no | 98.1 | 94.7 | 94.2 | 100 | 80.9 | 86.5 |
| yes | no | no | 98.2 | 95.2 | 94.5 | 100 | 81.3 | 86.9 |
| no | yes | no | 98.4 | 98.0 | 95.8 | 100 | 85.93 | 91.0 |
| yes | yes | no | 98.5 | 98.0 | 96.0 | 100 | 86.0 | 91.1 |
| yes | yes | yes | 98.5 | 98.0 | 96.0 | 100 | 86.7 | 91.4 |

The text rounds `85.93` to `85.9`. It attributes the largest gain to FF-PGRPO, PC-MTS primarily to avoiding a poor GRPO initialization, and APR to EP improvement while preserving the displayed NC/DAC/TTC/C values. The table does not state seeds, errors, or whether rows use equal total optimization steps. The `FF-PGRPO=yes, PC-MTS=no` row establishes that FF-PGRPO can be applied without PC-MTS, but the initialization for that row is not explicitly labeled beyond the surrounding “GT-only policy” prose.

### Table 5: APR rounds (p. 7)

| Policy | PDMS | Gain per round |
|---|---:|---:|
| Round 0, before APR | 91.10 | -- |
| Round 1 | 91.21 | +0.11 |
| Round 2 | 91.37 | +0.16 |
| Round 3 | 91.45 | +0.08 |

The deltas are arithmetically consistent and monotonic. The final value is reported to two decimals here and rounded to 91.4 in the main tables/abstract. The paper interprets monotonicity as support for dynamic teacher reconstruction, but it provides no fixed-teacher, one-shot, no-interpolation, no-retention, or extra-steps control and no multi-seed uncertainty; therefore Table 5 alone does not isolate that causal mechanism.

### Failure recovery (pp. 5 and 7)

- Initial hard set: **658 scenes**, selected from the ReCogDrive IL policy because collision or drivable-area violation occurred in all **5 rounds of sampling**, yielding PDMS 0 (p. 5).
- Original scalar GRPO: **367/658** recovered, reported as **55.8%** (p. 7).
- AMPT: **440/658** recovered, reported as **66.9%** (p. 7).
- Difference: **73 scenes** and **11.1 percentage points** (pp. 2 and 7).

The displayed percentages match the counts after rounding. The abstract says “11.1% higher,” but the body correctly says “11.1 percentage points.” Relative improvement over 55.8% would be a different quantity; the appendix should use “percentage points.” The PDF reports gross recovery only: it does not provide scene IDs, persistent failures, retained successes, new failures, net recovery, a transition matrix, failure types, seeds, or a confidence interval.

## Sanity-check against the user's target numbers

Every target number supplied by the user is present and consistent with the PDF:

| Target | PDF evidence | Status |
|---|---|---|
| NAVSIM v1 AMPT: NC 98.5, DAC 98.0, TTC 96.0, C 100, EP 86.7, PDMS 91.4 | Table 1, p. 6 | Exact match. |
| NAVSIM v2 AMPT: NC 98.4, DAC 97.7, DDC 99.0, TLC 99.7, TTC 97.7, EP 89.4, LK 91.7, HC 97.1, EC 87.7, EPDMS 89.1 | Table 3, p. 7 | Exact match. |
| GT 86.4 -> 90.3; Score 87.2 -> 85.8; Pareto 87.5 -> 86.7; PC-MTS 86.9 -> 91.1 | Figure 1, p. 1 and Table 2, p. 6 | Exact match. |
| Stage ablation PDMS 86.5, 86.9, 91.0, 91.1, 91.4 | Table 4, p. 7 | Exact match. |
| APR rounds 91.10 -> 91.21 -> 91.37 -> 91.45 | Table 5, p. 7 | Exact match. |
| 658 failures; scalar GRPO 367; AMPT 440; difference 73 | pp. 2, 5, and 7 | Exact match. |
| InternVL3-2B + ReCogDrive diffusion; IL 200 epochs; GRPO 16 rollouts/scene for 10 epochs; APR 3 rounds; 8 A800 GPUs | Implementation details, p. 5 | Exact match. |

No target conflicts with the PDF. The only interpretation issue is the abstract's “11.1% higher,” which should be written as an 11.1-percentage-point absolute recovery-rate gain.

## Naming and wording audit of the PDF

- **AMPT** is expanded as **Aligned Multi-Trajectory Policy Training** (p. 2) and is used consistently for the full method.
- **PC-MTS** is consistently **Policy-Compatible Multi-Trajectory Supervision**.
- **FF-PGRPO** is consistently **Feasibility-First Pareto GRPO**.
- The current Stage 3 name is used in the method heading, Figure 2, compared-variant definition, Table 4, and conclusion as **APR — Adaptive Pareto-Guided Policy Refinement**.
- **Naming defect:** the third contribution bullet on p. 2 instead says **“Iterative Pareto-guided policy refinement (IPR)”**. This is the only `IPR` occurrence visible in the PDF and must be normalized to APR in the supplement. Earlier on p. 2 the introductory overview shortens the expansion to “adaptive Pareto refinement (APR)”; the supplement should use the full current expansion.
- No `A/B/C`, `A1/A2/A`, or other old stage labels appear in this PDF.
- Minor text defects visible in the PDF: “driveing” in the Figure 2 caption (p. 3), missing spaces around “are removed” and “improvements relative” in the APR-round discussion (p. 7), and the abstract's ambiguous use of percent rather than percentage points for recovery. These should be suggestions for the main text, not silently altered experimental claims.

## Explicit supplementary-material commitments and evidence gaps

The main paper explicitly promises only two categories of deferred detail:

1. **“Complete hyperparameters are provided in the supplementary materials”** (Implementation details, p. 5).
2. **Detailed constrained task-vector merging rules are left to the supplementary material** (APR, p. 5).

To make the supplement consistent with the main paper without inventing evidence, the following absent items need either repository evidence or an explicit missing-evidence marker:

- PC-MTS candidate sources/counts, score and metric thresholds, policy rollout-bank size, `K`, command calibration quantile and held-out separation, distance normalization, perturbation process/count/pass threshold, target sampling probabilities, and fallback behavior.
- FF-PGRPO exact reference fallback, feasibility/EP/DDC guards, mixed-group shaping, `lambda_unsafe`, violation severity, normalization/clipping, group downweight, recovery/BC/KL weights, schedules, and diffusion-step credit implementation.
- APR teacher composition, audit margins/tolerances, distance/smoothness definitions, interpolation list, advantage weighting, retention sampling/loss, per-round budget/acceptance/expiry/stop rules, and constrained task-vector merge parameters.
- Optimizer/LR/batch/accumulation/precision/warmup/weight decay/gradient clipping/diffusion sampler and steps; model freezing; split and scorer versions; seeds and number of independent runs; runtime and storage; baseline reported-vs-reproduced protocol.
- Scene-level support for the 658-scene recovery counts and any net-recovery, new-failure, failure-type, or confidence-interval statement.
- Candidate-count/source/budget matching and rollout diagnostics needed to establish that Table 2 is not a candidate-count or training-budget effect.
- Controls needed to attribute APR's small gain specifically to dynamic teacher rebuilding rather than extra training steps.

## Reference inventory as printed (32 entries)

The following entries are present on pp. 8--9; this is an inventory of the manuscript bibliography, not an external verification of bibliographic correctness.

1. Ang et al. (2026), *CLOVER: Closed-Loop Value Estimation & Ranking for End-to-End Autonomous Driving Planning*, arXiv:2605.15120.
2. Cao et al. (2025), *Pseudo-simulation for autonomous driving*, arXiv:2506.04218.
3. Chen et al. (2026), *Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models*, arXiv:2603.06049.
4. Chi et al. (2023), *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS.
5. Chitta et al. (2022), *Transfuser: Imitation with transformer-based sensor fusion for autonomous driving*, IEEE TPAMI 45(11), 12878--12895.
6. Dauner et al. (2024), *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*, NeurIPS 37 Datasets and Benchmarks Track.
7. Hwang et al. (2024), *EMMA: End-to-End Multimodal Model for Autonomous Driving*, arXiv:2410.23262.
8. Jiang et al. (2025), *AlphaDrive: Unleashing the Power of VLMs in Autonomous Driving via Reinforcement Learning and Reasoning*, arXiv:2503.07608.
9. Jiao et al. (2025), *EvaDrive: Evolutionary Adversarial Policy Optimization for End-to-End Autonomous Driving*, arXiv:2508.09158.
10. Kumar et al. (2019), *Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction*, NeurIPS 32, 11761--11771.
11. Laroche, Trichelair, and Tachet des Combes (2019), *Safe Policy Improvement with Baseline Bootstrapping*, ICML/PMLR 97, 3652--3661.
12. Li et al. (2025a), *Hydra-MDP++: Advancing end-to-end driving via expert-guided hydra-distillation*, arXiv:2503.12820.
13. Li et al. (2025b), *Enhancing End-to-End Autonomous Driving with Latent World Model*, ICLR.
14. Li et al. (2025c), *DriveVLA-W0: World models amplify data scaling law in autonomous driving*, arXiv:2510.12796.
15. Li et al. (2025d), *ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving*, arXiv:2506.08052.
16. Li et al. (2024), *Hydra-MDP: End-to-end multimodal planning with multi-target hydra-distillation*, arXiv:2406.06978.
17. Liao et al. (2025), *DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving*, CVPR, 12037--12047.
18. Luo et al. (2025), *AdathinkDrive: Adaptive thinking via reinforcement learning for autonomous driving*, arXiv:2509.13769.
19. Nair et al. (2020), *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*, arXiv:2006.09359.
20. Tang et al. (2025), *Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling*, arXiv:2505.17659.
21. Tian et al. (2025), *DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models*, CoRL/PMLR 270, 4698--4726.
22. Wang et al. (2025a), *HMAD: Advancing E2E Driving with Anchored Offset Proposals and Simulation-Supervised Multi-Target Scoring*, arXiv:2505.23129.
23. Wang et al. (2026), *Drive-JEPA: Video JEPA Meets Multimodal Trajectory Distillation for End-to-End Driving*, arXiv:2601.22032.
24. Wang et al. (2025b), *OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning*, CVPR, 22442--22452.
25. Xing et al. (2025), *GoalFlow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving*, CVPR, 1602--1611.
26. Yao et al. (2026), *HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving*, arXiv:2604.03581.
27. Zeng et al. (2026), *FutureSightDrive: Thinking visually with spatio-temporal cot for autonomous driving*, NeurIPS 38, 67299--67318.
28. Zhao et al. (2026), *From forecasting to planning: Policy world model for collaborative state-action prediction*, NeurIPS 38, 134585--134611.
29. Zhou et al. (2025a), *OpenDriveVLA: Towards End-to-End Autonomous Driving with Large Vision Language Action Model*, arXiv:2503.23463.
30. Zhou et al. (2025b), *AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning*, NeurIPS 38.
31. Zhu et al. (2025), *InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models*, arXiv:2504.10479.
32. Zou et al. (2025), *DiffusionDriveV2: Reinforcement Learning-Constrained Truncated Diffusion Modeling in End-to-End Autonomous Driving*, arXiv:2512.07745.

## Claims that should be worded cautiously in the supplement

- “SOTA” on NAVSIM v1 is qualified in the paper as best **among the listed methods**. The supplement should preserve that scope or avoid the label because baseline protocol matching is not documented in the PDF.
- The supervision--optimization mismatch is supported by Table 2's four point estimates, but candidate-count/source matching and multi-seed uncertainty are absent from the PDF. Do not elevate it to a statistically stable conclusion without repository evidence.
- APR's three single-run increments are small (`0.11`, `0.16`, `0.08`) and monotonic, but no control isolates dynamic reconstruction from extra training. The supplement should call this “consistent with” the mechanism, matching the paper's phrasing, unless stronger evidence exists.
- Failure recovery is **gross recovery**, not net recovery: no new-failure count is reported.
- Table 1 and Table 3 are single-trajectory result tables, but the PDF does not distinguish local reproduction from externally reported baselines or document identical sensors/training data. Any fairness table must source those details elsewhere.
