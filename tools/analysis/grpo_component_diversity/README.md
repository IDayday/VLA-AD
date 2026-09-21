# GRPO component diversity: frozen IL vs historical original GRPO

This read-only analysis uses all 5,000 previously frozen scenes. It does not
train, choose successful scenes, modify historical caches, or reinterpret
outcome signatures as semantic driving modes. Internal Navtrain diagnostic;
not an untouched Navtest generalization benchmark.

## Execution

Use `/root/miniconda3/envs/navsim/bin/python` in this worktree.

1. `prepare.py` freezes primary configuration and verifies model/scene identities.
2. `sample_extra.py --model MODEL --smoke` on each of the two models checks native
   forward-GRPO output parity at G=8,16,32,64,128 and the inherited 64 draws.
3. `launch.py` runs eight inference workers (four per model), 80 persistent NAVSIM
   scoring workers, native reward/advantage audit, then component analysis.
4. `noise_decomposition.py --model MODEL` for both models, after GPU completion.
5. `outcome_summary.py`, `figures.py`, `final_audit.py`, `report.py`.

Only groups 4-7 are new, using the existing independent seed namespace and
actual native training sampler. Old groups 0-3 are read in place. Model draws
are paired with common random numbers. Native group-size parity uses fixed
per-noise-block Gaussian streams; it does not assume a single torch seed has
identical member/step alignment when the tensor shape changes.

## Statistical units and interpretations

- Primary: nested first G actual draws from each scene's 128 draws. Secondary:
  average metrics over all nonoverlapping groups of G. Original historical
  training used G8; these are read-only hypothetical groups, not training logs.
- 3,000 scene-paired and log-cluster bootstrap replicates. Empty conditional
  metrics remain NA; each output has its actual scene denominator. Holm
  adjustment covers only eight prespecified G16 endpoints. Bootstrap sign-tail
  diagnostics are not a reason to promote an exploratory result to causal proof.
- Four safety components use their actual discrete levels, including 0.5.
  Outcome entropy/count is sample-size dependent; disagreement is the fraction
  of distinct sampled pairs with different safety vectors. Neither is a
  semantic mode count. Greater failure entropy is not a success criterion.
- Safe EP headroom is best minus mean normalized EP among feasible members.
  Opportunity means at least two feasible members differing by 0.05 in EP.
  Its all-scene event denominator includes groups without any feasible member.
- Native FP32 CUDA advantage statements are extracted from the actual archived
  `forward_grpo`, not a reimplementation of generic PPO/GRPO. Reported component
  covariance with advantage describes reward coefficient alignment. It is not
  a measured parameter gradient or a prediction of realized training gain.
- Eval PDMS weights EP/TTC/Comfort 5/5/2; historical reward weights 10/5/2.
  Both multiply NC and DAC; DDC weight is zero. Scalar-equivalent NAVSIM scoring
  and real native train-score algebra are audited on four fixed scenes/model.
- Final-step conditional means in the small noise diagnostic are intermediate
  network outputs; no new deployable policy or PDMS is claimed for them.

Large NPZ banks and complete block-level table stay local. Publish code,
configuration, manifests, audit, scene-prefix tables, summaries and plot data.
