# Five-checkpoint training/distribution analysis

Read-only follow-up using the original 1000-scene, 5-checkpoint, 64-draw caches.
Adds actual MTS support/archive and TensorBoard analysis, true NAVSIM scoring of
center/residual geometric interventions, and the verified historical IL→GRPO chain.

All new artifacts use `outputs/five_checkpoint_training_distribution/`; no training
or original-cache writes occur. Historical MTS models are not direct fine-tunes
of the released IL action head. The report distinguishes this confound.

Run from the project root with the `navsim` environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python tools/analysis/five_checkpoint_training_distribution/prepare.py
python tools/analysis/five_checkpoint_training_distribution/score_counterfactuals.py --workers 96
python tools/analysis/five_checkpoint_training_distribution/teacher_audit.py
python tools/analysis/five_checkpoint_training_distribution/analyze.py
python tools/analysis/five_checkpoint_training_distribution/figures.py
python tools/analysis/five_checkpoint_training_distribution/test_mechanisms.py
```

For provenance replay, run `replay_smoke.py --model MODEL` once per checkpoint,
in separate processes (CUDA_VISIBLE_DEVICES can assign available GPUs). The
current-runtime repeat must be deterministic. Bitwise cache matching and actual
score differences are reported independently, never hidden by rewriting caches.

Then run `audit.py`, `pack_metrics.py`, and `report.py`. Tables larger than3MiB
are published as lossless Parquet with all rows and columns; CSVs remain local.
The historical GRPO identity audit verifies
that the training run's epoch8-step11970 file and requested90.41 file have identical
SHA256; its evidence is saved in `audits/grpo_9041_lineage.json`.

Important definitions:

- Main sampler is evaluation DDIM5/eta1/FP32, not the GRPO training sampler.
- Main HQ threshold is per-scene IL64 mean +1 point with NC/DAC/TTC/DDC all1.
- Teacher expected weights are pre-residual-budget reconstructions; historical
  post-budget weights are read from actual TensorBoard, not reconstructed.
- Counterfactual scores are true NAVSIM outputs. Center/residual attribution is
  an offline geometric intervention, not a training causal-effect estimate.
- All1000 scenes remain in the main tables. Undefined useful diversity/teacher
  metrics preserve NA and paired-intersection denominators.
- The configuration freezes an exploratory follow-up, not a hypothesis-blind
  preregistration, since V1 outcomes were already known.
