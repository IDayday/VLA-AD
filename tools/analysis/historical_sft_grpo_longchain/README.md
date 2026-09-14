# Historical SFT/GRPO long-chain audit

Read-only June/July evidence. No model inference, training, GPU allocation, or recent quick-test input.

Run from `/mnt/project/VLA-AD` with the existing navsim environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/miniconda3/envs/navsim/bin/python tools/analysis/historical_sft_grpo_longchain/analyze.py
/root/miniconda3/envs/navsim/bin/python tools/analysis/historical_sft_grpo_longchain/supplement.py
/root/miniconda3/envs/navsim/bin/python tools/analysis/historical_sft_grpo_longchain/figures.py
/root/miniconda3/envs/navsim/bin/python -m unittest discover -s tools/analysis/historical_sft_grpo_longchain -p 'test_*.py' -v
/root/miniconda3/envs/navsim/bin/python tools/analysis/historical_sft_grpo_longchain/finalize.py
```

`analyze.py` parses historical token-level evaluations and TensorBoard, with 3000 scene-cluster bootstrap replicates. `supplement.py` audits resolved training configs, hashes/reads representative existing checkpoints, reconstructs saved V6 code, separates wrong-VLM evaluations, and reads a supplementary historical long-run control. It never creates or updates a model. Run the two analysis stages sequentially because they merge a shared input manifest.

Only small metrics/manifests, figures and the report are committed. Large derived scalar caches and reconstructed source trees remain local. Historical checkpoint peaks are descriptive selections, not unbiased test-selected causal endpoints. Original/PSI comparisons are not CRN; A5/V6 primary comparisons use their historical same-noise reevaluations. V6 `quality` is Comfort in NAVSIM v1, not PDMS; LFP feasibility is not the report's conservative feasibility.
