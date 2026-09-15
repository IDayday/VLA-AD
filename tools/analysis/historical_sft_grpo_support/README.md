# Historical SFT support and native G16 mechanism diagnostic

The question is whether historical multi-target SFT learned its supervision and
how ordinary inference differs from actual GRPO sampling. No optimizer is created
and no weight update is performed. Parent commit is `4716261be46b2bc3283a80f89377bb6eaf7dc3fa`.

Four SFT policies: released official GT-IL; PSI IL-initialized candidate SFT;
random-action-head A5 and V6 multi-target SFT. The VLM was frozen in all cases.
Five historical GRPO snapshots are included where real checkpoint entities exist.
PSI SR-PGRPO checkpoint entities remain missing; its historical logs are kept
separate from newly sampled evidence. APR is not substituted for that missing chain.

All1000 fixed scenes are NAVTRAIN.835 are common actual training scenes;165 were
PSI validation scenes. Teacher learning claims use the common835 primary scope;
trajectory distributions retain Full1000. These are internal fitting diagnostics,
not untouched Navtest performance estimates or a new training experiment.

Each scene/policy/protocol has4 independent groups of16 trajectories. Evaluation
and native GRPO use common initial and transition RNG streams. Native sampling
executes the actual archived `forward_grpo` / `forward_lfp_grpo` preamble, including
PTA conditioning; a hook captures the current chain before rewards or updates.
The original historical GRPO used G8; G16 is a standardized diagnostic, not a replay
of historical G8 advantages. Zero hits in64 draws are not proof of impossibility.

`primary.yaml` and all scene/model identities are frozen before sampling. The256
factorial scenes are token-hash selected. Changing only evaluation noise floor
and/or noise clipping is a diagnostic intervention, never a tuned training recipe.
New caches are atomic and validated by protocol/checkpoint/source hashes.

Historical runtimes: git-archived A5; git-archived V6 plus saved source patch and
one absent untracked dependency recovered from the V6 worktree; recoverable later
PSI commit. PSI byte-identical June source and absence of uncommitted A5 changes
cannot be proven. Runtime paths and hashes are exported, and loaded head schemas
are strict. Four-group batching is checked against the same unbatched RNG stream;
FP32 GEMM roundoff is explicitly reported (tolerance1e-4m/rad).

Teacher tensors are recovered from the original per-token A5/V6 support archives
and PSI support index. Every model is tested against the same union, as well as
its own supervision, so differences in teacher source/difficulty remain visible.
A5/V6 reconstructed weights are pre-residual-budget expectation. Real post-budget
mass comes only from historical logs. Exact duplicate supervision identities
remain traceable but are not called additional output modes.

The fitting diagnostic calls each model's native `forward`, observes the epsilon
MSE without rewriting normalization, and uses the same uniform timestep/epsilon
per scene and draw for all teachers/models. Observation dropout is disabled for
this frozen fitting measurement. Legacy and FS raw loss magnitudes have different
coordinate scaling and cannot be interpreted as a universal likelihood.

NAVSIM scoring is reused from the existing scalar-equivalent batch evaluator.
PDMS tables use0–100points. Training reward uses the actual10/5/2weights instead
of standard5/5/2. In this v1 evaluator DDC is diagnostic with zero score weight;
only NC and DAC multiply the weighted score. Conservative feasibility separately
requires NC=DAC=TTC=DDC=1. Every group mean/max/min is calculated on16members, then
groups and scenes are averaged equally. Best-safe is NA if no safe member exists.

Native LFP advantage code is evaluated on fixed diagnostic batches of64scenes
using the real historical reference cache. This is not historical minibatch or
curriculum replay. The universal vanilla-zscore proxy is clearly named separately;
it is not PSI SR-PGRPO or LFP credit assignment.

Commands (Python `/root/miniconda3/envs/navsim/bin/python`):

```bash
python tools/analysis/historical_sft_grpo_support/prepare.py
python tools/analysis/historical_sft_grpo_support/teachers.py
python tools/analysis/historical_sft_grpo_support/sample.py --model official_il --smoke
python tools/analysis/historical_sft_grpo_support/sample.py --model official_il --rank 0 --world 2
python tools/analysis/historical_sft_grpo_support/score.py --workers 80 --watch
python tools/analysis/historical_sft_grpo_support/batched.py --model original_grpo_11970 --rank 0 --world 2
python tools/analysis/historical_sft_grpo_support/fitting.py --model official_il
python tools/analysis/historical_sft_grpo_support/historical.py
python tools/analysis/historical_sft_grpo_support/advantage.py --model a5_sft
python tools/analysis/historical_sft_grpo_support/analyze.py
python tools/analysis/historical_sft_grpo_support/figures.py
python tools/analysis/historical_sft_grpo_support/report.py
python tools/analysis/historical_sft_grpo_support/publish_tables.py
python -m unittest discover -s tools/analysis/historical_sft_grpo_support -p 'test_*.py' -v
python tools/analysis/historical_sft_grpo_support/audit.py --scoring-smoke
python tools/analysis/historical_sft_grpo_support/audit.py
```

BLAS threads are1; CUDA TF32 is disabled. Large caches, checkpoints and runtime
source trees stay local. Existing historical/V1/V2/V3 files are read-only.
