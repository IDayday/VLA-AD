# Matched continuation from official GT-IL

Primary config: `configs/psi_matched_sft/primary.yaml`; frozen hash in
`outputs/psi_matched_sft/manifests/protocol.json`. All historical assets are read-only.

Four arms: GT-only continuation, Score, Pareto, historical PSI selection kernel.
Each: two seeds, 512 updates, batch128, same native epsilon loss and full original
action head. 3072 training scenes have no log overlap with the existing 5000
evaluation scenes. No downstream GRPO updates are performed.

Original PSI full raw AWAC buffer is missing. This is a prospective selection
ablation on a shared available archived candidate bank, re-scored with NAVSIM-v1;
it is not an exact replay of historical PSI training. Raw V6 candidates are used,
never V6 admission masks or selected modes. `policy` source is the archived A5
checkpoint, not official IL. Source provenance is preserved.

Execution order:

1. `prepare.py` freezes token/log split and protocol before candidate scoring.
2. `features.py` via 8-GPU torchrun; `candidates.py` uses persistent CPU workers.
3. `audit.py` and `test_matched.py` validate scoring, identities and selection.
4. `orchestrate.py` waits for preparations, trains eight runs, evaluates fixed
   checkpoints using four disjoint server shards and persistent NAVSIM scoring.
5. `finalize.py` waits for completion, then analyzes, plots, audits and reports.
6. `parent_absorption.py` adds an explicitly descriptive post-training audit of
   actually presented parents on the previously frozen 512 training probes.
   Re-run `report.py` after placing the reviewed quantitative interpretation in
   `outputs/psi_matched_sft/report/interpretation.md`.

`multiscene.py` batches four scenes with four independent G16 groups each.
Every group retains its original random stream. Per-host parity must pass
against unbatched native GRPO and the old baseline cache (FP32 max error<1e-4).
This is a compute batching change, not a sampler/precision change.

Primary final results use every one of the 5000 scenes and average the two
training seeds within scene before paired bootstrap. Loss probes use common
targets, timestep and noise. Report actual teacher presentation counts alongside
fitting loss and rollout coverage; do not infer learnability from finite-sample
zero hits. Negative results never change thresholds, scenes or checkpoint choice.

Large local checkpoints/features/rollouts are excluded from version control.
Published target tables retain actual trajectories, source hashes and parent IDs.

Primary figures have CSV, PNG, PDF and SVG forms. `selected_source_composition.csv`
separates unique-parent counts from actual intended supervision weight. PSI changes
the target-count/quality/diversity/source bundle; this experiment does not isolate
one of those factors as the sole cause of a result. A sampler diagnostic is not a
downstream GRPO training intervention.
