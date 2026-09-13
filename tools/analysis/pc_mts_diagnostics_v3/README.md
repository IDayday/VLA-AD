# PC-MTS V3: preregistered mechanism and controlled optimization diagnostics

Base: `f499be15d693a6910593ec2c8f081c249cdbf871`. V1/V2 are immutable inputs.
The 192-candidate reservoir is a **controlled diagnostic space, not historical training data**.
All new artifacts live under `outputs/pc_mts_diagnostics_v3/`.

## Frozen protocol

`configs/pc_mts_diagnostics_v3/primary.yaml` was hashed before Experiment A.
`manifests/protocol_frozen.json` contains the exact hash, UTC timestamp and choices.
`prepare.py --verify` verifies every original V1/V2 byte; only append-only output
from preexisting live pressure-test logs is permitted. No unrelated process is stopped.

The Conditional selector uses hard NC/DAC safety, calibrated `q<95`, reference
quality and light local safety **before** recomputing Pareto ranks. It visits all
conditional fronts among Boundary candidates before Core, and uses Core nearest
the boundary first. A separate equal-eligibility Front≤2 comparison isolates gate
order; it must not be conflated with the whole selector's coverage change.
`q` is a held-out empirical distance rank, never a probability or likelihood.

Primary pools contain only unique raw parents. Primary PDMS≥reference and
secondary PDMS≥reference−1 are recorded separately. For training, repeating a
parent into 16 slots uses weight `1/(unique_count * parent_multiplicity)` per slot.
Zero-parent scenes explicitly use a GT training target; this is never counted as
strict PC coverage. The frozen Old-PC baseline is also reported in its original
16-slot form, while new primary comparisons exclude its synthetic fillers.

## Execution

Use `/root/miniconda3/envs/navsim/bin/python`, FP32, and environment variables
`PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

1. `prepare.py`; `static.py A`; `static.py B`; `native.py`; `matching.py`.
2. `torchrun --standalone --nproc_per_node=8 learnability.py`; `analyze_c.py`.
3. `torchrun --standalone --nproc_per_node=8 gradient.py`; `analyze_d.py`; `tests.py`.
4. `launch_training.py sft`; `torchrun --standalone --nproc_per_node=8 evaluate.py sft`;
   `scoring.py sft`; `analyze_training.py sft`.
5. `grpo_audit.py`; `launch_training.py grpo`; corresponding evaluation/scoring/analysis.
6. `torchrun --standalone --nproc_per_node=8 progressive.py`; `analyze_progressive.py`.
7. `torchrun --standalone --nproc_per_node=8 null_support.py`; `analyze_null.py`.
   This unchanged-IL finite-bank turnover control was declared separately after
   A–E and before any G measurement, in `calibration_control.yaml`. It is an
   additive supplementary control, not part of the original primary protocol.
8. `additional_audits.py`; `initialization_audit.py`; `supplementary.py`;
   `figures.py`; `tests.py`; `prepare.py --verify`; `final_audit.py`; `report.py`.
9. Review all figures and conclusions, then `package_results.py --stage`, commit
   and ordinary push to the V3 branch. It stages V3 paths only, includes tables
   up to 20 MB each, and excludes checkpoints, rollout caches and runtime logs.
   `PUBLICATION_INDEX.json` records published file hashes and the local-only
   inventory identities. After GPU work, `restore_idle_pressure.py` honors
   `AGENT.md` using new V3 logs and never signals existing processes.

`pipeline.py` records and executes the sequence after C inference. `run_phase.py`
preserves commands, exit codes, code hashes and passive resource telemetry.
Failures stop the pipeline; scientific negative findings do not change parameters.
The one-time `repair_c_schema.py` converts a task-generated NPZ string identifier
array to non-pickle Unicode and verifies that every numerical measurement is
bit-identical. It never reruns or changes model measurements.

## Interpretation limits

The 700/300 split holds out tokens **from the new updates**. These are Navtrain
scenes, potentially seen during historical model training. Some training and
holdout tokens share logs, and the frozen V1 GT-distance threshold was calibrated
using the original full 1000 scenes. This is not an untouched Navtest benchmark.
All actual new SFT/GRPO updates assert membership in the 700-token training set.
No future labels or scores enter model inference conditioning.

Diffusion MSE/reconstruction comparisons use exact-source and PDMS matching,
common timestep/noise, pair bootstrap and scene-cluster bootstrap. Regressions
absorb scene and exact-source fixed effects; 3000 wild scene-cluster bootstrap
draws quantify fixed-design regression uncertainty. Training comparisons report
both independent seeds and scene-paired intervals; two seeds do not establish
robustness across the population of optimizer random seeds.

GRPO calls the original archived `forward_grpo` directly. The only hook batches
the native reward evaluator and logs raw metrics and advantages. Four-scene
scalar-reward and complete-loss parity must pass before GRPO updates. Training
reward uses the original EP/TTC/comfort weights 10/5/2; evaluation uses 5/5/2.
Reference policy is the same frozen official IL for every initialization.
`CRN_policy_change_ADE` is a paired-trajectory change proxy, not KL.

Checkpoints and large inference caches remain on the server. Published metrics,
paired manifests, audits and PNG/PDF figures trace to their SHA256 identities.
