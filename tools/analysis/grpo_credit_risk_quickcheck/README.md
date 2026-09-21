# GRPO credit / risk quickcheck

This isolated analysis worktree uses the archived A5 runtime in read-only mode.
Frozen input identities, scene manifest and cached trajectories live at:
`/mnt/project/VLA-AD/outputs/grpo_credit_risk_quickcheck/`.

- `prepare.py`: one-time freeze (refuses an existing resolved config).
- `features.py`, `sample.py`: observed inputs and real native/deployment sampling.
- `scoring.py`: replay interface using archived simulator and official scorer.
  `score(load_cache(scene), trajectories)` exposes official metrics, centerline
  progress/EP denominator and actual simulated states. `margins=True` adds
  time-aligned footprint descriptors. All metric scores are 0–1.
- `credit.py`: exact archived LFP credit plus two EP-neutral shadow chains.
- `matching.py`, `perturb.py`, `risk_probe.py`, `analyze_risk.py`: deterministic
  pairing and independent selection/evaluation perturbations.
- `pairwise_credit.py`: small detached, symmetric pairwise credit surrogate.
- `tiny_*.py`: gated tiny-update implementation, **campaign STOPPED** after the
  first-step identity failure. Do not resume or replace these failed runs.
- `test_probe.py`: 12 mathematical/identity regression tests; no optimizer calls.

The first two experiments are complete; offline signal is WITHIN_GROUP_SIGNAL.
The attempted tiny-update campaign is INVALID_STOPPED (8 total optimizer steps,
1 per arm/seed). Step8, updated-policy evaluation and norm control are NOT_RUN.
The corrected constructor-freeze handling was checked read-only. Do not confuse
this correction with validation of updated policy quality.

Full evidence: `reports/GRPO_CREDIT_RISK_QUICKCHECK.md` and the output directory's
`commands.sh`, `audits/`, `manifests/`, `logs/`. Historical training configuration,
canonical checkpoints, and original workspace modifications were preserved.
