# Missing Evidence

No manuscript-facing table contains an unexplained blank.  Unresolved
measurements are either printed as “not recorded/not established,” or supplied
as a concrete **proposed-unrun** control value that is never styled as an
observed result.  The authoritative path-level search record and commands are
in [`audit/missing_evidence.md`](audit/missing_evidence.md).

The remaining measured evidence is:

- three-seed outcomes for Score/Pareto/PC-MTS, Scalar-GRPO/FF-PGRPO, and
  no-APR/full-APR;
- paper-run candidate counts, matched acceptance rates, and rollout-bank
  diagnostics;
- full positive-credit traces and step-matched APR controls;
- end-to-end wall time/storage and exact resolved manifests for all reported
  stages.

The bounded reconstruction uses bank 16, K=3, quantile 0.95, eight local
perturbations with a 6/8 pass rule, and proposed training seeds
20260726/20260727/20260728.  These values fill the executable protocol, not the
missing result cells.  All launchers remain dry-run unless both `ALLOW_TRAIN=1`
and `--execute` are supplied.

The fixed 658-scene analysis is not missing: it is locked to the historical
paper pair, 367 versus 440 positive outcomes, with 93 repairs, 20 new failures,
and net repair 73.  The later 91.45 hard-set columns are excluded.
