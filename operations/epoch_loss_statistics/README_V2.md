# Single-seed epoch-statistics helper V2

The user's instruction on2026-10-27 cancels20261028 and retains four20261027
runs. This self-contained standard-library helper changes only the admitted run
set and completion label. Summary math, checkpoint SHA verification, sparse
per-run ETA scheduling and actual-functional-duration handling are preserved.
The training repository/HEAD and GPU jobs are outside this helper's scope.

Run:

```bash
python -B test_epoch_loss_statistics_v2.py
python -B epoch_loss_statistics_v2.py watch --state SCHEDULE_V2.json
```

The CLI rejects watch schedules containing cancelled/unregistered runs before
reading any training log. It reports `ALL_ACTIVE_RUNS_40_EPOCHS_COMPLETE` after
the four retained runs have40sealed complete epochs. The cancelled run entries
are archived separately as `CANCELLED_BY_USER`; they have no future read window
or completion expectation. Old V1 sources/schedules/proofs and all original
summaries remain intact. The new four-run summaries inherit the last verified
outputs and deadlines, without a fresh backfill.

`USER_INSTRUCTION.json` is referenced by SHA in the migrated schedule/admission.
Actual runtime schedule/result paths are explicit in the JSON, not hardcoded in
the helper. `summarize --results-root ... --out ...` remains an explicit one-shot
offline operation; this migration does not invoke it. Due-window reads by the
watcher are the existing approved normal schedule, never a new qualification.
