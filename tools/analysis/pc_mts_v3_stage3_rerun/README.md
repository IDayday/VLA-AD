# V3 formal stage3 GRPO rerun

The seven initializations reuse frozen V3 inputs; earlier experiment trees are
read-only. `primary.yaml` is frozen by `prepare.py` before training. Do not run
`prepare.py` again on an existing campaign.

The training adapter uses the actual `ReCogDriveAgent`, `AgentLightningDiT`,
native optimizer and epoch scheduler, native padding collate and Lightning
Trainer. Real NAVSIM reward batching must pass `parity.py native` first.
`parity.py host` verifies the unchanged V3 evaluation class and observations on
each execution host. Exact effective head tensors are checked on every load.

Set `PYTHONDONTWRITEBYTECODE=1`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
Use the audited Python 3.9 / Torch 2.5.1+cu124 / Lightning 2.6.0 environment.

Execution:

1. `python tests.py`, `python parity.py native`, `python parity.py host`.
2. `python host_runner.py` on each available complete eight-GPU host. This
   starts independent native DDP runs through an atomic shared job queue.
3. `torchrun --standalone --nproc_per_node=8 evaluate_r.py` on available GPUs.
   This is inference sharding, not distributed training; fewer GPUs are valid.
4. `python score_r.py --shard HOST_INDEX --shards HOST_COUNT --workers 32 --watch`.
5. Once all 14 trainings, 224 evaluation shards and 16,800 score files exist:
   `python analyze_r.py`, `python figures_r.py`, `python audit_r.py`,
   `python report_r.py`, `python tests.py`.

No best-checkpoint selection occurs. Primary endpoint is step 110 (10 actual
epochs on 700 scenes), while step 100 is the old V3 update-count comparison.
The repeated four sampler padding scenes per epoch are distinguished from
candidate-parent duplication. Fixed reference policy equals each run's own
initialization, following the formal launcher.

Interrupted task claims are retained for audit. Before retrying, inspect the
failure, move only that failed task's new artifacts to this namespace's
`invalid/` directory, and retry the same configuration and initialization.
Do not delete or invalidate successful earlier-version caches. Small CSV,
figures and manifests are suitable for Git; raw caches, checkpoints, training
logs and the large cache index stay on the server.
