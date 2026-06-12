# ReCogDrive Route Status And Pruning Plan

Date: 2026-06-12

This repository currently contains several generations of ReCogDrive improvement routes. The active development path should be kept narrow: A0 official-aligned baseline plus the current `two_expert_slot` route. Older routes should remain reproducible only as archived experiments unless they have measured navtest gains over A0.

## Keep As Active

| Route | Why keep | Active files |
| --- | --- | --- |
| A0 official-aligned Stage2 | Main baseline. Best full navtest PDMS is `0.864891` at `step_00100000`. | core ReCogDrive agent/planner configs |
| `two_expert_slot` | Current hypothesis: VLM-internal soft expert slots plus two teacher adapters and CoWorld-style horizon conditioning. Not yet production-ready, but direction matches the latest design constraints. | `navsim/agents/recogdrive/two_expert_*.py`, `configs/last_vla_v2/two_expert_slot/`, `scripts/last_vla_v2/two_expert_slot/`, `tests/test_two_expert_*.py` |

## Keep Only As Legacy / Comparison

| Route | Status | Reason |
| --- | --- | --- |
| A4 direct expert path | Legacy comparison only | It is explicitly excluded from strict LaST/two-expert adaptation because it injects direct expert context into the action side. |
| LastRD | Legacy comparison only | Useful as a prior latent-reasoning baseline, but it is not the current route and should not be extended while `two_expert_slot` is active. |
| Last-VLA action-side CoT / `LastVLACoTTransformer` | Archive/freeze | It is external to the VLM hidden sequence. The current design requires VLM-internal soft slots, not post-hoc action-side CoT. |
| VLM LoRA CoT alignment sweep | Archive/freeze | It did not show confirmed navtest improvement, and the old LoRA Stage2 path was observed to hit NaN during training. Keep only minimal adapter-loading patterns needed for explicit future experiments. |

## Prune Or Archive Candidates

| Candidate | Evidence | Recommendation |
| --- | --- | --- |
| Hard bottleneck Last-VLA configs/scripts | Already moved under archive in part; design is now forbidden. | Keep archived readme only; delete active launchers/config references after confirming no running jobs depend on them. |
| Summary replacement / teacher trajectory / residual diffusion variants | Latest constraints forbid summary replacement, teacher-traj final targets, and residual/coarse+residual outputs. | Remove from active configs and tests; keep minimal historical report links. |
| `highcap_no_risk` Server A | Full navtest best PDMS `0.471093`, far below A0 `0.864891`. | Archive all launchers/configs; do not keep in active runbooks. |
| `decoupled_highcap_no_risk` A progressive/COT-only | Full navtest: COT-only `0.561656`; progressive `0.374729` and `0.302877`; B LoRA direct-online `0.766702`, still below A0. | Archive as negative result. Do not extend this family. |
| Old LoRA Stage2 branch | User-reported Stage2 NaN during training, plus no confirmed PDMS gain. | Treat as failed/unstable. Do not reuse as a base for current two-expert work. |
| Residual-anchor live eval launchers | Residual diffusion is now explicitly forbidden for current routes. | Archive or delete after checking no downstream scripts import them. |
| Stage3 RL/GRPO scripts on this branch | Unrelated to two-expert slot readiness and currently dirty/untracked. | Move to a separate branch or keep out of this branch's cleanup commit. |

## Simplification Target

The long-term cleanup should split the current monolithic route logic:

1. Keep base A0 diffusion logic in `recogdrive_diffusion_planner.py`.
2. Move legacy A4/LastRD/Last-VLA action-side route code into isolated legacy modules.
3. Keep `two_expert_slot` in its own small set of modules and one explicit config switch.
4. Replace route combinations of booleans with a single route enum or strict mutual-exclusion validator.
5. Keep tests for archived routes only at smoke/import level, not full matrix coverage.

## Safe Immediate Cleanup

Safe without changing model behavior:

- Remove generated `__pycache__` and `.pyc` files from the workspace.
- Add route-status documentation and mark negative-result routes as archived.
- Stop adding new features to `LastVLACoTTransformer`.
- Keep current P0 readiness work scoped to `two_expert_slot`.
- Keep LoRA non-default in new work until there is a specific finite-loss smoke and hidden-cache compatibility test.

Defer until explicit deletion approval:

- Deleting legacy configs/scripts.
- Removing old planner branches from `recogdrive_diffusion_planner.py`.
- Removing old tests that still document historical invariants.
