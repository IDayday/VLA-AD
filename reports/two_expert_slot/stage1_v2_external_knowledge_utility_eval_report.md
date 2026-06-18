# Stage1-v2 External Knowledge Utility Evaluation

- checkpoint: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/stage1_clean/clean_replay_8gpu_local_warmup_status_20260616T013758Z/stage1.ckpt`
- eval root: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/eval/stage1_v2_full_eval_20260616_135252`
- git HEAD: `97773ec3798e1e2bba495e1b2fc77123a061f683`
- baseline A0 full navtest PDMS: `0.864891`

## Eval Subset

- count: `4096`
- source: `train_cache_deterministic_subset_not_true_holdout`
- strict held-out: `false`
- train overlap: `4096/4096`
- base chunk root was missing locally; subset used JEPA/VGGT/replay intersection.
- replay source: official NAVSIM-Traj JSONL, not NAVSIM generated fallback.

These results are a Stage1 representation diagnostic, not a final held-out PDMS conclusion.

## Stage1-v2 vs Random

| metric | Stage1-v2 | random | verdict |
| --- | ---: | ---: | --- |
| dyn loss | 0.0010175 | 0.0019739 | pass |
| geo loss | 0.0018299 | 0.0019356 | fail |
| dyn retrieval top1/top5/MRR | 0.1846 / 0.4568 / 0.3154 | 0.0002 / 0.0020 / 0.0028 | pass |
| geo retrieval top1/top5/MRR | 0.0144 / 0.0745 / 0.0590 | 0.0002 / 0.0012 / 0.0025 | weak/fail |

## Anti-Shortcut and Slot-Only

| metric | value | gate |
| --- | ---: | --- |
| image-only dyn / normal dyn | 1.2842 | pass |
| image-only geo / normal geo | 1.0839 | fail |
| no-signal dyn / slot-only dyn | 1.6214 | pass |
| no-signal geo / slot-only geo | 1.2028 | fail |
| random-slots dyn / normal dyn | 1.0050 | weak |
| random-slots geo / normal geo | 0.9879 | weak |

Bootstrap 95% CI:
- dyn image-only ratio: `1.2824 - 1.2863`
- geo image-only ratio: `1.0826 - 1.0852`
- dyn slot-only gain: `1.6189 - 1.6239`
- geo slot-only gain: `1.1985 - 1.2071`

## Planning Probe

| metric | value |
| --- | ---: |
| normal probe loss | 0.0004979 |
| zero H_dyn / normal probe | 29.67 |
| zero H_geo / normal probe | 66.93 |
| dyn-only probe loss | 0.03332 |
| geo-only probe loss | 0.01477 |
| no-signal/history-only probe loss | 0.26285 |

The probe is strongly slot-dependent, but this does not rescue the failed geometry teacher/retrieval checks.

## Replay and Direct Trajectory

- replay CE: `0.2526` on 64 evaluated batches, token count mean `1218.7`.
- direct parse OK: `1.0`
- direct trajectory L1: `0.22253`
- base direct trajectory L1: `0.22253`
- direct L1 relative to base: `1.0`

Replay/direct trajectory preservation passed on the sampled direct eval.

## Hidden Drift

Hidden drift was unavailable in this run: `hidden_eval_count=0`. The current PEFT wrapper path did not return frozen-base raw hidden states through `disable_adapter()`, so this is reported as unmeasured, not as passing evidence.

## Stage1-v2 vs Stage1-v1

| metric | Stage1-v2 | Stage1-v1 | direction |
| --- | ---: | ---: | --- |
| dyn loss | 0.0010175 | 0.0008786 | v1 lower |
| geo loss | 0.0018299 | 0.0001810 | v1 much lower |
| dyn retrieval top1 | 0.1846 | 0.0098 | v2 much better |
| geo retrieval top1 | 0.0144 | 0.0039 | v2 better but below gate |
| image-only dyn ratio | 1.2842 | 1.0078 | v2 better |
| image-only geo ratio | 1.0839 | 1.1569 | v2 worse |
| slot-only dyn gain | 1.6214 | 1.1114 | v2 better |
| slot-only geo gain | 1.2028 | 1.3574 | v2 worse |
| zero H_dyn probe ratio | 29.67 | 1.70 | v2 better |
| zero H_geo probe ratio | 66.93 | 2.53 | v2 better |

## Gate Decision

- Stage1-v2 gate: `NOT_READY`
- external knowledge injection useful enough for Stage2: `false`
- allow hidden cache build: `false`
- allow Stage2: `false`

Main failure items:
- `teacher_geo_trained_vs_random`
- `geo_top1`, `geo_top5`, `geo_positive_margin`
- `slot_only_geo_gain`
- `image_only_geo_ratio`
- old Stage1-v1 geometry comparison
- hidden drift unavailable

## Prohibited Actions Confirmation

- full hidden cache generated: `false`
- Stage2 launched: `false`
- Stage3 launched: `false`
- residual diffusion enabled: `false`
- action-side CoT restored: `false`
- H_dyn/H_geo token counts changed: `false`
