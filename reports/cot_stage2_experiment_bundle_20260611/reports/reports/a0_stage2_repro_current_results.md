# A0 Stage2 Reproduction Current Results

Generated at: 2026-05-31 23:00 UTC

Reference used in this brief: official public-weight full eval in our environment is treated as `PDMS=0.863000`.

Evaluation setting for completed rows:
- Split: full navtest
- `num_samples`: 12146
- `num_pdm_valid`: 12138
- Eval precision: fp32
- Expert target tokens in eval: disabled

## Best Summary

| run | training protocol | best checkpoint by PDMS | best PDMS | delta vs 0.863 | final PDMS | final delta vs 0.863 | status |
|---|---|---|---:|---:|---:|---:|---|
| A0-official-aligned | Official Lightning-style training on local chunk cache | `step_00100000` | 0.864891 | +0.001891 | 0.860038 | -0.002962 | train+eval complete |
| A0-local-fixed | Local chunked loop, fp32 weights + bf16 autocast | `step_00160000` | 0.855944 | -0.007056 | 0.852983 | -0.010017 | train+eval complete |

Key deltas:

| comparison | PDMS delta |
|---|---:|
| best official-aligned - best local-fixed | +0.008948 |
| final official-aligned - final local-fixed | +0.007056 |
| best official-aligned - public reference 0.863 | +0.001891 |
| best local-fixed - public reference 0.863 | -0.007056 |

## A0-official-aligned

Output root: `outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete`

Training completed at `epoch=199`, `global_step=133000`. Final scheduler LR is `1e-6`.

Lightning val/loss top-k state:
- best val/loss checkpoint: `epoch=192-step=128345.ckpt`
- best val/loss score: `0.2591`
- final/current val/loss score: `0.2600`

Current top-k val/loss checkpoints:
- `epoch=189-step=126350.ckpt`
- `epoch=190-step=127015.ckpt`
- `epoch=192-step=128345.ckpt`
- `epoch=197-step=131670.ckpt`
- `epoch=199-step=133000.ckpt`

| checkpoint | PDMS | trajectory_l1 | NC | DAC | TTC | comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step_00050000` | 0.822396 | 0.339054 | 0.971124 | 0.913660 | 0.915225 | 0.998764 | 0.779462 | 0.980104 |
| `step_00060000` | 0.855336 | 0.270285 | 0.983894 | 0.936316 | 0.946449 | 0.999670 | 0.795374 | 0.981010 |
| `step_00080000` | 0.846615 | 0.271580 | 0.977509 | 0.932526 | 0.928077 | 0.999588 | 0.799538 | 0.978168 |
| `step_00100000` | 0.864891 | 0.260491 | 0.981340 | 0.947273 | 0.942742 | 0.999506 | 0.808921 | 0.978333 |
| `step_00120000` | 0.860470 | 0.256599 | 0.979857 | 0.944554 | 0.938952 | 0.999835 | 0.806741 | 0.979198 |
| `topk_epoch=189-step=126350` | 0.861800 | 0.257219 | 0.980639 | 0.944307 | 0.940188 | 0.999753 | 0.808185 | 0.978250 |
| `topk_epoch=190-step=127015` | 0.861972 | 0.256862 | 0.981092 | 0.944472 | 0.941506 | 0.999835 | 0.807438 | 0.979115 |
| `topk_epoch=192-step=128345` | 0.860452 | 0.256279 | 0.980351 | 0.943978 | 0.940105 | 0.999835 | 0.805976 | 0.978868 |
| `topk_epoch=197-step=131670` | 0.860815 | 0.256565 | 0.980845 | 0.943813 | 0.941176 | 0.999835 | 0.805758 | 0.978909 |
| `topk_epoch=199-step=133000` | 0.862331 | 0.256073 | 0.980845 | 0.945131 | 0.940765 | 0.999835 | 0.808095 | 0.978950 |
| `final` | 0.860038 | 0.256931 | 0.980598 | 0.943648 | 0.939529 | 0.999670 | 0.806535 | 0.978538 |

Notes:
- `step_00140000` and `step_00160000` do not exist for this run because the official-aligned Lightning run ends at `global_step=133000`.
- The PDMS-best checkpoint is `step_00100000`, not the val/loss-best checkpoint.
- The final checkpoint is below the public-reference `0.863`, but the best measured checkpoint is above it.

## A0-local-fixed

Output root: `outputs/a0_stage2_repro_20260530_135148/a0_local_fixed`

Training completed at `global_step=161000`.

| checkpoint | PDMS | trajectory_l1 | NC | DAC | TTC | comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `step_00050000` | 0.829853 | 0.347216 | 0.979898 | 0.916214 | 0.939858 | 0.995634 | 0.769420 | 0.975490 |
| `step_00060000` | 0.804503 | 0.316820 | 0.973719 | 0.894958 | 0.932032 | 0.998352 | 0.743711 | 0.975325 |
| `step_00080000` | 0.830177 | 0.348740 | 0.971618 | 0.923134 | 0.923216 | 0.999094 | 0.779364 | 0.974955 |
| `step_00100000` | 0.811843 | 0.304098 | 0.972360 | 0.900890 | 0.925441 | 0.999176 | 0.762032 | 0.974419 |
| `step_00120000` | 0.837908 | 0.288389 | 0.973595 | 0.929642 | 0.915637 | 0.999176 | 0.795992 | 0.976767 |
| `step_00140000` | 0.846128 | 0.267060 | 0.978703 | 0.929395 | 0.935657 | 0.999753 | 0.795445 | 0.977097 |
| `step_00160000` | 0.855944 | 0.257567 | 0.980433 | 0.939034 | 0.938458 | 0.999506 | 0.803162 | 0.978827 |
| `final` | 0.852983 | 0.259059 | 0.979815 | 0.936398 | 0.934915 | 0.999588 | 0.802519 | 0.978209 |

## Current Readout

1. The official-aligned training path can reproduce and slightly exceed the `0.863` reference when selecting by measured PDMS over checkpoints.
2. The local-fixed chunked loop improved relative to the old failing A0 direction, but still trails official-aligned by about `0.009` PDMS at each run's best checkpoint.
3. For this run, val/loss top-k is useful but not identical to PDMS selection: the val/loss-best official checkpoint `epoch=192-step=128345` gets `0.860452`, while `step_00100000` gets `0.864891`.
4. The next comparison should treat official-aligned as the standard A0 Stage2 reproduction baseline for A4 work.
