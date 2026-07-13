# Two-Expert Slot Random-HMEF-v2 Stage2 前 200 Epoch 结果汇总

生成时间: 2026-06-15 UTC  
范围: 只整理本轮 `random_hmef_v2` Stage2 前 200 epoch 及其已完成评估；后续从 epoch200 继续训练的结果不纳入本文。

## 结论摘要

本轮主线是 **Stage2 随机初始化** 的 two-expert-slot `denoise_hmef_v2` 设计，不是 A0 初始化续训。它使用 Stage1 LoRA 后生成的 hidden cache 训练 Stage2。

与本机 A0 official-aligned 复现横向比较:

| 对比项 | checkpoint | full navtest PDMS | 100 分制 | vs A0 official best |
|---|---|---:|---:|---:|
| A0 official-aligned best | `step_00100000` | 0.864891 | 86.489 | 0.000000 |
| 本轮 random-HMEF-v2 best | `step_00120000` | 0.866397 | 86.640 | +0.001506 |
| 本轮 random-HMEF-v2 latest/final-200ep | `latest.ckpt`, epoch 200, step 132800 | 0.865064 | 86.506 | +0.000173 |
| A0 official-aligned final | `final` | 0.860038 | 86.004 | -0.004853 |

关键读数:

- 本轮 best full navtest 是 `step_00120000`，PDMS `0.866397`，比 A0 official-aligned best `0.864891` 高 `+0.001506`。
- epoch200 `latest.ckpt` 的 full navtest 是 `0.865064`，仍略高于 A0 official-aligned best `+0.000173`，但低于本轮 `step_00120000` `-0.001333`。
- val6000 曲线到 epoch200 仍在上升: `step_00120000=0.921523` 到 `latest=0.923449`。full navtest 没有同步继续上升，说明 val6000 对最后 20 epoch 的选择仍需用 full navtest 校验。
- 相比 A0 official-aligned best，本轮 `step_00120000` 的 `NC/DAC/TTC/DDC` 略高，`EP` 略低，`trajectory_l1` 基本持平略差。PDMS 增益主要来自规则/安全分量，而不是轨迹 L1 或 EP。
- 早期 A0-init two-expert `horizon_hmef_lite` 分支 best 是 `0.859800`，明显低于本轮 random-HMEF-v2 best `0.866397`，说明当前架构和训练策略相对旧分支有实质改善。

## 数据来源

本轮训练:

```text
/root/two_expert_slot_stage2_random_hmef_v2_8gpu_20260614T103147Z
```

本轮评估:

```text
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_random_hmef_v2_val6000_navtest_20260614T134330Z
```

本轮评估使用的主要 cache / token:

```text
val6000 token file:
/mnt/project/VLA-AD/outputs/last_vla_v2/navtrain_pdms_val6000_probe_20260611T0315Z/selected_valset/navtrain_pdms_val6000_tokens.txt

navtrain hidden cache:
/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16

navtest hidden cache:
/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_random_hmef_v2_val6000_navtest_20260614T134330Z/navtest_cache/hidden_navtest_stage1_lora_bf16
```

Stage1 LoRA 微调与结果:

```text
Stage1 run:
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/stage1/full_vlm_sft_lora

Stage1 checkpoint:
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/stage1/full_vlm_sft_lora/stage1.ckpt

Stage1 representation eval:
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/reports/stage1_representation_eval_256.md
```

A0 official-aligned baseline:

```text
/mnt/project/VLA-AD/outputs/a0_stage2_repro_20260531_003029/a0_official_aligned_a0complete
```

历史 A0-init two-expert 分支:

```text
/root/vla_ad_navtest_eval/two_expert_slot_stage2_navtest_20260614T004821Z
```

## 实验定义

### Stage1 LoRA 微调和 hidden cache 来源

本轮 Stage2 使用的 hidden cache 不是 A0 official 权重直接生成，而是由前置 two-expert-slot Stage1 LoRA 微调权重生成。相关代码和配置:

| 项 | 文件 |
|---|---|
| Stage1 config | `configs/last_vla_v2/two_expert_slot/stage1_vlm_sft.yaml` |
| Stage1 train entrypoint | `scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py` |
| Stage1 launcher | `scripts/last_vla_v2/two_expert_slot/run_stage1_two_expert_vlm_sft_8gpu.sh` |
| Stage1 module | `navsim/agents/recogdrive/two_expert_vlm_sft.py` |
| soft expert slots | `navsim/agents/recogdrive/two_expert_slots.py` |
| hidden cache builder | `scripts/last_vla_v2/two_expert_slot/build_two_expert_hidden_cache.py` |
| Stage1 eval | `scripts/last_vla_v2/two_expert_slot/evaluate_stage1_two_expert_ckpt.py` |
| Stage1 result summary | `reports/two_expert_slot/stage1_representation_evaluation_summary.md` |

Stage1 训练设置:

| 项 | 值 |
|---|---|
| VLM base | `/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B` |
| train mode | `lora` |
| full VLM SFT | disabled |
| prompt version | `two_expert_slot_prompt_v1` |
| teacher cache | `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446` |
| JEPA dynamic teacher | `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/jepa_dynamic` |
| VGGT Feature(23) teacher | `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/vggt_feature23` |
| base chunk root | `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks` |
| max epochs | 2 |
| precision | `bf16-mixed` |
| final effective batch | 8 GPUs x batch 4 x grad accumulation 2 = 64 |
| LR | VLM LoRA `1e-5`, slots/adapters `1e-4` |
| trainable params | total `50,293,062`; VLM LoRA `36,929,536`; adapters `11,468,800`; probe `1,814,854`; slots `79,872` |
| final steps | `forward_steps=6440`, `optimizer_steps=3220` |
| final train loss | `0.0005950928` |

Stage1 final launch command:

```bash
/root/miniconda3/envs/navsim/bin/torchrun --nproc_per_node=8 --master_port 29562 \
  scripts/last_vla_v2/two_expert_slot/run_two_expert_vlm_sft.py \
  --base-chunk-root /mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks \
  --teacher-cache-root /mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446 \
  --output-dir /mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/stage1/full_vlm_sft_lora \
  --vlm-path /mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B \
  --chunk-name-pattern 'train_full_chunk_*,train_backfill_chunk_*,train_backfill_p1_chunk_*' \
  --vlm-type internvl --train-mode lora --top-layers 2 \
  --batch-size 4 --grad-accum 2 --max-epochs 2 \
  --lr-vlm 1e-5 --lr-slots-adapters 1e-4 --weight-decay 1e-4 \
  --precision bf16-mixed --teacher-lru-size 128 --max-image-patches 12 --log-every-steps 20
```

Stage1 256 样本表征评估结论:

| 指标 | 值 |
|---|---:|
| assessment | `partial_positive_not_decisive` |
| overall_stage1_representation_supported | `true` |
| teacher_alignment_better_than_random | `true` |
| slot_only_beats_no_signal | `true` |
| planner_probe_uses_slots | `true` |
| strong_slot_teacher_contribution_over_image_only | `false` |
| trained_dyn_loss | `0.00088973343` |
| random_dyn_loss | `0.001989007` |
| dynamic trained-vs-random ratio | `2.2355x` |
| trained_geo_loss | `0.0001712665` |
| random_geo_loss | `0.0019342601` |
| geometry trained-vs-random ratio | `11.2939x` |
| trained_probe_loss | `0.0014735367` |
| image_only_probe_loss / trained_probe_loss | `159.9065x` |
| zero_dyn_probe_loss / trained_probe_loss | `15.0850x` |
| zero_geo_probe_loss / trained_probe_loss | `26.3859x` |

解释: Stage1 表征评估支持 H_dyn/H_geo 确实承载 planning-relevant 信息，并且 teacher alignment 明显好于随机初始化；但 `strong_slot_teacher_contribution_over_image_only=false` 表明 teacher 重构本身仍有一部分可由 image hidden 解释，所以最终是否有用仍以 Stage2 PDMS 和专家消融为准。

Stage1 之后生成的 hidden cache:

| split | cache |
|---|---|
| navtrain | `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16` |
| navtest | `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_random_hmef_v2_val6000_navtest_20260614T134330Z/navtest_cache/hidden_navtest_stage1_lora_bf16` |

### 本轮 random-HMEF-v2 Stage2

| 项 | 值 |
|---|---|
| config | `configs/last_vla_v2/two_expert_slot/stage2_dit_sft_random_hmef_v2.yaml` |
| Hydra experiment | `two_expert_slot_stage2_dit_sft_random_hmef_v2` |
| Stage2 初始化 | `random`, 没有使用 A0 checkpoint 初始化 |
| cache | `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16` |
| cache 来源 | Stage1 LoRA 后权重生成的 hidden cache |
| train/val split | train `84918`, val `18118`, no overlap |
| Stage2 target | GT normalized trajectory, no residual diffusion |
| conditioning | `denoise_hmef_v2`, raw VLM context to DiT enabled, expert memory tokens to DiT enabled |
| disabled | Last-VLA, Last-RD, residual diffusion, teacher trajectory residual anchor, expert target loss |
| devices | 8 GPU |
| per-GPU batch | 16 |
| effective batch | 128 |
| precision | `bf16-mixed` |
| LR schedule | base/expert/action/gate LR `1e-4`, warmup 5 epoch, cosine to min LR `1e-6`, scheduler epochs 200 |
| max epoch | 200 |
| final checkpoint metadata | `epoch=200`, `global_step=132800` |
| wall time | 2026-06-14 10:31:47 到 2026-06-15 02:56:18 UTC, 约 16h24m31s |

训练命令:

```bash
/root/miniconda3/envs/navsim/bin/torchrun --nproc_per_node=8 --master_port 29643 \
  navsim/planning/script/run_training_recogdrive.py \
  +experiment=two_expert_slot_stage2_dit_sft_random_hmef_v2 \
  cache_path=/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16 \
  use_cache_without_dataset=true force_cache_computation=false train_test_split=navtrain \
  output_dir=/root/two_expert_slot_stage2_random_hmef_v2_8gpu_20260614T103147Z \
  seed=0 trainer.params.devices=8 trainer.params.strategy=ddp_find_unused_parameters_true
```

### A0 official-aligned

| 项 | 值 |
|---|---|
| run | `a0_official_aligned_a0complete` |
| protocol | Official Lightning-style training on local chunk cache |
| completed | `epoch=199`, `global_step=133000` |
| final scheduler LR | `1e-6` |
| best by full navtest PDMS | `step_00100000`, PDMS `0.864891` |
| final full navtest PDMS | `0.860038` |
| public reference used in old report | about `0.863` |

## Full Navtest 横向对比

Full navtest 共同设置:

- `num_samples=12146`
- `num_pdm_valid=12138`
- `num_pdm_missing_metric_cache=8`
- `num_pdm_failed=0`

### 同步 step 对比

| step/checkpoint | 本轮 random-HMEF-v2 PDMS | A0 official-aligned PDMS | delta | 备注 |
|---|---:|---:|---:|---|
| 60k | 0.852886 | 0.855336 | -0.002450 | 本轮仍低于 A0 |
| 80k | 0.861313 | 0.846615 | +0.014698 | 本轮明显超过 A0 同 step |
| 100k | 0.862516 | 0.864891 | -0.002375 | A0 best 在 100k |
| 120k | 0.866397 | 0.860470 | +0.005927 | 本轮 best |
| final/latest | 0.865064 at 132.8k | 0.860038 final | +0.005026 | 两者终点 step 很接近 |
| best vs best | 0.866397 | 0.864891 | +0.001506 | 本轮 best 小幅超过 A0 best |

### 本轮 random-HMEF-v2 full navtest 明细

| checkpoint | epoch | step | PDMS | 100 分制 | L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_35-step_23904` | 35 | 23904 | 0.834917 | 83.492 | 0.298987 | 0.976520 | 0.922557 | 0.931043 | 0.999011 | 0.783846 | 0.979568 |
| `epoch_39-step_26560` | 39 | 26560 | 0.838073 | 83.807 | 0.285132 | 0.980516 | 0.922557 | 0.939611 | 0.999094 | 0.781513 | 0.979692 |
| `epoch_40-step_27224` | 40 | 27224 | 0.837116 | 83.712 | 0.277836 | 0.974625 | 0.926017 | 0.923546 | 0.997446 | 0.792067 | 0.978291 |
| `epoch_42-step_28552` | 42 | 28552 | 0.844418 | 84.442 | 0.290786 | 0.977138 | 0.930466 | 0.932279 | 0.999259 | 0.794217 | 0.980021 |
| `epoch_45-step_30544` | 45 | 30544 | 0.836825 | 83.682 | 0.293975 | 0.978456 | 0.922392 | 0.924946 | 0.999670 | 0.792923 | 0.978044 |
| `epoch_51-step_34528` | 51 | 34528 | 0.835256 | 83.526 | 0.279681 | 0.976891 | 0.924123 | 0.931208 | 0.999259 | 0.781133 | 0.980845 |
| `epoch_54-step_36520` | 54 | 36520 | 0.850259 | 85.026 | 0.295583 | 0.982658 | 0.929890 | 0.946037 | 0.999259 | 0.791670 | 0.978291 |
| `epoch_56-step_37848` | 56 | 37848 | 0.848744 | 84.874 | 0.277412 | 0.981381 | 0.931537 | 0.939941 | 0.999259 | 0.794088 | 0.978827 |
| `epoch_68-step_45816` | 68 | 45816 | 0.854502 | 85.450 | 0.267873 | 0.984676 | 0.936151 | 0.946943 | 0.999259 | 0.793380 | 0.981092 |
| `epoch_71-step_47808` | 71 | 47808 | 0.854672 | 85.467 | 0.270822 | 0.981175 | 0.939117 | 0.938952 | 0.999423 | 0.797954 | 0.981257 |
| `step_00060000` | 90 | 60000 | 0.852886 | 85.289 | 0.318910 | 0.978044 | 0.936398 | 0.930549 | 0.999341 | 0.806242 | 0.980722 |
| `step_00080000` | 120 | 80000 | 0.861313 | 86.131 | 0.264934 | 0.979321 | 0.944719 | 0.937634 | 0.999753 | 0.810046 | 0.980104 |
| `step_00100000` | 150 | 100000 | 0.862516 | 86.252 | 0.262248 | 0.980763 | 0.943730 | 0.942247 | 0.999506 | 0.807756 | 0.981381 |
| `step_00120000` | 180 | 120000 | 0.866397 | 86.640 | 0.260623 | 0.983688 | 0.947273 | 0.947685 | 0.999918 | 0.806896 | 0.980104 |
| `latest.ckpt` | 200 | 132800 | 0.865064 | 86.506 | 0.259775 | 0.982575 | 0.945955 | 0.945625 | 0.999835 | 0.808411 | 0.979980 |

### A0 official-aligned full navtest 明细

| checkpoint | epoch | step | PDMS | 100 分制 | L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `step_00050000` | - | 50000 | 0.822396 | 82.240 | 0.339054 | 0.971124 | 0.913660 | 0.915225 | 0.998764 | 0.779462 | 0.980104 |
| `step_00060000` | - | 60000 | 0.855336 | 85.534 | 0.270285 | 0.983894 | 0.936316 | 0.946449 | 0.999670 | 0.795374 | 0.981010 |
| `step_00080000` | - | 80000 | 0.846615 | 84.662 | 0.271580 | 0.977509 | 0.932526 | 0.928077 | 0.999588 | 0.799538 | 0.978168 |
| `step_00100000` | - | 100000 | 0.864891 | 86.489 | 0.260491 | 0.981340 | 0.947273 | 0.942742 | 0.999506 | 0.808921 | 0.978333 |
| `step_00120000` | - | 120000 | 0.860470 | 86.047 | 0.256599 | 0.979857 | 0.944554 | 0.938952 | 0.999835 | 0.806741 | 0.979198 |
| `topk_epoch=189-step=126350` | 189 | 126350 | 0.861800 | 86.180 | 0.257219 | 0.980639 | 0.944307 | 0.940188 | 0.999753 | 0.808185 | 0.978250 |
| `topk_epoch=190-step=127015` | 190 | 127015 | 0.861972 | 86.197 | 0.256862 | 0.981092 | 0.944472 | 0.941506 | 0.999835 | 0.807438 | 0.979115 |
| `topk_epoch=192-step=128345` | 192 | 128345 | 0.860452 | 86.045 | 0.256279 | 0.980351 | 0.943978 | 0.940105 | 0.999835 | 0.805976 | 0.978868 |
| `topk_epoch=197-step=131670` | 197 | 131670 | 0.860815 | 86.081 | 0.256565 | 0.980845 | 0.943813 | 0.941176 | 0.999835 | 0.805758 | 0.978909 |
| `topk_epoch=199-step=133000` | 199 | 133000 | 0.862331 | 86.233 | 0.256073 | 0.980845 | 0.945131 | 0.940765 | 0.999835 | 0.808095 | 0.978950 |
| `final` | - | - | 0.860038 | 86.004 | 0.256931 | 0.980598 | 0.943648 | 0.939529 | 0.999670 | 0.806535 | 0.978538 |

## Val6000 评估明细

说明:

- val6000 是本轮专门整理的 6000 样本验证集，主要用于快速选点。
- A0 official-aligned 没有同一 val6000 评估，因此本节不做 A0 横向比较。
- `epoch=xx-step=yy` 文件名来自 Lightning top-k mirror；表中 epoch 为评估 summary 中记录的 completed epoch。step checkpoint 的 epoch 来自 checkpoint metadata 或按 664 step/epoch 对齐。

| checkpoint | epoch | step | PDMS | 100 分制 | L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch=24-step=16600` | 25 | 16600 | 0.871511 | 87.151 | 0.258140 | 0.985417 | 0.943500 | 0.948333 | 0.998000 | 0.822385 | 0.979167 |
| `epoch=32-step=21912` | 33 | 21912 | 0.846381 | 84.638 | 0.373659 | 0.978417 | 0.919167 | 0.934000 | 0.998667 | 0.812066 | 0.974500 |
| `epoch=33-step=22576` | 34 | 22576 | 0.883585 | 88.358 | 0.269359 | 0.989333 | 0.951167 | 0.954667 | 0.998500 | 0.833239 | 0.981083 |
| `epoch=34-step=23240` | 35 | 23240 | 0.836943 | 83.694 | 0.287022 | 0.979917 | 0.914000 | 0.942500 | 0.999500 | 0.788255 | 0.975167 |
| `epoch=35-step=23904` | 36 | 23904 | 0.885973 | 88.597 | 0.263161 | 0.987917 | 0.950167 | 0.958333 | 0.999000 | 0.838118 | 0.981750 |
| `epoch=39-step=26560` | 40 | 26560 | 0.888685 | 88.869 | 0.243255 | 0.990333 | 0.951000 | 0.966333 | 0.998833 | 0.835547 | 0.981667 |
| `epoch=40-step=27224` | 41 | 27224 | 0.889256 | 88.926 | 0.236754 | 0.985500 | 0.954833 | 0.951333 | 0.998333 | 0.848543 | 0.981250 |
| `epoch=41-step=27888` | 42 | 27888 | 0.887203 | 88.720 | 0.256281 | 0.982333 | 0.952333 | 0.943500 | 0.999167 | 0.854295 | 0.981500 |
| `epoch=42-step=28552` | 43 | 28552 | 0.897582 | 89.758 | 0.243395 | 0.987917 | 0.960000 | 0.960500 | 0.999500 | 0.850858 | 0.981917 |
| `epoch=45-step=30544` | 46 | 30544 | 0.889187 | 88.919 | 0.250392 | 0.989250 | 0.952833 | 0.953333 | 0.999500 | 0.848835 | 0.981333 |
| `epoch=49-step=33200` | 50 | 33200 | 0.888950 | 88.895 | 0.246208 | 0.989667 | 0.957500 | 0.963333 | 0.999500 | 0.832030 | 0.980833 |
| `epoch=51-step=34528` | 52 | 34528 | 0.891008 | 89.101 | 0.235106 | 0.988000 | 0.956167 | 0.959500 | 0.999500 | 0.841101 | 0.982917 |
| `epoch=54-step=36520` | 55 | 36520 | 0.896169 | 89.617 | 0.255970 | 0.989833 | 0.958333 | 0.964667 | 0.998833 | 0.845207 | 0.982000 |
| `epoch=56-step=37848` | 57 | 37848 | 0.901161 | 90.116 | 0.231015 | 0.991583 | 0.962000 | 0.966167 | 0.998667 | 0.851201 | 0.981667 |
| `epoch=61-step=41168` | 62 | 41168 | 0.875835 | 87.583 | 0.231593 | 0.986917 | 0.941167 | 0.954500 | 0.999667 | 0.830856 | 0.980500 |
| `epoch=64-step=43160` | 65 | 43160 | 0.873585 | 87.359 | 0.250717 | 0.988083 | 0.939333 | 0.947000 | 0.998667 | 0.834373 | 0.981083 |
| `epoch=68-step=45816` | 69 | 45816 | 0.906413 | 90.641 | 0.207805 | 0.994083 | 0.968167 | 0.973333 | 0.999667 | 0.847417 | 0.983083 |
| `epoch=71-step=47808` | 72 | 47808 | 0.907918 | 90.792 | 0.213084 | 0.992417 | 0.970667 | 0.968500 | 0.999333 | 0.851518 | 0.984500 |
| `step_00050000` | 75 | 50000 | 0.911142 | 91.114 | 0.218341 | 0.990500 | 0.971000 | 0.966000 | 0.998833 | 0.862819 | 0.982667 |
| `step_00060000` | 90 | 60000 | 0.912108 | 91.211 | 0.265934 | 0.991083 | 0.971167 | 0.964500 | 0.998667 | 0.866091 | 0.982167 |
| `step_00080000` | 120 | 80000 | 0.919329 | 91.933 | 0.164764 | 0.995167 | 0.976667 | 0.974500 | 0.999167 | 0.865811 | 0.983667 |
| `step_00100000` | 150 | 100000 | 0.920127 | 92.013 | 0.142698 | 0.996000 | 0.976167 | 0.978833 | 0.999333 | 0.863361 | 0.984417 |
| `step_00120000` | 180 | 120000 | 0.921523 | 92.152 | 0.130644 | 0.996667 | 0.978333 | 0.983667 | 0.998833 | 0.859018 | 0.984167 |
| `latest.ckpt` | 200 | 132800 | 0.923449 | 92.345 | 0.126391 | 0.996500 | 0.979667 | 0.982000 | 0.999167 | 0.863496 | 0.983833 |

Val6000 读数:

- 从 `step_00050000` 到 `latest.ckpt`，PDMS 从 `0.911142` 上升到 `0.923449`。
- `trajectory_l1` 从 50k 的 `0.218341` 降到 latest 的 `0.126391`，验证集轨迹误差持续下降。
- full navtest 在 120k 达到最高，但 val6000 在 132.8k/latest 仍最高，说明 val6000 和 full navtest 在最后阶段有轻微排序偏差。

## 知识注入影响诊断

诊断 checkpoint: `epoch=71-step=47808` 附近的 val6000 ablation。  
normal 参考: `epoch=71-step=47808` val6000 PDMS `0.907918`。

| mode | PDMS | 100 分制 | delta vs normal | L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.907918 | 90.792 | 0.000000 | 0.213084 | 0.992417 | 0.970667 | 0.968500 | 0.999333 | 0.851518 | 0.984500 |
| `zero_h_geo` | 0.884418 | 88.442 | -0.023500 | 0.346132 | 0.992250 | 0.970833 | 0.965333 | 0.999167 | 0.797765 | 0.983583 |
| `dyn_only` | 0.882353 | 88.235 | -0.025564 | 0.345224 | 0.991833 | 0.969500 | 0.963333 | 0.999500 | 0.796389 | 0.983083 |
| `geo_only` | 0.748188 | 74.819 | -0.159730 | 0.630424 | 0.983083 | 0.874833 | 0.927000 | 0.999833 | 0.628610 | 0.957250 |
| `zero_h_dyn` | 0.747294 | 74.729 | -0.160624 | 0.629640 | 0.981583 | 0.874500 | 0.927000 | 0.999500 | 0.627877 | 0.957583 |
| `raw_vlm_only` | 0.734114 | 73.411 | -0.173804 | 0.829735 | 0.973667 | 0.900667 | 0.918667 | 0.999500 | 0.582439 | 0.968083 |
| `zero_all_experts` | 0.733460 | 73.346 | -0.174458 | 0.830159 | 0.973667 | 0.899833 | 0.918500 | 0.999500 | 0.581277 | 0.968583 |

解释:

- `raw_vlm_only` 和 `zero_all_experts` 都在 `0.733-0.734`，远低于 normal `0.907918`，说明轨迹输出强依赖 two-expert hidden 条件。
- `dyn_only` / `zero_h_geo` 在 `0.882-0.884`，说明动态分支 `H_dyn` 承担了主要有效信息。
- `geo_only` / `zero_h_dyn` 在 `0.747-0.748`，说明几何分支单独不足以支撑当前输出，但在 normal 中与动态分支联合仍贡献约 `+0.0235` PDMS。

## 历史 A0-init Two-Expert 分支对照

这是早期 `horizon_hmef_lite` 分支，不是当前 `denoise_hmef_v2` 主线。它使用 A0 official-aligned 初始化，4GPU + grad accumulation 保持 effective batch 128。结果列在这里仅用于说明“当前 random-HMEF-v2 与旧 A0-init two-expert 分支”的差异。

| checkpoint | PDMS | 100 分制 | L1 | NC | DAC | TTC | Comfort | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch=5-step=3984` | 0.859800 | 85.980 | 0.267105 | 0.982864 | 0.940600 | 0.945872 | 0.999670 | 0.801477 | 0.977591 |
| `epoch=13-step=9296` | 0.858414 | 85.841 | 0.267405 | 0.980722 | 0.943483 | 0.939446 | 0.999506 | 0.802938 | 0.978621 |
| `epoch=11-step=7968` | 0.858268 | 85.827 | 0.271180 | 0.977467 | 0.943319 | 0.934750 | 0.999506 | 0.807967 | 0.978497 |
| `epoch=6-step=4648` | 0.848072 | 84.807 | 0.267556 | 0.978827 | 0.933185 | 0.934998 | 0.999835 | 0.795490 | 0.978044 |
| `last` | 0.846044 | 84.604 | 0.266649 | 0.978127 | 0.930302 | 0.934009 | 0.999670 | 0.795717 | 0.979115 |
| `epoch=16-step=11288` | 0.845564 | 84.556 | 0.268104 | 0.978580 | 0.930384 | 0.931867 | 0.999588 | 0.796566 | 0.979280 |

相对关系:

- 旧 A0-init two-expert best: `0.859800`
- A0 official-aligned best: `0.864891`
- 本轮 random-HMEF-v2 best: `0.866397`

因此当前 best 比旧 A0-init two-expert best 高 `+0.006597`，比 A0 official-aligned best 高 `+0.001506`。

## 判断

1. 当前 random-HMEF-v2 的 full navtest 最佳点已经超过本机 A0 official-aligned 复现，但增益较小，属于 `+0.15` PDMS 百分点量级。
2. epoch200 latest 仍高于 A0 best，但低于 120k，说明继续训练和选点应依赖 full navtest 或更可靠的 val-to-navtest 校准。
3. val6000 仍在上涨，说明 200 epoch 不是训练损失意义上的明显终点；但 full navtest 出现 120k 到 132.8k 的小幅回落，说明学习率下限续训需要密切评估。
4. 知识注入诊断显示模型确实显著使用 external expert hidden 条件，特别是 `H_dyn`。这支持“Stage1 LoRA cache + random Stage2 denoise-HMEF-v2”这条路线不是只靠 raw VLM context。
5. 后续继续训练的评估应优先看 `140k/160k/180k/199.2k` full navtest 和每 10 epoch val6000 的一致性，不能只按 val6000 排名做最终结论。
