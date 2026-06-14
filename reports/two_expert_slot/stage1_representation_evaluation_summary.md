# Stage1 表征评估方案总结

## 评估目标

Stage1 的评估目标不是直接看 PDMS，而是回答：

> Stage1 学到的 `H_dyn / H_geo` 是否真的包含有用的动态/几何表征，而不是 adapter 只靠 `image_hidden` 重构 teacher？

当前 two-expert slot Stage1 路线是 VLM-side soft expert slots：

- `H_dyn`: dynamic expert slots，对齐 JEPA dynamic teacher。
- `H_geo`: geometry expert slots，对齐 VGGT Feature(23) teacher。
- adapters 输入是 `random-masked image_hidden + slot hidden`。
- Stage1 trajectory probe 输入是 `H_dyn + H_geo + status + command + history`。

因此评估重点是三件事：

1. Teacher feature 是否能被重构。
2. 重构是否真的依赖 `H_dyn / H_geo`。
3. `H_dyn / H_geo` 是否对 trajectory probe 有贡献。

## 评估工具

新增脚本：

```bash
scripts/last_vla_v2/two_expert_slot/evaluate_stage1_two_expert_ckpt.py
```

它加载：

```text
Stage1 compact checkpoint
VLM LoRA adapter
TwoExpertSoftSlots
JEPADynamicAdapter
VGGTFeature23Adapter
TrajectoryProbe
```

脚本只做 eval forward：

```text
不训练
不生成 hidden cache
不跑 Stage2
不跑 full navtest eval
```

本次输出：

```text
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/reports/stage1_representation_eval_256.json
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/reports/stage1_representation_eval_256.md
```

## 评估视图 / Ablation

同一批样本上构造多种输入视图：

| 视图 | 含义 | 用途 |
|---|---|---|
| `trained` | 正常 Stage1 ckpt，使用 `image_hidden + H_dyn/H_geo` | 主结果 |
| `random` | 同一个 VLM/LoRA backbone，但 slots/adapters/probe 随机初始化 | 证明训练是否有效 |
| `slot_only` | 保留 `H_dyn/H_geo`，清零 `image_hidden` | 测 slots 本身是否有信息 |
| `image_only` | 保留 `image_hidden`，清零 `H_dyn/H_geo` | 测 adapter 是否主要靠图像 hidden |
| `no_signal` | `image_hidden` 和 `H_dyn/H_geo` 都清零 | 无信息下限 |
| `high_mask` | 保留 slots，遮掉 90% `image_hidden` | 测强图像缺失时是否还能对齐 |

## Teacher Reconstruction 指标

### Dynamic Teacher

评估：

```text
H_dyn + image_hidden -> JEPA dynamic teacher tokens
```

指标：

```text
dyn_loss = normalized MSE(pred_jepa, target_jepa)
```

同时分组看：

```text
dynamic_loss_short
dynamic_loss_mid
dynamic_loss_long
```

JEPA dynamic teacher shape：

```text
[3, 12, 1024]
```

三组分别代表 short / mid / long horizon dynamic features。

### Geometry Teacher

评估：

```text
H_geo + image_hidden -> VGGT Feature(23) teacher tokens
```

指标：

```text
geo_loss = normalized MSE(pred_vggt, target_vggt)
```

VGGT teacher shape：

```text
[12, 1024]
```

## Slot Contribution 指标

核心不是只看 `trained loss` 低不低，而是看对照关系。

### trained vs random

证明训练是否有效：

```text
random_dyn_loss / trained_dyn_loss
random_geo_loss / trained_geo_loss
```

如果比值明显大于 1，说明 trained Stage1 确实学到了 teacher alignment。

### slot_only vs no_signal

证明 slots 本身是否携带 teacher 信息：

```text
no_signal_dyn_loss / slot_only_dyn_loss
no_signal_geo_loss / slot_only_geo_loss
```

如果 `slot_only` 明显好于 `no_signal`，说明 `H_dyn/H_geo` 本身有信息。

### trained vs image_only

证明 teacher decoder 是否真的依赖 slots：

```text
image_only_dyn_loss / trained_dyn_loss
image_only_geo_loss / trained_geo_loss
```

如果 `image_only` 和 `trained` 很接近，说明 adapter 很可能主要靠 `image_hidden`，slots 的增量贡献偏弱。

这是当前最关键的诊断指标。

## Trajectory Probe 指标

Stage1 里有一个 weak planning probe：

```text
H_dyn + H_geo + status + command + history -> GT normalized trajectory
```

它不是最终 planner，只用于判断 slots 是否包含 planning-relevant 信息。

指标：

```text
probe_loss
probe_heading_loss
probe_progress_loss
```

然后做 corruption：

| Corruption | 含义 |
|---|---|
| `zero_dyn` | 清零 `H_dyn` |
| `zero_geo` | 清零 `H_geo` |
| `image_only` | `H_dyn/H_geo` 全清零 |
| `no_signal` | `image_hidden` 和 slots 全清零 |

重点看：

```text
zero_dyn_probe_loss / trained_probe_loss
zero_geo_probe_loss / trained_probe_loss
```

如果清零 slots 后 probe loss 大幅上升，说明 trajectory probe 强依赖 slots。

## Batch Retrieval 指标

为了避免只看 MSE，还加入 batch 内 retrieval：

```text
pred_teacher_feature 在 batch teacher targets 中找同一样本
```

分别对 dynamic 和 geometry 做：

```text
trained_dyn retrieval
trained_geo retrieval
slot_only_dyn retrieval
slot_only_geo retrieval
image_only_dyn retrieval
image_only_geo retrieval
random_dyn retrieval
random_geo retrieval
```

指标：

| 指标 | 含义 |
|---|---|
| `top1` | 预测 feature 最近邻是否是同一样本 teacher |
| `top5` | top-5 命中率 |
| `chance_top1` | 随机命中率 |
| `mean_rank` | 正样本平均排名 |
| `mrr` | mean reciprocal rank |
| `positive_cosine` | 正样本 cosine |
| `hardest_negative_cosine` | 最难负样本 cosine |
| `positive_margin` | 正样本 cosine - 最难负样本 cosine |

如果 `trained top1/top5/MRR` 明显高于 random/chance，说明 teacher feature 空间里有一定样本区分能力。

## 当前 256 样本评估结果

评估文件：

```text
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/reports/stage1_representation_eval_256.json
/mnt/project/VLA-AD/experiments/two_expert_slot_real_20260612_172446/reports/stage1_representation_eval_256.md
```

### Evidence Flags

```text
assessment = partial_positive_not_decisive
teacher_alignment_better_than_random = true
slot_only_beats_no_signal = true
planner_probe_uses_slots = true
strong_slot_teacher_contribution_over_image_only = false
overall_stage1_representation_supported = true
```

### Teacher Reconstruction

Dynamic teacher：

```text
trained_dyn_loss = 0.0008897
random_dyn_loss  = 0.0019890
```

Dynamic teacher 上 trained 约比 random 好：

```text
2.24x
```

Geometry teacher：

```text
trained_geo_loss = 0.0001713
random_geo_loss  = 0.0019343
```

Geometry teacher 上 trained 约比 random 好：

```text
11.29x
```

说明 Stage1 确实学到了 teacher reconstruction。

### Slot-Only vs No-Signal

```text
slot_only_dyn_loss = 0.0009053
no_signal_dyn_loss = 0.0010185
```

```text
slot_only_geo_loss = 0.0002281
no_signal_geo_loss = 0.0003154
```

说明 slots 本身有一定 teacher 信息。

### Image-Only 诊断

```text
trained_dyn_loss    = 0.0008897
image_only_dyn_loss = 0.0009037
```

Dynamic 上 image-only 只比 trained 差约：

```text
1.6%
```

```text
trained_geo_loss    = 0.0001713
image_only_geo_loss = 0.0001963
```

Geometry 上 image-only 只比 trained 差约：

```text
14.6%
```

这说明当前 teacher adapters 对 `image_hidden` 的依赖仍然很强，尤其 dynamic adapter。  
因此目前还不能强证明“主要表征已经压进 H_dyn/H_geo”。

### Probe Corruption

```text
trained_probe_loss = 0.0014735
```

清零 `H_dyn`：

```text
zero_dyn_probe_loss / trained_probe_loss = 15.08x
```

清零 `H_geo`：

```text
zero_geo_probe_loss / trained_probe_loss = 26.39x
```

说明 trajectory probe 明显依赖 `H_dyn/H_geo`。

### Retrieval

Dynamic retrieval：

```text
trained_dyn_top1 = 4.69%
chance_top1      = 0.39%
trained_dyn_top5 = 17.58%
```

Geometry retrieval：

```text
trained_geo_top1 = 3.52%
chance_top1      = 0.39%
trained_geo_top5 = 16.41%
```

说明 trained feature 有样本区分能力，但 margin 仍不够强。

## 当前结论

当前 Stage1 ckpt 有正向证据：

```text
1. trained 明显优于 random；
2. slot_only 优于 no_signal；
3. trajectory probe 强依赖 H_dyn/H_geo；
4. retrieval 明显高于 chance。
```

但还不是强证明：

```text
1. image_only 已经接近 trained；
2. dynamic adapter 对 H_dyn 的增量依赖偏弱；
3. retrieval positive margin 仍然偏弱；
4. 当前样本来自 navtrain，不是严格 held-out。
```

因此当前判断是：

```text
Stage1 表征有学习到东西，但证明力度是 partial positive，不是 decisive。
```

## 后续建议

为了更强证明 Stage1 表征有效，建议加：

```text
1. slot-only auxiliary loss
2. 更高比例 image_hidden masking
3. image-hidden dropout / adapter input dropout
4. H_dyn/H_geo contrastive retrieval loss
5. held-out teacher eval cache
6. Stage2 downstream corruption:
   normal
   raw_vlm_only
   zero_h_dyn
   zero_h_geo
   zero_all
   random_slots
```

最终还是要通过 Stage2 验证：

```text
normal > raw_vlm_only
zero_h_dyn 主要伤 TTC/dynamics
zero_h_geo 主要伤 NC/DAC/geometry
```

这才是证明 Stage1 表征对 planning 有用的关键闭环。
