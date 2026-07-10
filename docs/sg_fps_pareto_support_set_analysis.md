# SG-FPS Pareto 支持集分析整理

本文档整理当前 SG-FPS / DPSI 实验使用的 Pareto 支持集，包括已有文档位置、最终 archive 路径、构建流程、审计结果、已修复问题和剩余风险。

## 核心术语说明

| 术语 | 中文解释 |
| --- | --- |
| SG-FPS | Scorer-Guided Feasible Pareto Support，本项目提出的“评分器引导的可行 Pareto 支持集”方案。目标是为 diffusion planner 提供多样且高质量的训练目标，而不是只模仿单条 GT 轨迹。 |
| Pareto 支持集 | 每个场景保留的一组候选轨迹。这些轨迹不只按 PDMS 单一最高分选择，还综合考虑安全、进度、DDC、轨迹可行性和多样性。 |
| candidate / 候选轨迹 | 支持集筛选前生成的轨迹，可能来自 GT、外部模型、当前模型或结构化生成。候选轨迹需要经过 evaluator 和质量门控后才可能进入支持集。 |
| selected support / 入选支持轨迹 | 最终保留下来用于训练的轨迹。它们是候选轨迹的子集。 |
| archive / 支持集档案 | 保存每个场景候选轨迹、入选支持轨迹、评价指标和元信息的数据目录。本文中的最终 archive 是 `support_v3`。 |
| scene / token | NAVSIM 中的一个驾驶场景。`token` 是场景唯一标识。 |
| GT | Ground Truth，数据集原始真值轨迹。它是训练和对比的基础参考。 |
| IL | Imitation Learning，模仿学习产生的轨迹。本文中通常指已有 ReCogDrive / Stage3 权重生成的轨迹。 |
| current-policy | 当前训练中的策略模型生成的轨迹。质量可能随训练阶段变化。 |
| DDV2 | DiffusionDriveV2，外部自动驾驶轨迹生成模型。这里主要用它预生成的轨迹作为候选来源。 |
| DriveOR | 另一个外部自动驾驶轨迹生成模型。这里也主要使用预生成轨迹作为候选来源。 |
| structured generation / 结构化生成 | 不依赖随机噪声乱采样，而是根据进度、横向位置、速度曲线、延迟/让行、几何修复等规则生成候选轨迹。 |
| anchor / 锚点轨迹 | 用于后续扩展的基础轨迹。例如 GT、DDV2 高分轨迹或 DriveOR 高分轨迹。anchor 如果驾驶语义错了，后续扩展也会错，因此需要先做语义对齐。 |
| expand / 扩展轨迹 | 从 anchor 经过结构化扰动、修复或重组得到的新候选轨迹。 |
| evaluator | NAVSIM / PDM 评价器，用来给轨迹计算真实评价指标。支持集最终标签来自 evaluator，不来自预测器估计。 |
| scorer | 评分器模型，用来帮助减少无效候选、提高候选筛选效率。scorer 可以用于排序和 acquisition，但不能代替 evaluator 作为最终标签。 |
| PDMS | NAVSIM 主要综合分数，越高越好。它综合安全、进度、舒适性等指标，但单看 PDMS 不能完全保证驾驶语义正确。 |
| DDC | Driving Direction Compliance，行驶方向一致性。用于判断轨迹是否符合道路/路线方向。 |
| NC | No at-fault Collision，无责任碰撞指标。通常 `1` 表示通过，`0` 表示发生相关碰撞问题。 |
| DAC | Drivable Area Compliance，可行驶区域合规性。通常 `1` 表示轨迹在可行驶区域内。 |
| TTC | Time To Collision，碰撞时间安全相关指标。越高通常越安全。 |
| EP | Ego Progress，自车前进进度。用于衡量轨迹是否有效向前推进。 |
| feasibility / 可行性 | 轨迹几何和控制层面的可执行性，例如是否局部折线、尾部倒退、曲率过大或 heading 跳变。 |
| reference-relative | 相对参考轨迹的筛选策略。例如 DDC 不强制所有场景都超过固定阈值，而是参考 GT 或当前场景基线，避免误杀本身困难的场景。 |
| DPSI | Diverse Pareto-Support Imitation，多样 Pareto 支持集模仿学习。它用支持集中的多条高质量轨迹训练 diffusion planner，而不是只训练单条 GT。 |
| FS-Norm | Feasibility-aware Step-wise Normalization，面向轨迹可行性的逐步增量归一化，用于改善 diffusion 轨迹学习的数值表示。 |
| x0 auxiliary loss | diffusion 训练中的辅助损失，用模型估计的去噪后轨迹 `x0` 直接约束轨迹重建和几何质量。 |

## 已有记录位置

主要实现说明：

- `docs/sg_fps_implementation_notes.md`

已纳入仓库管理的审计报告：

- `reports/sg_fps/support_audit_20260705/README.md`
- `reports/sg_fps/support_audit_20260705/full_audit.md`
- `reports/sg_fps/support_audit_20260705/full_audit.json`
- `reports/sg_fps/support_audit_20260705/gap_summary_rank0.json`

当前 DPSI / FS-Norm 训练使用的最终支持集：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3`

最终完整审计来源目录：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/audit_full_20260705T190919Z`

最终 gap scene 扫描来源目录：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/gap_find_20260705T191148Z`

较早一版可视化案例，适合回看问题来源，但不代表最终支持集质量：

- `reports/psi_drive/sg_fps_support_visual_cases_20260704_validrank_v2/README.md`
- `reports/psi_drive/sg_fps_support_visual_cases_20260704_validrank_v2/*.png`

## 总体结论

当前最终版 Pareto 支持集覆盖 `103288` 个场景，覆盖了本轮 SG-FPS 支持集构建使用的 NAVSIM train/val 风格数据。

最终审计结果显示，入选支持轨迹已经通过了当前实现中的质量门控。下面每个英文字段都是审计脚本中的原始字段名，括号中给出中文含义：

- selected valid ratio mean（入选轨迹有效比例的平均值）: `1.000000`
- selected non-GT low reward scene ratio（入选的非 GT 轨迹中低分场景比例）: `0.000000`
- selected non-GT first-point mismatch scene ratio（入选的非 GT 轨迹起点明显错位比例）: `0.000000`
- selected non-GT local kink scene ratio（入选的非 GT 轨迹局部折线比例）: `0.000000`
- selected non-GT semantic mismatch scene ratio（入选的非 GT 轨迹驾驶语义错配比例）: `0.000000`
- selected improver over GT ratio（存在优于 GT 的入选轨迹的场景比例）: `0.584860`
- fallback tag ratio（因常规类别不足而使用兜底选择的支持轨迹比例）: `0.007789`

剩余问题主要是部分场景支持集数量偏少，而不是入选轨迹明显无效：

- `low_support_count`（高质量支持轨迹数量偏少的场景数）: `12702`
- `poor_gt_few_candidates`（GT 本身较差且候选也很少的场景数）: `3`
- `record_count`（总场景记录数）: `103288`

因此，这版支持集可以用于 DPSI 训练。需要注意的是，部分场景在严格语义、起点、几何和 evaluator 质量筛选后，剩余高质量候选较少。如果后续要继续提升支持集密度，应只针对这些 gap scene 补充生成，而不是重新全量生成候选池。

## 最终支持集关键指标

以下指标来自：

- `reports/sg_fps/support_audit_20260705/full_audit.json`

| 指标 | 数值 | 中文解释 |
| --- | ---: | --- |
| record_count | 103288 | 总场景记录数。每条记录对应一个 NAVSIM 场景。 |
| rank_count | 16 | 分布式构建 / 审计时使用的分片数量。 |
| candidate_count mean | 23.2844 | 每个场景平均生成的候选轨迹数量。 |
| candidate_count p50 | 24 | 一半场景的候选轨迹数不超过 24。 |
| candidate_count p90 | 27 | 90% 场景的候选轨迹数不超过 27。 |
| candidate_count min / max | 2 / 31 | 单个场景候选轨迹数量的最小值和最大值。 |
| selected_count mean | 10.2792 | 每个场景平均入选的支持轨迹数量。 |
| selected_count p50 | 12 | 一半场景入选支持轨迹数达到或不超过 12。 |
| selected_count p90 | 12 | 90% 场景入选支持轨迹数达到或不超过 12，说明大部分场景已经接近目标上限。 |
| selected_count min / max | 1 / 12 | 单个场景最终入选支持轨迹数量的最小值和最大值。 |
| has_valid_candidate_ratio | 1.0000 | 每个场景是否至少有一条 evaluator-valid 候选轨迹。`1.0` 表示全部场景都有。 |
| selected_has_improver_over_gt_ratio | 0.5849 | 有入选轨迹优于 GT 的场景比例。 |
| best_selected_above_gt_ratio | 0.5849 | 每个场景最佳入选轨迹优于 GT 的比例。这里与上一项一致。 |
| external_candidate_scene_ratio | 0.9632 | 至少有 DDV2 / DriveOR 等外部候选的场景比例。 |
| external_selected_scene_ratio | 0.9112 | 最终入选支持集中包含外部模型轨迹的场景比例。 |
| fallback_tag_ratio | 0.0078 | 兜底入选轨迹占比。数值越低，说明正常 Pareto / 多样性选择越充分。 |

支持集标签统计：

| 标签 | 数量 | 中文解释 |
| --- | ---: | --- |
| diversity_max | 431941 | 多样性最大轨迹。用于补充与已有支持轨迹形态不同、但仍满足质量要求的轨迹。 |
| vector_pareto | 278650 | Pareto 前沿轨迹。在 EP、TTC、DDC、可行性等多个目标上不被其他候选明显支配。 |
| best_pdms | 256100 | 当前场景中 PDMS 表现最好的候选之一。 |
| gt_anchor | 86752 | GT 锚点轨迹。用于保留数据集原始轨迹作为训练和比较基准。 |
| fallback_best | 8270 | 常规类别不足时的兜底高质量轨迹。比例较低，说明大部分场景不依赖兜底逻辑。 |

入选轨迹来源大类统计：

| 来源大类 | 数量 | 中文解释 |
| --- | ---: | --- |
| external | 713166 | 外部模型来源，主要包括 DDV2 和 DriveOR 预生成轨迹。 |
| progress | 160069 | 进度相关结构化生成轨迹，例如适度加快、延后恢复或调整前进速度曲线。 |
| gt | 102224 | 数据集 GT 轨迹。 |
| lateral | 46091 | 横向 / 路径相关结构化生成轨迹，例如轻微路径修复或横向位置调整。 |
| timing | 40163 | 时间策略相关轨迹，例如 slow-first、yield-delay、creep-stop 等时序变化。 |

## 支持集构建流程

最终定稿流程如下：

1. 构建 seed candidates（初始候选轨迹），来源包括 GT、ReCogDrive Stage3 / IL / current-policy 输出、DDV2 预生成输出、DriveOR 预生成输出，以及结构化轨迹生成。
2. 在 anchor（锚点轨迹）选择和扩展前先做 command / driving semantics（驾驶指令 / 驾驶语义）对齐，避免把驾驶意图错误的轨迹作为后续 expand（扩展生成）的父轨迹。
3. 做起点一致性检查，过滤初始位置明显偏离场景状态或 GT 的候选轨迹。
4. 做局部几何检查，重点检查 early kink（轨迹前段突然折线）、tail reverse（轨迹尾部倒退）、heading jump（朝向角突变）和不平滑轨迹。
5. 使用 evaluator-backed metrics（由真实评价器计算的指标）作为最终标签，不把 scorer（预测评分器）的预测值当作最终支持集标签。
6. 支持集选择以轨迹质量、Pareto 多样性和 evaluator 验证后的 utility（综合效用分数）为核心，不再按来源或类别强行保留。
7. GT 默认作为 hard anchor（强制保留的基础锚点）保留。只有当 GT 本身质量差，并且找到明显更好的轨迹时，才允许下调 GT 的作用。
8. DDC 和 feasibility（轨迹可行性）采用 reference-relative（相对参考轨迹）策略。部分场景本身 DDC 条件困难，不适合使用过硬的全局阈值。
9. 合并补充候选后重新做全量 audit（审计）和 gap scan（缺口扫描）。

## 此前发现的问题与修复

### 1. 驾驶语义不一致

较早的可视化中发现，有些场景 GT 是直行，但外部 anchor 或 expand 候选是右转，并且仍然可能获得较高 PDMS。这说明仅看 PDMS 不足以保证轨迹语义正确。

修复方式：

- 在 anchor 阶段加入语义 / command 对齐；
- 不再允许语义错误的轨迹进入 expand 父轨迹集合；
- 后续结构化扩展只能基于通过语义检查的 anchor。

### 2. GT 被误删或场景看起来没有轨迹

此前部分可视化中出现 GT 消失的情况，这是不合理的，因为 GT 是训练、对比和调试的稳定参考。

修复方式：

- GT 默认保留为 anchor；
- 只有 GT 很差并且存在明显更好的替代轨迹时，才允许弱化 GT；
- audit 中把低支持数量 scene 和无效轨迹质量问题分开统计。

### 3. IL anchor 质量不稳定

IL / current-policy anchor 有时质量较低。如果为了类别覆盖强行保留，会降低支持集质量。

修复方式：

- 支持集选择不再按来源 quota 强制保留；
- IL 质量好时可以入选，质量差时不强留；
- 最终目标是高质量、多样性和 Pareto 前沿，而不是来源丰富。

### 4. 起点错位

此前一些非 GT 轨迹的初始位置和 GT / 场景状态差距过大，属于不可用轨迹。

修复方式：

- 将起点一致性纳入 selected support audit；
- 最终审计中 selected non-GT first-point mismatch scene ratio 为 `0.000000`。

### 5. 局部弯折和尾部倒退

较早候选集中存在前几个点局部折线、尾部轻微倒退等问题。

修复方式：

- feasibility gate 中加入 early kink 和 tail reverse 检查；
- 最终审计中 selected non-GT local kink scene ratio 为 `0.000000`。

## 质量判断

当前支持集不是简单的大候选池，而是经过 evaluator 验证和质量筛选后的 Pareto support archive。

最重要的质量信号有两个：

1. `58.49%` 左右的场景存在优于 GT 的入选支持轨迹。
2. 最终入选轨迹在语义错配、起点错位、局部折线、低分非 GT 轨迹等关键风险上，审计结果均为 `0`。

当前主要不足是部分场景支持数量不够。`12702` 个 scene 被标记为 `low_support_count`，说明严格筛选后高质量候选不足。这类问题会影响 DPSI 中一部分场景的多样性，但不代表当前支持集不能用于训练。

## 当前训练使用方式

当前 DPSI / FS-Norm 训练使用的支持集：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3`

当前训练输出目录：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z/dpsi_fs_norm_200ep`

记录在 `docs/sg_fps_implementation_notes.md` 中的关键训练设置：

- `max_epochs=200`
- 每个 epoch 保存一次 checkpoint
- 每 `10000` train steps 保存一次 checkpoint
- `use_fs_norm=true`
- `x0_aux_weight=0.1`
- `geo_aux_weight=0.05`

训练参数含义：

| 参数 | 中文解释 |
| --- | --- |
| `max_epochs=200` | 训练总轮数为 200 轮。这里的 epoch 表示完整遍历训练数据一遍。 |
| `checkpoint` | 模型权重保存点。用于恢复训练、选择最佳权重和后续评估。 |
| `train steps` | 训练迭代步数。每一步通常对应一个 batch 的前向和反向传播。 |
| `use_fs_norm=true` | 启用 FS-Norm 轨迹归一化，让 diffusion 模型学习逐步增量轨迹表示。 |
| `x0_aux_weight=0.1` | `x0` 轨迹重建辅助损失权重。该损失直接约束去噪后的轨迹接近目标轨迹。 |
| `geo_aux_weight=0.05` | 几何辅助损失权重。用于惩罚局部折线、尾部倒退等不可行轨迹形态。 |

## 如何快速核查

查看已入库审计摘要：

```bash
sed -n '1,220p' reports/sg_fps/support_audit_20260705/full_audit.md
```

读取 JSON 中的关键指标：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('reports/sg_fps/support_audit_20260705/full_audit.json')
data = json.loads(p.read_text())

for k in [
    'record_count',
    'rank_count',
    'selected_has_improver_over_gt_ratio',
    'external_selected_scene_ratio',
    'fallback_tag_ratio',
]:
    print(k, data.get(k))

print('selected_count', data.get('selected_count'))
print('support_tag_counts', data.get('support_tag_counts'))
PY
```

查看 gap scene 摘要：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path('reports/sg_fps/support_audit_20260705/gap_summary_rank0.json')
data = json.loads(p.read_text())
print(data)
PY
```

查看旧版可视化案例：

```bash
ls reports/psi_drive/sg_fps_support_visual_cases_20260704_validrank_v2
sed -n '1,160p' reports/psi_drive/sg_fps_support_visual_cases_20260704_validrank_v2/README.md
```

## 注意事项

`reports/psi_drive/sg_fps_support_visual_cases_20260704_validrank_v2` 中的可视化来自较早 archive，主要用于暴露问题，不应作为最终支持集质量的依据。

最终 audit 报告已经入库，但完整 raw archive 和分片输出仍在 `outputs/` 下，没有纳入 git，因为体积较大。

后续如果继续补强支持集，建议只针对 `low_support_count` 和 `poor_gt_few_candidates` 这类 gap scene 做定向补充，避免反复重建全量候选池。
