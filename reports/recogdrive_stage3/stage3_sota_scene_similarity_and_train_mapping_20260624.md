# ReCogDrive Stage3 SOTA 场景相似性与训练集映射分析

日期：2026-06-24

## 结论摘要

本次分析使用 SOTA Core-Pareto GRPO v2 的 navtest step21600 逐场景评估结果，并用 hidden cache 构建 navtest 到 hidden_navtrain 的全量精确 top-20 近邻映射。相似性不是只看轨迹，而是组合了 VLM/two-expert hidden slot、历史轨迹、状态、命令和专家目标轨迹。

核心结论：

- SOTA navtest PDMS = **0.910274**，Core = **0.923721**；对齐 hidden cache 后覆盖 **12138** 个 navtest 场景。
- navtest 零分场景 **415/12138 = 3.42%**，全部主要来自 NC/DAC 乘法约束失败：`zero_dac_offroad=229`，`zero_nc_collision=178`，`zero_nc_and_dac=8`。
- 零分场景的 top-10 训练近邻相似度均值 **0.9436**，高分场景为 **0.9543**。这说明多数零分场景并不是完全没有相似训练样本，更像是策略在已有相似分布中仍然没有稳定学会安全可行轨迹。
- `turning` 类最值得优先处理：PDMS **0.8941**，零分率 **5.30%**，相比 91.04 开发版逐 token 基线平均下降 **-0.0081**，但 top-10 相似度 **0.9720** 很高，说明训练覆盖存在，问题更偏向优化目标、采样稳定性或转弯边界决策。
- 非零低分主要是 `low_ep`：**1806** 个场景，均值 PDMS **0.8423**。这与 Core-Pareto 设计关注 EP 是一致的，后续应继续检查组内 advantage 是否仍对慢/保守候选压制不足。
- 与本地可对齐的 91.04 开发版 CSV 对比，整体均值几乎持平：SOTA - baseline = **-0.00008**。但逐场景变化很大：`gain >= 0.02` 有 **16.74%**，`drop <= -0.02` 有 **15.41%**。这说明新方法有明确的场景迁移，而不是所有场景等比例变化。

注意：本地目前没有找到原版 90.54/90.55 Stage3 的逐 token CSV，只找到文档记录的均值和路径。因此本报告的逐场景对比使用的是本地可对齐的 `recogdrive_new_9104_dev_epoch9_step13300` CSV，不能把它当作原版 90.55 基线。

## 数据源

| 项目 | 路径 |
| --- | --- |
| SOTA eval CSV | `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_4to7_sharded_conc4_watch_from5700_v1/eval_step-step_21600/local_sharded_aggregated.csv` |
| 可逐 token 对齐基线 | `/mnt/project/tsinghua_code/evaluation_record/original_eval_navtest_epoch9_2026.05.14.16.49.31.csv` |
| hidden_navtrain | `/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z` |
| hidden_navtest | `/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtest_stage1_v2_clean_bf16_20260617T020404Z` |
| 输出目录 | `/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624` |

hidden_navtrain index 有 **103036** 条，hidden_navtest index 有 **12146** 条，两者 token 无重叠。这个 hidden_navtrain 数量大于官方 train split 85109，因此本报告称为 hidden_navtrain 覆盖集，不直接等同官方训练监督集合。

## 相似性指标设计

用户关心“是否还有别的指标和参数能判断两个场景是否类似，不仅仅是轨迹类似”。答案是有，而且应该组合使用：

| 维度 | 当前是否使用 | 用途 |
| --- | --- | --- |
| VLM/two-expert hidden slot | 已使用 | 表征视觉语义、动态/几何上下文，是当前最重要的非轨迹相似性来源 |
| 历史轨迹 | 已使用 | 判断自车进入场景前的运动状态 |
| 高层命令 | 已使用 | 区分直行、转向、路线意图 |
| 状态特征 | 已使用 | 包含速度/状态类信号 |
| 专家目标轨迹形态 | 已使用 | 描述应该怎么开，但不是唯一相似性依据 |
| 地图拓扑、车道、路线、可行驶区域 | 未纳入本轮全量向量 | 应作为下一轮补充，尤其用于 DAC/offroad 失败分析 |
| 交通参与者数量、距离、类别 | 未纳入本轮全量向量 | 应作为下一轮补充，尤其用于 NC/TTC 失败分析 |
| traffic light / rule state | 未纳入本轮全量向量 | 应作为下一轮补充，尤其用于红绿灯/规则场景 |
| 模型输出失败类型 | 不参与近邻，只参与分类 | 防止用结果泄漏定义相似性，但用于解释为何失败 |

本轮实现方式：

- 从 hidden-cache `.pt` zip 中直接读取小 storage，不完整 `torch.load` 8.7MB 文件。
- 每个样本抽取 `two_expert_h_dyn` 和 `two_expert_h_geo` 的 block-pooled 视觉/语义向量。
- 拼接 history/status/command/trajectory 的结构化特征。
- 使用 8 张 A800 做全量精确 cosine top-k，不使用近似 ANN，不降低分析质量。

## SOTA 分数分桶

| bucket | count | PDMS | Core | NC | DAC | EP | TTC | DDC | NN1 sim | NN10 sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero | 415 | 0.0000 | 0.3876 | 0.5554 | 0.4289 | 0.0000 | 0.5301 | 0.8988 | 0.9562 | 0.9436 |
| low_0_0p5 | 37 | 0.3960 | 0.4603 | 0.8649 | 1.0000 | 0.6777 | 0.0270 | 0.9595 | 0.9444 | 0.9302 |
| mid_0p5_0p85 | 1086 | 0.7415 | 0.7415 | 1.0000 | 1.0000 | 0.6348 | 0.7449 | 0.9797 | 0.9507 | 0.9381 |
| good_0p85_0p95 | 3401 | 0.9106 | 0.9106 | 1.0000 | 1.0000 | 0.7854 | 1.0000 | 0.9822 | 0.9564 | 0.9444 |
| high_0p95_1 | 7199 | 0.9907 | 0.9907 | 1.0000 | 1.0000 | 0.9777 | 1.0000 | 0.9793 | 0.9644 | 0.9543 |

解释：

- zero 场景的近邻相似度并不低，说明很多失败不是“训练集中找不到类似场景”。
- mid/good 分桶的主要差异来自 EP，而不是 NC/DAC；这正是 Core-Pareto 后续还应继续关注的方向。
- `low_0_0p5` 样本很少，但 TTC 均值只有 0.027，需要单独抽查是否为极端交互/碰撞风险场景。

## 失败类型

| failure_reason | count | PDMS | NN10 sim | 主要含义 |
| --- | ---: | ---: | ---: | --- |
| nonzero_ok | 9304 | 0.9746 | 0.9527 | 主体高分场景 |
| low_ep | 1806 | 0.8423 | 0.9369 | 最主要非零低分问题，偏保守/进度不足 |
| low_ttc | 265 | 0.5679 | 0.9423 | TTC 低，交互/时序风险 |
| low_ddc | 263 | 0.9742 | 0.9489 | DDC 低但 PDMS 不直接惩罚，仍需 guard |
| zero_dac_offroad | 229 | 0.0000 | 0.9515 | 越界/不可行驶区域导致乘法清零 |
| zero_nc_collision | 178 | 0.0000 | 0.9337 | 碰撞导致乘法清零 |
| zero_nc_and_dac | 8 | 0.0000 | 0.9370 | 碰撞和越界同时失败 |

优先级建议：

1. `zero_dac_offroad`：229 个，且相似度高，说明可从相似训练场景中强化车道/边界可行性。
2. `zero_nc_collision`：178 个，直行/横移/快速直行都有，需要结合 traffic actor 距离补充分析。
3. `low_ep`：1806 个，是影响均值最大的非零低分簇，应重点看 GRPO 组内候选是否仍然奖励了慢/保守轨迹。

## 运动类别

| motion_phenotype | count | PDMS | zero ratio | delta vs 91.04 dev | NN10 sim |
| --- | ---: | ---: | ---: | ---: | ---: |
| turning | 2436 | 0.8941 | 5.30% | -0.0081 | 0.9720 |
| fast_straight | 1990 | 0.9025 | 3.57% | +0.0014 | 0.9457 |
| straight | 3637 | 0.9156 | 1.90% | -0.0005 | 0.9342 |
| lateral_shift | 3282 | 0.9180 | 3.66% | +0.0056 | 0.9500 |
| stop_or_creep | 793 | 0.9230 | 3.28% | -0.0012 | 0.9598 |

最关键发现：

- `turning` 是当前最弱类别，而且训练近邻相似度最高。这不是简单的数据覆盖不足，更可能是转弯场景中 GRPO 候选的可行性/边界优势分配仍不够精准。
- `lateral_shift` 相比 91.04 开发版是正向变化，说明 Core-Pareto 并非全局无效，而是在一些横向调整场景中确实带来收益。

## 与 91.04 开发版逐 token 对比

| delta bucket | count | delta PDMS | delta Core | delta EP | delta TTC | delta NC | delta DAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large_gain_ge_0p10 | 423 | +0.6111 | +0.4454 | +0.4709 | +0.5981 | +0.2955 | +0.2104 |
| gain_0p02_0p10 | 1609 | +0.0341 | +0.0341 | +0.0819 | +0.0000 | +0.0000 | +0.0000 |
| flat_abs_lt_0p02 | 8236 | +0.0012 | +0.0011 | +0.0029 | -0.0002 | +0.0001 | -0.0001 |
| drop_minus_0p10_0p02 | 1395 | -0.0442 | -0.0442 | -0.1061 | +0.0000 | +0.0000 | +0.0000 |
| large_drop_le_minus_0p10 | 475 | -0.5531 | -0.3687 | -0.4785 | -0.4063 | -0.1674 | -0.2842 |

解释：

- 中等 gain/drop 基本是 EP 变化驱动：`gain_0p02_0p10` 的 delta EP 是 +0.0819，`drop_0p02_0p10` 的 delta EP 是 -0.1061。
- 大幅 gain/drop 同时牵涉 NC/DAC/TTC，说明这些是安全可行性发生离散变化的场景。
- SOTA 不是简单“更保守”或“更激进”，而是重排了场景表现：一部分场景明显变好，一部分转弯/边界场景明显变差。

## 零分场景细分

| failure + motion | count | NN10 sim | baseline PDMS | delta |
| --- | ---: | ---: | ---: | ---: |
| zero_dac_offroad + turning | 107 | 0.9616 | 0.5375 | -0.5375 |
| zero_dac_offroad + lateral_shift | 75 | 0.9444 | 0.6216 | -0.6216 |
| zero_nc_collision + straight | 53 | 0.9174 | 0.3829 | -0.3829 |
| zero_nc_collision + lateral_shift | 45 | 0.9336 | 0.4245 | -0.4245 |
| zero_nc_collision + fast_straight | 43 | 0.9405 | 0.3744 | -0.3744 |
| zero_dac_offroad + fast_straight | 26 | 0.9503 | 0.4110 | -0.4110 |
| zero_nc_collision + stop_or_creep | 19 | 0.9334 | 0.1316 | -0.1316 |
| zero_nc_collision + turning | 18 | 0.9659 | 0.2675 | -0.2675 |

这张表的含义很直接：

- 最大零分簇是转弯越界，且训练近邻非常相似。这不是“训练集没类似场景”的主要证据，而是“已有类似场景但策略输出没有稳定吸收”的证据。
- 直行碰撞类的 NN10 相似度相对低一些，但仍不算极低，需要加 actor 距离/速度特征再判断是否真的覆盖不足。

## 训练场景映射产物

完整映射文件：

- 每个 navtest 场景的 top-20 train 近邻：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/navtest_scene_category_train_mapping.csv`
- 每个类别聚合后的 top train tokens/logs：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/category_to_train_scene_mapping.csv`
- top-20 近邻长表：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/train_neighbor_to_navtest_long_top20.csv`
- train token 反向聚合到 navtest 表现：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/train_scene_to_navtest_outcome_mapping.csv`
- SOTA 零分场景及近邻：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/sota_zero_score_scenes_with_train_neighbors.csv`
- 自动生成报告：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/recogdrive_stage3_scene_similarity_report.md`
- 特征缓存：`/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/hidden_features_navtrain.npz` 和 `hidden_features_navtest.npz`

这些 CSV 可以直接用于以场景为单位做训练/分析：

- 选定 `failure_reason=zero_dac_offroad` 且 `motion_phenotype=turning`，取其 top train tokens 作为 DAC/turning replay 或 oracle 搜索池。
- 选定 `failure_reason=low_ep`，按 `motion_phenotype` 分别构造 EP 强化样本，避免把直行、转弯、横移混成一个目标。
- 选定 `delta_bucket=large_drop_le_minus_0p10`，对比 SOTA 和 91.04 开发版预测轨迹，找 Core-Pareto 在哪些安全约束上退化。
- 选定 `nn10_similarity` 低的零分样本，作为可能的数据覆盖不足样本，优先补充 metric-cache 地图/actor 特征后复核。

## Train 近邻簇与 Test 表现的对应关系

这一步做了反向映射：把每个 train token 在 navtest top-20 近邻中出现的所有测试场景聚合起来，看这个 train 场景簇对应的测试表现。它回答的是“某个训练场景簇附近的测试场景表现如何”，但还不是“模型在这个 train token 自身的推理 PDMS”。后者需要额外对这些 train token 子集跑 SOTA 权重推理和 PDM scoring。

输出文件：

- `/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/train_neighbor_to_navtest_long_top20.csv`
- `/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/train_scene_to_navtest_outcome_mapping.csv`

反向聚合结果：

| proxy label | train tokens | linked navtest mean | linked navtest median | mean navtest PDMS | zero ratio | mean similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maps_to_good_tests | 7750 | 8.47 | 7 | 0.9828 | 0.0000 | 0.9536 |
| maps_to_medium_good_tests | 4330 | 9.65 | 8 | 0.9272 | 0.0000 | 0.9396 |
| mixed | 36524 | 2.69 | 2 | 0.9240 | 0.0262 | 0.9401 |
| maps_to_low_or_zero_tests | 5463 | 6.75 | 5 | 0.7118 | 0.1873 | 0.9392 |

这说明存在明显的对应关系：

- 有一批 train token 簇稳定映射到高分测试：`maps_to_good_tests` 的测试均分 0.9828，zero ratio 为 0，相似度也最高。
- 也有一批 train token 簇稳定映射到低分/零分测试：`maps_to_low_or_zero_tests` 的测试均分只有 0.7118，zero ratio 18.73%。
- 低分簇的平均相似度 0.9392，并不低。因此“附近有训练样本”不等价于“策略能在该类场景稳定输出好轨迹”。

典型高分对应 train token：

| train token | linked navtest | mean PDMS | zero ratio | main mix |
| --- | ---: | ---: | ---: | --- |
| `8d8ed7c58fe75d5c` | 80 | 0.9798 | 0.0000 | straight, mostly nonzero_ok |
| `4409da28f8ec507c` | 61 | 0.9857 | 0.0000 | straight, mostly nonzero_ok |
| `cc4521f323975486` | 52 | 1.0000 | 0.0000 | stop_or_creep, all nonzero_ok |
| `1247a72bfb245c3f` | 47 | 0.9959 | 0.0000 | lateral_shift / fast_straight, all nonzero_ok |

典型低分/零分对应 train token：

| train token | linked navtest | mean PDMS | zero ratio | main mix |
| --- | ---: | ---: | ---: | --- |
| `ec68dc7254c75650` | 69 | 0.8449 | 0.0000 | low_ep-heavy straight |
| `9972a2a47f395872` | 56 | 0.7949 | 0.1071 | fast_straight/lateral_shift, mixed low_ttc/zero |
| `ab3b94de4d54553c` | 38 | 0.7471 | 0.1316 | fast_straight, collision/low_ttc mixed |
| `81e6aa29dc135c4f` | 29 | 0.7731 | 0.1724 | lateral_shift/fast_straight, collision-heavy |

因此目前更准确的表述是：

- “测试低分/零分场景不是没有训练近邻覆盖”这个结论成立。
- “训练集中相似场景本身也没有训好”这个结论还没有严格证明，因为缺 SOTA 在这些 train token 上的真实推理 PDMS。
- 但“低分测试确实聚集在一批固定的训练近邻簇周围”已经有证据。这些簇是下一步最适合补跑 train-subset 推理、PDM scoring 和定向 GRPO/replay 的对象。

## Train-Subset 推理验证：训练簇学得好，测试簇是否也好？

为了验证“训练集中学得好的，在相似 navtest 场景中是否也表现好”，额外从反向映射中抽取两组 train token，并用 SOTA step21600 权重在 navtrain/train metric cache 上做真实推理和 exact PDM scoring：

- `maps_to_low_or_zero_tests` 前 500 个 train token。
- `maps_to_good_tests` 前 500 个 train token。

注意：这不是全量 train 随机抽样，而是按 navtest 反向映射结果构造的对照验证集，用来验证对应关系和失败类型。

输出目录：

- `/mnt/project/VLA-AD/outputs/recogdrive_stage3_train_subset_sota_validation_20260624/full500`

整体结果：

| subset | train tokens | train PDMS | train zero | train Core | train NC | train DAC | train EP | train TTC | train DDC | linked navtest PDMS | linked navtest zero | mean similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| good | 500 | 0.9813 | 0.40% | 0.9820 | 0.996 | 1.000 | 0.9668 | 0.990 | 0.998 | 0.9817 | 0.00% | 0.9558 |
| low_or_zero | 500 | 0.9038 | 2.00% | 0.9142 | 0.995 | 0.984 | 0.8242 | 0.970 | 0.983 | 0.7800 | 12.65% | 0.9402 |

这说明存在真实对应关系：

- 映射到高分测试簇的 train token，自身推理分数也很高：train PDMS 0.9813，linked navtest PDMS 0.9817。
- 映射到低分/零分测试簇的 train token，自身推理分数明显更低：train PDMS 0.9038，主要差距来自 EP 和 DAC。
- 在这 1000 个有偏对照样本上，train PDMS 与 linked navtest PDMS 的 Pearson 相关为 0.3055，Spearman 相关为 0.5033；说明排序关系存在，但不是一一决定。

按 train score 分桶看：

| subset | train score bucket | n | train PDMS | linked navtest PDMS | linked zero |
| --- | --- | ---: | ---: | ---: | ---: |
| good | >0.95 | 461 | 0.9931 | 0.9833 | 0.00% |
| good | 0.90-0.95 | 27 | 0.9348 | 0.9583 | 0.00% |
| low_or_zero | >0.95 | 246 | 0.9877 | 0.7725 | 14.97% |
| low_or_zero | 0.90-0.95 | 112 | 0.9266 | 0.7928 | 11.85% |
| low_or_zero | 0.85-0.90 | 57 | 0.8744 | 0.7910 | 8.64% |
| low_or_zero | 0.5-0.85 | 73 | 0.7472 | 0.7902 | 7.47% |
| low_or_zero | <=0.5 | 12 | 0.0605 | 0.6998 | 23.11% |

因此需要更精确地说：

- “train 学得差会提高相似 test 低分概率”成立。train PDMS < 0.90 的样本里，linked navtest PDMS < 0.85 的比例约 92.21%；train PDMS >= 0.90 时该比例降到 42.32%。
- “train 学得好就一定能推到相似 test 好”不成立。在 `low_or_zero` 簇里，train PDMS > 0.95 的 246 个样本，linked navtest PDMS 仍只有 0.7725，linked zero ratio 为 14.97%。
- 这表明当前问题是两部分叠加：一部分是训练簇自身仍没有完全学好，尤其 EP/DAC；另一部分是 train-good 到 test-low 的泛化缺口，可能来自 actor 交互、地图边界、转弯半径、时序细节等当前相似性向量没有显式建模的因素。

进一步按 linked navtest 行拆分：

| case | linked rows | unique train tokens | linked navtest PDMS | linked zero | similarity | train PDMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_high_test_high | 10795 | 697 | 0.9930 | 0.00% | 0.9571 | 0.9935 |
| train_high_test_low | 1453 | 347 | 0.4109 | 40.33% | 0.9403 | 0.9907 |
| train_low_test_low | 1154 | 148 | 0.6098 | 18.63% | 0.9305 | 0.7649 |

`train_high_test_low` 的 linked navtest 失败构成：

| failure reason | ratio | mean score | mean similarity |
| --- | ---: | ---: | ---: |
| low_ep | 35.38% | 0.7815 | 0.9254 |
| zero_nc_collision | 22.16% | 0.0000 | 0.9466 |
| low_ttc | 20.99% | 0.5689 | 0.9449 |
| zero_dac_offroad | 17.76% | 0.0000 | 0.9568 |

这个结果很关键：即使相似 train token 自身预测为高分，测试侧仍可能因为碰撞、越界、TTC 或 EP 掉分。这说明后续不能只把高分 buffer/GT 强监督灌给 diffusion，还要让 Stage3 优化显式覆盖“相似但交互/边界细节不同”的鲁棒性。

扩展到 top2000 后，结论保持一致且更稳定：

输出目录：

- `/mnt/project/VLA-AD/outputs/recogdrive_stage3_train_subset_sota_validation_20260624/full2000`

| subset | train tokens | train PDMS | train zero | train Core | train NC | train DAC | train EP | train TTC | train DDC | linked navtest PDMS | linked navtest zero | mean similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| good | 2000 | 0.9755 | 0.55% | 0.9777 | 0.9965 | 0.9975 | 0.9590 | 0.9875 | 0.9980 | 0.9814 | 0.00% | 0.9559 |
| low_or_zero | 2000 | 0.8947 | 3.30% | 0.9121 | 0.9955 | 0.9710 | 0.8170 | 0.9720 | 0.9710 | 0.7452 | 16.24% | 0.9429 |

top2000 的对应关系统计：

- train PDMS 与 linked navtest PDMS 的 Pearson 相关为 0.2629，Spearman 相关为 0.5065。
- train PDMS < 0.90 时，linked navtest PDMS < 0.85 的比例为 89.51%；train PDMS >= 0.90 时该比例为 42.49%。
- train PDMS > 0.95 的样本整体 linked navtest PDMS 更高，但在 `low_or_zero` 簇内部仍只有 0.7408，linked zero ratio 为 17.92%。

top2000 的 linked navtest case 拆分：

| case | linked rows | unique train tokens | linked navtest PDMS | linked zero | similarity | train PDMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train_high_test_high | 27531 | 2681 | 0.9918 | 0.00% | 0.9571 | 0.9926 |
| train_high_test_low | 3772 | 1180 | 0.3571 | 46.61% | 0.9415 | 0.9891 |
| train_low_test_low | 2972 | 590 | 0.5502 | 26.72% | 0.9360 | 0.7432 |

`train_high_test_low` 的失败构成在 top2000 中更清晰：

| failure reason | ratio | mean score | mean similarity |
| --- | ---: | ---: | ---: |
| low_ep | 26.78% | 0.7833 | 0.9261 |
| zero_dac_offroad | 23.75% | 0.0000 | 0.9562 |
| low_ttc | 22.43% | 0.5660 | 0.9445 |
| zero_nc_collision | 22.16% | 0.0000 | 0.9414 |

top2000 进一步确认：训练近邻自身学得弱会显著提高相似测试低分概率；但相似 train 已高分时，测试仍可能因为 DAC/NC/TTC/EP 失败而低分。这说明下一步应针对 train-high/test-low 的鲁棒泛化缺口做专项分析，而不是简单假设“只要把训练集高分轨迹监督进去，navtest 就会自然提升”。

## 下一步建议

1. 对 `zero_dac_offroad + turning` 做专项复盘：抽取 top train neighbors 的 GT、91.04 开发版预测、SOTA 预测，检查是否是转弯半径、lane boundary、route centerline 或 heading 约束问题。
2. 对 `zero_nc_collision` 增加 actor-aware 相似性：从 metric cache 抽取前车/横穿车/行人最近距离、相对速度、TTC proxy。当前 hidden slot 有视觉语义，但没有显式 actor 距离表。
3. 对 `low_ep` 类别检查 GRPO 训练日志：关注 positive advantage 是否仍给了慢轨迹、all-slow group 比例、EP floor pass ratio 和 Core delta。
4. 对 `train_scene_to_navtest_outcome_mapping.csv` 中的 `maps_to_low_or_zero_tests` 取前 500-2000 个 train tokens，补跑 SOTA 权重在 train-subset 上的 PDMS。若 train-subset 也低分，说明训练内也没学好；若 train-subset 高分但相似 navtest 低分，说明泛化或细粒度环境因素不足。
5. 如果后续找到原版 90.54/90.55 的逐 token CSV，直接复用本次特征缓存，只重跑 CSV join 和 delta 分析即可。

## 复现命令

```bash
/root/miniconda3/envs/navsim/bin/python scripts/evaluation/analyze_recogdrive_scene_similarity_hidden.py \
  --train-hidden-root /mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z \
  --navtest-hidden-root /mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtest_stage1_v2_clean_bf16_20260617T020404Z \
  --sota-eval-csv /mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_4to7_sharded_conc4_watch_from5700_v1/eval_step-step_21600/local_sharded_aggregated.csv \
  --baseline-eval-csv /mnt/project/tsinghua_code/evaluation_record/original_eval_navtest_epoch9_2026.05.14.16.49.31.csv \
  --baseline-name recogdrive_new_9104_dev_epoch9_step13300 \
  --output-dir /mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624 \
  --workers 48 \
  --knn-k 20 \
  --query-block 512 \
  --gpu-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
```
