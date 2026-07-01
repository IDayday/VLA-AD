# ReCogDrive Stage3 SOTA 随机 10k 训练集失败模式与采样敏感性分析

日期：2026-06-24

参考论文：<https://arxiv.org/pdf/2603.06049>

Checkpoint：

```text
/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_sharded_conc4_watch_from5400_v1/checkpoint_archive/step-step_21600.ckpt
```

对应 SOTA 备份：

```text
/mnt/project/VLA-AD/checkpoints/recogdrive/stage3_sota_backups/core_pareto_grpo_v2_step21600_pdms0.910274_20260621/recogdrive_stage3_core_pareto_grpo_v2_step21600_pdms0.910274.ckpt
```

NAVTEST 单次评估参考：

```text
/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_4to7_sharded_conc4_watch_from5700_v1/eval_step-step_21600/local_sharded_aggregated.csv
```

该单次 NAVTEST 的 `average` 行为 `PDMS=0.910274`。因此后文所有训练集 10k 的 `mean-PDMS@8=0.947543` 都不能解释成 full-NAVTEST PDMS 93 或 95。它是训练集随机场景上每个 token 采样 8 条候选轨迹后的组内平均 PDM。

## 摘要结论

这份文档重写了此前基于 4k/定向子集的 failure mode 和 sensitivity 分析。新的分析使用随机抽取的 10000 个 navtrain 场景 token，并补齐策略多样性、采样敏感性、hidden-neighbor train/test 对齐和失败类型分布。

核心结论如下：

- 当前 Stage3 SOTA 是高 PDMS、低方差、局部随机的策略。随机 10k navtrain 上 `mean-PDMS@8=0.9475`，`best-PDMS@8=0.9550`，但 `std-PDMS@8=0.00578`，说明同一场景内 8 条候选的 PDM 差异很小。
- 策略多样性比论文中的 ReCogDrive baseline 略高，但远低于 Curious-VLA。我们的 `mean-pFDE=0.390m`，高于论文 ReCogDrive 的 `0.325m`，但只有 Curious-VLA `1.415m` 的 `27.5%`。
- 采样多次能带来有限 best-of 收益，但大多数场景没有真正的多模态候选。`62.91%` 的 token 属于 `stable_high`，即 8/8 候选都 `PDMS>=0.95`；只有 `1.29%` 的 token 组内 `best-min >= 0.10`。
- 训练集高分不保证相似 navtest 高分。在 5287 个可映射到 hidden-neighbor navtest 簇的随机样本中，`train_high_test_low` 有 400 个：训练侧 `mean-PDMS@8=0.9879`，但相似 navtest 均分只有 `0.6307`，zero ratio 为 `0.2595`。
- 低分测试簇并不主要是“完全没有训练近邻”。`maps_to_low_or_zero_tests` 的平均 hidden 相似度为 `0.9357`，但 linked navtest PDMS 只有 `0.6997`，zero ratio `0.2019`。这更像是目标/场景细节吸收不足，而不是简单数据覆盖缺失。

## 与旧版 4k/定向分析的差异

旧版文件分析的是定向 train proxy slice：从 hidden-state train/navtest 映射里挑出 `maps_to_good_tests` 和 `maps_to_low_or_zero_tests` 两组，合计 13213 个 train scene，并额外跑了一个 320 个 navtest scene 的 best-of-8 sensitivity 小样本。

本版改为：

- 训练集场景 token 随机抽样，而不是按 navtest 失败簇定向抽样。
- 样本数为 10000，来自 `navtrain.yaml` 与 `metric_cache_train_full` 的交集。
- 每个 token 采样 8 条完整轨迹，计算论文式 behavioral diagnostics：quality、diversity、perf。
- 对 10000 个 token 中能够命中 hidden-neighbor 反向映射的 5287 个 token，再做 train/test family 对齐。

因此本版的 `train score` 是 `mean-PDMS@8`，不是旧版中的单条轨迹 train PDMS。两者都能说明问题，但语义不同：

- 单条 train PDMS 更接近常规 evaluation。
- `mean-PDMS@8` 更适合分析当前随机采样策略的稳定性、best-of 潜力和 reward diversity。

## 数据产物

随机 10k token：

```text
/mnt/project/VLA-AD/outputs/stage3_policy_diversity_k8_navtrain10k_random_step21600_20260624T0630Z/navtrain_random10000_seed260306049_tokens.txt
```

K=8 策略多样性聚合结果：

```text
/mnt/project/VLA-AD/outputs/stage3_policy_diversity_k8_navtrain10k_random_step21600_thread_20260624T0640Z/local_sharded_aggregated.csv
```

注意：聚合 CSV 里有一行 `token=average` 的统计行。本文所有 10k 数字都先按 token 文件过滤，排除了该 average 行。过滤后为 `10000/10000` 有效 token，所有 token 的 `candidate_valid_count=8`。

本次补充生成的分析产物：

```text
/mnt/project/VLA-AD/outputs/stage3_policy_diversity_k8_navtrain10k_random_step21600_thread_20260624T0640Z/failure_mode_analysis_10k
```

主要文件：

- `policy_diversity_10k_filtered_with_candidate_stats.csv`
- `overall_mean_summary.csv`
- `percentile_summary.csv`
- `threshold_summary.csv`
- `sensitivity_class_summary.csv`
- `train_perf_bucket_summary.csv`
- `pfde_paper_bucket_summary.csv`
- `policy_diversity_10k_joined_with_navtest_mapping.csv`
- `mapping_proxy_summary.csv`
- `train_test_case_summary.csv`
- `matched_correlation_summary.csv`
- `train_threshold_vs_linked_test_low_summary.csv`
- `linked_case_mix_distribution.csv`
- `examples_*.csv`

Hidden-neighbor train/test 反向映射：

```text
/mnt/project/VLA-AD/outputs/recogdrive_stage3_scene_similarity_hidden_20260624/train_scene_to_navtest_outcome_mapping.csv
```

NAVTEST best-of-6 参考结果：

```text
/mnt/project/VLA-AD/outputs/stage3_best_of6_step21600_navtest_full_20260624T0345Z/local_sharded_aggregated.csv
```

该 best-of-6 full NAVTEST 的 `average` 行为 `PDMS=0.923203`，相对单次 `0.910274` 提升 `+1.293` points。这个结果用于解释 best-of 潜力，但不是本文 10k navtrain 的主表。

## 评估设置

| 项目 | 设置 |
| --- | --- |
| Split | `navtrain` |
| Token 池 | `navtrain.yaml` 与 `metric_cache_train_full` 交集 |
| Token 池大小 | 103288 |
| 随机样本数 | 10000 |
| 随机种子 | 260306049 |
| 每 token 采样数 | `K=8` |
| 采样对象 | ReCogDrive diffusion head 的完整未来轨迹 |
| 每条轨迹 horizon | 4 秒，8 个 future poses |
| 几何指标 | 使用 `(x, y)` 轨迹点 |
| PDM 指标 | 使用完整轨迹并由 NAVSIM exact PDM 评分 |
| async backend | `thread` |
| 有效样本 | 10000/10000 |

这里的 `K=8` 用于论文式策略多样性诊断。它不是逐 token 筛选，也不是逐轨迹点筛选。

一个 `token` 是一个场景 sample token。一个候选是该 token 下采样出的一条完整未来轨迹。PDM/PDMS 必须基于完整轨迹与场景约束计算，不能对单个 future pose 单独计算有意义的 PDMS。

## 指标定义

对每个场景 token，模型采样 8 条候选轨迹：

```text
Y = {y_1, y_2, ..., y_8}
```

每条 `y_i` 是 4 秒未来轨迹，包含 8 个 future poses。

### Quality: minADE / minFDE

Quality 衡量 8 条候选里是否至少有一条接近 logged human GT：

```text
ADE_i = mean_t || y_i,t - y*_t ||_2
FDE_i = || y_i,T - y*_T ||_2

minADE = min_i ADE_i
minFDE = min_i FDE_i
```

该指标越小越好。它是 imitation-style GT 贴近度指标，不是安全性指标。一个模型可以 PDM 很高，但 minFDE 较差，因为它输出了不同于 logged human 的可行驾驶方式。

### Diversity: mean-pADE / mean-pFDE

Diversity 衡量同一个 token 下 8 条候选轨迹彼此是否展开：

```text
pADE_ij = mean_t || y_i,t - y_j,t ||_2
pFDE_ij = || y_i,T - y_j,T ||_2

mean-pADE = mean_{i<j} pADE_ij
mean-pFDE = mean_{i<j} pFDE_ij
```

该指标越大越好。`mean-pFDE` 可以近似理解为候选终点的平均分散距离。

### Perf: mean-PDMS

Perf 衡量 8 条候选整体是否安全、合规、有进展：

```text
mean-PDMS = mean_i PDM(y_i)
```

该指标越大越好。它不是 best-of-N。Best-of-N 会取 8 条里的最高分：

```text
best-PDMS@8 = max_i PDM(y_i)
```

本文同时报告 `perf_best_pdms`、`perf_min_pdms`、`perf_std_pdms`、`sample_range`，用于判断采样敏感性。

## 论文指标对比

论文 Table 4 的参考结果和本次随机 10k navtrain 结果如下：

| 方法 | Quality minADE/FDE ↓ | Diversity mean-pADE/pFDE ↑ | Perf mean-PDMS ↑ |
| --- | ---: | ---: | ---: |
| 论文 ReCogDrive | 0.295 / 0.621 | 0.148 / 0.325 | 90.95 |
| 论文 Curious-VLA + FTE + RL | 0.269 / 0.547 | 0.641 / 1.415 | 91.55 |
| 我们 Stage3 SOTA, random 10k navtrain | 0.743 / 1.342 | 0.199 / 0.390 | 94.75 |

相对论文 ReCogDrive：

| 指标 | 我们 | 论文 ReCogDrive | 相对关系 |
| --- | ---: | ---: | ---: |
| `mean-pADE` | 0.199 | 0.148 | 1.35x |
| `mean-pFDE` | 0.390 | 0.325 | 1.20x |
| `mean-PDMS` | 94.75 | 90.95 | +3.80 points |
| `minADE` | 0.743 | 0.295 | 2.52x worse |
| `minFDE` | 1.342 | 0.621 | 2.16x worse |

相对论文 Curious-VLA + FTE + RL：

| 指标 | 我们 | 论文 Curious-VLA | 相对关系 |
| --- | ---: | ---: | ---: |
| `mean-pADE` | 0.199 | 0.641 | 0.31x |
| `mean-pFDE` | 0.390 | 1.415 | 0.28x |
| `mean-PDMS` | 94.75 | 91.55 | +3.20 points |
| `minADE` | 0.743 | 0.269 | 2.76x worse |
| `minFDE` | 1.342 | 0.547 | 2.45x worse |

解释：

- 我们的策略不是完全 collapse。它的 pairwise diversity 高于论文 ReCogDrive baseline。
- 但它也不是 Curious-VLA 那种行为级多模态策略。`mean-pFDE=0.390m` 仍然只是局部扰动级别，而不是多种驾驶意图级别。
- `mean-PDMS` 很高，说明当前模型会稳定输出 PDM 友好的轨迹。
- `minADE/minFDE` 明显较差，说明高 PDM 策略和 logged human GT 有偏移。这个偏移不必然是坏驾驶，但说明 8 次采样也没有覆盖到人类轨迹附近。

## 10k 总体结果

| 指标 | 均值 |
| --- | ---: |
| `quality_min_ade` | 0.742700 |
| `quality_min_fde` | 1.342238 |
| `diversity_mean_pade` | 0.199117 |
| `diversity_mean_pfde` | 0.389558 |
| `perf_mean_pdms` | 0.947543 |
| `perf_best_pdms` | 0.955026 |
| `perf_min_pdms` | 0.939057 |
| `perf_std_pdms` | 0.005777 |
| `sample_range = best-min` | 0.015969 |
| `zero_rate@8, PDM<=0.05` | 0.009675 |
| `success_rate@8, PDM>=0.85` | 0.901075 |
| `high_success_rate@8, PDM>=0.95` | 0.671913 |
| `best_gain_over_mean` | 0.007483 |
| `best_gain_over_candidate0` | 0.007021 |

`best_gain_over_mean` 只有 `0.00748`，说明在随机训练集上，best-of-8 相对 mean candidate 的平均提升很小。这和 full NAVTEST best-of-6 只有 `+1.293` points 的提升方向一致：采样有收益，但候选分布不够展开。

## 分布统计

| 指标 | P5 | P25 | P50 | P75 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quality_min_ade` | 0.296456 | 0.456995 | 0.658323 | 0.926776 | 1.505076 |
| `quality_min_fde` | 0.107381 | 0.432167 | 1.052098 | 1.967736 | 3.551905 |
| `diversity_mean_pade` | 0.102844 | 0.146562 | 0.183843 | 0.236518 | 0.348761 |
| `diversity_mean_pfde` | 0.120710 | 0.218446 | 0.318675 | 0.499062 | 0.888276 |
| `perf_mean_pdms` | 0.819016 | 0.928847 | 0.995641 | 1.000000 | 1.000000 |
| `perf_best_pdms` | 0.836642 | 0.939869 | 1.000000 | 1.000000 | 1.000000 |
| `perf_min_pdms` | 0.804672 | 0.918839 | 0.987645 | 1.000000 | 1.000000 |
| `perf_std_pdms` | 0.000000 | 0.000000 | 0.001626 | 0.004944 | 0.010246 |
| `sample_range` | 0.000000 | 0.000000 | 0.004976 | 0.015349 | 0.031908 |
| `best_gain_over_mean` | 0.000000 | 0.000000 | 0.001757 | 0.006992 | 0.015479 |

阈值统计：

| 条件 | token 数 | 比例 |
| --- | ---: | ---: |
| `mean-pADE >= 0.148`，达到论文 ReCogDrive 水平 | 7388 | 73.88% |
| `mean-pFDE >= 0.325`，达到论文 ReCogDrive 水平 | 4867 | 48.67% |
| `mean-pADE >= 0.641`，达到论文 Curious-VLA 水平 | 2 | 0.02% |
| `mean-pFDE >= 1.415`，达到论文 Curious-VLA 水平 | 20 | 0.20% |
| `minFDE <= 0.621`，达到论文 ReCogDrive quality 水平 | 3365 | 33.65% |
| `minFDE <= 0.547`，达到论文 Curious-VLA quality 水平 | 3039 | 30.39% |
| `mean-PDMS >= 0.90` | 8380 | 83.80% |
| `mean-PDMS >= 0.95` | 6679 | 66.79% |
| `mean-PDMS <= 0.85` | 777 | 7.77% |
| 至少一个候选 `PDM<=0.05` | 122 | 1.22% |
| 8 个候选全部 `PDM<=0.05` | 73 | 0.73% |
| 至少一个候选 `PDM>=0.85` | 9378 | 93.78% |
| 8 个候选全部 `PDM>=0.95` | 6291 | 62.91% |
| `sample_range >= 0.10` | 129 | 1.29% |
| `sample_range >= 0.50` | 51 | 0.51% |

最关键的分布事实是：`mean-PDMS` 中位数已经到 `0.9956`，而 `sample_range` 中位数只有 `0.0050`。这说明多数 token 上，8 次采样只是围绕同一个高分局部模式做小扰动。

## 采样敏感性

敏感性分桶沿用旧分析脚本的规则，基于每个 token 的 8 条候选完整轨迹 PDM：

- `knife_edge_zero_to_success`：候选最小分 `<=0.05`，且最大分 `>=0.85`。
- `large_range`：非 knife-edge，且 `best-min >=0.50`。
- `stable_low`：候选最大分 `<0.85`。
- `stable_high`：候选最小分 `>=0.95`。
- `moderate_range`：剩余场景。注意这个名字来自旧脚本，并不表示 `range>=0.10`，而是表示不属于稳定高/稳定低/大跨度/刀刃型。

| sensitivity class | scenes | candidate0 | mean | best | min | range | zero rate | success@0.85 | success@0.95 | mean-pFDE | minFDE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stable_high | 6291 | 0.9954 | 0.9953 | 0.9970 | 0.9931 | 0.0039 | 0.0000 | 1.0000 | 1.0000 | 0.3512 | 1.4422 |
| moderate_range | 3051 | 0.9128 | 0.9123 | 0.9250 | 0.8975 | 0.0275 | 0.0000 | 0.9647 | 0.1372 | 0.4784 | 1.1258 |
| stable_low | 607 | 0.6725 | 0.6732 | 0.6802 | 0.6662 | 0.0139 | 0.1203 | 0.0000 | 0.0000 | 0.3182 | 1.3608 |
| knife_edge_zero_to_success | 34 | 0.5199 | 0.4702 | 0.9605 | 0.0000 | 0.9605 | 0.4853 | 0.4596 | 0.2537 | 0.6244 | 1.5791 |
| large_range | 17 | 0.4238 | 0.3449 | 0.6302 | 0.0278 | 0.6024 | 0.4265 | 0.0588 | 0.0588 | 0.7145 | 2.0648 |

这个表说明：

- `stable_high` 是主导类别。62.91% token 的 8 条候选全都 `>=0.95`，这些场景对 best-of-N 不敏感。
- `stable_low` 有 607 个。它们不是 selector 选错候选，而是当前候选分布内没有高分模式。Best-of-N 很难救这类样本。
- 真正有 best-of 潜力的是 `knife_edge_zero_to_success` 和一部分 `large_range/moderate_range`，但数量很少。`knife_edge_zero_to_success` 只有 34 个，占 0.34%。
- `large_range` 的 mean-pFDE 更高，说明轨迹几何展开和 PDM 分数跨度相关，但数量只有 17 个，不能支撑大幅 best-of 提升。

因此，适合我们算法的 best-of-N 评估方式不是逐 token 或逐轨迹点筛选，而是：

1. 对同一个 scene token 复用同一 VLM context。
2. 从 diffusion head 采样 N 条完整轨迹。
3. 对每条完整轨迹独立计算 exact PDM。
4. 用 `max_i PDM(y_i)` 得到 best-of-N。
5. 用 `min/max/range/std/zero_rate/success_rate` 诊断是否存在可选择的好候选。

如果只看最终 best-of 分数，会掩盖两种完全不同的情况：

- 候选分布里真的有好有坏，selector 或 oracle best-of 可以提升。
- 候选分布整体都很高或整体都很低，best-of 几乎没有信息量。

## 为什么 PDMS 提升时一些指标会下降

这个现象在本次 10k 和 full NAVTEST best-of-6 中都符合预期。原因不是评估脚本逐 token 乱选，而是目标函数不同。

PDM/PDMS 是安全、合规、进展、舒适、方向等组件的组合。Best-of-PDM 选择的是 PDM 最高的完整轨迹，不是选择最接近 human GT 的轨迹，也不是选择多样性最大的轨迹。因此：

- `minADE/minFDE` 可能变差：PDM 最优轨迹可能比 logged human 更保守、更靠中线、更早减速或更少横移。
- `mean-pADE/mean-pFDE` 不一定提高：best-of 只输出一条轨迹，策略分布是否多样是采样集合的属性，不是最终选中轨迹的属性。
- EP、TTC、DAC、NC 等子指标可能有 trade-off：例如更安全的 TTC 轨迹可能进展更慢；更高 EP 的轨迹可能更接近边界或更容易触发 DAC/TTC 风险。
- 单个 future pose 不能评价 PDMS：PDMS 依赖整条轨迹与 drivable area、碰撞、进展、方向、TTC 的时序关系。

所以，最终 PDMS 提升但某些 imitation 或子项指标下降，并不自动表示评估错误。它表示 best-of 选择目标是 PDM，而不是 human trajectory matching 或单一组件最大化。

## 按 train-side perf 分桶

| train perf bucket | scenes | mean-PDMS | best-PDMS | min-PDMS | zero rate | success@0.95 | mean-pADE | mean-pFDE | minADE | minFDE | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `train_high_ge_0p95` | 6679 | 0.9930 | 0.9952 | 0.9902 | 0.0000 | 0.9850 | 0.1938 | 0.3668 | 0.7882 | 1.4481 | 0.0051 |
| `train_mid_0p90_0p95` | 1701 | 0.9270 | 0.9362 | 0.9142 | 0.0000 | 0.0676 | 0.2330 | 0.5161 | 0.6622 | 1.1093 | 0.0220 |
| `train_lowmid_0p85_0p90` | 843 | 0.8785 | 0.8868 | 0.8646 | 0.0001 | 0.0085 | 0.1841 | 0.3456 | 0.5790 | 0.9467 | 0.0222 |
| `train_low_lt_0p85` | 777 | 0.6762 | 0.7245 | 0.6349 | 0.1244 | 0.0238 | 0.1868 | 0.3564 | 0.7058 | 1.3716 | 0.0896 |

几个观察：

- 高分桶的 `success@0.95=0.9850`，几乎所有候选都高分。
- 中分桶的 `mean-pFDE=0.5161`，比高分桶更高，说明模型在中等难度场景里有更多几何扰动，但这些扰动没有变成高 reward 多模态。
- 低分桶的 `sample_range=0.0896` 最大，zero rate 也最高。这是 best-of 最可能有用的区域，但只占 7.77%。

## 按论文 pFDE 阈值分桶

| pFDE bucket | scenes | mean-pFDE | mean-PDMS | best-PDMS | minFDE | range | zero rate | success@0.95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| below ReCogDrive `<0.325` | 5133 | 0.2147 | 0.9584 | 0.9609 | 1.2082 | 0.0055 | 0.0016 | 0.7115 |
| ReCogDrive+ `>=0.325` | 4847 | 0.5701 | 0.9365 | 0.9491 | 1.4810 | 0.0268 | 0.0178 | 0.6307 |
| Curious level `>=1.415` | 20 | 1.5246 | 0.8530 | 0.8746 | 2.1112 | 0.0750 | 0.1000 | 0.5125 |

该表对策略多样性的解释很关键：

- 多样性提高不是免费的。pFDE 越高，mean-PDMS 越低，zero rate 越高。
- 达到 Curious-VLA pFDE 水平的 token 只有 20 个，而且 mean-PDMS 降到 0.8530。这说明当前模型只有在困难或不稳定场景里才偶发较大轨迹展开。
- 理想状态不是简单把 pFDE 拉大，而是像论文 Curious-VLA 那样扩大可行轨迹模式，同时保持高 mean-PDMS。

## 全局相关性

| relation | Pearson | Spearman |
| --- | ---: | ---: |
| `mean-pFDE` vs `mean-PDMS` | -0.1131 | -0.2134 |
| `mean-pADE` vs `mean-PDMS` | -0.0887 | -0.1875 |
| `minFDE` vs `mean-PDMS` | -0.0224 | 0.1153 |
| `minFDE` vs `mean-pFDE` | 0.1217 | 0.1076 |
| `std-PDMS@8` vs `mean-pFDE` | 0.1664 | 0.4733 |
| `sample_range` vs `mean-pFDE` | 0.1882 | 0.4682 |
| `sample_range` vs `mean-PDMS` | -0.3588 | -0.8028 |
| `zero_rate` vs `mean-PDMS` | -0.7972 | -0.1990 |
| `success@0.95` vs `mean-pFDE` | -0.1238 | -0.1176 |

结论：

- 几何多样性和 PDM 分数跨度正相关，但相关性主要体现在排序上。
- PDM 分数越高，组内 range 越小。这是典型的 narrow high-reward mode。
- `minFDE` 与 `mean-PDMS` 几乎不线性相关，说明 imitation closeness 和 PDM quality 在当前模型上已经明显解耦。

## Train/Test hidden-neighbor 对齐

10000 个随机 token 中，能在 `train_scene_to_navtest_outcome_mapping.csv` 找到反向映射的有 5287 个，覆盖率 `52.87%`。

| 映射状态 | token 数 | 占 10k 比例 |
| --- | ---: | ---: |
| no mapping | 4713 | 47.13% |
| mixed | 3526 | 35.26% |
| maps_to_good_tests | 825 | 8.25% |
| maps_to_low_or_zero_tests | 517 | 5.17% |
| maps_to_medium_good_tests | 419 | 4.19% |

这里的 `test_outcome_proxy` 来自 hidden-neighbor 反向聚合：先在 navtest 上找每个测试场景的 top-20 train 近邻，再把每个 train token 被命中的 navtest 表现聚合起来。因此它表示“这个 train token 附近的 navtest family 表现如何”，不是模型在该 train token 上的直接测试表现。

全量 hidden mapping 中共有 54067 个 train token：

| proxy label | train tokens | linked navtest mean | linked navtest median | mean navtest PDMS | zero ratio | mean similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maps_to_good_tests | 7750 | 8.47 | 7 | 0.9828 | 0.0000 | 0.9536 |
| maps_to_medium_good_tests | 4330 | 9.65 | 8 | 0.9272 | 0.0000 | 0.9396 |
| mixed | 36524 | 2.69 | 2 | 0.9240 | 0.0262 | 0.9401 |
| maps_to_low_or_zero_tests | 5463 | 6.75 | 5 | 0.7118 | 0.1873 | 0.9392 |

随机 10k 的可映射子集统计如下：

| test outcome proxy | scenes | train mean-PDMS@8 | train best | train min | train zero | train success@0.95 | mean-pFDE | minFDE | linked navtest PDMS | linked navtest zero | linked low ratio | linked high ratio | similarity | linked count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maps_to_good_tests | 825 | 0.9753 | 0.9812 | 0.9697 | 0.0039 | 0.8702 | 0.4466 | 1.4268 | 0.9820 | 0.0000 | 0.0060 | 0.8697 | 0.9534 | 8.3552 |
| maps_to_medium_good_tests | 419 | 0.9143 | 0.9268 | 0.8978 | 0.0191 | 0.4365 | 0.4983 | 1.3239 | 0.9273 | 0.0000 | 0.0798 | 0.3922 | 0.9371 | 9.8974 |
| mixed | 3526 | 0.9464 | 0.9539 | 0.9384 | 0.0093 | 0.6614 | 0.3876 | 1.3395 | 0.9260 | 0.0231 | 0.1087 | 0.6321 | 0.9400 | 2.7805 |
| maps_to_low_or_zero_tests | 517 | 0.9091 | 0.9201 | 0.8936 | 0.0276 | 0.5172 | 0.4763 | 1.4203 | 0.6997 | 0.2019 | 0.4295 | 0.3225 | 0.9357 | 6.7795 |

主要含义：

- `maps_to_good_tests` 在训练侧也明显更高分，说明 mapping 有实际信号。
- `maps_to_low_or_zero_tests` 在训练侧也下降到 `0.9091`，但没有低到 navtest 的 `0.6997`。这说明低分测试 family 不是完全由 train-side 能力缺失解释，还存在测试场景细节、地图/actor 差异、泛化和目标函数差异。
- `mean-pFDE` 在 low/medium proxy 中更高，但 train mean-PDMS 更低。这再次说明当前“更展开”的样本往往是更不稳定，而不是真正的高质量多模态。

## Train/Test case split

沿用旧脚本阈值：

- train high：`train mean-PDMS@8 > 0.95`
- train low：`train mean-PDMS@8 < 0.90`
- test high：`linked navtest mean_sota_score > 0.95`
- test low：`linked navtest mean_sota_score < 0.85`

| case | scenes | train mean-PDMS@8 | train best | train min | train zero | train success@0.95 | mean-pFDE | minFDE | linked navtest PDMS | linked zero | similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train_high_test_high | 2427 | 0.9939 | 0.9961 | 0.9911 | 0.0000 | 0.9893 | 0.3758 | 1.4145 | 0.9900 | 0.0000 | 0.9504 |
| other | 1871 | 0.9175 | 0.9275 | 0.9057 | 0.0107 | 0.3668 | 0.4562 | 1.2720 | 0.9169 | 0.0032 | 0.9348 |
| train_high_test_low | 400 | 0.9879 | 0.9917 | 0.9834 | 0.0000 | 0.9706 | 0.4276 | 1.5963 | 0.6307 | 0.2595 | 0.9347 |
| train_low_test_low | 218 | 0.7584 | 0.7813 | 0.7308 | 0.0837 | 0.0120 | 0.3897 | 1.2619 | 0.6721 | 0.1761 | 0.9295 |
| train_low_test_high | 201 | 0.7344 | 0.7861 | 0.6932 | 0.0995 | 0.0367 | 0.3791 | 1.2265 | 0.9848 | 0.0000 | 0.9277 |
| train_mid_test_low | 170 | 0.9274 | 0.9373 | 0.9110 | 0.0000 | 0.0860 | 0.5419 | 1.2716 | 0.6497 | 0.2217 | 0.9338 |

相关性：

| relation | Pearson | Spearman |
| --- | ---: | ---: |
| train `mean-PDMS@8` vs linked navtest PDMS | 0.1562 | 0.4760 |
| train `mean-PDMS@8` vs linked zero ratio | -0.0745 | -0.0755 |
| train `best-PDMS@8` vs linked navtest PDMS | 0.1553 | 0.4672 |
| train `mean-pFDE` vs linked navtest PDMS | -0.0973 | -0.1931 |
| train `mean-pFDE` vs linked zero ratio | 0.0803 | 0.1170 |
| train `minFDE` vs linked navtest PDMS | -0.0596 | 0.0154 |
| train `sample_range` vs linked navtest PDMS | -0.0577 | -0.3878 |

阈值关系：

| train mean-PDMS@8 threshold | pct train below | pct linked test-low if below | pct linked test-low if above |
| ---: | ---: | ---: | ---: |
| 0.85 | 0.0787 | 0.2861 | 0.1373 |
| 0.90 | 0.1661 | 0.2483 | 0.1293 |
| 0.95 | 0.3427 | 0.2141 | 0.1151 |

解释：

- train-side 低分会增加 linked test-low 概率，但不是决定性关系。
- Spearman 高于 Pearson，说明排序信号存在，但不是线性对应。
- `train_high_test_low` 是最值得关注的 bucket：train 侧采样稳定高分，测试相似 family 却大量 zero/low。这类问题靠继续提升 train-side oracle best-of 不一定解决，需要更细的环境特征、风险建模或 selector 训练。

## Linked failure mix

`train_high_test_low` 的 linked navtest failure mix：

| failure reason | linked rows | ratio |
| --- | ---: | ---: |
| nonzero_ok | 1069 | 0.5210 |
| low_ep | 275 | 0.1340 |
| zero_dac_offroad | 227 | 0.1106 |
| zero_nc_collision | 203 | 0.0989 |
| low_ttc | 162 | 0.0789 |
| low_ddc | 76 | 0.0370 |

`train_low_test_low` 的 linked navtest failure mix：

| failure reason | linked rows | ratio |
| --- | ---: | ---: |
| low_ep | 482 | 0.4806 |
| nonzero_ok | 291 | 0.2901 |
| zero_dac_offroad | 81 | 0.0808 |
| zero_nc_collision | 61 | 0.0608 |
| low_ttc | 44 | 0.0439 |
| low_ep+low_ttc | 19 | 0.0189 |

`train_mid_test_low` 的 linked navtest failure mix：

| failure reason | linked rows | ratio |
| --- | ---: | ---: |
| nonzero_ok | 378 | 0.4621 |
| low_ep | 160 | 0.1956 |
| zero_dac_offroad | 90 | 0.1100 |
| zero_nc_collision | 70 | 0.0856 |
| low_ttc | 63 | 0.0770 |
| low_ddc | 24 | 0.0293 |

Motion mix：

| case | top motion mix |
| --- | --- |
| train_high_test_low | lateral_shift 28.54%, turning 27.52%, straight 20.60%, fast_straight 15.78%, stop_or_creep 7.55% |
| train_low_test_low | straight 40.34%, turning 20.82%, lateral_shift 17.73%, fast_straight 11.25%, stop_or_creep 9.86% |
| train_mid_test_low | lateral_shift 32.07%, turning 25.00%, straight 23.29%, fast_straight 19.39%, stop_or_creep 0.24% |
| train_high_test_high | lateral_shift 33.89%, straight 26.54%, turning 20.75%, fast_straight 10.82%, stop_or_creep 7.99% |

这些分布说明：

- `train_high_test_low` 不是纯粹的低进展问题，它同时包含 DAC offroad、NC collision、TTC、DDC 等安全相关失败，且 lateral/turning 比例高。
- `train_low_test_low` 更偏 EP 问题，说明训练侧低分 family 和测试侧低分 family 在“进展不足”上有一致性。
- `train_high_test_low` 中 `nonzero_ok` 仍占 52.10%，表示很多 linked navtest 不是零分，而是 PDM 低到 `<0.85` 或 mixed family。这类场景需要继续拆分，不应只按 zero/non-zero 处理。

## 典型样本

稳定高分但多样性极低：

| token | mean-PDMS | best | min | mean-pFDE | minFDE | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1552b4b20abd55fa` | 1.0000 | 1.0000 | 1.0000 | 0.0073 | 2.3093 | 0.0000 |
| `680854f51b515483` | 1.0000 | 1.0000 | 1.0000 | 0.0100 | 2.8358 | 0.0000 |
| `7fd917f219c254db` | 1.0000 | 1.0000 | 1.0000 | 0.0113 | 3.6153 | 0.0000 |
| `30e2a85cc85d585f` | 1.0000 | 1.0000 | 1.0000 | 0.0119 | 1.9709 | 0.0000 |

这些 token 是 narrow high-reward mode 的典型：PDM 完全稳定，但采样几乎不展开，且可能离 GT 较远。

高 PDMS 且高 pFDE：

| token | mean-PDMS | best | min | mean-pFDE | minFDE | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a19a65c5bf0c5965` | 0.9896 | 1.0000 | 0.9675 | 1.6374 | 2.8086 | 0.0325 |
| `f1527c1d7fb7514e` | 0.9625 | 0.9886 | 0.9401 | 1.5830 | 4.5724 | 0.0484 |
| `9dc02d23dbf75845` | 0.9991 | 1.0000 | 0.9926 | 1.5582 | 2.1675 | 0.0074 |
| `c1a3efdd543154a9` | 0.9641 | 0.9808 | 0.9501 | 1.4975 | 1.3757 | 0.0307 |

这些是最接近“可行多样性”的样本，但数量极少。全 10k 中 `mean-pFDE>=1.415` 的只有 20 个。

Knife-edge zero-to-success：

| token | mean-PDMS | best | min | zero rate | success@0.95 | mean-pFDE | minFDE | range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `e24e68bd02a8588a` | 0.2500 | 1.0000 | 0.0000 | 0.750 | 0.250 | 0.9493 | 0.7092 | 1.0000 |
| `28e609264c295deb` | 0.4167 | 1.0000 | 0.0000 | 0.375 | 0.125 | 0.5120 | 2.5122 | 1.0000 |
| `576177f2e0715644` | 0.5000 | 1.0000 | 0.0000 | 0.500 | 0.500 | 0.3746 | 2.4074 | 1.0000 |
| `1da09ddc10b55a5f` | 0.5938 | 1.0000 | 0.0000 | 0.250 | 0.375 | 0.6650 | 0.7845 | 1.0000 |

这些场景说明模型分布里偶尔存在高分模式，但概率不稳定。它们适合训练 selector/value/risk head，也适合用作 GRPO 的高方差组样本。

`train_high_test_low` 例子：

| token | train mean-PDMS@8 | mean-pFDE | minFDE | linked navtest PDMS | zero ratio | similarity | proxy | failure mix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `270900cc875b5448` | 1.0000 | 0.2744 | 1.6627 | 0.0000 | 1.0000 | 0.9459 | mixed | zero_nc_collision:1 |
| `30b7a3bf71b956de` | 1.0000 | 0.7928 | 2.9421 | 0.0000 | 1.0000 | 0.9877 | maps_to_low_or_zero_tests | zero_nc_collision:3 |
| `48ff23d4f1d15802` | 1.0000 | 0.2255 | 1.3539 | 0.0000 | 1.0000 | 0.8749 | mixed | zero_nc_collision:1 |
| `b65e09c4d8125da0` | 0.9584 | 0.5818 | 3.9565 | 0.0000 | 1.0000 | 0.9140 | mixed | zero_dac_offroad:2 |

这类样本说明：train 侧自己已经稳定高分，但 hidden-neighbor 相似测试 family 仍会出现 collision/offroad zero。下一步不能只继续做 train-side best-of，而应该检查相似测试场景的 map topology、actor 交互、交通状态和风险边界。

## 深度分析

### 策略多样性到底如何

按照参考论文的 behavioral diagnostics，我们的策略多样性可以概括为：

```text
高于 ReCogDrive baseline，但远低于 Curious-VLA；多数展开只是局部扰动，不是行为级多模态。
```

证据有三层：

1. 平均值层面：`mean-pFDE=0.390m`，只比论文 ReCogDrive 的 `0.325m` 高 `19.9%`，但距离 Curious-VLA 的 `1.415m` 很远。
2. 分布层面：`mean-pFDE` 中位数只有 `0.319m`，P95 也只有 `0.888m`。多数场景采样 8 次后终点仍然聚在很近范围内。
3. reward 层面：`std-PDMS@8` 中位数 `0.00163`，`sample_range` 中位数 `0.00498`。也就是说，候选轨迹即使几何上有轻微扰动，PDM reward 也几乎没有区分度。

论文强调 Curious-VLA 的动机是避免 VLA 在 IL/RL 后进入 narrow policy。传统模型容易围绕单一专家模式或单一高 reward 模式收缩，导致 GRPO 这类组内相对优势方法缺少有效 reward difference。我们的结果正好呈现这个形态：策略很会输出高 PDM 轨迹，但同组 reward variance 太低。

### 为什么 quality 差但 PDMS 高

`minFDE=1.342m` 明显劣于论文 ReCogDrive 和 Curious-VLA，但 `mean-PDMS=94.75` 更高。这不是矛盾，而是目标函数差异。

NAVSIM PDM 奖励的是：

- 是否碰撞；
- 是否在可行驶区域；
- 是否有足够进展；
- 是否保持方向和 comfort；
- 是否满足 TTC 等安全约束。

它不奖励“必须贴近 logged human 的唯一轨迹”。训练到 Stage3 后，模型可能学到一种 PDM 更稳定的保守/中线/低风险风格，而不是覆盖人类所有可能驾驶模式。因此：

- quality 指标提示我们 GT 覆盖不足；
- PDM 指标提示这些偏离 GT 的轨迹很多仍然可行；
- diversity 指标提示偏离不是因为覆盖了更多人类模式，而是收缩到 PDM 友好的局部模式。

### 为什么 high train 会对应 low test

`train_high_test_low` 的存在说明 hidden-neighbor 层面存在泛化裂缝。400 个可映射随机样本在 train 侧 `mean-PDMS@8=0.9879`，但 linked navtest 平均只有 `0.6307`，zero ratio `0.2595`。

这不等于“同一个场景 train 高 test 低”。它表示：某个 train token 的 hidden-neighbor 测试 family 低分。可能原因包括：

- hidden descriptor 没包含完整 map topology、actor 速度/距离、交通灯状态，导致“看起来近”的场景在 PDM 风险上并不完全等价；
- 训练场景是该 family 的容易版本，测试场景是更靠边界的困难版本；
- 当前策略学到了 train family 的主模式，但没有学到测试 family 需要的风险边界；
- PDM 高分模式在 train 中稳定，在 test 中因地图/actor 微差触发 DAC/NC/TTC。

从 failure mix 看，`train_high_test_low` 里 zero DAC 和 zero NC 合计接近 21%，再加 low TTC、low DDC，说明这些不是单纯 EP 不足，而是安全边界问题。

### 为什么 low train 也会对应 high test

`train_low_test_high` 有 201 个，训练侧 `mean-PDMS@8=0.7344`，但 linked navtest `0.9848`。这说明 train/test mapping 不是单调真值标签。

可能原因：

- hidden-neighbor 反向聚合是一对多 family 关系，不是 pairwise exact match；
- train token 自身可能有局部地图/actor 风险，linked 测试 family 反而简单；
- train 侧 K=8 采样刚好落入低分模式，但测试 family 的 SOTA 单次轨迹更稳；
- PDM 对单场景边界非常敏感，相似度高不保证风险边界相同。

这也是为什么本文把 hidden mapping 作为“family-level diagnostic”，而不是把它当成 strict generalization label。

### 对 best-of-N 的含义

Full NAVTEST best-of-6 从 `0.910274` 提到 `0.923203`，说明 best-of-N 确实有用。但 10k 训练集诊断说明这个提升上限受策略分布限制：

- 若大多数 token 是 `stable_high`，best-of 没什么可选；
- 若低分 token 是 `stable_low`，best-of 也选不出高分；
- 只有 `knife_edge` 或大 range token 才能显著受益；
- 当前这类 token 占比很小。

所以后续如果想继续提高 best-of-N，关键不是把 N 无限增大，而是让候选分布中出现更多“高质量且互异”的可行轨迹模式。换句话说，需要提升 feasible diversity，而不是单纯 stochastic noise。

## 建议

1. 继续保留 full NAVTEST best-of-6 作为 benchmark-facing 指标，但不要用它判断策略多样性。策略多样性应继续用 `mean-pADE/mean-pFDE`、`std-PDMS@K`、`sample_range`、`zero/success rate`。
2. 对 `train_high_test_low` 做二次诊断。优先补充 map topology、actor 距离/速度、交通灯状态等特征，确认 linked navtest zero 是相似度误配还是真实泛化失败。
3. 用 `knife_edge_zero_to_success` 和 `large_range` token 训练 selector/value/risk head。这些样本存在同场景好坏候选，最适合学习候选选择。
4. 对 `stable_low` 不要指望 best-of。它们需要改变生成分布本身，例如 replay、FTE-style 多可行轨迹扩展、risk-aware objective 或 targeted GRPO。
5. 对 `stable_high_low_diversity` 不应继续只优化 PDM。它们已经高分但多样性极低，继续 PDM-only 训练可能加剧 narrow policy。
6. 若目标是接近论文 Curious-VLA，需要同时提升 diversity 和 quality，而不是只提高 mean-PDMS。当前的主要缺口是 feasible behavior modes 不够，且对 GT family 覆盖不足。

## 最终判断

当前 Stage3 SOTA 的策略分布可以总结为：

```text
它已经学会稳定产生高 PDM 轨迹，但采样分布偏窄，组内 reward 方差很低，且高 PDM 模式与 human GT 存在明显偏移。
```

从工程上看，这个模型是强单模态/弱多模态的高分策略。它不是不会开，而是太稳定地开成少数 PDM 友好的模式。Best-of-N 可以利用少量 knife-edge 和 large-range 场景带来提升，但无法从根本上解决策略多样性不足。要接近论文 Curious-VLA 的效果，下一步需要增加高质量可行模式的覆盖，而不是只扩大采样次数。
