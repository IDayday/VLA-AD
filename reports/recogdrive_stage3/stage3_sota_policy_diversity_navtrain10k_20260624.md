# Stage3 SOTA 策略多样性诊断：随机 10k Navtrain

日期：2026-06-24

本文基于参考论文 arXiv:2603.06049 的 Behavioral Diagnostics 设置，对当前 Stage3 SOTA checkpoint 做策略多样性分析。核心目标不是重新报告单次 NAVTEST PDMS，而是回答：同一个场景下模型采样多条轨迹时，策略分布是否足够展开，是否存在 narrow policy。

参考论文链接：https://arxiv.org/html/2603.06049v1

## 评估设置

本次结果使用随机抽取的 10k 个训练集场景 token，而不是前缀子集。

| 项目 | 设置 |
|---|---|
| Checkpoint | `/mnt/project/VLA-AD/outputs/stage3_grpo_core_pareto_s16_lr1e4_b2acc4_20e_local8_20260616T020136Z/direct_local_4gpu_sharded_conc4_watch_from5400_v1/checkpoint_archive/step-step_21600.ckpt` |
| Split | `navtrain` |
| Token 池 | `navtrain.yaml` 与 `metric_cache_train_full` 交集 |
| Token 池大小 | `103288` |
| 随机样本数 | `10000` |
| 随机种子 | `260306049` |
| 每 token 采样数 | `k=8` |
| 采样方式 | ReCogDrive diffusion head 随机采样；每个 token 复用一次 VLM context |
| PDM 评分 | exact PDM |
| 有效样本 | `10000/10000` |

随机 token 文件：

```text
/mnt/project/VLA-AD/outputs/stage3_policy_diversity_k8_navtrain10k_random_step21600_20260624T0630Z/navtrain_random10000_seed260306049_tokens.txt
```

聚合结果：

```text
/mnt/project/VLA-AD/outputs/stage3_policy_diversity_k8_navtrain10k_random_step21600_thread_20260624T0640Z/local_sharded_aggregated.csv
```

注意：本次 `mean-PDMS` 是论文诊断里的 `mean-PDMS@8`，不是单次 NAVTEST PDMS。它是在随机 10k 个训练 token 上，每个 token 采 8 条候选轨迹，分别打 PDM 后取平均。它不能和 full-navtest 单样本 `0.910274` 或 best-of-6 NAVTEST `0.923203` 直接等同。

## 指标定义

对每个场景 token，模型采样 `k=8` 条未来轨迹：

```text
Y = {y_1, y_2, ..., y_8}
```

每条轨迹包含 4 秒 horizon 的 8 个 future pose，每个 pose 为 `(x, y, heading)`。几何指标只使用 `(x, y)` 位移；PDM 指标使用完整轨迹并由 NAVSIM exact PDM scoring 计算。

### Quality: minADE / minFDE

Quality 衡量的是：8 条候选轨迹里，是否至少有一条接近 logged human GT。

对第 `i` 条候选轨迹：

```text
ADE_i = mean_t || y_i,t - y*_t ||_2
FDE_i = || y_i,T - y*_T ||_2
```

然后取：

```text
minADE = min_i ADE_i
minFDE = min_i FDE_i
```

该指标越小越好。它不是安全性指标，而是 imitation-style 的 GT 贴近度指标。一个模型可以 PDMS 很高，但 minADE/minFDE 较差，因为它选择了与 logged human 不同但仍然可行的驾驶方式。

### Diversity: mean-pADE / mean-pFDE

Diversity 衡量的是：8 条候选轨迹彼此之间是否真的展开。

对每一对候选轨迹 `(i, j)`：

```text
pADE_ij = mean_t || y_i,t - y_j,t ||_2
pFDE_ij = || y_i,T - y_j,T ||_2
```

对全部 `C(8, 2)=28` 对取平均：

```text
mean-pADE = mean_{i<j} pADE_ij
mean-pFDE = mean_{i<j} pFDE_ij
```

该指标越大越好。`mean-pFDE` 尤其直观：它近似反映 8 条候选轨迹终点能分散多远。若 `mean-pFDE` 很小，说明采样多次也只是同一局部动作附近的小扰动。

### Perf: mean-PDMS

Perf 衡量的是：8 条候选轨迹整体是否仍然安全、可行、进展好。

```text
mean-PDMS = mean_i PDM(y_i)
```

该指标越大越好。它不是 best-of-N；best-of-N 会取 8 条里的最高分，而这里取平均。论文用它判断“多样性是否以牺牲驾驶质量为代价”。

## 主要结果

| 方法 | Quality minADE/FDE ↓ | Diversity mean-pADE/pFDE ↑ | Perf mean-PDMS ↑ |
|---|---:|---:|---:|
| 论文 ReCogDrive | 0.295 / 0.621 | 0.148 / 0.325 | 90.95 |
| 论文 Curious-VLA + FTE + RL | 0.269 / 0.547 | 0.641 / 1.415 | 91.55 |
| 我们 Stage3 SOTA, random 10k navtrain | 0.743 / 1.342 | 0.199 / 0.390 | 94.75 |

完整均值如下：

| 指标 | 数值 |
|---|---:|
| `quality_min_ade` | 0.742700 |
| `quality_min_fde` | 1.342238 |
| `diversity_mean_pade` | 0.199117 |
| `diversity_mean_pfde` | 0.389558 |
| `perf_mean_pdms` | 0.947543 |
| `perf_best_pdms` | 0.955026 |
| `perf_min_pdms` | 0.939057 |
| `perf_std_pdms` | 0.005777 |
| `mean_ade` | 0.875372 |
| `mean_fde` | 1.746361 |

## 分布特征

### Diversity 分布

| 指标 | P5 | P25 | P50 | P75 | P95 |
|---|---:|---:|---:|---:|---:|
| `mean-pADE` | 0.102844 | 0.146562 | 0.183843 | 0.236518 | 0.348761 |
| `mean-pFDE` | 0.120710 | 0.218446 | 0.318675 | 0.499062 | 0.888276 |

解释：

- `mean-pFDE` 中位数只有 `0.319m`，说明多数场景下 8 条轨迹终点仍然聚在很近的位置。
- `mean-pFDE` 的 P95 为 `0.888m`，只有少数场景产生接近 1m 级别的终点差异。
- 和 Curious-VLA 的 `mean-pFDE=1.415m` 相比，我们的大多数样本远没有达到行为级多模态。

阈值统计：

| 阈值 | 达标比例 |
|---|---:|
| `mean-pADE >= 0.148`，论文 ReCogDrive 水平 | 73.88% |
| `mean-pFDE >= 0.325`，论文 ReCogDrive 水平 | 48.67% |
| `mean-pADE >= 0.641`，论文 Curious-VLA 水平 | 0.02% |
| `mean-pFDE >= 1.415`，论文 Curious-VLA 水平 | 0.20% |

这说明我们的策略分布比论文 ReCogDrive baseline 更宽，但远远没有达到 Curious-VLA 的展开程度。

### Quality 分布

| 指标 | P5 | P25 | P50 | P75 | P95 |
|---|---:|---:|---:|---:|---:|
| `minADE` | 0.296456 | 0.456995 | 0.658323 | 0.926776 | 1.505076 |
| `minFDE` | 0.107381 | 0.432167 | 1.052098 | 1.967736 | 3.551905 |

解释：

- `minFDE` 中位数为 `1.052m`，说明即便采 8 条，最接近 GT 的终点也常常离 logged human 轨迹超过 1m。
- `minFDE >= 0.621` 的样本占 66.35%，也就是说大多数样本达不到论文 ReCogDrive 的 GT 贴近水平。
- 这不等价于“不安全”。它更像是我们的模型在 PDM 优化后偏向某种高 PDMS 驾驶模式，而不是模仿 logged human 的那条具体轨迹。

### Perf 分布

| 指标 | P5 | P25 | P50 | P75 | P95 |
|---|---:|---:|---:|---:|---:|
| `mean-PDMS` | 0.819016 | 0.928847 | 0.995641 | 1.000000 | 1.000000 |
| `std-PDMS@8` | 0.000000 | 0.000000 | 0.001626 | 0.004944 | 0.010246 |

阈值统计：

| 阈值 | 样本比例 |
|---|---:|
| `mean-PDMS <= 0.8` | 3.99% |
| `mean-PDMS <= 0.9` | 16.20% |
| `mean-PDMS >= 0.95` | 66.79% |
| `mean-PDMS` 中位数 | 0.995641 |

解释：

- 大多数 token 上，8 条候选轨迹的平均 PDM 都很高。
- `std-PDMS@8` 极低，中位数只有 `0.001626`，说明同一 token 下 8 条候选轨迹的 PDM 分数差异很小。
- 这与 narrow policy 的风险相关：如果同组 rollout 的 reward 差异太小，GRPO/AWAC 类训练很难从组内样本中获得强 advantage 信号。

## 和论文结论的关系

参考论文认为，传统 VLA 在 IL 后容易形成 narrow policy：模型学会输出单一专家模式，采样多次也缺少行为级差异。随后即使使用 RL，组内样本 reward 差异不足，GRPO 之类的 critic-free 方法容易出现 advantage collapse，训练很快饱和。

论文提出 Curious-VLA 的关键动机是：

1. 用 FTE 扩充可行轨迹模式，而不是只学习 logged human 单轨迹。
2. 用 step-wise normalization 让模型学会多模式轨迹分布。
3. 用 ADAS / SDR 在 RL 阶段维持 reward diversity，让训练不只停留在低方差局部策略附近。

我们的 10k 结果与这个叙事高度相关：

- 我们的 `mean-PDMS=94.75` 很高，说明当前 Stage3 SOTA 已经学到非常稳的高 PDM 行为。
- 但 `mean-pFDE=0.390m` 仍然很低，说明采样分布主要是同一高分策略附近的小扰动。
- `std-PDMS@8=0.00578` 很低，说明同一 token 下 8 条候选之间 reward 差异也很小。
- `minFDE=1.342m` 明显高于论文 ReCogDrive / Curious-VLA，说明我们的高 PDM 策略和 logged human GT 存在明显偏移，并且 8 次采样也没有覆盖到 GT 附近。

因此，我们的问题不是简单的“模型不会开”。更准确的判断是：

```text
模型会稳定地产生高 PDM 轨迹，但采样分布偏窄，且偏向 PDM 最优的局部模式，而不是覆盖多种人类可行策略。
```

这解释了为什么 best-of-N 可以提升 NAVTEST PDMS，但提升幅度有限：候选之间有差异，所以 best-of-N 有收益；但差异主要是局部扰动，不是大范围行为模式，所以收益不会像真正多模态策略那样大。

## 对比结论

### 相对论文 ReCogDrive

我们的多样性高于论文 ReCogDrive baseline：

| 指标 | 我们 | 论文 ReCogDrive | 相对比例 |
|---|---:|---:|---:|
| `mean-pADE` | 0.199 | 0.148 | 1.35x |
| `mean-pFDE` | 0.390 | 0.325 | 1.20x |
| `mean-PDMS` | 94.75 | 90.95 | 1.04x |

这说明 Stage3 训练确实让采样分布比原始 ReCogDrive 更宽，并且没有牺牲 PDM 性能。

### 相对 Curious-VLA

和 Curious-VLA 相比，我们的策略多样性仍明显不足：

| 指标 | 我们 | Curious-VLA + FTE + RL | 相对比例 |
|---|---:|---:|---:|
| `mean-pADE` | 0.199 | 0.641 | 0.31x |
| `mean-pFDE` | 0.390 | 1.415 | 0.28x |
| `mean-PDMS` | 94.75 | 91.55 | 1.04x |

这说明我们的模型更像“高 PDM、低方差、局部随机”的策略，而不是 Curious-VLA 的“高质量、多模式、广探索”的策略。

## 风险与解释边界

1. 论文没有公开完整 token 列表。本文使用随机 10k navtrain token，并固定 seed，保证本地可复现，但不是论文精确 token 复刻。
2. `mean-PDMS@8` 不等于单样本 NAVTEST PDMS。它是在训练 split 上对 8 条候选取平均。
3. `minADE/minFDE` 与 PDMS 优化方向不同。我们的 Quality 低，不代表轨迹不安全；它说明轨迹不贴近 logged human。
4. 本次 PDM 并发使用 thread backend。第一次使用 process backend 时，multiprocessing worker 在首个 token 处出现 `ConnectionRefusedError`，因此改为 thread backend。该改动只影响执行稳定性，不改变 token、checkpoint、采样种子或指标定义。

## 后续建议

1. 训练目标上显式增加 diversity-preserving 项，而不是只追求 PDM 均值。
2. 针对低 `mean-pFDE` 场景构建 hard subset，观察是否集中在直行、跟车、路口让行等特定类型。
3. 若继续做 RL，应监控组内 reward std。当前 `std-PDMS@8` 很低，说明 GRPO 可能缺少有效 advantage 信号。
4. 可以尝试论文式 FTE 思路：从 expert / PDM planner / perturbation 中生成多条可行轨迹，先做多模式蒸馏，再进入 RL。
5. best-of-N 评估仍有意义，但如果目标是接近 Curious-VLA，需要提升候选间行为级差异，而不是只提高局部噪声。

## 一句话结论

当前 Stage3 SOTA 已经具备很强的 PDM 稳定性，并且比论文 ReCogDrive baseline 有更高采样多样性；但它离 Curious-VLA 的行为级多模态仍有明显差距。主要瓶颈不是平均驾驶质量，而是采样分布太窄、组内 reward 方差太小、无法覆盖多种可行驾驶策略。
