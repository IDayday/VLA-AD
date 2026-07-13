# ReCogDrive Stage2 支持对齐策略多样性评价

## 1. 结论摘要

本报告提出并实测 **Scene-Normalized Support-Aligned Diversity v2（SNSAD v2）**。它只要求算法输出轨迹样本，不读取训练目标、Planning Token、FS-Norm 表示或模型内部状态，因此同时适用于：

- 确定性单轨迹算法；
- 只学习单一 GT、但推理时可随机采样的 diffusion 算法；
- 多假设或显式分布策略。

在固定的 1,024 个 navtrain 场景、每场景 32 次 5-step DDIM 采样下，当前 A5 top3 相比官方 ReCogDrive Stage2 的主要结论是：

1. A5 明显提高了支持邻域精度和 GT 精度；
2. A5 没有学习到比原版更宽的策略分布；
3. A5 从 epoch 155 到 174 持续变准，但分布继续收缩；
4. epoch 155 是 NAVTEST top3 中多样性最好者，epoch 174 是支持精度和 GT 误差最好者；
5. 当前结果证明 A5 是更好的单峰/窄分布拟合器，尚不能证明它实现了预期的支持集分布覆盖。

该评估使用训练支持集中的场景，只能验证支持集拟合机制，不能作为未见场景上的多样性泛化结论。

## 2. 为什么不使用原有 pairwise ADE

原有 policy-diversity evaluator 统计样本之间的 pairwise ADE/FDE、相对 GT 的 min ADE/FDE 和候选 PDMS。它存在三个问题：

1. 随机发散可以获得较高 pairwise ADE；
2. 天然多模态场景的绝对宽度大于单模态场景，跨场景直接平均不公平；
3. 它不读取 v3 正支持，无法区分“覆盖有效策略”和“离开支持流形”。

因此新指标必须同时测量：支持邻域精度、支持覆盖、有效模式数量和相对场景固有宽度。

## 3. 参考集与评估单位

对每个场景 `s`，定义：

- 正支持轨迹 `S_s = {z_i}`；
- 算法生成轨迹 `Y_s = {y_j}`；
- 每个场景先独立计算指标，最后对场景做等权宏平均。

当前部署的 v3 archive 是 legacy array schema：`candidates[N,8,3] + support_indices`。评估只读取 `support_indices` 指向的正支持；未选 candidate 和 hard negative 不作为正参考。实现同时兼容 dataclass-style `support_set` schema。

本次 1,024 场景中：

- selected support 数量均值为 10.274，中位数为 12；
- 仅用于诊断的 complete-linkage 模式数均值为 4.861；
- 模式阈值从 0.3 到 0.6 会显著改变离散模式数，因此离散聚类不再进入主指标。

## 4. 固定轨迹距离

距离在原始轨迹空间计算，因而不依赖被评估模型是否使用 FS-Norm。对两条轨迹定义五个分量：

```text
endpoint XY / 3.0 m                 weight 0.30
mean XY / 1.5 m                    weight 0.30
mean |delta progress| / 1.5 m      weight 0.15
mean |delta lateral| / 0.8 m       weight 0.15
mean wrapped heading / 0.35 rad    weight 0.10
```

最终距离是归一化分量的加权和。航向差使用 `atan2(sin(delta), cos(delta))`。尺度来自 v3 支持选择使用的行为差异尺度，评估时固定，不针对 checkpoint 调参。

## 5. SNSAD v2 指标

### 5.1 密度均衡支持权重

支持集中可能存在多个近重复轨迹。对支持间距离矩阵 `D`，定义：

```text
K_ij = exp(-0.5 * (D_ij / h_density)^2)
w_i  = (1 / sum_j K_ij) / sum_l (1 / sum_j K_lj)
```

默认 `h_density=0.4`。同一局部簇中有更多近重复轨迹不会获得更多总投票权。

### 5.2 Support Precision

在带宽 `h` 下：

```text
P_h = mean_j max_i exp(-0.5 * (d(y_j, z_i) / h)^2)
```

它惩罚离开正支持集的随机发散。主结果使用 `h={0.25,0.5,0.75,1.0}` 的平均值 `P_AUC`。

### 5.3 Density-Corrected Support Recall

```text
R_h = sum_i w_i * max_j exp(-0.5 * (d(y_j, z_i) / h)^2)
```

`R_AUC` 是四个带宽上的平均。它从支持到预测计算，回答生成样本是否覆盖参考可行策略，同时避免密集近重复区域主导结果。

### 5.4 Kernel Effective Mode Coverage

硬聚类模式数对阈值过于敏感。v2 改用 kernel participation ratio。给定样本权重 `q`：

```text
N_eff(X, h) = 1 / sum_ij q_i q_j exp(-(d(x_i, x_j) / h)^2)
```

策略样本使用经验均匀权重；参考支持使用密度均衡权重。定义：

```text
KEMR_h = min(N_eff(policy, h) / N_eff(support, h), 1)
```

`KEMR_AUC` 是四个带宽上的平均。它不要求模型训练时使用支持集，也不要求离散模式标签。

### 5.5 Relative Dispersion Calibration

分别计算策略和密度均衡支持的 pairwise trajectory distance：

```text
rho_s = (D_policy + 0.05) / (D_support + 0.05)
RDC_s = exp(-abs(log(rho_s)))
```

- `rho < 1`：相对场景固有宽度发生收缩；
- `rho ~= 1`：宽度匹配；
- `rho > 1`：过度发散。

### 5.6 汇总分

```text
SNSAD_s = (P_AUC * R_AUC * KEMR_AUC * RDC)^(1/4)
```

SNSAD 是二级汇总分。论文主结果必须同时报告四个分量，不能只报告 SNSAD。

### 5.7 Oracle-Normalized Gain

为直接量化相对单 GT delta policy 的增益，增加校准指标：

```text
ONG(Q)_s = (Q_model_s - Q_GT-repeat_s)
           / (Q_support-oracle_s - Q_GT-repeat_s)
```

只在 oracle 与 GT-repeat 的场景差值大于 0.05 时计算。报告 raw mean、median、`[0,1]` clipped mean 和有效场景数。ONG 不替代原始分量，只用于解释“完成了多少场景可用多样性空间”。

## 6. 单 GT 算法如何使用

- 真正确定性的算法：重复其唯一输出 K 次，得到合法的 delta distribution；
- 单 GT 训练但具有随机采样器的算法：按原生采样器生成 K 个样本；
- 当场景只有一条参考轨迹时，确定性正确轨迹的 precision、recall、KEMR 和 RDC 都等于 1；若一个行为簇仍含有局部轨迹变化，RDC 会继续测量这部分局部宽度；
- 当场景有多个有效模式时，单点策略保持较高 precision，但 recall、KEMR 和相对宽度下降；
- 不向确定性算法人为注入噪声。

这保证指标不会因为场景天然缺乏候选策略而惩罚单轨迹算法，也不会把随机噪声误认为有效多样性。

## 7. 实验协议

### 7.1 数据

- 支持集：`support_v3`，共 103,288 个 archive；
- 固定场景清单：从 hidden-cache 与 support archive 交集按 SHA-256(seed, token) 选取 1,024 个场景；
- 清单：`outputs/support_aligned_diversity_eval_20260711/navtrain_v3_random1024_seed260711.tsv`；
- 所有模型使用完全相同的 token、`seed=260711`、K 和采样批大小。

### 7.2 推理

- VLM 条件来自官方 Stage1 hidden cache；
- sampler：5-step DDIM，保留模型原生随机采样；
- 每场景 `K=32`；
- `sample_batch_size=4`；
- FP32；
- 32 shards，8 张 A800，每卡 4 shards。

### 7.3 权重

```text
Official Stage2:
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/
ReCogDrive_Diffusion_Planner_2B_IL.ckpt

A5 epoch 155/157/174:
outputs/stage2_pta_fs_dit_a5_full103k_standardv2_clean_20260710T091344Z/
```

官方 checkpoint 加载 347 个 planner key；A5 加载 785 个 planner key。两者均无 shape mismatch，输出均为 finite。

## 8. 主结果

所有数值均为 1,024 个场景的宏平均。

| 方法 | Precision AUC | Recall AUC | KEMR AUC | Width ratio | RDC | SNSAD | Pairwise ADE (m) | Mean GT ADE (m) | Best GT ADE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GT-repeat control | 0.994675 | 0.586014 | 0.490504 | 0.124850 | 0.124850 | 0.401714 | 0.000000 | 0.000000 | 0.000000 |
| Official Stage2 | 0.905771 | 0.617563 | 0.504625 | 0.302735 | 0.216681 | **0.485336** | 0.142481 | 0.371585 | 0.224643 |
| A5 epoch 155 | 0.984085 | **0.639157** | 0.498695 | 0.221943 | 0.175484 | 0.464131 | **0.087649** | 0.170181 | 0.091723 |
| A5 epoch 157 | 0.984503 | 0.636728 | 0.496872 | 0.217130 | 0.171299 | 0.460440 | 0.079650 | 0.178139 | 0.096962 |
| A5 epoch 174 | **0.986757** | 0.632190 | 0.496728 | 0.207028 | 0.168388 | 0.456894 | 0.075287 | **0.163082** | **0.089146** |
| Support-oracle control | 1.000000 | 0.994112 | 0.945382 | 0.891299 | 0.889494 | 0.955277 | 1.163738 | 1.259282 | 0.028395 |

Support-oracle 的 mean GT ADE 较大是预期现象：有效支持策略不要求都接近唯一 GT；其 best GT ADE 仍接近 0。

## 9. 配对显著性

A5 epoch 155 相对 Official Stage2，5,000 次按场景 bootstrap：

| 指标 | 配对均值差 | 95% CI | A5 场景胜率 |
|---|---:|---:|---:|
| Precision AUC | +0.078314 | [+0.071530, +0.085661] | 93.6% |
| Recall AUC | +0.021593 | [+0.013694, +0.029873] | 41.4% |
| KEMR AUC | -0.005931 | [-0.007570, -0.003956] | 12.1% |
| Width ratio | -0.080791 | [-0.094911, -0.066192] | 10.1% |
| RDC | -0.041197 | [-0.046948, -0.035464] | 15.2% |
| SNSAD | -0.021204 | [-0.026311, -0.015958] | 23.0% |
| Pairwise ADE | -0.054832 m | [-0.062566, -0.046118] | 10.5% |
| Mean GT ADE | -0.201403 m | [-0.231237, -0.167266] | 9.2%* |

`*` GT ADE 越低越好，因此该行表示 A5 明显更准确。

## 10. 按场景固有多模态程度分层

离散模式数仅用于分层解释，不进入主分数。

| 场景层 | 场景数 | Official SNSAD | A5-155 SNSAD | Official width ratio | A5-155 width ratio |
|---|---:|---:|---:|---:|---:|
| 1 mode | 80 | 0.824689 | **0.895232** | 1.701592 | **1.286102** |
| 2-3 modes | 157 | **0.544906** | 0.528265 | **0.238733** | 0.184191 |
| 4+ modes | 787 | **0.438956** | 0.407515 | **0.173306** | 0.121301 |

A5 在单模态场景更好，但在具有多个参考模式的场景中更窄。这正是绝对 pairwise ADE 或单一 PDMS 无法揭示的差异。

## 11. Oracle-Normalized Gain

| 方法 | Recall ONG | KEMR ONG | RDC ONG | SNSAD ONG |
|---|---:|---:|---:|---:|
| Official Stage2 | 0.09393 | **0.03720** | **0.15736** | **0.18001** |
| A5 epoch 155 | **0.13026** | 0.01965 | 0.11095 | 0.13773 |
| A5 epoch 157 | 0.13153 | 0.01625 | 0.10488 | 0.13242 |
| A5 epoch 174 | 0.11525 | 0.01553 | 0.10001 | 0.12378 |

A5 覆盖了更多支持邻域，但没有把概率质量有效分配到更多模式，且宽度增益较低。

## 12. 稳定性检查

### 12.1 采样预算

| K | Precision 差 A5-Official | Recall 差 | KEMR 差 | Width ratio 差 | SNSAD 差 |
|---:|---:|---:|---:|---:|---:|
| 8 | +0.078586 | +0.013280 | -0.004463 | -0.076540 | -0.022432 |
| 16 | +0.078366 | +0.013908 | -0.005905 | -0.076991 | -0.023734 |
| 32 | +0.078314 | +0.021593 | -0.005931 | -0.080791 | -0.021204 |

结论在 K=8/16/32 下方向一致。

### 12.2 密度带宽

将 `h_density` 从 0.3、0.4 改到 0.5，A5-Official 的 SNSAD 差分别为 `-0.020765/-0.021204/-0.021517`，width ratio 差分别为 `-0.079835/-0.080791/-0.081920`。结论不依赖默认 `h_density=0.4`。

## 13. 对当前 Stage2 设计的判断

当前 A5 学到了：

- 更高的支持邻域 precision；
- 更低的 GT 误差；
- 略高的 density-corrected support recall。

当前 A5 没有学到：

- 更高的 kernel effective mode coverage；
- 与 v3 支持集接近的相对分布宽度；
- 随训练后期保持或扩大策略多样性。

因此不能把当前 Stage2 结果表述为“学习了宽且支持对齐的策略分布”。更准确的表述是“支持对齐和轨迹精度显著改善，但生成分布仍然收缩”。在改变训练目标前，应先检查 selected-target 采样频率、每场景 support exposure、diffusion 输出方差和 sampler 对模式覆盖的影响。

## 14. 局限与正式论文协议

1. 当前 1,024 场景来自训练支持 archive，只能测机制拟合；
2. 正式论文需要在未参与 Stage2 训练的场景上独立构建冻结支持集；
3. v3 支持是可行策略集合，不是真实行为概率分布，因此主指标使用密度均衡 set coverage，不声称校准真实概率；
4. 模式阈值只用于分层，不进入 SNSAD；
5. SNSAD 是汇总指标，不能替代四个分量和 precision-recall-width 图；
6. 本评估未重新计算每个随机样本的 PDMS；NAVTEST 仍作为独立质量/安全指标报告。

建议正式协议至少包含：held-out support archive、K=32 或 64、3 组采样 seed、按场景配对 bootstrap、单模态/多模态分层、GT-repeat 与 support-oracle 校准。

## 15. 代码与产物

实现：

```text
navsim/agents/recogdrive/support_aligned_diversity.py
scripts/evaluation/evaluate_recogdrive_support_aligned_diversity.py
scripts/evaluation/recompute_recogdrive_support_aligned_diversity.py
scripts/evaluation/build_support_aligned_diversity_controls.py
scripts/evaluation/aggregate_recogdrive_support_aligned_diversity.py
scripts/evaluation/run_recogdrive_support_aligned_diversity_sharded.sh
tests/test_support_aligned_diversity.py
```

最终结果：

```text
outputs/support_aligned_diversity_eval_20260711/final_comparison_v2/summary.json
outputs/support_aligned_diversity_eval_20260711/final_comparison_v2/*_scene_metrics.csv
```

核心复现命令：

```bash
pytest -q tests/test_support_aligned_diversity.py

python scripts/evaluation/aggregate_recogdrive_support_aligned_diversity.py \
  --model original_stage2=outputs/support_aligned_diversity_eval_20260711/original_stage2_k32_n1024_v2 \
  --model a5_epoch155=outputs/support_aligned_diversity_eval_20260711/a5_epoch155_k32_n1024_v2 \
  --model a5_epoch157=outputs/support_aligned_diversity_eval_20260711/a5_epoch157_k32_n1024_v2 \
  --model a5_epoch174=outputs/support_aligned_diversity_eval_20260711/a5_epoch174_k32_n1024_v2 \
  --model gt_repeat=outputs/support_aligned_diversity_eval_20260711/controls_k32_n1024/gt_repeat \
  --model support_oracle=outputs/support_aligned_diversity_eval_20260711/controls_k32_n1024/support_oracle \
  --output-dir outputs/support_aligned_diversity_eval_20260711/final_comparison_v2 \
  --bootstrap-samples 5000 \
  --seed 260711
```
