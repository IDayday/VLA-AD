# PSI、Score、Pareto与GT-only的等预算IL续训实验

状态：实际训练、采样、NAVSIM评分及审计完成。结论仅针对本次预算与共享候选库；不代表完整训练性能，也不是 downstream GRPO 更新实验。

## 核心结论：本轮没有证明PSI的必要性

**NOT_SUPPORTED：在本轮冻结候选库、3072训练场景、512更新、两seed的等预算对照中，PSI没有优于Score/Pareto，也没有比GT-only提供更好的质量—安全组合。** 这不是说PSI没有学到候选：它对共同PSI候选的拟合和输出覆盖最好，但GT保留更差、分布中心移动更大，额外位移未转化为更好的held-out安全质量。

这是一项新的受控重训结果，不是历史PSI checkpoint的原数据/原预算复现。历史完整AWAC raw库缺失；因此不能据此否定那个历史checkpoint的实测成绩，也不能把此次负结果藏起来以支持原论文叙事。

### 1. 三种对照相对PSI到底发生了什么

以下为5000场景、每场景64条普通eval轨迹、两训练seed先在scene内平均的结果；中心位移相对同sampler的初始IL，单位米。

| 权重 | Mean sampled PDMS | 联合可行率 | Pairwise ADE64 | 中心位移 |
| --- | ---: | ---: | ---: | ---: |
| 初始官方IL | 91.414 | 93.131% | 0.1455 | 0 |
| IL-SFT / GT-only续训 | 90.645 | 92.121% | 0.1422 | 0.1748 |
| Score续训 | 90.710 | 87.647% | 0.2857 | 1.1915 |
| Pareto续训 | 90.657 | 87.565% | 0.2862 | 1.1937 |
| PSI续训 | 90.453 | 86.837% | 0.2855 | 1.2951 |

数据：`metrics/summary.csv`，`split=holdout, protocol=eval`；分母见`summary_denominators.csv`，每方法均5000，无缺失。

- **IL-SFT**保留了与初始IL接近的分布宽度和较小的中心位移；它没有学到非GT高分候选。该有限数据续训本身也使held-out平均PDMS下降，不能把所有变化都归因于多候选。
- **Score/Pareto**把分布中心移向候选，宽度约为初始IL的1.96倍，但联合可行率下降约5.5个百分点。二者训练选集在96.58%的场景完全相同，结果接近有明确的数据原因。
- **PSI**与Score/Pareto的最终普通eval宽度基本相同，没有额外的宽度收益；它主要多移动了约0.10m，并付出了额外安全与平均质量代价。

PSI减baseline的scene-paired差值与95% CI（3000次bootstrap）：

| Baseline | ΔPDMS | Δ联合可行率（百分点） | Δ中心位移（m） |
| --- | ---: | ---: | ---: |
| Score | -0.256 [-0.375, -0.139] | -0.810 [-0.987, -0.638] | +0.104 [0.099, 0.108] |
| Pareto | -0.204 [-0.328, -0.092] | -0.727 [-0.894, -0.553] | +0.101 [0.097, 0.106] |
| IL-SFT | -0.191 [-0.630, 0.269] | -5.284 [-5.900, -4.680] | +1.120 [1.093, 1.148] |

数据：`paired_comparisons.csv`，`left=psi, split=holdout, protocol=eval`。PSI相对IL-SFT的平均PDMS差区间跨0，不能称为明确的平均分下降；安全差与位移差则很清楚。PSI相对Score/Pareto的质量、安全劣势在log-cluster敏感性中仍存在。两seed各自的PSI平均PDMS和可行率也均低于Score/Pareto；这些CI仍条件于两次训练，并不充分覆盖训练随机性。

### 2. 确实学到了候选，但同时损害了GT保留

所有权重评价**完全相同的PSI non-GT目标**，不是各比各的目标：

| 权重 | 共同PSI目标epsilon MSE | 共同PSI目标Hit64@0.5m | 相同GT epsilon MSE | GT Hit64@0.5m |
| --- | ---: | ---: | ---: | ---: |
| 初始IL | 0.029261 | 16.90% | 0.003756 | 90.12% |
| IL-SFT | 0.031208 | 16.42% | 0.003970 | 88.29% |
| Score | 0.010881 | 49.55% | 0.009585 | 41.61% |
| Pareto | 0.010867 | 49.54% | 0.009589 | 41.52% |
| PSI | 0.010181 | 52.99% | 0.010999 | 38.05% |

共同PSI目标覆盖分母4728个含non-GT目标的holdout场景；MSE为固定1000探针中满足该条件的943场景、每目标16次共同noise/timestep。GT覆盖5000、MSE1000场景。目标权重固定且所有模型相同；表中Hit64为场景内目标加权命中率，再场景等权。数据：`teacher_scene_metrics.parquet`、`teacher_summary.csv`。

相对Score，PSI共同目标Hit64提高3.44个百分点，95% CI [2.73, 4.17]；epsilon MSE降低0.000700，CI [-0.000938, -0.000490]。但相同GT的Hit64下降3.56个百分点，CI [-4.08, -3.07]，GT MSE反而增加0.001414。数据：`teacher_paired.csv`，`teacher:psi:NON_GT`及`teacher:il_sft:GT`。

**SUPPORTED：PSI更强地吸收了它选出的候选。NOT_SUPPORTED：更强吸收自动意味着更有用、更安全的policy扩展。** 有些目标虽被监督过，64次采样仍未覆盖：固定512训练probe中，PSI实际被训练过的unique parent在0.5m阈值下零命中比例为两seed平均28.94%（parent计数）；场景等权为24.59%。Score对应22.04%/22.05%。这里各方法自己的目标不同，难度不可直接对齐，不能用它取代上面的共同目标对照。原生GRPO的0.5m零命中几乎饱和，1.0m补充与完整分母一并保存在`parent_absorption_summary.csv`；有限采样零命中不是“不可学习”。

### 3. 质量收益没有在held-out安全性上兑现

PSI相对初始IL，普通eval的EP由85.274升到89.958，但TTC由97.451降至91.847，NC由99.508降至97.717；联合可行率由93.131%降至86.837%。因此更大的进度和更高的少数样本上限，不能抵消总体风险增加。

PSI训练目标自身的NC、DAC、TTC加权均值均为100，EP为96.252；可行率未达100主要与DDC有关（`supervision_metric_components.csv`）。因此，不能简单归因于“PSI选入了很多TTC不安全的标签”：标签的安全性与学得policy输出的安全性确实发生了分离。是否来自轨迹间插值、条件泛化或周围扰动不稳，仍需额外实验区分。

固定512训练probe中，PSI平均PDMS从92.513升至95.686；held-out 5000却从91.414降至90.453。GT-only的训练probe也从92.513升至92.914，held-out降至90.645。这里比较的是各自场景内相对初始化的变化，不把train/holdout的原始绝对分数差当作过拟合证据。结果更符合**监督拟合与训练场景收益已出现，但其安全收益没有转移到当前held-out场景**，不符合“训练完全没起作用”的解释。尚不能仅靠这些统计确定是哪一类网络表征或损失项造成该转移失败。

PSI的选择实际上同时改变了几个因素：训练场景平均1.635个目标，57.29%场景只选1个；多目标场景内pairwise ADE为1.149m，Score/Pareto约0.672/0.675m。PSI对DDV2/DrivOR的监督权重约32.59%，Score约23.46%；归档A5 policy候选权重则为9.95%与24.62%。PSI每个unique目标实际平均获得13.04次监督，Score/Pareto约7.11次。因此这里验证的是整个选择机制，不能将结果单独归因于多样性、目标数量、来源、或某个未实现的policy-compatibility gate。

数据：`supervision_composition.csv`、`target_count_and_diversity.csv`、`selected_source_composition.csv`、`target_presentation_summary.csv`。PSI训练目标自身加权PDMS为98.438，Score为98.086；更高的离线监督分数没有保证更好的模型输出。

### 4. 真实GRPO采样：上限略高，均值与安全更差

同一组权重通过真实forward_grpo采样，每场景4组、每组16条；没有GRPO更新：

| 权重 | 组内Mean PDMS | 组内Min PDMS | 组内Max PDMS | 联合可行率 | Pairwise ADE64 | 中心位移 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 初始IL | 69.020 | 22.115 | 96.685 | 69.247% | 1.9937 | 0 |
| IL-SFT | 68.860 | 22.066 | 96.650 | 68.823% | 1.9905 | 0.1797 |
| Score | 67.717 | 20.251 | 97.837 | 64.118% | 2.0659 | 1.2065 |
| Pareto | 67.677 | 20.220 | 97.831 | 64.057% | 2.0663 | 1.2086 |
| PSI | 67.525 | 20.131 | 97.891 | 63.625% | 2.0675 | 1.3086 |

min/max为真实G16的组内最小/最大值，再平均组、场景和训练seed；不是跨全数据的极端值。数据：`summary.csv`的`native_grpo`与`group16_metrics.parquet`。

PSI相对Score的native mean PDMS差为-0.192，95% CI [-0.257, -0.125]；相对Pareto为-0.152 [-0.218, -0.081]；相对IL-SFT为-1.334 [-1.569, -1.102]。PSI的native CVaR25为33.220，初始IL为35.800，说明低分尾部也变差。其Hit8虽由初始IL的44.70%升至56.19%，却与Score的56.14%接近，不能据此证明PSI有独特优势。

这说明高分样本仍能出现，但组内更多样本处于较差或不安全区域；这种采样起点值得警惕。**UNTESTED：它是否最终导致GRPO优化退化。** 本轮没有advantage更新、optimizer step或下游学习曲线，不能把起点统计替代真实训练。

### 5. 最保守的论文表述

当前证据支持：“在本次受控IL续训中，多候选监督使policy明显靠近高分候选并扩大几何采样范围；PSI进一步提高了对其候选的吸收，但伴随更大的中心位移、更弱的GT保留和更差的held-out安全性。离线高质量和描述符多样性本身不足以保证有效的policy扩展。”

| 待验证主张 | 本轮状态 |
| --- | --- |
| PSI确实改变监督内容并增强共同候选的拟合/覆盖 | SUPPORTED |
| PSI比纯Score/Pareto带来更好的安全质量组合 | NOT_SUPPORTED |
| PSI比GT-only提供更好的安全质量权衡 | NOT_SUPPORTED |
| PSI带来额外的普通eval平均分布宽度收益 | NOT_SUPPORTED |
| 更强候选吸收伴随GT保留与安全上的代价 | SUPPORTED（本配置范围内） |
| PSI可改善后续GRPO实际学习效率或最终收益 | UNTESTED |
| 精确复现历史PSI原始完整训练 | MISSING（完整原raw库未恢复） |

**因此，本轮不足以把当前PSI写成“必要且优于简单高分筛选”的论文核心贡献。** 需要保留此次负结果，而不是换场景、阈值或最好checkpoint重新包装为正结果。


## 1. 这次真正控制了什么

- 初始化：同一个官方GT-IL checkpoint，SHA256 `4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443`。
- 训练3072场景，评估沿用原随机5000场景，两个集合的log交集为0。原5000场景覆盖991个log；排除后可用3268场景，按预定哈希取3072。这个可用性调整发生在protocol冻结和任何评分/训练之前。
- 四方法各2个seed（1701/2903），512次AdamW更新；batch128=microbatch16×accumulation8；每run65536次场景监督，约21.33个训练集遍历。相同seed下场景顺序、noise/timestep随机数完全对应。
- 全部训练同一34329219个原生action-head参数；VLM不更新。FP32；AdamW betas=(0.9,0.95)，weight decay=1e-4，clip=1；峰值LR5e-5、warmup96步、固定cosine降至1e-6。沿用历史优化器关键设置，缩短并预先冻结了schedule；不是历史60-epoch重放。
- 只改变目标选择。都使用原生trajectory normalization、uniform t∈[0,99]、epsilon prediction MSE。每场景每次按目标权重抽1条；没有额外GT retention loss。IL-SFT始终抽GT。
- 全量主结果使用step512；step128只作预先定义1000场景上的学习曲线，不按得分选最好checkpoint。
- 5000场景每个模型分别普通eval和真实forward_grpo采样：每种4个独立G16，共64条。官方IL原始缓存按hash复用；新checkpoint全部实采实评。输入仅观察特征，GT/teacher仅作为训练标签或离线评估对象。
- 这些是Navtrain内部、相对本轮微调log隔离的诊断场景；初始公开权重和历史teacher未必未见过它们。不得称为untouched Navtest。

冻结配置SHA256：`86be187b0a1c3b4a95257da0d35891cd78119817ca5f06e4000a69be70f6b605`。配置：`configs/psi_matched_sft/primary.yaml`。manifest：`outputs/psi_matched_sft/manifests/`。

## 2. PSI究竟是什么

本轮沿用可恢复历史PSI实现 `pareto_support.py` 与 `build_stage2_pareto_support_index.py` 的选择函数。它没有current-policy KNN、q-percentile、GRPO probability或diffusion-loss gate。

共同候选首先满足finite、NC=1、DAC=1、DDC≥0.95或不低于参考DDC-0.01、EP不低于参考EP-0.02。参考优先GT/精确IL来源；本次raw库没有确定性official IL条目，使用GT。**没有复用V6的valid_mask、teacher_eligible或SG-FPS选集。**

| 方法 | 共同合格集合内的选择 |
| --- | --- |
| IL-SFT | 只继续拟合GT，无候选监督 |
| Score | PDMS降序，最多3个unique目标 |
| Pareto | EP/TTC/Comfort非支配排序，逐层取、同层PDMS，最多3个 |
| PSI | 历史core-adjusted score + Pareto bonus，在best score-0.02的质量范围内，按标准化行为描述符最远点选择；最小距离0.75，最多3个，自适应停止 |

PSI score = PDMS + 0.3(core-reference core) - 0.5 slow violation - 0.2 tradeoff violation + 0.2 Pareto-front indicator。core=(5EP+5TTC+2Comfort)/12。score的0.02 band不是纯PDMS的2分band。

三种候选方法统一使用原PSI权重函数：best权重0.5、被选GT权重0.2、其他目标共享0.3；GT未被选则其0.2给best；无其他目标时0.3也给best；单目标权重1。不同选集会产生不同实际GT监督比例，这是selection bundle的组成部分，不能声称实际GT比例完全相同。目标描述符均值/方差只从3072训练场景计算。

**数据恢复限制**：历史PSI完整AWAC raw库已缺失，只有历史最终support index仍在。因此本次是原PSI选择机制在共享可恢复raw库上的前瞻受控对照，不能冒称重新训练了原始历史PSI数据。

共享库来自V6 archive的selection之前原始轨迹：GT、结构化扰动、8条A5策略候选，以及实际DDV2/DrivOR输出。此处`policy`来源是A5 checkpoint，并非当前official IL。所有候选重新以同一NAVSIM-v1 evaluator评分，跨来源完全重复合并保留标签。`raw_source_summary.csv`公开真实来源数量，`selected_targets.parquet`保存可恢复轨迹、archive hash和原索引。

## 3. 监督目标差异

| split | method | count | weighted_PDMS | weighted_feasible | GT_weight | target_pairwise_ADE | fallback |
| --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | il_sft | 1.0000 | 94.8587 | 0.9796 | 1.0000 | 0.0000 | 0.0000 |
| holdout | pareto | 2.9992 | 97.8858 | 0.9757 | 0.2706 | 0.6855 | 0.0000 |
| holdout | psi | 1.5826 | 98.4492 | 0.9808 | 0.2860 | 0.4758 | 0.0000 |
| holdout | score | 2.9992 | 98.0837 | 0.9807 | 0.2708 | 0.6808 | 0.0000 |
| train | il_sft | 1.0000 | 94.9078 | 0.9710 | 1.0000 | 0.0000 | 0.0000 |
| train | pareto | 2.9993 | 97.8906 | 0.9674 | 0.2824 | 0.6747 | 0.0000 |
| train | psi | 1.6354 | 98.4383 | 0.9728 | 0.2952 | 0.4906 | 0.0000 |
| train | score | 2.9993 | 98.0855 | 0.9724 | 0.2825 | 0.6717 | 0.0000 |

上表场景等权；PDMS为0–100 points，可行率与GT_weight为0–1。Score/Pareto训练集完全相同选集比例为96.5820%。因此它们接近可能来自实际目标退化，不能临时换Pareto目标制造差异。

## 4. Full-5000最终分布与质量

| method | protocol | mean_PDMS | feasible_% | pairwise_ADE64 | centroid_displacement | Spread_AUC | min_PDMS | max_PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| il_sft | eval | 90.6446 | 92.1213 | 0.1422 | 0.1748 | 0.1187 | 87.2114 | 93.3696 |
| official_il | eval | 91.4140 | 93.1312 | 0.1455 | 0.0000 | 0.1217 | 87.8746 | 93.7665 |
| pareto | eval | 90.6569 | 87.5647 | 0.2862 | 1.1937 | 0.2413 | 84.5477 | 94.7660 |
| psi | eval | 90.4532 | 86.8373 | 0.2855 | 1.2951 | 0.2406 | 84.2887 | 94.7331 |
| score | eval | 90.7095 | 87.6472 | 0.2857 | 1.1915 | 0.2408 | 84.6097 | 94.7955 |
| il_sft | native_grpo | 68.8597 | 68.8225 | 1.9905 | 0.1797 | 1.5931 | 22.0657 | 96.6504 |
| official_il | native_grpo | 69.0197 | 69.2466 | 1.9937 | 0.0000 | 1.5958 | 22.1146 | 96.6847 |
| pareto | native_grpo | 67.6769 | 64.0572 | 2.0663 | 1.2086 | 1.6567 | 20.2202 | 97.8308 |
| psi | native_grpo | 67.5254 | 63.6255 | 2.0675 | 1.3086 | 1.6577 | 20.1314 | 97.8912 |
| score | native_grpo | 67.7175 | 64.1180 | 2.0659 | 1.2065 | 1.6564 | 20.2506 | 97.8368 |

先在场景内平均64次采样，再对5000场景等权；最后平均两训练seed。Pairwise ADE64为64条两两XY-ADE；centroid displacement为同sampler下相对初始IL的平均轨迹中心ADE。min/max PDMS是每真实G16内min/max后平均，不是全数据极值。联合可行定义为NC/DAC/TTC/DDC全1。

沿用原诊断代码的Spread_AUC定义：每个G16内、每个未来时刻计算sqrt(var(x)+var(y))（样本方差ddof=1），对8个时刻取均值，再平均4组；它是离散等权spread摘要，不是额外拟合连续曲线的积分。单位为米。

`metrics/summary.csv`与`summary_denominators.csv`；原始场景/组表为`scene_metrics.parquet`与`group16_metrics.parquet`。

质量分量及组内尾部（同一5000场景、同一两seed平均；分量均为0–100）：

| method | protocol | EP | NC | DAC | TTC | DDC | Comfort | std_PDMS | CVaR25_PDMS | Hit8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| il_sft | eval | 84.8314 | 99.3019 | 96.9598 | 97.0803 | 98.3745 | 99.9791 | 2.2618 | 88.4082 | 0.2636 |
| official_il | eval | 85.2739 | 99.5077 | 97.6359 | 97.4513 | 98.3928 | 99.9825 | 2.1759 | 89.1044 | 0.2296 |
| pareto | eval | 89.7257 | 98.0355 | 96.9895 | 92.5223 | 98.0932 | 99.9981 | 3.7512 | 86.8441 | 0.5709 |
| psi | eval | 89.9581 | 97.7168 | 97.0383 | 91.8473 | 97.9943 | 99.9872 | 3.8348 | 86.5570 | 0.5695 |
| score | eval | 89.7499 | 98.0680 | 97.0022 | 92.5980 | 98.0946 | 99.9983 | 3.7382 | 86.9118 | 0.5715 |
| il_sft | native_grpo | 70.8234 | 95.2460 | 85.6773 | 83.3620 | 96.9589 | 68.7562 | 27.1363 | 35.8490 | 0.4378 |
| official_il | native_grpo | 70.6535 | 95.3542 | 85.8137 | 83.6309 | 97.0544 | 69.0716 | 27.1704 | 35.7999 | 0.4470 |
| pareto | native_grpo | 73.3911 | 92.8417 | 85.4802 | 78.7503 | 96.0646 | 66.9927 | 28.6051 | 33.4546 | 0.5613 |
| psi | native_grpo | 73.4881 | 92.4145 | 85.6084 | 78.3177 | 95.8464 | 66.7370 | 28.6871 | 33.2200 | 0.5619 |
| score | native_grpo | 73.4190 | 92.8822 | 85.4869 | 78.8070 | 96.0693 | 67.0025 | 28.5891 | 33.4997 | 0.5614 |

CVaR25为每个真实G16中最低4条PDMS的均值，再场景等权；Hit8为真实G16的固定连续8条子组中至少一条同时满足联合可行与PDMS≥初始IL普通eval场景均分+1的比例。阈值在所有方法/采样器中一致；超过100分的阈值保留在分母，不事后换阈值。

## 5. PSI相对三个微调baseline的场景配对差

| protocol | right | metric | n | mean_difference | median_difference | ci_low | ci_high | log_cluster_ci_low | log_cluster_ci_high | win_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eval | il_sft | mean_PDMS | 5000 | -0.1915 | 0.9751 | -0.6305 | 0.2693 | -0.6802 | 0.3043 | 0.5660 |
| eval | il_sft | feasible_rate | 5000 | -0.0528 | 0.0000 | -0.0590 | -0.0468 | -0.0589 | -0.0464 | 0.0556 |
| eval | il_sft | pairwise_ADE64 | 5000 | 0.1433 | 0.1179 | 0.1413 | 0.1455 | 0.1406 | 0.1461 | 1.0000 |
| eval | il_sft | centroid_displacement | 5000 | 1.1202 | 0.9224 | 1.0935 | 1.1476 | 1.0726 | 1.1704 | 0.9518 |
| eval | score | mean_PDMS | 5000 | -0.2564 | 0.0000 | -0.3751 | -0.1389 | -0.3834 | -0.1360 | 0.4338 |
| eval | score | feasible_rate | 5000 | -0.0081 | 0.0000 | -0.0099 | -0.0064 | -0.0098 | -0.0063 | 0.0856 |
| eval | score | pairwise_ADE64 | 5000 | -0.0001 | 0.0078 | -0.0009 | 0.0007 | -0.0015 | 0.0011 | 0.6780 |
| eval | score | centroid_displacement | 5000 | 0.1036 | 0.0935 | 0.0992 | 0.1084 | 0.0977 | 0.1096 | 0.8048 |
| eval | pareto | mean_PDMS | 5000 | -0.2038 | 0.0000 | -0.3282 | -0.0918 | -0.3285 | -0.0820 | 0.4342 |
| eval | pareto | feasible_rate | 5000 | -0.0073 | 0.0000 | -0.0089 | -0.0055 | -0.0091 | -0.0055 | 0.0864 |
| eval | pareto | pairwise_ADE64 | 5000 | -0.0007 | 0.0079 | -0.0015 | 0.0002 | -0.0021 | 0.0007 | 0.6754 |
| eval | pareto | centroid_displacement | 5000 | 0.1014 | 0.0900 | 0.0971 | 0.1060 | 0.0952 | 0.1075 | 0.7982 |
| native_grpo | il_sft | mean_PDMS | 5000 | -1.3344 | -0.1046 | -1.5693 | -1.1019 | -1.6235 | -1.0466 | 0.4884 |
| native_grpo | il_sft | feasible_rate | 5000 | -0.0520 | -0.0234 | -0.0550 | -0.0490 | -0.0552 | -0.0485 | 0.2626 |
| native_grpo | il_sft | pairwise_ADE64 | 5000 | 0.0770 | 0.0649 | 0.0758 | 0.0782 | 0.0756 | 0.0785 | 1.0000 |
| native_grpo | il_sft | centroid_displacement | 5000 | 1.1290 | 0.9377 | 1.1027 | 1.1560 | 1.0805 | 1.1787 | 0.9506 |
| native_grpo | score | mean_PDMS | 5000 | -0.1921 | -0.0651 | -0.2574 | -0.1248 | -0.2661 | -0.1159 | 0.4708 |
| native_grpo | score | feasible_rate | 5000 | -0.0049 | 0.0000 | -0.0057 | -0.0040 | -0.0059 | -0.0040 | 0.3218 |
| native_grpo | score | pairwise_ADE64 | 5000 | 0.0017 | 0.0054 | 0.0012 | 0.0021 | 0.0010 | 0.0023 | 0.7268 |
| native_grpo | score | centroid_displacement | 5000 | 0.1021 | 0.0934 | 0.0974 | 0.1066 | 0.0961 | 0.1078 | 0.8022 |
| native_grpo | pareto | mean_PDMS | 5000 | -0.1516 | -0.0532 | -0.2184 | -0.0814 | -0.2232 | -0.0781 | 0.4718 |
| native_grpo | pareto | feasible_rate | 5000 | -0.0043 | 0.0000 | -0.0052 | -0.0035 | -0.0053 | -0.0034 | 0.3260 |
| native_grpo | pareto | pairwise_ADE64 | 5000 | 0.0013 | 0.0052 | 0.0008 | 0.0017 | 0.0006 | 0.0019 | 0.7192 |
| native_grpo | pareto | centroid_displacement | 5000 | 0.1000 | 0.0899 | 0.0955 | 0.1043 | 0.0935 | 0.1061 | 0.7960 |

这里差值=PSI-baseline；feasible_rate差值单位是比例，乘100得到百分点。win_fraction仅表示差值>0，对loss/center等不等同于性能胜率。3000次scene-paired bootstrap，同时报告log-cluster sensitivity。先在每scene平均两个seed，未把10000个scene-seed当10000独立场景；CI条件于这两个已训练模型，不能代表充分的训练seed不确定性。

## 6. 两seed是否一致

| method | seed | protocol | mean_PDMS_mean | feasible_rate_mean | centroid_displacement_mean | pairwise_ADE64_mean |
| --- | --- | --- | --- | --- | --- | --- |
| il_sft | 1701 | eval | 90.5756 | 0.9204 | 0.1739 | 0.1442 |
| il_sft | 1701 | native_grpo | 68.7016 | 0.6863 | 0.1783 | 1.9911 |
| il_sft | 2903 | eval | 90.7137 | 0.9220 | 0.1757 | 0.1402 |
| il_sft | 2903 | native_grpo | 69.0179 | 0.6902 | 0.1810 | 1.9900 |
| pareto | 1701 | eval | 90.6852 | 0.8754 | 1.1978 | 0.2827 |
| pareto | 1701 | native_grpo | 67.5989 | 0.6386 | 1.2072 | 2.0613 |
| pareto | 2903 | eval | 90.6287 | 0.8759 | 1.1896 | 0.2897 |
| pareto | 2903 | native_grpo | 67.7550 | 0.6426 | 1.2101 | 2.0713 |
| psi | 1701 | eval | 90.6142 | 0.8697 | 1.2926 | 0.2823 |
| psi | 1701 | native_grpo | 67.5758 | 0.6356 | 1.3039 | 2.0633 |
| psi | 2903 | eval | 90.2921 | 0.8670 | 1.2976 | 0.2888 |
| psi | 2903 | native_grpo | 67.4749 | 0.6369 | 1.3134 | 2.0718 |
| score | 1701 | eval | 90.7165 | 0.8759 | 1.1960 | 0.2830 |
| score | 1701 | native_grpo | 67.6212 | 0.6391 | 1.2067 | 2.0614 |
| score | 2903 | eval | 90.7026 | 0.8771 | 1.1869 | 0.2884 |
| score | 2903 | native_grpo | 67.8138 | 0.6433 | 1.2064 | 2.0703 |

必须同时看两个seed，不能只挑PSI较好的一次。完整配对表`paired_by_seed.csv`。step0/128/512使用相同1000场景的曲线见`matched1000_evolution.csv`。

## 7. 有无真正吸收相同监督信号

所有模型评价同一PSI non-GT target，避免“每个方法只看自己的teacher”造成直接难度混杂：

| split | method | scenes | loss_scenes | epsilon_MSE | Hit64 | mass64 | nearest_ADE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | il_sft | 4728 | 943 | 0.0312 | 0.1642 | 0.1010 | 1.4661 |
| holdout | official_il | 4728 | 943 | 0.0293 | 0.1690 | 0.1034 | 1.4186 |
| holdout | pareto | 4728 | 943 | 0.0109 | 0.4954 | 0.2520 | 0.6621 |
| holdout | psi | 4728 | 943 | 0.0102 | 0.5299 | 0.2681 | 0.6391 |
| holdout | score | 4728 | 943 | 0.0109 | 0.4955 | 0.2518 | 0.6633 |
| train | il_sft | 492 | 492 | 0.0269 | 0.1622 | 0.0943 | 1.4277 |
| train | official_il | 492 | 492 | 0.0253 | 0.1746 | 0.1059 | 1.3859 |
| train | pareto | 492 | 492 | 0.0083 | 0.6499 | 0.3500 | 0.4934 |
| train | psi | 492 | 492 | 0.0073 | 0.7198 | 0.4003 | 0.4306 |
| train | score | 492 | 492 | 0.0083 | 0.6488 | 0.3498 | 0.4935 |

相同GT目标上的保留情况（所有方法都评价同一个GT，避免GT仅在部分PSI选集中出现造成分母混淆）：

| split | method | loss_scenes | GT_epsilon_MSE | GT_Hit64 | GT_nearest_ADE |
| --- | --- | --- | --- | --- | --- |
| holdout | il_sft | 1000 | 0.0040 | 0.8829 | 0.2488 |
| holdout | official_il | 1000 | 0.0038 | 0.9012 | 0.2326 |
| holdout | pareto | 1000 | 0.0096 | 0.4152 | 0.8878 |
| holdout | psi | 1000 | 0.0110 | 0.3805 | 0.9924 |
| holdout | score | 1000 | 0.0096 | 0.4161 | 0.8864 |
| train | il_sft | 512 | 0.0018 | 0.9795 | 0.1410 |
| train | official_il | 512 | 0.0024 | 0.9258 | 0.1984 |
| train | pareto | 512 | 0.0087 | 0.4941 | 0.8048 |
| train | psi | 512 | 0.0100 | 0.4414 | 0.9024 |
| train | score | 512 | 0.0087 | 0.4922 | 0.8022 |

覆盖指标在全部5000 holdout及固定512个train probe计算；native epsilon loss在固定1000 holdout及512 train probe上，16个共同noise/timestep draws。无non-GT teacher的scene不进入该teacher分母，表中公开scenes/loss_scenes。Hit64是64次中至少一条ADE≤0.5m，不是概率或可学习性的充分判据；mass64是64次中的附近频率。`teacher_summary.csv`同时包含各来源自己的teacher、GT与共同跨方法teacher比较。

训练loss降低、输出覆盖增加、可行高质量样本增加是三个不同检验。即使noise-MSE更低，64次没采到也只能说明本采样预算和邻域下低覆盖，不能断言永远生成不了。`target_presentations.parquet`记录目标究竟被实际抽到训练多少次，零次不伪称学过。

额外的训练后描述性审计 `parent_absorption_summary.csv` 在预先固定的512个训练probe场景中，逐parent连接真实训练ledger，只对实际至少被监督一次的parent报告64次输出零命中率，并同时给出场景等权及parent计数版本。该补充审计在训练后增加，不冒称预注册主终点；未改变目标、训练、场景或checkpoint选择。

## 8. 为什么必须分开普通eval和GRPO采样

两者同一权重，但原实现采样floor/clip不同：eval floor0.0001、noise clip±1；native GRPO floor0.04、clip±5；logprob floor0.1是独立参数。在当前legacy归一化下，native最后一步0.04的噪声floor对应约1.3348m的X标准差、0.84m的Y标准差（裁剪前）；这也解释了为何0.5m/8点ADE邻域中的有限次命中可能极低。GRPO组内宽度与尾部风险因此不能从普通eval宽度直接外推。这里只调用真实forward_grpo采样、在reward/advantage/optimizer前截取；没有GRPO权重更新。G16是统一诊断group，历史成功训练原本用G8，不冒称重放历史G8更新。

## 9. 证据范围与暂不能成立的因果表述

- 受控干预范围：本轮直接改变相同IL初始化之后的监督选择，预算、loss、网络和随机流一致；因此可讨论本次selection bundle对SFT输出的影响。实验设计通过审计本身并不等于PSI优越性得到支持。
- 不得把结果分解成“单独diversity门槛”“单独quality band”“单独目标数量”的因果贡献：这些在PSI bundle中同时不同；需后续消融才能拆分。
- Pairwise ADE和centroid displacement是轨迹几何统计；增大本身不能证明产生了新的驾驶语义模式，也不能证明新增行为安全或有用。
- UNTESTED：downstream GRPO训练是否因此更稳/收益更高。本轮只测native GRPO rollout分布，未做GRPO更新。
- UNTESTED：这套方法是否在完整训练规模和独立Navtest优于所有baseline，或是否具有普遍必要性。两seed、3072训练场景的受控结果不能替代正式规模复验。
- 最终支持/反对PSI的定量判定应同时读取第4–7节，不允许因负结果改阈值、换scene或按得分挑snapshot。

## 10. 审计与产物

全部新增测试通过；原生loss/梯度smoke通过；scalar/batch NAVSIM评分差0；native抽取和跨scene批处理通过FP32 parity；四台服务器逐一核验旧IL CRN缓存；训练初始化hash一致、梯度预算一致、冻结buffer未变；eval权重/状态未变；旧V1/V2/V3指标和报告hash未变。

- 报告：`reports/PSI_MATCHED_SFT_20260920.md`
- 图：`outputs/psi_matched_sft/figures/Fig1_matched_distribution_quality.*`
- 图：`outputs/psi_matched_sft/figures/Fig2_scene_paired_differences.*`
- 图：`outputs/psi_matched_sft/figures/Fig3_common_supervision_absorption.*`
- 图：`outputs/psi_matched_sft/figures/Fig4_fixed1000_training_evolution.*`
- 图：`outputs/psi_matched_sft/figures/Fig5_scene_distribution_ECDF.*`
- 图同时保存PNG/PDF/SVG/CSV。代码：`tools/analysis/psi_matched_sft/`。
- checkpoint、raw rollouts、特征与大缓存仅保存在本地独立namespace，不提交Git。
