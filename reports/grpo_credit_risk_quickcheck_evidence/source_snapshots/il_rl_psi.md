# IL → GRPO 与 PSI-SFT：5000场景轨迹分布复核（2026-09-20）

本报告使用真实缓存与新增采样，不按期望结论选择场景。三个checkpoint全程冻结，无SFT或GRPO权重更新。

## 证据支持的解释

**SUPPORTED：原版GRPO主要改善真实组内采样的质量和低分尾部，没有明显收窄其几何宽度。** Native GRPO下均分+9.834 points（95% CI [9.502,10.174]），可行率+8.018个百分点；宽度差−0.00014m（CI跨0）。组内最低分+9.222，最高分仅+1.086，说明收益不只是偶尔采到更高的峰值。新增4000独立token复现。

**PARTIALLY_SUPPORTED：PSI SFT拓宽了采样轨迹范围，并小幅提高普通推理均分；扩大部分同时包含失败轨迹。** Eval均分+0.691 [0.323,1.066]，ADE64由0.1455升到0.3066m，但可行率−1.407个百分点。Native采样均分−3.229 [−3.455,−2.994]、可行率−6.285个百分点、宽度+0.2607m。这些方向在新增4000中均复现。

**NOT_SUPPORTED：这组PSI SFT带来更高可行率，以及比GRPO更大的中心位移。** Eval中心位移PSI=0.4295m，GRPO=0.5014m；PSI−GRPO=−0.0719m，CI [−0.0785,−0.0649]。Native下两者约0.49–0.50m，PSI也没有更大。不能把“更宽”与“中心移动更大”混为一谈，也不能把约0.5m位移直接命名为语义mode不变。

**PARTIALLY_SUPPORTED：PSI更接近部分训练教师，但监督覆盖改善较弱。** 在3677个确有非GT监督的训练场景中，平均最近ADE下降0.1440m；Eval Hit@64（0.5m）由9.50%升到10.17%，差+0.671个百分点 [0.064,1.269]。新增4000中的2910个相应场景只提升0.355个百分点 [−0.292,1.031]，独立复核未排除零差异。同时teacher邻域的平均采样mass由6.40%降到5.33%。更接近教师、至少命中一次、频繁复现教师是三个不同概念。

**诊断限制：0.5m教师命中在native sampler中全部为零，不能用于区分三个checkpoint。** 这不代表数学概率为零。事前固定的1.0m敏感性下Hit@64为IL 19.49%、GRPO 20.75%、PSI 14.25%；主阈值没有被替换。Native更强的逐步噪声会显著改变有限采样下的近邻命中，不能把未命中直接当作未学会。

**安全维度并非全部同向改善。** GRPO的joint feasibility上升，但DDC均值下降：Eval 98.393→97.476、Native 97.054→95.535。其主要安全收益来自DAC和TTC；PSI普通推理均分提高伴随EP提高，而NC/DAC/TTC/DDC均值均下降。Native下PSI的TTC由83.631降到78.551，是需要关注的变化。

**UNTESTED：上述分布变化是否单独造成后续GRPO成败。** 本轮没有进行权重更新，也没有找到PSI历史GRPO实体；这些数据不能替代相同GRPO recipe、相同更新预算下的初始化对照。PSI训练／验证身份分层都出现native可行率与均分下降，说明观察到的现象不只出现在其训练场景；但这仍不是损失函数、监督来源与后续优化之间的完整因果证明。

完成 **5000/5000场景、120000组G16、1920000条轨迹**；新增4000场景、1536000条轨迹，其余复用经SHA256核验的旧缓存。缺失场景0、未解决评分异常0。

## 设计和解释范围

母体为canonical Navtrain的103288个token。保留先前随机1000场景，按指令类别分层、token hash随机增加4000；合计直行3170、左转1257、右转573。旧1000、新4000、全5000分别公开。新4000与旧1000 token不重叠，但可以来自相同驾驶日志，因此额外给出log-cluster区间。

每个checkpoint、每个场景分别使用普通evaluation sampler与真实forward_grpo采样入口，各采4组×16=64条。模型间和采样协议间使用common random numbers；不同场景／组独立。使用FP32，同一观察特征、统一真实NAVSIM-v1评分。G16是本次标准化诊断；历史正式original GRPO用G8，本报告没有将G16的统计冒充历史G8训练日志。

这批场景来自Navtrain，可能参与过历史训练。它们是内部机制分析，不是untouched Navtest泛化结果；表中采样均分不能与checkpoint名称中的90.41等历史单条评测分数直接比较。

比较范围是Official IL→original GRPO与Official IL→PSI候选SFT。它不是MTS-86.92／87.51从头联合训练的5000场景复现，也不是PSI-SFT→GRPO因果链：后者历史GRPO权重仍未找到，不以APR替代。

## 主要实测结果

Pairwise ADE64：全部64条轨迹之间的平均XY ADE；不含heading。Centroid displacement：64条均值轨迹相对同采样协议Official IL均值轨迹的ADE。Feasible：NC、DAC、TTC、DDC全部为1；PDMS以0–100 points显示。每个指标先在场景内算，再对5000场景等权平均。

### eval

| Policy | Pairwise ADE64 (m) | Centroid displacement (m) | Feasible (%) | Mean PDMS |
| --- | --- | --- | --- | --- |
| Initial GT-IL | 0.1455 | 0.0000 | 93.1312 | 91.4140 |
| Original GRPO (90.41 checkpoint) | 0.1715 | 0.5014 | 94.6534 | 94.7978 |
| PSI candidate-SFT | 0.3066 | 0.4295 | 91.7247 | 92.1052 |

### native_grpo

| Policy | Pairwise ADE64 (m) | Centroid displacement (m) | Feasible (%) | Mean PDMS |
| --- | --- | --- | --- | --- |
| Initial GT-IL | 1.9937 | 0.0000 | 69.2466 | 69.0197 |
| Original GRPO (90.41 checkpoint) | 1.9936 | 0.4997 | 77.2644 | 78.8532 |
| PSI candidate-SFT | 2.2544 | 0.4875 | 62.9619 | 65.7908 |

来源：`metrics/summary.csv`，scope=FULL5000，各行分母5000场景／320000条轨迹。组内16条Pairwise ADE另存`pairwise_ADE`字段；主表使用更充分的64条估计。

## 配对差值与置信区间

差值方向为当前checkpoint−Official IL。95% CI来自3000次scene-paired bootstrap；log-cluster CI对整段日志重采样。下面的概率相关差值以百分点呈现。

### eval 配对比较

| Checkpoint | Metric | Mean Δ | Median Δ | 95% CI | Log-cluster CI | Scene Δ>0 (%) | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original GRPO (90.41 checkpoint) | pairwise_ADE64 | 0.0260 | 0.0256 | [0.0255, 0.0265] | [0.0254, 0.0267] | 95.0400 | 5000 |
| Original GRPO (90.41 checkpoint) | centroid_displacement | 0.5014 | 0.4674 | [0.4957, 0.5069] | [0.4930, 0.5103] | 100.0000 | 5000 |
| Original GRPO (90.41 checkpoint) | feasible_rate | 1.5222 | 0.0000 | [0.9753, 2.1072] | [0.8485, 2.1861] | 8.6400 | 5000 |
| Original GRPO (90.41 checkpoint) | mean_PDMS | 3.3838 | 0.5389 | [3.0088, 3.7603] | [2.9964, 3.7958] | 57.0800 | 5000 |
| Original GRPO (90.41 checkpoint) | min_PDMS | 5.8544 | 0.7016 | [5.3169, 6.3928] | [5.2860, 6.4226] | 57.8400 | 5000 |
| Original GRPO (90.41 checkpoint) | max_PDMS | 1.9052 | 0.0000 | [1.6109, 2.2199] | [1.5726, 2.2532] | 48.8800 | 5000 |
| PSI candidate-SFT | pairwise_ADE64 | 0.1611 | 0.1538 | [0.1596, 0.1625] | [0.1587, 0.1635] | 100.0000 | 5000 |
| PSI candidate-SFT | centroid_displacement | 0.4295 | 0.3889 | [0.4226, 0.4368] | [0.4188, 0.4405] | 100.0000 | 5000 |
| PSI candidate-SFT | feasible_rate | -1.4066 | 0.0000 | [-1.8934, -0.9397] | [-1.8850, -0.9341] | 5.8200 | 5000 |
| PSI candidate-SFT | mean_PDMS | 0.6912 | 0.3332 | [0.3230, 1.0659] | [0.3163, 1.0865] | 54.1000 | 5000 |
| PSI candidate-SFT | min_PDMS | -2.0698 | 0.0000 | [-2.6484, -1.4698] | [-2.7010, -1.4662] | 48.1400 | 5000 |
| PSI candidate-SFT | max_PDMS | 1.9638 | 0.6596 | [1.6657, 2.2564] | [1.6642, 2.2779] | 55.6200 | 5000 |

### native_grpo 配对比较

| Checkpoint | Metric | Mean Δ | Median Δ | 95% CI | Log-cluster CI | Scene Δ>0 (%) | N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original GRPO (90.41 checkpoint) | pairwise_ADE64 | -0.0001 | 0.0026 | [-0.0006, 0.0003] | [-0.0008, 0.0006] | 57.9200 | 5000 |
| Original GRPO (90.41 checkpoint) | centroid_displacement | 0.4997 | 0.4694 | [0.4942, 0.5052] | [0.4915, 0.5083] | 100.0000 | 5000 |
| Original GRPO (90.41 checkpoint) | feasible_rate | 8.0178 | 4.6875 | [7.6068, 8.4197] | [7.4669, 8.5385] | 63.4400 | 5000 |
| Original GRPO (90.41 checkpoint) | mean_PDMS | 9.8336 | 5.8405 | [9.5019, 10.1736] | [9.4145, 10.2453] | 88.3400 | 5000 |
| Original GRPO (90.41 checkpoint) | min_PDMS | 9.2217 | 1.7207 | [8.7464, 9.6837] | [8.6903, 9.7756] | 57.0000 | 5000 |
| Original GRPO (90.41 checkpoint) | max_PDMS | 1.0863 | 0.0617 | [1.0003, 1.1781] | [0.9825, 1.1908] | 50.8800 | 5000 |
| PSI candidate-SFT | pairwise_ADE64 | 0.2607 | 0.2616 | [0.2591, 0.2622] | [0.2583, 0.2633] | 100.0000 | 5000 |
| PSI candidate-SFT | centroid_displacement | 0.4875 | 0.4572 | [0.4808, 0.4943] | [0.4769, 0.4988] | 100.0000 | 5000 |
| PSI candidate-SFT | feasible_rate | -6.2847 | -4.6875 | [-6.5719, -5.9916] | [-6.6209, -5.9614] | 19.9600 | 5000 |
| PSI candidate-SFT | mean_PDMS | -3.2289 | -2.4752 | [-3.4554, -2.9943] | [-3.4914, -2.9597] | 30.8000 | 5000 |
| PSI candidate-SFT | min_PDMS | -4.7703 | 0.0000 | [-5.1082, -4.4170] | [-5.1507, -4.4173] | 14.8800 | 5000 |
| PSI candidate-SFT | max_PDMS | 0.6650 | 0.0000 | [0.5479, 0.7847] | [0.5332, 0.8027] | 48.5600 | 5000 |

来源：`metrics/paired_comparisons.csv`。Δ>0仅表示数值增加，并非所有指标上都意味着更好；几何宽度与中心位移尤其不能直接按大小判优。

## GRPO组内的质量分布

下表每个最小值／最大值均先在真实16条组内求，再对4组和5000场景平均。它们不是整批数据的单个极值。

| model | protocol | min_PDMS | mean_PDMS | max_PDMS | std_PDMS | CVaR25_PDMS | No-safe group (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official_il | eval | 87.8746 | 91.4140 | 93.7665 | 2.1759 | 89.1044 | 4.3850 |
| official_il | native_grpo | 22.1146 | 69.0197 | 96.6847 | 27.1704 | 35.7999 | 1.9100 |
| original_grpo_11970 | eval | 93.7289 | 94.7978 | 95.6717 | 0.6569 | 94.0943 | 4.1850 |
| original_grpo_11970 | native_grpo | 31.3363 | 78.8532 | 97.7710 | 22.1126 | 50.1786 | 1.9900 |
| psi_sft | eval | 85.8047 | 92.1052 | 95.7302 | 3.5115 | 88.4602 | 3.9900 |
| psi_sft | native_grpo | 17.3443 | 65.7908 | 97.3497 | 29.5113 | 30.1348 | 2.3400 |

来源：`metrics/group16_metrics.parquet`（120000组）与`metrics/summary.csv`。组内没有安全轨迹时best-safe PDMS记NA，并报告该类组比例。

补充描述：各组件均分（0–100）及ego坐标系X/Y平均绝对中心偏移（m）；不把X/Y偏移直接等同于沿道路进度／横向偏移。

| model | protocol | EP | NC | DAC | TTC | DDC | Comfort | centroid_abs_dx | centroid_abs_dy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_il | eval | 85.2739 | 99.5077 | 97.6359 | 97.4513 | 98.3928 | 99.9825 | 0.0000 | 0.0000 |
| official_il | native_grpo | 70.6535 | 95.3542 | 85.8137 | 83.6309 | 97.0544 | 69.0716 | 0.0000 | 0.0000 |
| original_grpo_11970 | eval | 90.0352 | 99.5384 | 99.3966 | 98.4450 | 97.4762 | 100.0000 | 0.4395 | 0.1619 |
| original_grpo_11970 | native_grpo | 81.3649 | 96.7780 | 93.6628 | 87.2834 | 95.5355 | 70.7269 | 0.4370 | 0.1639 |
| psi_sft | eval | 88.3470 | 99.1898 | 97.2922 | 96.6012 | 98.0394 | 99.9947 | 0.3958 | 0.1043 |
| psi_sft | native_grpo | 70.9179 | 92.5483 | 84.9313 | 78.5509 | 95.1988 | 63.0497 | 0.4419 | 0.1334 |


## 新增4000场景是否复现原结论

| scope | protocol | model | metric | mean_difference | ci_low | ci_high | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FULL5000 | eval | original_grpo_11970 | pairwise_ADE64 | 0.02601 | 0.02554 | 0.02649 | 5000 |
| FULL5000 | eval | original_grpo_11970 | feasible_rate | 0.01522 | 0.00975 | 0.02107 | 5000 |
| FULL5000 | eval | original_grpo_11970 | mean_PDMS | 3.38380 | 3.00876 | 3.76028 | 5000 |
| FULL5000 | eval | psi_sft | pairwise_ADE64 | 0.16111 | 0.15957 | 0.16253 | 5000 |
| FULL5000 | eval | psi_sft | feasible_rate | -0.01407 | -0.01893 | -0.00940 | 5000 |
| FULL5000 | eval | psi_sft | mean_PDMS | 0.69120 | 0.32296 | 1.06590 | 5000 |
| FULL5000 | native_grpo | original_grpo_11970 | pairwise_ADE64 | -0.00014 | -0.00063 | 0.00034 | 5000 |
| FULL5000 | native_grpo | original_grpo_11970 | feasible_rate | 0.08018 | 0.07607 | 0.08420 | 5000 |
| FULL5000 | native_grpo | original_grpo_11970 | mean_PDMS | 9.83358 | 9.50187 | 10.17358 | 5000 |
| FULL5000 | native_grpo | psi_sft | pairwise_ADE64 | 0.26065 | 0.25905 | 0.26219 | 5000 |
| FULL5000 | native_grpo | psi_sft | feasible_rate | -0.06285 | -0.06572 | -0.05992 | 5000 |
| FULL5000 | native_grpo | psi_sft | mean_PDMS | -3.22888 | -3.45540 | -2.99434 | 5000 |
| NEW4000 | eval | original_grpo_11970 | pairwise_ADE64 | 0.02616 | 0.02562 | 0.02669 | 4000 |
| NEW4000 | eval | original_grpo_11970 | feasible_rate | 0.01739 | 0.01107 | 0.02422 | 4000 |
| NEW4000 | eval | original_grpo_11970 | mean_PDMS | 3.50972 | 3.07429 | 3.96439 | 4000 |
| NEW4000 | eval | psi_sft | pairwise_ADE64 | 0.16086 | 0.15925 | 0.16263 | 4000 |
| NEW4000 | eval | psi_sft | feasible_rate | -0.01238 | -0.01741 | -0.00727 | 4000 |
| NEW4000 | eval | psi_sft | mean_PDMS | 0.75780 | 0.35466 | 1.15304 | 4000 |
| NEW4000 | native_grpo | original_grpo_11970 | pairwise_ADE64 | -0.00005 | -0.00062 | 0.00048 | 4000 |
| NEW4000 | native_grpo | original_grpo_11970 | feasible_rate | 0.08032 | 0.07589 | 0.08497 | 4000 |
| NEW4000 | native_grpo | original_grpo_11970 | mean_PDMS | 9.83478 | 9.47286 | 10.20788 | 4000 |
| NEW4000 | native_grpo | psi_sft | pairwise_ADE64 | 0.26005 | 0.25824 | 0.26190 | 4000 |
| NEW4000 | native_grpo | psi_sft | feasible_rate | -0.06175 | -0.06484 | -0.05848 | 4000 |
| NEW4000 | native_grpo | psi_sft | mean_PDMS | -3.15553 | -3.41201 | -2.89156 | 4000 |
| OLD1000 | eval | original_grpo_11970 | pairwise_ADE64 | 0.02541 | 0.02439 | 0.02641 | 1000 |
| OLD1000 | eval | original_grpo_11970 | feasible_rate | 0.00653 | -0.00505 | 0.01734 | 1000 |
| OLD1000 | eval | original_grpo_11970 | mean_PDMS | 2.88012 | 2.19208 | 3.60983 | 1000 |
| OLD1000 | eval | psi_sft | pairwise_ADE64 | 0.16215 | 0.15902 | 0.16530 | 1000 |
| OLD1000 | eval | psi_sft | feasible_rate | -0.02080 | -0.03116 | -0.01112 | 1000 |
| OLD1000 | eval | psi_sft | mean_PDMS | 0.42477 | -0.29746 | 1.13746 | 1000 |
| OLD1000 | native_grpo | original_grpo_11970 | pairwise_ADE64 | -0.00051 | -0.00159 | 0.00058 | 1000 |
| OLD1000 | native_grpo | original_grpo_11970 | feasible_rate | 0.07963 | 0.07036 | 0.08886 | 1000 |
| OLD1000 | native_grpo | original_grpo_11970 | mean_PDMS | 9.82878 | 9.13386 | 10.54342 | 1000 |
| OLD1000 | native_grpo | psi_sft | pairwise_ADE64 | 0.26306 | 0.25969 | 0.26663 | 1000 |
| OLD1000 | native_grpo | psi_sft | feasible_rate | -0.06722 | -0.07363 | -0.06081 | 1000 |
| OLD1000 | native_grpo | psi_sft | mean_PDMS | -3.52229 | -4.03847 | -3.00713 | 1000 |

这一表中的feasible_rate仍为0–1单位，乘100即百分点。NEW4000来自事前抽样，不因模型输赢而筛选。FULL5000包含旧1000，不能把它当作完全独立复现；独立token复核应看NEW4000。

### PSI历史训练／验证身份分层（补充）

| psi_membership | model | protocol | scenes | pairwise_ADE64 | centroid_displacement | feasible_rate | mean_PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PSI_TRAIN | official_il | eval | 4111 | 0.14565 | 0.00000 | 0.94199 | 92.55488 |
| PSI_TRAIN | official_il | native_grpo | 4111 | 1.99413 | 0.00000 | 0.70067 | 69.91791 |
| PSI_TRAIN | original_grpo_11970 | eval | 4111 | 0.17076 | 0.49317 | 0.95943 | 95.76098 |
| PSI_TRAIN | original_grpo_11970 | native_grpo | 4111 | 1.99298 | 0.49130 | 0.78715 | 79.91297 |
| PSI_TRAIN | psi_sft | eval | 4111 | 0.30285 | 0.41835 | 0.92799 | 93.18615 |
| PSI_TRAIN | psi_sft | native_grpo | 4111 | 2.25026 | 0.47465 | 0.63825 | 66.65997 |
| PSI_VALIDATION | official_il | eval | 889 | 0.14466 | 0.00000 | 0.88192 | 86.13820 |
| PSI_VALIDATION | official_il | native_grpo | 889 | 1.99185 | 0.00000 | 0.65451 | 64.86594 |
| PSI_VALIDATION | original_grpo_11970 | eval | 889 | 0.17485 | 0.53952 | 0.88688 | 90.34372 |
| PSI_VALIDATION | original_grpo_11970 | native_grpo | 889 | 1.99636 | 0.53856 | 0.70555 | 73.95277 |
| PSI_VALIDATION | psi_sft | eval | 889 | 0.32386 | 0.48101 | 0.86758 | 87.10653 |
| PSI_VALIDATION | psi_sft | native_grpo | 889 | 2.27336 | 0.54715 | 0.58973 | 61.77140 |

这里的验证身份只针对PSI历史run，不表示这些场景对所有模型都是未见数据。完整配对CI见`psi_membership_distribution_comparisons.csv`。

## 真实PSI监督轨迹的采样覆盖（5000场景补充复核）

依据原train_args中的train/val log名单核验：4111个PSI训练场景、889个验证场景；其中3677个训练场景具有正权重非GT教师。教师张量与权重直接读取原support index，其SHA256与历史审计一致。

下表只统计实际训练场景内的正权重非GT监督。对每条教师，64次rollout中至少一条XY ADE≤0.5m即命中；先在场景内平均教师命中，再对场景等权平均。表中命中率／mass为0–1单位；这不是native diffusion loss，也不是精确轨迹生成概率。

| model | protocol | scenes | teachers | hit64_0p5 | hit16_0p5 | mass64_0p5 | nearest_ADE | weighted_hit64_0p5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_il | eval | 3677 | 4594 | 0.09500 | 0.08920 | 0.06403 | 1.25626 | 0.09776 |
| official_il | native_grpo | 3677 | 4594 | 0.00000 | 0.00000 | 0.00000 | 1.23361 | 0.00000 |
| original_grpo_11970 | eval | 3677 | 4594 | 0.05811 | 0.05098 | 0.02824 | 1.21564 | 0.05955 |
| original_grpo_11970 | native_grpo | 3677 | 4594 | 0.00000 | 0.00000 | 0.00000 | 1.19850 | 0.00000 |
| psi_sft | eval | 3677 | 4594 | 0.10171 | 0.09146 | 0.05333 | 1.11222 | 0.10626 |
| psi_sft | native_grpo | 3677 | 4594 | 0.00000 | 0.00000 | 0.00000 | 1.24136 | 0.00000 |

相对IL的Hit@64配对差值：

| model | protocol | mean_difference | ci_low | ci_high | log_cluster_ci_low | log_cluster_ci_high | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original_grpo_11970 | eval | -0.03690 | -0.04329 | -0.03060 | -0.04395 | -0.02942 | 3677 |
| psi_sft | eval | 0.00671 | 0.00063 | 0.01269 | 0.00031 | 0.01287 | 3677 |
| original_grpo_11970 | native_grpo | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3677 |
| psi_sft | native_grpo | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 3677 |

完整结果含GT教师、验证场景、0.25/1.0m敏感性、旧1000／新4000拆分，见`psi_teacher_summary.csv`、`psi_teacher_comparisons.csv`。64次未命中仅表示有限采样下没有观测到，不能推断概率为零、严格不可达或不可学。

## 可以支持什么，不能支持什么

- Original GRPO (90.41 checkpoint) / eval：相对IL，均分+3.384，可行率+1.522个百分点，宽度+0.0260m；组内最低分+5.854、最高分+1.905。
- Original GRPO (90.41 checkpoint) / native_grpo：相对IL，均分+9.834，可行率+8.018个百分点，宽度-0.0001m；组内最低分+9.222、最高分+1.086。
- PSI candidate-SFT / eval：相对IL，均分+0.691，可行率-1.407个百分点，宽度+0.1611m；组内最低分-2.070、最高分+1.964。
- PSI candidate-SFT / native_grpo：相对IL，均分-3.229，可行率-6.285个百分点，宽度+0.2607m；组内最低分-4.770、最高分+0.665。

解释应区分三个维度：几何宽度、分布中心和质量尾部。组内最低分改善而宽度近似不变，符合“相同探索噪声下提高采样质量”的观察；它不自动证明策略只在原有语义mode内微调。宽度增大且可行率下降，则意味着扩大的采样范围同时包含更多失败行为，不能称为新增feasible modes。中心位移必须在同一sampler下比较；0.5m左右的位移也不能直接称为无变化。

普通推理与GRPO采样不是同一个分布。已审计的runtime中，evaluation noise floor=0.0001，native sampling floor=0.04；evaluation noise clip=±1，native=±5，logprob std floor=0.1。0.1用于密度计算，不能当成实际采样标准差。具体步骤、模块train/eval状态和实际import路径见`audits/sampling_*.json`。前次1000场景factorial诊断保留用于解释噪声机制，本轮没有再修改sampling floor或clip。

本轮把实际教师的几何采样覆盖扩展到了5000场景，但没有重算5000场景的native epsilon loss；历史loss分析仍只有原1000场景。几何命中和训练loss衡量不同问题，即使平均loss下降或教师命中增加，也不能声称所有候选teacher都已被学会。

SUPPORTED / NOT_SUPPORTED的判定需结合实测方向与区间：只有均分上升、可行率也上升、且扩展独立复现时，才能描述为质量与安全同时改善。宽度扩大本身不是价值。无新权重更新、无matched训练消融，本轮仍不能给出“特定SFT损失导致后续GRPO退化”的唯一因果解释。

## Provenance 与审计

- 新分支：`analysis/il-rl-psi-distribution-5000-20260920`；parent `636a07668d7e5446a082562a4fd6e2cfe87f2a83`。
- 冻结配置SHA256：`d03c25ee4c7c99a9b662bb702933cd77bcb74bab17e5cd126230407da79a36fd`。
- Scene manifest SHA256：`a9ee4f3e8af6a332fcaf4f448e6c7dea970f2ff80a9ec589b8bd9ba319f9bf07`。
- 三个checkpoint路径／SHA256、实际归档runtime路径／SHA256见`manifests/models.json`；旧缓存原protocol metadata不改写。
- 原6000份rollout／score文件哈希一致；4scene FP32特征重建与原缓存完全一致；scalar-vs-batch NAVSIM评分完全一致。
- 每个checkpoint重新运行native采样入口parity和独立组批处理parity；新采样与旧缓存误差阈值1e-4m/rad；全部rank完成后参数与buffer SHA一致。
- 输入仅含4帧历史观察；未来GT／metric cache仅离线评分使用；没有optimizer step。
- PSI checkpoint真实存在，但其原始当日源码快照未完整恢复；使用经严格state-dict与采样一致性核验的可恢复归档runtime。这一历史provenance限制并未因样本扩增而消失。

## 文件索引

- 配置：`configs/il_rl_psi_distribution_5000/primary.yaml`。
- 原始分场景与分组统计：`outputs/il_rl_psi_distribution_5000/metrics/scene_metrics.parquet`、`group16_metrics.parquet`。
- 汇总／配对CI：`metrics/summary.csv`、`paired_comparisons.csv`；分驾驶指令：`command_strata.csv`。
- 图1：`figures/Fig1_distribution_and_quality.{png,pdf,svg,csv}`。
- 图2：`figures/Fig2_group16_quality_tails.{png,pdf,svg,csv}`。
- 图3：`figures/Fig3_independent_replication.{png,pdf,svg,csv}`。
- 图4：`figures/Fig4_scene_distribution_ECDF.{png,pdf,svg,csv}`；显示完整场景分布范围，不截尾。
- 最终完整性：`audits/final.json`；全部输入／输出hash：`manifests/observations.json`、`cache_hashes.json`。

统计和图形只读取实际缓存，没有生成理想结果，没有按checkpoint表现更换场景，没有修改旧实验数据。
