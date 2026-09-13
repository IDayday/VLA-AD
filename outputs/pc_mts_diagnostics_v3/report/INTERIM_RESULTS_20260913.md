# PC-MTS V3 阶段性结果：A–E 完成，F/G 继续运行

整理时间：2026-09-13T09:30:37Z。这是用户要求的首次阶段性推送，**不是最终完成报告**。

分支：`analysis/pc-mts-policy-diagnostics-v3-20260913`；base：`f499be15d693a6910593ec2c8f081c249cdbf871`；primary YAML SHA256：`e0decb6e64854f06f29ce81af3031c74149eb6d1b9041f2b0a1bd79cf85cceaf`。

V1/V2 的代码、结果、缓存、报告与图表保留。所有新增产物位于 V3 namespace；192-candidate reservoir 是受控诊断候选空间，不是历史训练数据。

## 已完成与待完成

A selection audit、B quality–compatibility frontier、C source/quality-matched native learnability、D exact gradient diagnostic、E controlled Micro-SFT 均已完成。12 个 Micro-SFT run、12 个短程 original-GRPO run 的训练均完成。

F 的快照评测正在运行；本次整理时完成 7703/10800 个非零 GRPO 快照×场景缓存。每个快照固定300个holdout scenes、每场景64条CRN轨迹。训练成功不等于下游有效，**本次不报告尚未完整评分的GRPO收益**。

待完成：F 全部快照评分、gain与safety统计；G current-policy独立R128/Q128重新校准及unchanged-IL重采样对照；最终审计、Fig6/7与完整报告。后台实验持续运行，完成后第二次commit/push。

## 预先固定的实验与边界

1000 scenes×192 unique raw candidates；token hash固定700 train/300 holdout；seeds=1701,2903。SFT 200 updates，snapshots=0/20/100/200；GRPO 100 updates，snapshots=0/10/50/100。优化器、LR、sampling budget、GT retention及unique-parent weighting对所有方法一致。

新optimizer更新未使用300个holdout tokens。但其中195个scene与训练共享log，且历史checkpoint可能已见过这些Navtrain scenes。冻结V1 GT-distance阈值利用了原1000场景校准。因此这是内部机制诊断，不能称为完全未接触的独立测试集。

核心比较使用3000次bootstrap；配对候选同时报告pair-level和scene-cluster区间。训练结果先在同scene内平均两个seeds，再进行scene-paired比较；同时保留每个seed。q是Q→R距离经验秩，不是概率/likelihood。

## A：Conditional selector与gate order

实际顺序：hard safety NC=DAC=1 → q<95 → PDMS≥reference → selection-local≥0.75 → compatible集合内Pareto → Boundary优先 → diversity；strict不足时secondary才允许reference−1。Core按靠近Boundary优先。只计unique raw parents，禁止tiny/duplicate构成新模式。训练重复slot按1/(unique count×parent multiplicity)赋权，空pool使用显式GT fallback。

| method | strict_candidate_count | unique_candidate_count | zero_strict_scene | scene_has_4 | scene_has_8 | scene_has_16 | PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| conditional_pc | 15.113 | 15.208 | 0.04 | 0.953 | 0.948 | 0.919 | 95.012 |
| gt_distance | 2.91 | 16 | 0.431 | 0.249 | 0.14 | 0.025 | 95.857 |
| old_pc | 6.303 | 6.375 | 0.043 | 0.361 | 0.356 | 0.336 | 94.345 |
| pareto | 1.248 | 16 | 0.568 | 0.067 | 0.04 | 0.004 | 97.123 |
| quality_only | 15.113 | 15.208 | 0.04 | 0.953 | 0.948 | 0.919 | 95.012 |
| score | 1.074 | 16 | 0.846 | 0.101 | 0.069 | 0.005 | 97.4 |

Conditional平均strict selected unique数15.113，Old-PC按同一strict定义为6.303；Conditional有40个零strict scene、38个空pool，全部保留在Full-1000 coverage中。候选PDMS对空pool未定义，因此不能拿Conditional的962-scene均值直接对比其他方法1000-scene均值。

共1135条rescued candidates，涉及582/1000 scenes；far-dominator poisoning为100%。保持V2其他eligibility不变，仅改变Pareto排名集合时，平均Front≤2可用数由12.655变为13.798。完整新selector还修改了其他gate，不能把全部覆盖收益归因于gate order。

同一962个可比较scene上的Conditional-minus-baseline候选PDMS：

| method | n | mean | median | ci_low | ci_high | win_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| score | 962 | -2.371 | -1.1437 | -2.585 | -2.1539 | 0 |
| pareto | 962 | -2.1543 | -1.0768 | -2.4076 | -1.9134 | 0.017672 |
| gt_distance | 962 | -0.82587 | -0.076613 | -0.98654 | -0.66114 | 0.11746 |
| old_pc | 962 | -0.44514 | -0.27566 | -0.48144 | -0.4097 | 0.006237 |

**Pareto degeneration：** strict eligible集合内TTC/DDC恒为1，Pareto几乎退化为EP排序；没有事后换目标。覆盖提高不代表没有质量代价。

## B：质量与兼容性frontier

| metric | n | mean | median | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| fraction_within_oracle_0.25 | 1000 | 0.42 | 0 | 0.39 | 0.45003 |
| fraction_within_oracle_0.5 | 1000 | 0.445 | 0 | 0.414 | 0.476 |
| fraction_within_oracle_1.0 | 1000 | 0.498 | 0 | 0.466 | 0.529 |
| boundary_headroom | 959 | 0.20734 | 0.02829 | 0.18301 | 0.23223 |
| boundary_headroom_positive_full1000 | 1000 | 0.503 | 1 | 0.472 | 0.53403 |
| boundary_gain_over_reference | 962 | 1.9647 | 0.60952 | 1.3997 | 2.6386 |
| boundary_gain_over_reference_positive_full1000 | 1000 | 0.623 | 1 | 0.595 | 0.653 |

Full-1000中，q<95可行oracle距global feasible oracle不超过0.25/0.5/1 point的场景比例为42.0%/44.5%/49.8%。Boundary相对Core有正headroom的场景为50.3%，相对IL reference为62.3%。有限的高质量frontier存在，但compatibility tax也真实存在。

## C：source/quality-matched learnability

primary gap≤0.25 point，共3274对、890 scenes；3209对Boundary-vs-Far、65对Core-vs-Far。同scene、exact source、NC/DAC class、一对一无replacement，pair内完全相同timestep/epsilon。原生epsilon-prediction MSE，未构造不存在的x0预测头。

| metric | near_mean | far_mean | mean | cluster_ci_low | cluster_ci_high | scene_win_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| diffusion_loss | 0.0012201 | 0.0025757 | 0.0013556 | 0.00085797 | 0.0019252 | 0.80899 |
| loss_t20 | 0.00035086 | 0.0008792 | 0.00052835 | 0.00035536 | 0.00072666 | 0.8618 |
| loss_t50 | 6.6823e-05 | 0.00012813 | 6.1307e-05 | 4.1936e-05 | 8.3576e-05 | 0.80562 |
| loss_t80 | 2.5506e-05 | 3.1716e-05 | 6.2102e-06 | 4.2904e-06 | 8.3836e-06 | 0.74607 |
| E_rec | 0.21277 | 0.24926 | 0.036483 | 0.034258 | 0.038859 | 0.89326 |
| return_rate | 0.93479 | 0.88648 | -0.04831 | -0.053389 | -0.043227 | 0.057303 |

控制scene/source固定效应、PDMS及d_GT后的r_knn系数：

| outcome | coefficient | ci_low | ci_high | n | scenes |
| --- | --- | --- | --- | --- | --- |
| diffusion_loss | 0.0011468 | 0.00057752 | 0.0017226 | 8000 | 1000 |
| E_rec | 0.031138 | 0.028376 | 0.033985 | 8000 | 1000 |

**SUPPORTED：** 离线quality、source与GT-distance不足以替代policy compatibility对原生拟合难度的描述；仍不等于距离本身的完整因果效应。

## D：gradient compatibility

固定256 scenes，最后DiT block+action_decoder共2453219个参数，exact gradients，没有sketch。matched gradient analysis使用393对候选。

| metric | near_mean | far_mean | mean | cluster_ci_low | cluster_ci_high |
| --- | --- | --- | --- | --- | --- |
| loss | 0.0010524 | 0.0013688 | 0.00031646 | 6.1981e-05 | 0.00062601 |
| gradient_norm | 0.31846 | 0.40548 | 0.087022 | 0.057184 | 0.1213 |
| cosine | 0.20717 | 0.16236 | -0.044816 | -0.10237 | 0.0073849 |
| dot | 0.0003683 | 0.0051914 | 0.0048231 | -0.0085151 | 0.018298 |
| predicted_delta_L_ref | -3.683e-09 | -5.1914e-08 | -4.8231e-08 | -1.8599e-07 | 8.0928e-08 |

**SUPPORTED：** Far梯度范数更大。**NOT SUPPORTED：** Far具有更强gradient conflict/reference interference；相关区间包含0。不能把更大范数等同于有害冲突。

## E：Controlled Micro-SFT

所有方法从相同official IL action head开始，原生diffusion MSE+0.25 GT retention；200 updates、effective scene batch8、16 candidate slots。以下是final的两个seed结果：

| method | seed | PDMS | feasible_rate | Spread_AUC | center_shift | IL_retention_loss |
| --- | --- | --- | --- | --- | --- | --- |
| conditional_pc | 1701 | 92.73 | 0.94214 | 0.11832 | 0.08961 | 0.0010702 |
| conditional_pc | 2903 | 92.147 | 0.93724 | 0.11587 | 0.081257 | 0.0010553 |
| gt_distance | 1701 | 92.781 | 0.93234 | 0.14939 | 0.18655 | 0.0016845 |
| gt_distance | 2903 | 92.732 | 0.93641 | 0.14062 | 0.14671 | 0.001495 |
| gt_only | 1701 | 92.521 | 0.93901 | 0.14229 | 0.12206 | 0.0014042 |
| gt_only | 2903 | 92.329 | 0.93771 | 0.13788 | 0.15708 | 0.0014626 |
| old_pc | 1701 | 92.712 | 0.93807 | 0.11626 | 0.12183 | 0.0010825 |
| old_pc | 2903 | 92.174 | 0.93448 | 0.11314 | 0.097196 | 0.0010393 |
| pareto | 1701 | 93.239 | 0.92536 | 0.16581 | 0.41072 | 0.0026479 |
| pareto | 2903 | 92.942 | 0.92281 | 0.16167 | 0.37169 | 0.0026185 |
| score | 1701 | 92.847 | 0.92016 | 0.16271 | 0.39919 | 0.0024949 |
| score | 2903 | 92.745 | 0.92099 | 0.15886 | 0.35191 | 0.0024411 |

Conditional-minus-baseline的scene-paired差值：

| method | metric | mean | median | ci_low | ci_high | win_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| gt_only | PDMS | 0.014079 | 0 | -0.36676 | 0.36563 | 0.41333 |
| gt_only | feasible_rate | 0.0013281 | 0 | -0.0033594 | 0.0064062 | 0.063333 |
| gt_only | center_shift | -0.054138 | -0.044911 | -0.060826 | -0.047558 | 0.14333 |
| gt_only | IL_retention_loss | -0.00037066 | -0.00028055 | -0.00041567 | -0.00032493 | 0.046667 |
| score | PDMS | -0.35724 | -0.86506 | -0.96599 | 0.31189 | 0.10667 |
| score | feasible_rate | 0.019115 | 0 | 0.0095573 | 0.029324 | 0.11667 |
| score | center_shift | -0.29012 | -0.21912 | -0.31462 | -0.26689 | 0.016667 |
| score | IL_retention_loss | -0.0014053 | -0.00091621 | -0.0015528 | -0.0012636 | 0.01 |
| pareto | PDMS | -0.65141 | -0.96023 | -1.2886 | -0.021621 | 0.1 |
| pareto | feasible_rate | 0.015599 | 0 | 0.0065352 | 0.025964 | 0.1 |
| pareto | center_shift | -0.30577 | -0.2378 | -0.33098 | -0.28013 | 0.01 |
| pareto | IL_retention_loss | -0.0015705 | -0.0010667 | -0.0017411 | -0.0014111 | 0.0066667 |
| gt_distance | PDMS | -0.31757 | -0.28258 | -0.66187 | 0.037058 | 0.08 |
| gt_distance | feasible_rate | 0.0053125 | 0 | 0.00041667 | 0.010859 | 0.076667 |
| gt_distance | center_shift | -0.081193 | -0.075297 | -0.087944 | -0.074486 | 0.04 |
| gt_distance | IL_retention_loss | -0.00052707 | -0.00042043 | -0.0005818 | -0.00047251 | 0.03 |
| old_pc | PDMS | -0.0042543 | -0.14551 | -0.29921 | 0.32917 | 0.073333 |
| old_pc | feasible_rate | 0.0034115 | 0 | -7.8125e-05 | 0.0070059 | 0.063333 |
| old_pc | center_shift | -0.024078 | -0.019182 | -0.029041 | -0.019159 | 0.28667 |
| old_pc | IL_retention_loss | 1.8579e-06 | -2.5092e-06 | -1.7821e-05 | 2.243e-05 | 0.48667 |

**PARTIALLY SUPPORTED：** Conditional位移和retention loss更小，相对Score/Pareto安全率更高；但相对GT-only PDMS仅+0.0141 point，CI包含0，没有证明额外quality headroom。Score/Pareto-MTS本轮分布更宽，不能将历史“Multi-SFT更窄”当成所有多候选SFT的必然性质。

## 当前可以与不能得出的结论

SUPPORTED：global gate会过滤compatible candidates；存在有限Boundary headroom；匹配与固定效应支持support distance和native fitting difficulty相关。

PARTIALLY SUPPORTED：strict coverage和策略保留改善；但质量存在取舍，实际pool仍混合source、quality、support属性，不能唯一归因于compatibility。

NOT SUPPORTED：更强gradient conflict、相对GT-only明确的SFT质量headroom，以及“多候选SFT必然压缩support”的普遍表述。NOT SUPPORTED表示当前未获支持，不是证明普遍无效。

UNTESTED/评测未完成：Conditional-PC是否改善GT-SFT→GRPO的gain/safety tradeoff，以及更强policy是否促进Far→compatible。尚不足以将PC-MTS写成已验证的核心下游性能贡献。

## 图表与审计

全部表位于`outputs/pc_mts_diagnostics_v3/metrics/`，配置位于`configs/pc_mts_diagnostics_v3/`。数值无改动的C字符串序列化修复及A计数范围澄清均有独立审计；未修改阈值、seed、split或rollout数值。大型缓存/checkpoint只保留服务器。

Fig-V3-1: [PNG](../figures/Fig-V3-1_Gate_order.png) / [PDF](../figures/Fig-V3-1_Gate_order.pdf)。

Fig-V3-2: [PNG](../figures/Fig-V3-2_Quality_compatibility_frontier.png) / [PDF](../figures/Fig-V3-2_Quality_compatibility_frontier.pdf)。

Fig-V3-3: [PNG](../figures/Fig-V3-3_Matched_learnability.png) / [PDF](../figures/Fig-V3-3_Matched_learnability.pdf)。

Fig-V3-4: [PNG](../figures/Fig-V3-4_Gradient_interference.png) / [PDF](../figures/Fig-V3-4_Gradient_interference.pdf)。

Fig-V3-5: [PNG](../figures/Fig-V3-5_Micro_SFT_shift.png) / [PDF](../figures/Fig-V3-5_Micro_SFT_shift.pdf)。
