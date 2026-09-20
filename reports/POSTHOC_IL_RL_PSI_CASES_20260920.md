# 事后按结果筛选的IL/RL/PSI案例集

**OUTCOME-SELECTED ILLUSTRATIONS — NOT POPULATION VALIDATION。** 这是应用户要求专门按期望结果选择的案例集。不能将此表作为随机1000场景结果、无偏评测、预注册验证或总体机制证据。原完整1000场景报告及负结果保持不变。

## 数据与筛选目标

可复用且具备三模型、同场景、多次采样、真实评分和同口径中心的完整缓存是1000场景。没有新增推理，没有把历史单输出评分当成64次分布，没有重复或加权复制场景来凑1000。

分别在eval、native_grpo中按相同scene ID选择，三模型使用同一选集且场景等权，禁止跨采样器拼指标。先报告每个场景单独满足条件的数量，再通过整数规划寻找子集均值满足条件的案例集。后者允许个别场景不满足趋势，由其他场景补偿；所有选中及排除ID完整公开。

directional只要求RL/PSI均分上升且RL增幅更大、两者可行率上升且PSI增幅更大、RL宽度不升、PSI宽度升、PSI中心位移更大。数值guard只避免浮点误差，不代表差异明显或显著。

practical把“小幅/明显”操作化为一套明确但并非唯一的事后示例阈值：RL均分至少+1分、PSI +0.1至+1分；RL可行率+0.1至+2个百分点、PSI至少+2个百分点且比RL高至少0.1个百分点；RL中心位移≤0.5m、PSI至少比RL多0.1m；RL ADE不增加、PSI至少增加0.01m。这些不是经用户确认或验证过的科学界限，不把它们包装成预注册标准。

目标为在上述条件下尽量保留更多unique scenes。求解时间上限20秒；只有solver_status=OPTIMAL才可称为该有限候选全集、该规则下的最大集合，其他只是找到的可行集合。

## 筛选数量

| protocol | rule | available_scenes | individual_all_conditions_count | selected_count | solver_status |
| --- | --- | --- | --- | --- | --- |
| eval | directional | 1000 | 0 | 138 | OPTIMAL |
| eval | practical | 1000 | 0 | 114 | OPTIMAL |
| native_grpo | directional | 1000 | 1 | 293 | OPTIMAL |
| native_grpo | practical | 1000 | 0 | 183 | OPTIMAL |

## 选中案例的描述性统计

以下数字已经被筛选条件优化过。不能对这些条件性均值套普通bootstrap CI后声称验证了原假说。Feasible单位%，ADE/中心位移单位m，PDMS单位0–100。

| protocol | rule | model | scenes | pairwise_ADE | center_shift_m | feasible_rate | mean_PDMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval | directional | official_il | 138 | 0.1691 | 0.0000 | 90.5118 | 94.0876 |
| eval | directional | original_grpo_11970 | 138 | 0.1691 | 0.4524 | 91.7006 | 95.2855 |
| eval | directional | psi_sft | 138 | 0.3100 | 0.4527 | 91.7233 | 95.2429 |
| eval | practical | official_il | 114 | 0.1725 | 0.0000 | 88.1579 | 92.2050 |
| eval | practical | original_grpo_11970 | 114 | 0.1725 | 0.4402 | 90.0082 | 93.3949 |
| eval | practical | psi_sft | 114 | 0.3132 | 0.5421 | 90.2823 | 93.1386 |
| native_grpo | directional | official_il | 293 | 1.9954 | 0.0000 | 75.3840 | 75.6182 |
| native_grpo | directional | original_grpo_11970 | 293 | 1.9924 | 0.5251 | 75.4053 | 80.6189 |
| native_grpo | directional | psi_sft | 293 | 2.2489 | 0.5255 | 75.4106 | 76.7153 |
| native_grpo | practical | official_il | 183 | 1.9957 | 0.0000 | 68.6219 | 72.3050 |
| native_grpo | practical | original_grpo_11970 | 183 | 1.9931 | 0.4751 | 70.5174 | 77.3031 |
| native_grpo | practical | psi_sft | 183 | 2.2474 | 0.5751 | 70.6284 | 73.3033 |

A/B半组复查完整公开，但FULL64选择已经用了A和B，两者均不是未见验证数据。未选场景与完整1000场景同表提供，不能隐藏被排除的负结果。实际半组复查并非全部通过：practical-eval在A/B分别满足8/11、10/11条条件，practical-native分别满足10/11、8/11条；完整64次则都满足11/11条。这显示均值门槛附近的筛选结果对采样有敏感性，不能把FULL64满足门槛当作稳定泛化。

当前不能交付1000个不同且满足这些条件的案例：现有完整候选全集本身只有1000，选择全部就回到已经不满足期望趋势的总体。这里不声称在其他未采样场景中不存在这种案例；也没有扩大搜索直到得到期望结论后将其冒称随机数据。

完整原始对照：[1000场景假设核查](IL_RL_PSI_TREND_AUDIT_1000_20260920.md)。选集路径：outputs/historical_sft_grpo_support/posthoc_case_selection_20260920/scene_selection_with_exclusions.csv。结果用途仅为描述“在哪些已观测场景中能展示这种现象”。
