# PC-MTS Policy Distribution Diagnostics — 2026-09-13

状态：五个checkpoint的1000场景分布/PDMS/readiness，以及四方法的64,000条候选和768,000条held-out扰动全部完成。PC-MTS包含用户授权的coverage扩展，原规则零父轨迹率仍单独保留。所有表格中的比例均为0–1，PDMS为0–100 points。

## 1. Experimental Setup

主实验：Navtrain，1000 个固定场景，每 checkpoint 每场景 64 条随机轨迹。五个真实 checkpoint 与四种 reconstructed candidate selection rule 是两条独立分析线，没有重新训练或把 MTS 权重对应到 M1–M4。

共同采样器：FP32 VLM/DiT，DDIM 5 步，eta=1.0，temperature=1，不使用 CFG；每个场景共享初始噪声 seed 和中间采样 seed。轨迹为未来 4 秒、8 个点、米制 XY 与弧度 heading。A5/V6 的训练专属 FS-Norm 与 adapter 必须保留，因此内部归一化无法作为严格相同的训练算法对照。

## 2. Checkpoint Manifest

- **Official IL** (gt_sft)：`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt`；SHA256 `4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443`。历史 PDMS 为 None，仅识别权重。
- **MTS-86.92** (multi_sft)：`/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_pta_fs_dit_v6_full103k_from_random_20260713T1706Z/epoch_165.ckpt`；SHA256 `bf7183e1bbc49fd4ee1f275d293be944996dde4bc9518b82c03fdd0300fa6ff2`。历史 PDMS 为 86.92391718739543，仅识别权重。
- **MTS-87.51** (multi_sft)：`/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_pta_fs_dit_a5_full103k_standardv2_clean_20260710T091344Z/epoch_155.ckpt`；SHA256 `79257dffba07792c77835aadd31e05bdab8c2f9afec75bc9772bac772a32e4b7`。历史 PDMS 为 87.51984244441522，仅识别权重。
- **GRPO-90.41** (rl_optimized)：`/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl/outputs/pdms_eval_stage3_ckpt_watcher_20260609T093852Z/ckpts/epoch8_step11970.ckpt`；SHA256 `b51951abbba86e9661fc82f85406d09ae26c6eff6aa3651a40462363866fb326`。历史 PDMS 为 90.41989766469273，仅识别权重。
- **APR-91.45** (rl_optimized)：`/mnt/project/VLA-AD/outputs/pdms914497_reverse_conflict_residual_trust105_scale14p3_20260727/reverse_conflict_residual_scale14p3.ckpt`；SHA256 `aeea278ce1bcc473b42ade97c0705bcc674dea315a051f18e1fe0173582d3274`。历史 PDMS 为 91.450465，仅识别权重。

完整来源、config、代码版本见 `manifests/checkpoint_manifest.yaml` 和 `manifests/assets.json`。

## 3. Scene Sampling

从官方 Navtrain 的 103,288 个 token 中按 command 比例分层，seed=20260913；{'right': 115, 'straight': 634, 'left': 251}，覆盖 512 个日志。无基于模型输出或 score 的筛选。场景文件 `manifests/scenes_1000.json`。代表场景在模型结果之前按 2 个 straight、2 个 turn、2 个 heading 总变化不低于 0.55 rad 的 high-curvature 场景固定。

## 4. Parallelization and Compute

每次一个 checkpoint，8 个 torchrun rank 分片；每 rank 4 个 DataLoader workers。rollout 先落盘，再运行最多 96 个 CPU scoring workers，嵌套 BLAS/OpenMP 线程为 1。`logs/*_resources.jsonl` 记录 GPU、CPU 和内存；`manifests/*_execution.json` 记录实际命令与时长。每个缓存包含场景/config/checkpoint 身份或指向其父缓存哈希。

初始20场景smoke验证了五权重和M1–M3；原M4因零父轨迹无法通过。用户要求完整覆盖后，先完成扩展规则的20场景全四方法smoke，再运行1000场景覆盖评测。Smoke已验证真权重、轨迹坐标、采样回放、批量评分、原生 GRPO 奖励、几何控制、扰动幅度和绘图。50 场景 benchmark 的运行记录位于 `benchmark/manifests/`。批量与逐条 PDM 的全部七项分数误差为 0；GT 变换误差为 0。

## 5. Policy Distribution Width / Sharpness

TABLE A：数值先在场景内聚合，再对场景平均；所有距离以米计。D_positive 为可行且超过默认 IL reference 的轨迹之间的两两 ADE；不足两条的场景记为缺失，不当作零多样性。

|checkpoint|category|mean_pdms|feasible_rate|pairwise_ade|spread_auc|r90|endpoint_spread|effective_rank|center_shift_from_il|oracle_64|D_positive|
|---|---|---|---|---|---|---|---|---|---|---|---|
|official_il|gt_sft|92.1547|0.9394|0.1448|0.1223|0.1863|0.2703|1.6656|0.0000|94.6682|0.1186|
|mts_8692|multi_sft|93.7139|0.9676|0.0841|0.0901|0.1059|0.1225|1.7369|0.4126|95.0598|0.0914|
|mts_8751|multi_sft|94.1829|0.9676|0.0835|0.0996|0.0951|0.1874|1.7944|0.4677|95.9507|0.1900|
|grpo_9041|rl_optimized|95.0581|0.9464|0.1705|0.1440|0.2179|0.2696|1.8774|0.5008|95.9391|0.1717|
|apr_9145|rl_optimized|95.3553|0.9568|0.1831|0.1528|0.2192|0.2828|2.9973|0.8092|96.2440|0.1949|

完整时序 spread、R50/R90、终点 covariance trace、半径、center shift 及子集有效场景数在 CSV/Parquet 与 `policy_summary.csv`。Fig-0a–g 展示宽度、中心移动和预注册场景。

有效秩均值 IL/MTS86.92/MTS87.51/GRPO/APR 为1.666/1.737/1.794/1.877/2.997；APR变化分布在更多线性方向，但这不等于证明存在更多语义模态。

## 6. Reconstructed Candidate Reservoir

每个场景 128 条：1 exact GT + 31 条平滑结构扰动 + 四个非 IL 模型各 24 条独立种子输出。各方法读取同一个文件；官方 IL 只作为 support/reference，不作为 raw source。历史 candidate archive 确实找到，但未将新构建库冒称历史训练数据。

50 场景中，96 条库的 PC-MTS 主规则不足比例为 98%，按预注册规则扩为 128；扩容后仍为 98%，全部已允许回退后仍有 80% 场景没有合格父轨迹。详见 `reservoir_expansion_decision.json`。

## 7. Four Candidate Selection Rules

- Score-only：有效候选直接按标准 PDMS 降序，稳定 ID 打破完全相同的 score。
- Pareto：复用历史 `pareto_front_mask` 和三个 component（progress、TTC、driving direction），逐层 non-dominated sorting；同层先高 PDMS，差距不超过 0.01 point 时用几何多样性打破平局；不使用 GT 或 policy distance。
- Pareto + GT-distance：用 IL rollout-to-GT 的 command 条件 80% 分位作为主阈值，不足时 90%、95%，最后 filler；独立保存固定 80/90/95 的敏感性统计。
- PC-MTS：Pareto Front 1、保守可行、PDMS 不低于默认 IL reference、q∈[50,95)、四次 selection perturbation 中局部可行比例≥0.75；依次只放宽局部比例到0.5、q下界到40、q上界到99，之后才 filler。q 不参与“越大越好”的排序。

未在原需求中量化的“Pareto qualified”和“基本 quality”，在正式结果之前固定为 Front 1 与上述 reference-relative quality；这些具体定义会影响候选可得性，报告不将其包装成 PC-MTS 的唯一可能定义。

**用户授权的全覆盖扩展（PC-MTS + coverage）**：原规则与其192个非空池保持不变；仅对另外808个零父轨迹场景分层补充。Level 6取消q限制、保留Front 1/完整可行性/IL参考分数/局部可行性≥0.5；level 7保留完整可行性和局部≥0.5；level 8保留完整可行性；level 9保留NC=DAC=1。层内使用原PDMS与去冗余规则，从相同128库选择。新规则见 `configs/pc_mts_diagnostics/coverage_fallback.yaml`。这是知道前期结果后增加的探索性方案，在新增候选及其held-out评分前固定，没有通过反复调阈值追求预期结论。

后续表和图中的 `pc_mts` / “PC-MTS + coverage”均指此完整方法。原Boundary门限仍是[50,95)，因此coverage候选可能被如实判为OOD。

## 8. Candidate Pool Sanity Check

TABLE B（ALL-16）：

|method|candidate_pdms|d_GT|q_policy|core_fraction|boundary_fraction|ood_fraction|pool_diversity|filler_fraction|unique_fraction|construction_success|
|---|---|---|---|---|---|---|---|---|---|---|
|gt_distance|95.3764|0.2181|99.5247|0.0002|0.0224|0.9774|0.2114|0.0000|0.9988|1.0000|
|pareto|97.1218|0.7831|99.8319|0.0001|0.0074|0.9924|0.3771|0.0000|0.9998|1.0000|
|pc_mts|97.2901|0.7214|99.0137|0.0001|0.0579|0.9421|0.2966|0.0632|0.9888|1.0000|
|score|97.3267|0.6117|99.4677|0.0002|0.0258|0.9740|0.1860|0.0000|0.9998|1.0000|

UNIQUE-NON-FILL 与 construction success / 缺失场景数必须同时查看；不能将 filler 算作新的可学习模式。

TABLE B（UNIQUE-NON-FILL，全1000场景）：

|method|count|candidate_pdms|d_GT|q_policy|core_fraction|boundary_fraction|ood_fraction|pool_diversity|filler_fraction|unique_fraction|coverage_fraction|metric_valid_scenes|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|score|15.9970|97.3267|0.6117|99.4677|0.0002|0.0258|0.9740|0.1860|0.0000|1.0000|0.0000|1000|
|pareto|15.9970|97.1219|0.7831|99.8319|0.0001|0.0074|0.9924|0.3772|0.0000|1.0000|0.0000|1000|
|gt_distance|15.9810|95.3763|0.2182|99.5247|0.0002|0.0224|0.9774|0.2115|0.0000|1.0000|0.0000|1000|
|pc_mts|14.9810|97.2900|0.7214|99.0016|0.0001|0.0575|0.9424|0.3071|0.0000|1.0000|0.8080|967|

真实raw coverage候选计入UNIQUE-NON-FILL；另有QUALIFIED-NON-FILL排除所有额外回退候选。仅后者仍只在原规则可构建场景上有定义，详见完整覆盖补充报告。

## 9. Exp-1 Policy-relative Candidate Position

k=5；IL 自距离排除自身，candidate 距离取最近五条 IL rollout 的平均 ADE。q 为相对于 64 个 IL self-distance 的右连续经验 CDF；Core<50、Boundary∈[50,95)、OOD≥95。q 是局部几何参照的分位数，不是模型显式概率或经过校准的 OOD 检验。Fig-1a–e 与 candidate_records 保存位置、质量、来源、fallback、GT 距离和边界高质量质量占比。

## 10. Exp-2 Local Robustness

每个选中候选使用独立 held-out stream 生成 12 条平滑扰动：0.02/0.05/0.10/0.20m，各3个方向。相同轨迹在不同 pool 中共享相同扰动，selection perturbation 不复用。CVaR20 对最差20%经验质量精确加权（12条时为最差2条加第三差的0.4权重，再除2.4）。下降斜率定义为四档幅度对平均 PDMS 的线性回归斜率取负，单位 points/m；负值表示扰动后改善，不截成0。

TABLE C：

|method|original_pdms|local_mean|local_p10|local_cvar20|local_feasible_rate|local_drop|robustness_slope|
|---|---|---|---|---|---|---|---|
|gt_distance|95.3764|94.7081|93.3822|92.7096|0.9690|0.6682|8.7476|
|pareto|97.1218|96.7657|96.0687|95.6833|0.9650|0.3562|4.5983|
|pc_mts|97.2901|96.8940|96.1470|95.7076|0.9764|0.3960|5.2533|
|score|97.3267|96.7720|95.7569|95.2351|0.9601|0.5546|6.7748|

TABLE C（UNIQUE-NON-FILL，全1000场景）：

|method|count|original_pdms|local_mean|local_p10|local_cvar20|local_feasible_rate|local_drop|robustness_slope|metric_valid_scenes|
|---|---|---|---|---|---|---|---|---|---|
|score|15.9970|97.3267|96.7721|95.7569|95.2351|0.9601|0.5546|6.7748|1000|
|pareto|15.9970|97.1219|96.7657|96.0687|95.6833|0.9650|0.3561|4.5983|1000|
|gt_distance|15.9810|95.3763|94.7085|93.3842|92.7135|0.9690|0.6678|8.7405|1000|
|pc_mts|14.9810|97.2900|96.8886|96.1019|95.7057|0.9763|0.4014|5.2961|1000|

[完整覆盖及分层配对报告](../outputs/pc_mts_diagnostics/report/PC_MTS_FULL_COVERAGE.md)保存ALL-16、UNIQUE-NON-FILL、QUALIFIED-NON-FILL，以及原可构建192/额外覆盖808两个场景层的四方法同场景比较。

## 11. Exp-3 Rollout & GRPO Readiness

TABLE D：p_plus 与实际 Hit@8 使用标准 PDMS、Δ=0/1/2 point；64条严格按种子顺序分成8个G=8 group。Reference是额外采样的官方 IL 默认采样器 seed=0 单条轨迹，不是 IL Oracle@64。Oracle@K 是离线候选上界，不是部署策略得分。

|checkpoint|feasible_rate|mean_pdms|oracle_64|oracle_mean_gap|p_plus_0|p_plus_1|p_plus_2|hit8_0|hit8_1|hit8_2|all_infeasible_group_rate|mostly_infeasible_group_rate|unsafe_positive_advantage|feasible_positive_advantage|D_all|D_feasible|D_positive|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|official_il|0.9394|92.1547|94.6682|2.5135|0.4441|0.1037|0.0341|0.6192|0.2846|0.1334|0.0423|0.0520|0.0311|0.9689|0.1448|0.1432|0.1186|
|mts_8692|0.9676|93.7139|95.0598|1.3459|0.3155|0.1823|0.1183|0.3576|0.2198|0.1459|0.0266|0.0309|0.0159|0.9841|0.0841|0.0840|0.0914|
|mts_8751|0.9676|94.1829|95.9507|1.7678|0.3764|0.2235|0.1375|0.4289|0.2712|0.1802|0.0265|0.0297|0.0186|0.9814|0.0835|0.0837|0.1900|
|grpo_9041|0.9464|95.0581|95.9391|0.8810|0.5221|0.4084|0.2730|0.5579|0.4715|0.3299|0.0439|0.0510|0.0389|0.9611|0.1705|0.1692|0.1717|
|apr_9145|0.9568|95.3553|96.2440|0.8887|0.5131|0.4386|0.3456|0.5376|0.4800|0.3957|0.0369|0.0406|0.0340|0.9660|0.1831|0.1815|0.1949|

Unsafe Positive Advantage 使用从历史 GRPO 源文件提取并执行的 advantage 代码：group mean/std，PyTorch unbiased std +1e-8，实际默认 quantile clipping 0/1。历史训练 reward 使用 progress/TTC/comfort=10/5/2；统一展示的标准 PDMS 为5/5/2。因此 advantage 病态诊断与 PDMS 质量采用各自正确口径，原始代码和校验见 `grpo_advantage_source.json`。

保守 feasible 明确定义为 NC=DAC=TTC=DDC=1；hard failure 为 NC<1 或 DAC<1，两者不是互补。Comfort 单独作为质量指标。每个场景先计算 unsafe/positive 比例，正 advantage 分母为零时记缺失，同时保存分子/分母；不靠跨场景样本堆叠制造显著性。

Coverage共新增12,928条真实raw候选：level 6/7/8/9分别为5,793/6,869/1/265条，没有新增duplicate filler。Level 9只满足NC=DAC=1，不能称为满足完整可行性；局部评测按真实TTC/DDC等结果计分。

## 12. Statistical Tests

主比较按 scene 配对，3000 次场景 bootstrap；报告均值差、中位数差、95% CI、Wilcoxon signed-rank p、matched rank-biserial effect size、有效场景数。未将 64 次 rollout 当作独立场景。区间与p值为逐项结果，未作多重比较校正；同日志邻近场景仍存在相关性，是解释边界。相关系数为描述性，详见 candidate_correlations.json。

## 13. Hypothesis Validation

- **H1（SFT宽、GRPO/APR尖）：不支持。** 默认随机DDIM协议下，两个MTS的两两ADE比IL分别低0.0607m、0.0613m，95% CI均低于0；GRPO/APR分别高0.0257m [0.0247,0.0267]、0.0383m [0.0359,0.0407]。Spread-AUC和R90也支持这种方向。GRPO/APR相对IL的center shift约0.501/0.809m，中心移动的尺度明显大于宽度差。不能把这一结果外推为所有GRPO算法的定律。
- **H2（Score单点高分但更脆弱）：部分支持。** Score的原始PDMS最高是top-score选择定义的直接结果。与Pareto相比，Score局部平均PDMS没有可区分差异（Pareto−Score = −0.0064 point，95% CI [−0.1616,0.1412]）；Pareto局部CVaR20高0.4482 point [0.0895,0.8197]，Local Drop低0.1985 point [0.0912,0.3135]。不能把Score整体称为不稳的reward spike；Score的pool diversity反而小于Pareto，OOD比例也更低。
- **H3（Pareto改善多目标平衡，但不保证policy compatibility）：部分支持。** Pareto提高平均driving-direction component（0.9837 vs Score 0.9766），但TTC略降（0.995 vs 1.0），不是所有component同时改善。其局部尾部指标更好，但OOD比例为99.24%，高于Score的97.40%，不支持“Pareto自然更policy-near”。
- **H4（GT-distance降低OOD且大量进入Core）：部分支持，Core预测不支持。** GT约束将平均d_GT降到0.218m，相对其母方法Pareto，OOD降低1.506个百分点，95% CI [1.056,2.000]个百分点；但Core仅0.0188%，OOD仍97.74%，与Score的OOD差异CI包含0。固定80/90/95分位敏感性中，OOD约97.74/98.64/99.10%，没有观察到大量Core候选。该q定义下大量样本已饱和为100%，因此也不能仅凭q把所有OOD程度视为相同。
- **H5（PC-MTS更多Boundary、局部更稳）：全1000场景的扩展方法部分支持。** PC-MTS + coverage的Boundary占比5.7875%，高于Score 2.5813%、Pareto 0.7438%、GT-distance 2.2438%；对Pareto差值为5.0438个百分点，95% CI [3.8936,6.2377]个百分点。Local Mean比Pareto高0.1283 point [0.0082,0.2592]，Local Feasible高1.1464个百分点 [0.6995,1.6359]个百分点；但CVaR20只高0.0243 point [−0.2007,0.2402]，没有可区分差异。对Score的Local Mean/CVaR20/Local Feasible分别改善0.1220 point、0.4724 point、1.6375个百分点，区间均高于0。**不支持“绝大多数候选集中在Boundary”或“尾部全面优于Pareto”**：OOD仍94.2063%。808/1000场景使用取消policy约束等额外回退，这些改进可能来自可行性优先筛选，不能单独归因于policy boundary机制。原规则的192场景配对分析保留在条件性补充报告中；其局部Mean/CVaR20并无优于Pareto的证据。
- **H6（GRPO readiness不能只看Mean/Oracle）：数据支持联合分析的必要性，尚非后续训练效果的因果验证。** 两个MTS可行率约96.76%，高于GRPO/APR；但Hit@8(Δ=1)为21.98%/27.13%，低于GRPO/APR的47.15%/48.00%。MTS的Unsafe Positive Advantage约1.59%/1.86%，低于GRPO/APR的3.89%/3.40%。这些指标刻画了不同取舍。APR与GRPO的Mean PDMS差值0.297 point和Hit@8差值0.0085的95% CI都包含0，不能宣称在这1000场景上APR已显著优于GRPO。

D_positive的均值只在至少两条可行正提升轨迹的场景上定义：IL/MTS86.92/MTS87.51/GRPO/APR分别为623/380/458/566/543场景。跨模型对比应读取配对交集统计；例如APR−GRPO在529个有效配对场景上的D_positive差为0.0204m [0.0169,0.0241]，不能用不同场景子集的两列均值直接推导同一效应。

## 14. Limitations

- 这是一组已训练 checkpoint 的 Navtrain 机制诊断，不是独立测试集泛化结论。历史 APR 曾使用 Navtest 进行模型选择。
- MTS 权重与 IL/GRPO 的内部归一化、adapter 和初始化不同，五模型不是严格训练算法因果消融；官方 IL→原版 GRPO 的初始化关系有训练 manifest 支持。
- 64样本的kNN经验 support 可能很窄，q≥95并不自动证明真实策略概率为零。PC-MTS 缺少父轨迹时不能从无中生成合格监督。
- 小幅扰动衡量指定局部邻域的敏感性，不能证明现实驾驶安全。
- GRPO readiness 是机会、可行性与advantage结构诊断，不是重新训练 GRPO 的最终性能保证。

## 15. Conclusions

五个真实checkpoint在固定1000个Navtrain场景上的320,000条rollout、标准PDMS、40,000个真实G=8组的GRPO诊断全部完成。四方法每场景均构建16条候选，合计64,000条；768,000条独立held-out扰动全部评分并通过来源、形状、有限值与缓存哈希校验。没有删除场景，也没有按APR分数选场景。

默认随机DDIM协议下，两个MTS权重的分布比IL更窄，GRPO/APR比IL略宽且中心位移更大，不支持“SFT更宽、RL更尖”的简单假设。该结论限于这些真实权重及采样协议，不能解释成严格训练算法因果消融。

在全1000场景中，PC-MTS + coverage的Boundary占比最高，局部平均PDMS及局部可行率相对Pareto有所改善；其CVaR20与Pareto没有显著可区分差异。相对Score，局部平均、尾部与可行率均改善。与此同时，OOD仍占94.21%，80.8%的场景需要额外覆盖回退，因此这不是纯粹PC-MTS Boundary机制已经成功的证明。结果支持选择质量、局部稳健性和可达性需要联合考虑，不能写成所有预设猜想都得到验证。

原规则不足和新规则的作用分别记录：192个原非空PC-MTS池保持字节一致；808个原零父轨迹场景从相同128条共同库中分级选择，新增12,928条均为真实raw候选，没有新增duplicate filler。原192场景已有的1,011条tiny filler仍保留，最终全池filler比例6.3188%。

所有核心实验已完成；独立分析提交包含实现、冻结配置与本报告。完整覆盖报告、ALL-16/UNIQUE-NON-FILL/QUALIFIED-NON-FILL和分层配对统计位于outputs/pc_mts_diagnostics/report/PC_MTS_FULL_COVERAGE.md与metrics/coverage。


计算时长：正式观测特征、五模型rollout和PDM评分合计1173.59秒（19.56分钟）；M1–M3候选与扰动pipeline合计427.39秒（7.12分钟），不含此前资产审计、smoke、50场景benchmark和报告整理。

[PNG/PDF图目录](../outputs/pc_mts_diagnostics/figures/) · [CSV/Parquet与配对统计](../outputs/pc_mts_diagnostics/metrics/) · [冻结场景](../outputs/pc_mts_diagnostics/manifests/scenes_1000.json) · [完整性校验](../outputs/pc_mts_diagnostics/manifests/full_validation.json) · [已解决的覆盖规则说明](../outputs/pc_mts_diagnostics/report/OPEN_DECISIONS.md)

1000场景PC-MTS覆盖追加pipeline耗时270.42秒（4.51分钟）；复用已有候选与评分缓存。含全四方法重新分析、图表和覆盖校验；不含之前的smoke与人工报告整理。原192场景条件性pipeline另耗时97.70秒。

[实际8-GPU与CPU执行命令](../outputs/pc_mts_diagnostics/report/EXECUTED_COMMANDS.md) · [逐项完成验收](../outputs/pc_mts_diagnostics/manifests/completion_audit.json)
