# IL / RL / PSI：1000场景趋势假设核查

这是对已知总体结果的回顾性分析，不能称为预注册的新实验。固定原1000 NAVTRAIN场景，未根据结果替换或删除场景；没有重新训练或推理。每模型、每采样器64次（4组×16），共复用384,000条真实采样与统一NAVSIM评分。三个checkpoint对应IL→original GRPO、IL→PSI candidate SFT两个分支，不是三个连续阶段。PSI后续GRPO权重缺失。

## 数据集与口径

共有1000场景、512个log；直行634、左转251、右转115。835是PSI/A5/V6共同训练场景，165是PSI验证场景；不宣称untouched Navtest泛化。所有数据集都保留完整1000场景，命令/原训练身份只作公开分层。

Pairwise ADE按组内16条计算，再平均4组和场景。中心为64条XY均值轨迹，位移参照同一采样器下官方IL。Feasible要求NC=DAC=TTC=DDC=1，表中单位百分比，配对差为百分点。PDMS单位0–100。

## 普通推理（Full1000）

| Policy stage | pairwise_ADE | center_shift_m | feasible_rate | mean_PDMS |
| --- | --- | --- | --- | --- |
| Initial GT-IL | 0.1449 | 0.0000 | 93.9516 | 92.1767 |
| After original GRPO | 0.1703 | 0.4991 | 94.6047 | 95.0568 |
| After PSI candidate SFT | 0.3070 | 0.4411 | 91.8719 | 92.6015 |

## 真实GRPO采样（Full1000）

| Policy stage | pairwise_ADE | center_shift_m | feasible_rate | mean_PDMS |
| --- | --- | --- | --- | --- |
| Initial GT-IL | 1.9955 | 0.0000 | 69.0859 | 69.2287 |
| After original GRPO | 1.9951 | 0.4982 | 77.0484 | 79.0574 |
| After PSI candidate SFT | 2.2588 | 0.4998 | 62.3641 | 65.7064 |

## 场景配对差值

3000次scene-paired bootstrap；log-cluster区间及正/负/相等scene比例另存。没有把候选数当作独立场景数。以下CI是描述性区间，不从多指标中挑显著项定义整体成功。

| protocol | model | metric | effect_95CI |
| --- | --- | --- | --- |
| eval | original_grpo_11970 | pairwise_ADE | +0.0254 [+0.0244, +0.0264] |
| eval | original_grpo_11970 | center_shift_m | +0.4991 [+0.4871, +0.5122] |
| eval | original_grpo_11970 | feasible_rate | +0.6531 [-0.4485, +1.7236] |
| eval | original_grpo_11970 | mean_PDMS | +2.8801 [+2.1964, +3.5748] |
| eval | psi_sft | pairwise_ADE | +0.1621 [+0.1589, +0.1655] |
| eval | psi_sft | center_shift_m | +0.4411 [+0.4254, +0.4573] |
| eval | psi_sft | feasible_rate | -2.0797 [-3.0610, -1.1109] |
| eval | psi_sft | mean_PDMS | +0.4248 [-0.2793, +1.1445] |
| native_grpo | original_grpo_11970 | pairwise_ADE | -0.0005 [-0.0016, +0.0007] |
| native_grpo | original_grpo_11970 | center_shift_m | +0.4982 [+0.4864, +0.5101] |
| native_grpo | original_grpo_11970 | feasible_rate | +7.9625 [+7.0750, +8.8954] |
| native_grpo | original_grpo_11970 | mean_PDMS | +9.8288 [+9.1390, +10.5788] |
| native_grpo | psi_sft | pairwise_ADE | +0.2632 [+0.2596, +0.2665] |
| native_grpo | psi_sft | center_shift_m | +0.4998 [+0.4851, +0.5150] |
| native_grpo | psi_sft | feasible_rate | -6.7219 [-7.3766, -6.0266] |
| native_grpo | psi_sft | mean_PDMS | -3.5223 [-4.0544, -2.9914] |

## 独立采样组复核

固定group0/1为A、group2/3为B，每批每scene32条，二者随机流独立但场景相同。此前总体分析已包含这些样本，因此这只是采样稳定性检查，不是新盲测或独立场景验证。所有A/B绝对值及配对区间完整保存在absolute_statistics.csv和paired_deltas.csv；没有根据哪一半有利选择结果。本次PSI相对IL的可行率下降在两半均复现：eval A/B分别−2.150/−2.009个百分点，native A/B分别−6.919/−6.525个百分点。PSI的eval小幅均分增益在两半CI均包含0；native均分下降在两半CI均不包含0。这支持采样稳定性，不增加独立场景数。

## 是否增加有限IL参考库较少覆盖的可行轨迹

对目标A的32条轨迹，与IL B的32条参考比较，B对A同理。若到所有参考轨迹的最小ADE>0.5米且目标可行，则计入；分母始终64，包括不安全轨迹。IL自身两半互查给出有限参考库的非覆盖基线。0.25/1.0米敏感性全部公开。

| protocol | model | feasible_noncoverage_mass_percent | initial_finite_bank_null_percent | mean | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| eval | official_il | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| eval | original_grpo_11970 | 26.6484 | 0.0 | 26.6484 | 24.2453 | 29.2516 |
| eval | psi_sft | 21.8141 | 0.0 | 21.8141 | 19.903 | 23.7973 |
| native_grpo | official_il | 69.0859 | 69.0859 | 0.0 | 0.0 | 0.0 |
| native_grpo | original_grpo_11970 | 77.0484 | 69.0859 | 7.9625 | 7.0717 | 8.8766 |
| native_grpo | psi_sft | 62.3641 | 69.0859 | -6.7219 | -7.3563 | -6.1108 |

这是几何有限采样非覆盖，不是语义mode数量；高维强噪声下即使同一policy的两批样本也可能互不接近。必须对照IL自身基线，不能仅凭PSI数值大就宣称新模式。mean/CI列是目标相对该基线的百分点差。本次native在0.5米下，三模型几何非覆盖率均为100%，该指标已饱和，feasible_noncoverage_mass完全退化为可行率，不能提供额外的support扩张证据。eval下，PSI有21.814%的采样同时可行且不落在IL参考库的0.5米邻域内；original RL为26.648%，IL自身为0%。这不能证明新增语义mode，且这种几何变化也不是PSI独有。

## 对原假说的判定

- RL提高质量：SUPPORTED，两个采样协议均提高平均PDMS。
- RL可行率小幅上升：普通推理点估计小幅上升但CI包含0；native上升约7.96个百分点，不能把它描述成同一口径下的小幅变化。
- RL宽度持平或略降：native与该描述相容；普通推理ADE明确增加。不能混用普通推理的安全率与native的宽度拼成一行。
- PSI平均分小幅上升：普通推理点估计上升但CI包含0；native明确下降。PARTIALLY SUPPORTED至多适用于普通推理的描述性均值。
- PSI可行率明显上升：NOT SUPPORTED，两个协议均下降。
- PSI中心位移明显大于RL：NOT SUPPORTED，普通推理更小，native与RL接近。
- PSI宽度增加：SUPPORTED，两个协议都增加。
- RL仅在既有behavior modes内refinement、PSI创造新的feasible语义modes：UNTESTED。中心与宽度不能单独识别mode身份，有限参考库几何覆盖只提供补充证据。

现有数据不支持用户提出的完整对比叙事。不应根据目标趋势另选1000个成功场景或改可行性定义。如果进一步采集新的随机1000场景，应先冻结新manifest并完整报告结果；不能把结果最好的一批替代本报告。

数据文件：scenes_1000.csv、scene_dataset.parquet、group_dataset.parquet、crossfit_geometric_coverage.parquet、absolute_statistics.csv、paired_deltas.csv、all_prespecified_strata.csv、finite_bank_coverage_summary.csv。图trend_effects保存PNG/PDF/SVG及源CSV。源模型身份、轨迹缓存路径/hash和输入表hash可追溯；旧结果不变。
