from common import *
def table(f,columns=None):
 if columns:f=f[columns]
 def cell(v):return (f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])
def loadcsv(n):return pd.read_csv(OUT/n)
models=read(OUT/'manifests/models.json');gate=read(OUT/'audits/signal_gate.json');status=read(OUT/'audits/TINY_STEP1_COMPLETE.json')
ci=loadcsv('checkpoint_paired_ci.csv');ci=ci[(ci.split=='confirmation')&ci.metric.isin(['PDMS','NC','DAC','TTC','EP','raw_progress','end_speed','zero_score'])]
credit=loadcsv('credit_summary.csv');credit=credit[(credit.split=='confirmation')&credit.metric.isin(['all_infeasible','mixed_feasible','all_feasible','all_full_pdms','all_ep_saturated','all_components_identical','ep_only_components','ep_only_identical_gates','information_active','positive_weighted_raw_progress_shift','positive_weighted_end_speed_shift'])]
cov=loadcsv('native_vs_constructed_coverage.csv');cov=cov[(cov.model=='a5_sft')&(cov.protocol=='native_grpo')&(cov.split=='confirmation')]
sel=loadcsv('selector_comparisons.csv');seld=sel[(sel.split=='confirmation')&(sel.selector=='D')&sel.metric.isin(['evaluation_risk','evaluation_NC_fail','evaluation_DAC_fail','evaluation_TTC_fail','evaluation_hard_fail','nominal_PDMS','nominal_raw_progress','nominal_end_speed','evaluation_raw_progress'])]
frames=pd.read_parquet(OUT/'perturbation_results.parquet');events=frames.groupby(['split','stream']).agg(draws=('token','count'),risk_events=('risk','sum'),NC_events=('NC_fail','sum'),DAC_events=('DAC_fail','sum'),TTC_events=('TTC_fail','sum')).reset_index()
frozen=read(OUT/'manifests/frozen.json')
text=f'''# GRPO 信用与安全风险快速核验

日期：2026-09-21。分析分支 `analysis/grpo-credit-risk-quickcheck-20260921`；工作目录 `{WORK}`。
统一产物：`{OUT}`。初始代码提交 `{frozen['base_commit']}`。最终提交通过 `git log -1` 查看。

## 当前结论与实验状态

**值得继续验证风险标签和极短信用更新；不具备宣称新 GRPO 方法有效的依据。**

|阶段|状态|实际执行|
|---|---|---|
|真实失败链路、信用分析 A/B/C|PASS（只读）|512 场景、两个真实 checkpoint、65,536 条正式轨迹；未更新原模型|
|等进度候选与独立扰动|WITHIN_GROUP_SIGNAL|确认集 149 场景、112 个 log；独立风险比较通过工程门槛|
|两 seed × 四训练臂|INVALID_STOPPED|各执行一次 optimizer 更新，共 8 次；随后因实现的可训练参数身份错误停止|
|step8、更新后训练/部署评测、范数匹配 A|NOT_RUN|不补造分数，不将失效尝试用于方法比较|

第二阶段的离线选择器 D 相对名义 PDMS 选择器 A，独立扰动风险差 **−0.398 个百分点**，95% log-cluster 区间 **[−0.771, −0.088] 个百分点**；名义 PDMS 差 **−0.054 个百分点**，区间 [−0.071, −0.039]。相对随机选择也通过验证。这里只有风险标签分辨率证据；**离线选择器 D 与第三阶段训练臂 D 不是同一项实验**。

最主要未解决问题是：在严格保持归档可训练参数集合的实现中，这个 pairwise surrogate 能否改善实际更新方向，且迁移到部署输出。下一步唯一必要实验：修正身份合同后，复用冻结场景和共享 rollout，完成一次可审计的单步四臂对照，先保存更新权重并验证冻结参数不变；在这之前不进入第 8 步、不搜索超参数、不做 SFT 或长程 GRPO。

## 1. 真实链路与冻结身份

优先检查了当前 SimScale 任务、近期实际配置/日志和历史 manifest。近期任务并未提供更明确的同类失败 GRPO 链路，故按指令采用 **A5 epoch155 → 正式 LFP-GRPO r4 固定 step300**。不是 PSI-SFT 起点，也不是成功的官方 IL 链路。step300 在本轮评分之前固定，没有择优 checkpoint。

|对象|真实文件|SHA256|
|---|---|---|
|SFT 父本|`{models['a5_sft']['checkpoint_path']}`|`{models['a5_sft']['sha256']}`|
|早期 GRPO|`{models['a5_grpo_300']['checkpoint_path']}`|`{models['a5_grpo_300']['sha256']}`|

VLM：`{ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B`。FS standard-v2 统计、PTA/DiT 配置、全部顶层 VLM 权重/配置、reference cache 与 scorer 源码哈希见 `manifests/input_files.json`。归档 runtime commit `f350e56408584c0d375b1ff10744da2750442580`；runtime 在 `{RUNTIME}` 只读复用，未覆盖当前分支代码。

归档真实调用链是 `forward_grpo → forward_lfp_grpo → compute_lfp_advantages`。不是根据 `reward_mode` 或当前 dataclass 猜测。最终映射后的配置、函数原文及行号分别在 `manifests/resolved_runtime.json`、`source_snapshots/callable_*.py`。

实际 G=16；DDIM 5 步，训练噪声未使用 legacy 高斯裁剪，sampling/logprob std floor 均 0.04；部署 `get_action(deterministic=False)` 使用归档部署参数。BC=0、exact reverse-transition reference KL=0.005；reference policy 为同一个 A5 epoch155。训练 scalar 为 `NC*DAC*(10*EP+5*TTC+2*Comfort)/17`，官方 PDMS 为 `NC*DAC*(5*EP+5*TTC+2*Comfort)/12`。DDC 不进入这些加权分母，但进入实际 LFP 的 GT-relative 可行资格。

实际信用顺序：scalar → `0.2*clip((scalar-reference_scalar)/0.05,-1,1)` → 可行/EP-reference/TTC-reference/Pareto 资格 → 可行数至少 2 时按可行候选均值中心化，否则按全组均值 → 全局 masked moments、std floor 0.05 → 资格限制 → clip±3 → detach。TTC guard 在实际 r4 开启，不能沿用归档类的 False 默认值。该版本没有后来版本的 PDMS-span 低信息清零门控。当前 probe 的 scene/group loss 权重均为 1；历史 step300 属于 curriculum 首 epoch uniform warmup。

原历史 hidden cache 与 A5 未提交 dirty diff 已无法完整恢复。使用归档 BF16 VLM 从相同四帧已观测输入重新构建特征，归档 builder 将输出转为 FP32 存储；动作头 native 采样使用历史 `16-mixed` 对应 FP16 autocast，部署采样使用仓库 FP32 标准。原先 FP32 诊断轨迹没有冒充训练精度缓存。**这是重建身份一致的探针，不能声称历史运行逐位重现。**

已读取指定的两份历史报告、历史分支的 `primary.yaml` 和研究总结；冻结副本在 `source_snapshots/`。官方 IL→original GRPO 成功链路只保留既有报告作为描述性对照，没有额外采样，也没有将其 G16 诊断结果改称 G8 历史优势。

## 2. 样本、等价性与只读边界

512 个场景来自已有 outcome-unconditioned、command 分层的 Navtrain 1000 场景框架，再按 log hash 隔离、按指令配额 hash 抽取。校准 256 场景/162 log；确认 256 场景/174 log，无重叠 log。每份配额 straight163/left64/right29。16-scene smoke 是校准集内固定子集。未用 navtest 调阈值。

确认集只能称为“本次探针未用于更新/调参的确认集”；它来自历史 Navtrain，可能参加过原 SFT/GRPO，**不是从未参加训练的泛化集**。第三阶段仅选校准集训练。

16-scene smoke 检查通过：官方单条/批量评分、standard/fast scorer、添加同批候选后的评分不变、PDM baseline 与 EP 分母一致；容差预设 1e−8，未放宽。native 实际 forward 入口捕获和直接 native sampler 的相同链复放、重复采样误差为 0；两个动作头与 VLM 参数/buffer 前后 SHA 一致。正式 512 场景评分无异常。`audits/SMOKE_COMPLETE.json`、`scoring_formal.json` 与最终输入文件复核提供证据。

每 checkpoint 每场景 native 两个独立 G16 组；另取两个含 16 次真实部署 sampler 调用语义的随机输出池。部署均值不是 oracle@K，也不是部署时执行 GRPO。跨 checkpoint 共同随机数仅用于同协议场景配对，不代表相同动作。

新优势使用固定 hash 的 64 场景归一化批次，分别处理两组；保留归档全局 moments 语义，但不是恢复历史 minibatch、历史 curriculum 或历史训练优势。逐轨迹 `trajectory_credit.parquet` 保存每阶段优势、资格、reference 分项、归一化批次及权重。所有原始评分保存为 0–1，本文只有明确标为“百分点”的展示才乘 100。

原始道路进度来自真实 simulator 状态、官方 scorer ego 中心线投影；保存 raw/gated progress、PDM baseline raw/gated progress、每候选 pairwise EP 分母及低进度分支。不是 ego-x、欧氏终点距离、折线长或 capped EP。并保存末速度、速度峰值、加速度、jerk 等；`ttc_infraction_time` 仅是违规时刻，不当作连续最小 TTC。

## 3. 分析 A：正信用集中在哪里

正信用质量严格定义为 `sum(max(A,0))`，**不是参数梯度贡献，更不是因果归因**。

以下为确认集按场景等权的组统计；0/1 指标的 mean 是比例，位移列单位分别为米、米/秒，位移只在有正信用组中定义。完整 log 数和置信区间见 CSV。

{table(credit,['model','protocol','metric','mean','low','high'])}

“其他评分分项相同”与“所有实际门控状态也相同”必须区分。全 512 场景中，SFT native 的 component-only EP 组贡献 85.79% 正信用质量，部署为 84.88%；step300 则分别为 38.88%、53.67%。但严格要求其他分项及 **全部实际门控** 相同时，正信用质量贡献是 **0**。EP 的变化经常同时改变 Pareto/progress 门控，不能把前者冒称后者。

相对 coherent reference，SFT 确认集 native 正信用轨迹中，没有发现“EP 上升同时 TTC/Comfort 下降”的样本；这是实际 TTC guard 与资格门控下的结果。该检查不等价于连续安全裕度不下降，也不证明共享参数更新不会影响其他轨迹。

![正信用来源](../outputs/grpo_credit_risk_quickcheck/positive_credit_sources.png)

绘图原始数据 `positive_credit_plot_data.csv`。只有乘积门控相同的可行子集，才在 `linear_feasible_scalar_decomposition.parquet` 精确分解中心化 scalar 的 EP/TTC/Comfort；非线性 reference margin 单列，不强行分摊线性比例。

## 4. 分析 B：EP 比较信息移除

预定义 EP 常量为 1，对同场景候选与对应 coherent reference 的 EP 一致替换，重建训练 scalar 及非线性 reference margin。官方原始 PDMS 和轨迹从未覆盖。

- fixed 影子：保留原资格/Pareto门控和原 global mean/std，只重建 EP-neutral 分数链。回答分数差及 reference margin 的直接来源；残留原门控中可能含有的 EP 信息被明确保留。
- full 影子：按归档实现完整重算 scalar、margin、资格、Pareto、中心化和归一化。回答计算链如何响应，不是训练结果。

确认集 SFT native 原有效双向比较组 298/512（58.20%）；fixed 剩 28/512（5.47%），full 剩 14/512（2.73%）。full 下原正优势消失 82.73%，正信用质量下降 71.11%；fixed 下正质量下降 89.74%。两者差异包含归一化尺度与门控变化，不能当成奖励线性贡献占比。

step300 native full 下有效比较组从 275 降到 62，但正信用质量反而增加约 0.38%；这是归一化/非线性链响应的具体反例，说明不能把“移除 EP 后仍有多少总正信用”简单等同于 EP 因果贡献。所有正优势消失/翻转、排序与活跃组统计见 `shadow_fixed_effects.csv`、`shadow_full_effects.csv` 和逐组 parquet。1e−8 正信用计数阈值预先冻结，接近零的 float32 优势计数可能受归一化舍入影响；质量统计保留其实际极小权重，不将数量等同于强信用。

## 5. 分析 C：EP 主导是否就是安全退化

固定确认场景 step300−SFT 的场景等权差值及 log-cluster 95% 区间如下。评分均为原始 0–1 差值；raw_progress 为米，end_speed 为米/秒。

{table(ci,['protocol','metric','mean','low','high','scenes','logs'])}

本次确实重现“EP/道路进度上升、TTC 与安全尾部退化”，并非以成功链替代。训练 sampler 与部署 sampler 的程度不同，两种协议都存在该现象。确认集部署 TTC 差为 −3.235 个百分点，区间 [−5.040,−1.639]；native 为 −3.967 个百分点，区间 [−5.709,−2.366]。

{table(loadcsv('safety_transition_counts.csv'))}

上表是相同 scene/group/candidate 随机抽样索引下的**输出对**计数，不是独立场景样本量，也不代表跨模型同一个行为。完整数据保留 EP 上升且安全改善的反例：确认集 native 48 对、部署 33 对。没有只展示退化例子。

安全退化并未更集中于 SFT 的 EP-component-only 正信用场景：全 512 场景 native，有这类正信用的场景平均 failure 增加约 2.81 个百分点，无此类信用的场景约 4.12 个百分点；部署对应约 2.42 与 3.23。分 split、场景数及区间见 `ep_credit_association_summary.csv`。这里只是关联，未控制全部混杂，也不能用它排除通过共享参数发生的安全迁移。

![训练与部署协议变化](../outputs/grpo_credit_risk_quickcheck/train_deploy_changes.png)

数据 `protocol_change_plot_data.csv`。现有证据支持 **高分区域奖励分辨率不足确实存在**，也显示 **采样协议影响尾部表现**；失败链跨协议的共同退化使“仅仅协议不一致”不足以解释。共享参数更新后的安全迁移是尚待有效更新对照验证的解释，不能从这些离线关联断言原因。

## 6. 等进度配对与独立安全验证

原生 G16 组内为主分析。同场景、同 checkpoint A5 SFT、同 native 协议，NC=DAC=TTC=1、实际 GT-relative DDC 资格、两条 PDMS≥0.95、差≤0.01、**simulator 后原始道路进度差≤0.5m**。PDM 低进度特殊分支单列，排除出正常行驶主分析。每场景固定 hash 选最多两个不重复候选对，不使用之后的风险结果筛选。

{table(cov,['scope','tolerance_m','scenes','matched_scene','matched_groups','total_groups','high_score_groups','matched_high_score_groups','all_group_coverage','high_score_group_coverage'])}

确认集主阈值：149/256 场景、293/512 原生组可配对；高分组定义为至少两条符合名义安全且 PDMS≥0.95 的候选，覆盖293/344。合并两组后为152/256，只代表扩大候选预算，不能改称原生信号。0.25m/1m仅报告预声明的覆盖率敏感性；未据此替换0.5m主阈值，也未扩大风险评价候选预算。未匹配原因和低进度配对计数在 `unmatched_reasons.csv`。

实际选取校准302对/158场景、确认294对/149场景，共1192条不同候选。确认集中100个场景存在所选两条都PDMS=1的对子，其独立风险差 D−A 为−0.531个百分点，区间[−1.094,−0.119]。进度匹配不控制中间速度/减速时序等全部混杂。

名义连续描述量使用同时间、同车辆尺寸、同坐标的官方 ego footprint 与参与者 footprint；排除红灯伪对象，记录最近对象、时刻、几何重叠与初始 collision-ignore 状态。可驾驶边界描述量用官方相关语义区域的并集，full footprint 越界给负裕度，并单列越界标记。它比官方角点资格更细，校准中出现约−3mm的小幅 footprint 越界但名义DAC=1，不能将这直接称为官方违规。对象最近距离不是完整风险；16.50%的确认候选最小对象距离出现在共同初始时刻，可能无法区分行为。

扰动基函数 `16u²(1−u)²` 起终点位置及一阶导数为零，沿局部路径切向/法向移动，用 Hermite 基路径与解析导数重建 heading。保持初始状态、车辆尺寸、交通参与者、地图及 PDM baseline 不变；克隆评分所需缓存，不覆盖原 metric cache。零扰动评分和道路进度误差均为0。

校准冻结较小幅度：横向最大0.05m、纵向最大0.20m。校准9664/9664次通过预设数值/动力学检查；确认9408/9408次也通过，**没有因事件稀少而增大幅度**。幅度是工程邻域探针，不是真实误差分布估计。每条8次selection与8次evaluation，种子流不重叠，同场景候选使用相同扰动参数，manifest逐场景保存全部参数。无生成失败、动力学失败或scorer数值异常；接口区分这些错误，不以碰撞替代。

{table(events)}

确认evaluation的42个风险事件只分布在8个场景；不是4704个独立场景。8次零失败不能表述为真实失败概率0。日志里的周车未来不随新ego动作作真实反应；结果只能称“轨迹邻域安全敏感性”，不是闭环事故概率或安全保证。

四选择器：A按名义PDMS，平分精确平均两种选择期望；B均匀随机；C按预定义最小几何/道路裕度（上限10m）；D按selection风险，风险相同时精确平均且不宣称有偏好。确认294对中只有12对有不同selection风险；其余如实保留平局。所有主比较只用独立evaluation。

{table(seld,['comparator','metric','mean','low','high','scenes','logs'])}

几何裕度选择 C 相对 A 的独立风险差为−0.126个百分点，区间[−0.471,+0.217]，**未通过独立改善证据门槛**。D相对A的原道路进度差−0.0321m，末速度差−0.0152m/s，都随安全收益一起报告；不是严格等效率保证，也不能把略慢说成无代价。评价扰动后的实际道路进度也保留，避免将EP分母变化冒称安全收益。

![名义分数与独立风险配对](../outputs/grpo_credit_risk_quickcheck/matched_pdms_independent_risk.png)

数据 `matched_pair_plot_data.csv`。选择器逐对、逐场景、分项和区间在 `selector_pair_results.parquet`、`selector_paired_differences.csv`、`selector_comparisons.csv`。统计先场景内平均，再场景等权；3000次按log整体重采样，保持对照配对。

![不同候选预算覆盖率](../outputs/grpo_credit_risk_quickcheck/signal_coverage.png)

数据 `signal_coverage_plot_data.csv`。外部候选触发条件为原生确认场景<50或高分组覆盖<10%；本轮均不满足，因此 **外部构造NOT_RUN，生成0条**，不是外部构造没有信号。预冻结的64+64场景名单仍保留。没有外部候选或扰动轨迹进入GRPO loss。

## 7. 极短更新：实现、失败和边界

仅在上述只读身份/评分检查与WITHIN_GROUP_SIGNAL通过后启动。两个seed为2026092111/2026092112；起点同一A5 SFT；四臂A原始、B训练scalar的EP分子系数减半、C原始加非等进度risk credit、D再要求道路进度≤0.5m。

B保留分母17，因此没有暗中放大TTC/Comfort；同一选定reference轨迹的scalar也用5/5/2分子除17重算，非线性margin、中心化、归一化随之重算；官方PDMS、原EP与实际安全/进度资格保持真实。完整变化冻结在 `manifests/tiny_update.json`。

C/D为用户指定的 `B_i=sum_j Mij(c_j-c_i)/(G-1)`，lambda1、对称mask、零对角；D使用真实道路进度，固定G−1分母；风险相同或无匹配时零增量，不做风险差单位方差化。最后再次应用原正信用资格和clip，无其后再中心化。旧scalar全平分不会自动清零新风险信息。D的动作依赖mask是pairwise surrogate，不宣称无偏GRPO改写。

有效batch64、G16、lr1e−5、AdamW(.9,.95)、weight decay1e−4、clip1、BC0、KL.005。各臂单卡累计64场景梯度并行，区别于历史8卡×batch8的浮点归约顺序，不称精确历史优化器重放。第1步每seed共用1024条真实rollout与完整chain；训练扰动流独立于selection/evaluation。后续自己策略重采样接口已实现，但未执行。

归档原 `forward_grpo` 损失与提取后的真实loss/KL调用，在同一实际rollout、同一全局优势下误差0；Gaussian logprob公式核验误差0。**这些前向数值检查没有覆盖住随后暴露的requires_grad身份错误。**

失败属于本次实现：在从只读loader恢复训练时执行了全模型 `requires_grad_(True)`，破坏归档constructor的冻结集合，错误加入105个张量：固定 `eta.eta_logit` 与104个结构性禁用的planning分支张量。eta本来用`atanh(1)=+inf`编码固定有限采样eta，并被原代码冻结；对其计算`inf-inf`位移得到NaN。严格JSON写入报错时，每臂已经执行1次更新。**不能据此说模型发生发散，也不能把这8次当成有效的原recipe四臂结果。**

原实现把step1保存放在统计JSON之后，因此异常前没有留下step1权重；现有step0、共享真实chain、优势、risk mask和完整traceback均保留，没有重放优化器去伪造恢复权重。step1更新后评测、范数匹配控制与step8全部NOT_RUN。此缺失是明确限制，不能用step0评价冒充更新结果。

已修复为保留真实constructor flags，新增训练集合/有限值断言和checkpoint先保存，并禁止失败臂自动重启。修复后只读核验452个可训练张量、EtaFixed保持冻结；12项单元测试通过。没有重新执行任何optimizer更新。

`tiny_update_status.csv`逐seed逐臂列出INVALID_STOPPED与实际次数；`tiny/<seed>/<arm>/failure.log`、`logs/tiny_*_step1.log`、`source_snapshots/failure_identity_and_persistence.md`保存失败证据。D−A、D−B、D−C、D−范数匹配A以及训练→部署迁移结论均 **未验证**。

## 8. 可复现产物与计算成本

{table(loadcsv('compute_cost.csv'))}

计数按候选评分计，不含每次PDM baseline；共享第1步采样/评分成本只计一次，8个臂均复用。独立额外调试评分6条另记在 `manifests/cost_notes.json`。没有长训、SFT、额外seed搜索或扰动调参。

核心复用接口及数据：

- `manifests/scenes.json`、`models.json`、`input_files.json`、`resolved_runtime.json`、`trajectory_cache_index.parquet`；冻结总配置 `resolved_config.yaml`，第三阶段增量配置 `manifests/tiny_update.json`。
- `cache/features/`、`cache/rollouts/`、`cache/simulated/`、`trajectories.parquet`、`trajectory_credit.parquet`、`group_statistics.parquet`、`scene_credit_statistics.parquet`。
- `matched_pairs.parquet`、`nominal_margins.parquet`、`perturbation_manifest.json`、`perturbation_results.parquet`、`native_vs_constructed_coverage.csv`、`selector_comparisons.csv`。
- 全部配对差值/区间、四张真实数据图及各自CSV；`commands.sh`和`logs/`；无理想趋势图、无补造缺失结果。
- `tools/analysis/grpo_credit_risk_quickcheck/scoring.py`提供可重放评分接口；`README.md`说明阶段入口。analysis分支提交代码、测试、冻结配置、报告和紧凑审计/失败证据；大型模型、特征、轨迹缓存留在上述统一输出路径，索引及哈希可复用。

本轮最明确的进展是：原生组内确实存在可独立验证的邻域安全差异，且单一名义PDMS不能全部分辨。当前尚未取得任何有效的极早期更新对照证据；后续应先修复并验证单步实验合同，不转向完整框架，也不依据此次失效更新断言新信用有效或无效。
'''
path=WORK/'reports/GRPO_CREDIT_RISK_QUICKCHECK.md';path.parent.mkdir(exist_ok=True);path.write_text(text)
(OUT/'GRPO_CREDIT_RISK_QUICKCHECK.md').write_text(text)
print(path)
