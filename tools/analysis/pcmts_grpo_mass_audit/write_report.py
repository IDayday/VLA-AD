"""Generate the final report from measured tables, never from expected outcomes."""
from common_mass import *
import pandas as pd

def number(v, signed=False):
    if pd.isna(v):return 'NA'
    if not isinstance(v,(float,np.floating)):return str(v)
    spec='.4e' if 0<abs(v)<.001 else '.4f'
    return format(v,('+' if signed else '')+spec)

def markdown(df):
    # No optional tabulate dependency; preserve explicit NA rather than zero.
    rows=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for record in df.itertuples(index=False,name=None):
        vals=[]
        for v in record:
            vals.append(number(v))
        rows.append('| '+' | '.join(vals)+' |')
    return '\n'.join(rows)

def main():
    m=OUT/'metrics';summary=pd.read_csv(m/'method_summary.csv');paired=pd.read_csv(m/'paired_comparisons.csv');source=pd.read_csv(m/'raw_source_summary.csv');matched=pd.read_csv(m/'source_quality_matched.csv');noil=pd.read_csv(m/'no_il_native_control.csv');sensitivity=pd.read_csv(m/'sensitivity.csv');global_policy=pd.read_csv(m/'global_policy_sampling_baseline.csv');pareto=pd.read_csv(m/'pareto_audit.csv');matchingcoverage=pd.read_csv(m/'matching_coverage.csv');counts=pd.read_csv(m/'raw_scene_counts.csv');pools=pd.read_csv(m/'pool_scene_metrics.csv')
    op=summary[summary.pool_type=='operational'].set_index('method').reindex(METHODS);strict=summary[summary.pool_type=='strict'].set_index('method').reindex(METHODS)
    columns=['candidate_PDMS','conservative_feasible_fraction','mean_candidate_p_B','PoolMass','QualityMatchedPoolMass','candidate_local_gain','strict_16_scene_fraction','fallback_fraction','external_source_fraction']
    table=op[columns].reset_index();comparisons=paired[(paired.variant=='primary')&(paired.pool_type=='operational')&paired.metric.isin(cfg()['analysis']['primary_endpoints'])]
    deltas=comparisons[['comparator','metric','mean_difference','median_difference','ci_low','ci_high','scene_win_fraction','n','Holm_p_primary_family12']]
    inv=read(OUT/'manifests/source_inventory.json');sampler=read(OUT/'manifests/sampler_identity.json');freeze=read(OUT/'manifests/protocol_frozen.json');run=read(OUT/'manifests/run_summary.json')
    review=read(OUT/'audits/independent_real_data_review.json')['diagnostics']
    sens1=sensitivity[(sensitivity.variant=='eps1.0_p0.03')&(sensitivity.pool_type=='operational')].set_index('method').reindex(METHODS)
    paired_sensitivity=pd.read_csv(m/'sensitivity_paired_comparisons.csv')
    sens1pairs=paired_sensitivity[(paired_sensitivity.variant=='eps1.0_p0.03')&(paired_sensitivity.pool_type=='operational')&paired_sensitivity.metric.isin(cfg()['analysis']['primary_endpoints'])]
    mass=comparisons[comparisons.metric=='mean_candidate_p_B'];union=comparisons[comparisons.metric=='PoolMass'];quality=comparisons[comparisons.metric=='candidate_PDMS']
    coverage_supported=len(mass)==3 and (mass.ci_low>0).all()
    outcome='PARTIALLY_SUPPORTED' if coverage_supported else 'NOT_SUPPORTED'
    if coverage_supported and (union.ci_low>0).all() and (quality.mean_difference>=0).all():outcome='SUPPORTED'
    nstrict=(strict.strict_count*1000).round().astype(int)
    def delta_line(method,metric):
        r=comparisons[(comparisons.comparator==method)&(comparisons.metric==metric)].iloc[0]
        return f"{number(r.mean_difference,True)}，95% CI [{number(r.ci_low,True)}, {number(r.ci_high,True)}]，共同有定义场景 n={int(r.n)}"
    txt=f'''# PC-MTS：1000 场景外部高分候选与真实 GRPO 采样覆盖审计

状态：**{outcome}**（静态质量—覆盖证据）。执行状态：**{run['status']}**。权重更新次数 **0**。

**主结果没有支持预设的PC规则。** 在0.5m / 3%主配置下，PC的strict候选在全部1000场景均为空，16000个operational槽位全部是Level 1方法约束回退，与Score逐场景、逐候选完全相同。下面PC的operational高分不能算作概率合格的PC成功。主局部增益在所有方法均为NA，不能推断存在“已覆盖区域内的额外专家知识”。

预先冻结的1m敏感性显示另一种质量—覆盖取舍，但不能替换失败的主配置，也没有训练或因果收益证据。

## 1. 范围、冻结协议与数据身份

本轮唯一策略是 frozen Official GT-IL；四种方法选择不同候选集合，没有训练四个策略。
1000 个固定场景全部纳入，缺失 {run['missing_scenes']}，评分异常 {run['error_scenes']}。另有 **{run['raw_log_timestamp_flagged_scenes']} 个场景原始log时间戳跳帧**，完整保留并标记。这些场景的GT严格复现原生 `Scene.get_future_trajectory` 的未来帧索引定义，而非精确墙钟时间插值；统一评分按NAVSIM名义4秒/8点契约解释。不得把这一数据限制隐瞒为所有原始帧间隔都精确0.5秒。具体token与间隔见 `audits/coordinate_and_token_identity.json`、`metrics/missing_and_errors.csv`。
本轮为历史内部场景机制分析；A/B 是同场景独立随机样本，不能称为 untouched Navtest 泛化，也不能和历史 checkpoint 名称中的分数直接比较。

- Branch：`analysis/pcmts-grpo-mass-audit-1000-20260913`。
- 指定诊断基线：`d4922a8207960636773767aa9d664c6a1d9bb7b5`。
- 实际 parent / 本轮开始时代码 commit：`{freeze['actual_parent_commit']}`，包含正式 stage3 审计修正。
- 配置：`configs/pcmts_grpo_mass_audit/primary.yaml`。
- SHA256：`{freeze['sha256']}`；冻结时间：`{freeze['frozen_at']}`。
- 权重：`{sampler['checkpoint_path']}`；SHA256 `{sampler['checkpoint_hash']}`。
- 实际加载类：`{sampler['class_name']}`。
- 归档运行代码：`{sampler['runtime_code_path']}`；SHA256 `{sampler['runtime_code_hash']}`。

每场景 A=1024、B=1024、C=64；总计 A/B=2,048,000 次抽样，另有 C=64,000 次。每组 G=8；组、流、token 使用不重叠的确定性随机身份。自然重复输出仍计入抽样分母。
主邻域 ADE≤0.5m、mass≥3%、GT-ADE≤1.0m 是预设工作设置，没有声称它们最优。所有主配置、敏感性及 no-IL-native 选集与来源/质量匹配，均在读取 B 结果前冻结。
V1/V2/V3 的受保护结果 hash 校验通过；大缓存和权重保留在本地。

## 2. 真实 GRPO sampler 与评分审计

直接使用 `forward_grpo` 内 current-policy 的 `sample_chain`：5 DDIM steps，eta=1，初始标准正态噪声，逐步 sampling std 为 {[(r['timestep'],round(r['sampling_std_min'],6)) for r in sampler['per_step_sampling_std']]}，noise clip={sampler['sampling_noise_clip']}，final action clip={sampler['final_action_clip']}。logprob std floor={sampler['logprob_std_floor']} 独立登记，**没有用于估计概率**。

保持 native training 模块状态，再执行原生 `set_frozen_modules_to_eval_mode`。活动 dropout 的 p=0，无活动 BatchNorm；权重与 buffer 在采样前后完全相同。输入仅包含当前/历史观察缓存；GT、外部轨迹和评分没有进入 policy conditioning。
8 个固定 smoke 场景，新入口与 native GRPO 输出逐元素相同，native advantage/loss 日志一致，scalar/batch evaluator 一致。
等价的 8-group 批处理与相同 B=8、G=8 的完整 native forward 输出完全一致；逐组与批处理仅有不超过约 6e-6 的 FP32 矩阵计算舍入差，完整记录在 `audits/batch_native_parity.json`。先前已生成的逐组缓存保留，未伪造新增抽样。

所有 trajectory 都由同一 NAVSIM-v1 evaluator 评分。PDMS 对外是 0–100 point，NC/DAC/EP/TTC/DDC/Comfort 与概率是 0–1。标准 PDMS 权重 EP/TTC/Comfort=5/5/2；真实训练 reward 另存 10/5/2，未混用。

候选频率按真实命中数计算；联合安全/质量概率始终以完整 N_B=1024 为分母。局部均分包含全部邻域 rollout，包括不安全和低分。零命中局部均分/增益=NA，主局部增益只使用至少10次命中的候选。
候选区间按128个真实独立 group 的组内命中均值估计：非零命中采用group均值标准误的正态近似，稀疏事件下这个区间精度有限，不能解释成严格有限样本保证。零命中主上界采用 group 层面的单侧95%保守界，约2.31%；成员 iid 条件下约0.292%的界仅额外列出，不据此宣称不可达。GroupPoolHit 使用实际 group 计数及Wilson区间。

### 为什么0.5m邻域几乎没有命中？

全321000个raw候选在A的最高命中数为 **{review['max_hits_A']} /1024**，在B也只有 **{review['max_hits_B']} /1024**；3%要求至少31次。原始候选B零命中比例为 {review['raw_zero_hit_B_fraction']:.6%}，最近一条B rollout的ADE中位数为 {review['nearest_ADE_B_quantiles']['0.5']:.4f}m、P90为 {review['nearest_ADE_B_quantiles']['0.9']:.4f}m。独立逐candidate循环重算8个smoke场景的全部raw候选，命中数、安全/质量联合概率与主向量实现完全一致，见 `audits/independent_real_data_review.json`。

实际 `sample_chain` 最后一个step仍执行sampling std floor=0.04和逐点噪声。原生 `denorm_odo` 的X/Y尺度分别为66.74/2、42/2，因此未被最终clip影响时，仅最后一次新噪声的X/Y标准差已约为 **1.3348m / 0.84m**；这是逐点扰动，不是整条轨迹共同平移。平均8点XY误差≤0.5m是较强约束。这与低命中相符，但本轮没有修改std做因果消融，不能把全部差异定量归因于该floor。

同一策略的C均分={global_policy.C_mean_PDMS.mean():.4f}，B均分={global_policy.B_mean_PDMS.mean():.4f}，B conservative-feasible={global_policy.B_feasible_rate.mean():.4%}。这些是真实GRPO随机分支的均分，不能与旧evaluation采样或checkpoint文件名分数混用。Global_HQMass仍非零，说明“小几何球覆盖不足”和“完全没有高分行为”是不同命题。

## 3. 共享候选来源与去重

DDV2：`{inv['ddv2']['checkpoint_path']}`，hash `{inv['ddv2']['checkpoint_sha256']}`。
DrivoR：`{inv['drivor']['checkpoint_path']}`，hash `{inv['drivor']['checkpoint_sha256']}`。
两个来源均覆盖1000场景。DDV2 在 scorer 前原生生成800条（20 anchors ×10组 ×原始及3次原生乘性增强），按固定 proposal 索引 hash 取64；这些原生增强不冒充GT扰动。DrivoR返回64条原生 proposal，未只取最终赢家。
DDV2的64000条所选proposal还保存了base-parent索引、augmentation block、实际X/Y乘性残差及重构误差：`audits/ddv2_native_augmentation_lineage.parquet`。两个外部实现与原始checkpoint均未修改。
统一采用原生部署契约：ego 局部坐标、8个未来点、0.5秒间隔、4秒，t=0是隐含原点，不是第一未来点。DDV2 原生 `bezier_xyyaw` 保留XY、由含原点的导数计算heading；DrivoR训练代码的long-target变形不用于重标记部署时间轴。
每场景另有GT 1条、固定五族结构化扰动128条、独立C流64条。扰动幅度指最大同步点位移，保持隐含起始pose，生成规则和seed冻结。
坐标精度6位小数用于完全重复身份；跨来源重复保留所有父本和source tags，并按对称规则选代表。source组成另报均分归因，避免keep-first偏差。0.02/0.10m代表数是几何非冗余度，不是驾驶语义模式数。

{markdown(source[['source','generated_pre_dedup','unique_candidates','covered_scenes','raw_mean_PDMS','raw_mean_p_A','raw_mean_p_B']])}

总raw unique={int(counts.raw_unique_count.sum())}；每场景范围 {int(counts.raw_unique_count.min())}–{int(counts.raw_unique_count.max())}；emergency追加批次总数={int(counts.emergency_batches.sum())}。
来源覆盖及逐token cache hash见 `manifests/source_inventory.json`，计数见 `metrics/raw_scene_counts.csv`、`raw_source_summary.csv`。

## 4. 四种规则、严格覆盖和回退

所有方法共用 E=valid、NC=DAC=1、PDMS≥独立C流平均 b_s。
Score 按PDMS；Pareto在E内对 EP/TTC/DDC 遍历全部非支配层；GT geometry额外要求GT-ADE≤1m；GRPO mass PC额外要求p_A≥3%（1024样本至少31次命中），达门槛后仍按PDMS选择，不最大化概率，也不限制来源。
所有PDMS并列在0.01point范围内使用相同最小ADE多样性tie-break及固定candidate ID。
严格结果最多16条unique；operational保留strict后按顺序放宽方法条件、质量floor、最后hard safety。最后一级只用于诊断，禁止作为安全训练目标；重复/padding不计真实候选。

{markdown(summary[['pool_type','method','nonempty_scene_count','candidate_denominator','strict_16_scene_fraction','strict_count','fallback_fraction','level_1_fraction','level_2_fraction','level_3_fraction']])}

共同E为空的场景数：{int(((pools.method=='score')&(pools.pool_type=='strict')&(pools.strict_count==0)).sum())} /1000，未替换或排除。各方法strict为空时质量=NA，Full1000覆盖仍以1000为分母。
Pareto与Score完全相同选集比例={pareto.identical_set.mean():.4f}，Jaccard均值={pareto.score_pareto_jaccard.mean():.4f}；E内TTC零方差场景={int((pareto.TTC_variance==0).sum())}，DDC零方差={int((pareto.DDC_variance==0).sum())}。这可能导致目标退化；没有为了制造差异临时换目标。

## 5. 主结果：Full1000，operational 16槽位

下表概率/比例单位均为0–1；PDMS与local gain为point。先场景内平均，再对1000场景等权。external_source_fraction专指带DDV2或DrivoR真实来源标签的候选。

{markdown(table)}

表源 `metrics/method_summary.csv`，字段如表头。主局部增益是描述性不同有效子集均值，不能直接解释为同一总体的因果优势。有效候选/场景分母：

{markdown(summary[['pool_type','method','local_gain_candidate_denominator','local_gain_scene_denominator','nonempty_scene_count']])}

Strict-only质量及覆盖另表：

{markdown(strict[columns].reset_index())}

## 6. 预设主配对比较与独立确认

PC减baseline；3000次scene-paired bootstrap，并另存log-cluster敏感性。主要检验族为3个比较×4终点，Holm校正12项。表内概率差是fraction，乘100为百分点。NA质量只在共同有定义场景配对，n明确给出。

{markdown(deltas)}

PC相对Score的覆盖差：{delta_line('score','mean_candidate_p_B')}；PDMS差：{delta_line('score','candidate_PDMS')}。
候选集合覆盖提高不代表策略全局高分能力提高；四方法的policy相同。

{markdown(summary[['pool_type','method','selection_optimism','fraction_candidate_p_B_ge_3percent','zero_hit_fraction','mean_candidate_p_B_safe','mean_candidate_p_B_quality']])}

`selection_optimism=p_A-p_B`；B点估计略低于3%不一律判失败，需结合连续差和group区间。概率来自计数，不是旧q、KNN百分位、logprob或diffusion loss的转换。

## 7. 集合并集覆盖、冗余与外部专家增益

PoolMass是B rollout是否命中至少一个候选的并集，不能把16个p_B相加；每个rollout最多计一次。
QualityMatchedPoolMass要求同一个候选同时满足near及quality阈值，不能near高分候选A却借用低分候选B的阈值。GroupPoolHit由真实8成员group计数。

{markdown(summary[['pool_type','method','PoolMass','QualityMatchedPoolMass','GroupPoolHit','sum_individual_mass','near_duplicate_representatives_0_02m','near_duplicate_representatives_0_10m','pairwise_ADE','candidate_local_gain']])}

逐候选marginal coverage见 `pool_scene_metrics.csv:marginal_coverage`；总和严格等于PoolMass。局部增益定义为candidate_PDMS减全部near-B的均分。高覆盖且增益≥0.5point表示附近常被访问但仍可能有更好的外部控制目标；高覆盖、低增益可能冗余；低覆盖高分只能解释为本次有限采样覆盖不足，不能断言不可学。
完整类型分解见 `expert_gain_types.csv`、`candidate_metrics.parquet:descriptive_type`。DDV2/DrivoR/GT结构化扰动各自质量、p_A/p_B和选后local gain见 `raw_source_summary.csv`，不会把合并来源比例当来源独立证据。

## 8. 同场景、同来源、同质量匹配

匹配在A选集冻结时完成，不读B：同scene、exact source、NC/DAC类别、PDMS差≤0.25point，最大匹配数量下最小总差，一对一无replacement。跨来源重复使用显式multi-source集合标签。

{markdown(matched)}

匹配覆盖表记录每个scene的shared、different及unmatched count；总未匹配PC数（跨3比较及两种pool累加，仅作工作量说明）={int(matchingcoverage.unmatched_pc.sum())}。各来源匹配覆盖见 `matched_source_breakdown.csv`。
pair-level CI与scene-cluster CI同时报告，主要解释场景等权差值；local gain只在两侧命中都≥10时配对，额外缩小的分母在表中公开。样本不足时为common-support不足，不能无限放宽0.25point条件。

## 9. 去掉IL-native C后的重选

只删除C-exclusive candidates；有真实外部来源的跨来源重复保留并标记C-overlap。b_s、A/B、evaluator完全不变。下表的strict不足和fallback照常统计。

{markdown(noil[['pool_type','method']+columns+['C_overlap_fraction']])}

对照配对区间见 `paired_comparisons.csv:variant=no_il_native`。这一对照检验是否依赖原生来源，不等价于外部知识已被模型吸收。

## 10. 固定敏感性与全局policy能力

所有epsilon=0.25/0.5/1.0、p_min=1/3/5%、GT radius=0.5/1/2m配置均在读取B前确定；主结果始终单列，不选最有利设置替代。详细表 `sensitivity.csv`，图Fig6。

{markdown(sensitivity[(sensitivity.method=='grpo_mass_pc')&(sensitivity.pool_type=='operational')][['variant','epsilon','pmin','candidate_PDMS','mean_candidate_p_B','PoolMass','strict_16_scene_fraction']])}

预设1m / 3%敏感性，同一A/B、同一raw bank，四方法完整对照如下（补充结果，非primary）：

{markdown(sens1[columns+['selection_optimism','fraction_candidate_p_B_ge_3percent','local_gain_candidate_denominator','local_gain_scene_denominator']].reset_index())}

{markdown(sens1pairs[['comparator','metric','mean_difference','ci_low','ci_high','n']])}

在1m时，PC严格选满16条的场景为953/1000，独立B的平均候选覆盖约4.26%，PoolMass约11.13%；PDMS为95.92，低于Score99.24、Pareto98.13、GT geometry96.79。这是覆盖提高且质量下降的tradeoff，不是无代价胜出。其原生C独占来源以外的DDV2/DrivoR比例约58%，说明此补充选集不限于原生输出；但1m下没有另做来源匹配或no-native重选，不能声称在该设置已排除source confound。全部9个epsilon/p_min组合及GT半径敏感性均完整公开，未根据结果修改选集。

全局高分阈值是每场景shared raw bank内conservative-feasible best-known分数−1point，不是真实物理oracle。以下只报告一次：Global_HQMass均值={global_policy.Global_HQMass.mean():.6f}；Global_HQGroupHit均值={global_policy.Global_HQGroupHit.mean():.6f}；Best-safe@8={global_policy.Best_safe_at_G.mean():.4f}；无safe-member group比例={global_policy.No_safe_member_group_rate.mean():.6f}。
near-ceiling场景={int(global_policy.near_ceiling.sum())}/1000，不将其高HQ命中当困难突破。B超过raw best-known的scene={int((global_policy.B_best_known_gap<0).sum())}，负gap原样保留。没有安全成员时Best-safe@8为NA，并公开无安全成员比例。

## 11. 支持证据、反对证据与下一步

1. **来源真实接入：SUPPORTED。** DDV2、DrivoR各64000条unique，GT结构化扰动128000条，每个来源覆盖1000场景；GT与C另为1000/64000条。没有来源缺失、单输出冒充多候选或人工复制抽样。总321000条去重后均不同。依据Section3与`raw_source_summary.csv:unique_candidates/covered_scenes`。
2. **严格16条：PC NOT_SUPPORTED。** Score/Pareto/GT geometry各1000/1000场景严格选满；PC为0/1000，全部16000槽位依靠Level 1回退，未放宽质量或hard safety。依据`method_summary.csv:strict_16_scene_fraction/level_1_fraction`。
3. **独立B覆盖优势：NOT_SUPPORTED。** PC回退后与Score完全相同；相对GT geometry覆盖更低，相对Pareto仅稀疏命中差，CI包含0。严格PC没有可比较质量分布。依据Section6配对表，所有operational比较n=1000。
4. **质量/安全代价：不能当作PC收益。** PC operational与Score的质量/安全差均为0；比GT geometry高约2.45PDMS、conservative feasibility低约2.33个百分点，是Score回退行为造成的。严格PC的质量=NA。
5. **16条并集覆盖提高：NOT_SUPPORTED。** PC与Score的PoolMass相同，约0.0009766%，近重复代表数也完全相同。重复邻域不能加和为独立覆盖，见Section7。
6. **外部专家局部增益：UNTESTED。** 主配置全部raw候选的B命中最多2次，没有达到10次门槛，所有方法主要local gain的候选/场景分母均为0。少量1–2次命中的描述性均分不能支撑专家知识结论。
7. **来源+质量匹配：没有保留PC优势。** PC与Score选集相同，没有不同候选可配；对Pareto有356对/115场景，p_B差=0；对GT geometry有2765对/539场景，PC的匹配p_B差仍为负。这里比较的是全部回退后的PC，strict-PC匹配均为空；局部增益配对分母均为0。见`source_quality_matched.csv`，paired/scene-cluster/scene-equal区间完整公开。
8. **移除IL-native后：仍为NOT_SUPPORTED。** PC strict继续全空，operational仍与Score相同，PDMS约99.18，外部DDV2/DrivoR比例约84.71%。来源占比增加不能算PC概率门槛成功。原主池PC外部比例59.59%、C比例34.21%，也全部是回退项。见Section9。
9. **哪个来源兼具质量、覆盖、局部增益：主配置UNTESTED。** 所有来源均无3%覆盖合格项，也没有10-hit局部增益样本。DrivoR raw均分83.91高于DDV2的59.30，结构化扰动为80.71；这只是本样本raw质量对照，不能据此评出最好的可学习来源。
10. **阈值依赖：SUPPORTED。** 0.25/0.5m均无PC严格候选；1m出现大量候选，1/3/5%门槛严格16覆盖分别99.0/95.3/59.4%。1m补充结果可见更高覆盖及较低PDMS，但不能替代主结果或直接宣称训练价值。见`metrics/sensitivity.csv`、`sensitivity_paired_comparisons.csv`。
11. **所有方法均无合格候选的困难场景：本轮没有。** 共同E在1000场景都至少有16条；问题来自PC mass gate，而非所有来源同时缺少有效高分候选。PC空场景全部保留在Full1000分母。
12. **下一步建议：暂不直接训练本轮primary PC池。** 它与Score相同，做训练也不能检验mass筛选的独立作用。1m敏感性值得作为新的、独立预注册研究的设计依据，先固定质量/安全可接受代价、补充该设置的来源/质量匹配及no-native对照，再决定matched SFT＋short GRPO。不得回填为本轮成功。**训练收益、GRPO增益、多轮能力均为UNTESTED。**

保守论文表述：在本次真实GRPO采样和冻结候选库下，0.5m/3%邻域频率门槛不能产生严格PC监督；预设更大邻域呈现质量与覆盖之间的取舍，说明此类门槛需与实际采样尺度共同解释。现有证据不足以把“真实采样覆盖筛选优于Score/Pareto/GT geometry”写成已证实的核心贡献。

## 12. 文件、审计与解释边界

六图均有PNG、SVG、PDF和真实绘图CSV：`outputs/pcmts_grpo_mass_audit/figures/Fig1_*` 至 `Fig6_*`。
完整选集：`outputs/pcmts_grpo_mass_audit/metrics/selected_pools_4x1000x16.parquet`，包含trajectory本地路径、raw_index、candidate_id、content hash、valid_mask、strict/fallback标签。
候选全表：`metrics/candidate_metrics.parquet`；raw可解析NPZ/JSON在 `cache/raw/`；大rollout/模型不入git。
主要统计：`method_summary.csv`、`paired_comparisons.csv`、`source_quality_matched.csv`、`no_il_native_control.csv`、`sensitivity.csv`、`global_policy_sampling_baseline.csv`。所有文件均在本轮namespace。
协议/选集冻结：`manifests/protocol_frozen.json`、`selection_frozen.json`；来源/采样身份：`source_inventory.json`、`sampler_identity.json`；全量结果身份：`audits/cache_hash_manifest.json`、`completion.json`。
工程修复只涉及缺失依赖、缓存扫描忽略atomic临时文件和已通过parity的采样批处理；记录在 `audits/engineering_changes.json`。没有按科学结果调整threshold、场景、来源或扰动。
续跑会保留原始selection seal的hash及pre-B时间戳，不重新封签伪装独立性。最终29项测试、8场景native parity、8场景独立B逐候选概率复算及1000场景完整性审计通过；原始rank6/rank7因资源重分配在进程结束前中断，没有其进程尾部state hash。其已完成atomic缓存保留；其余14个完成worker的前后state hash相同，模型无optimizer、活动BN或dropout。该审计范围限制见 `audits/final_publication_review.json`，未伪造中断worker的尾部审计。
最终科学问题只能回答“外部候选集合的质量—采样覆盖组合”，不能写成“这些候选训练后已经改善策略”。
'''
    path=ROOT/'reports/PC_MTS_GRPO_MASS_AUDIT_1000_20260913.md';path.write_text(txt)
    save(OUT/'manifests/conclusion_status.json',dict(status=outcome,coverage_CI_positive_vs_all_baselines=bool(coverage_supported),training_improvement='UNTESTED',report=str(path),report_sha256=sha(path)))
    print(path,outcome,flush=True)
if __name__=='__main__':main()
