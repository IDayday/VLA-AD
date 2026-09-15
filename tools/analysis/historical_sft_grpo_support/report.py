"""Render evidence tables and explicitly bounded mechanism conclusions."""
from common_support import *
from figures import LABEL

def table(x):
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,x.columns))+' |','| '+' | '.join(['---']*len(x.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in x.itertuples(index=False,name=None)])
def main():
    sm=pd.read_csv(OUT/'metrics/group16_summary.csv');sm=sm[sm.scope=='FULL1000']
    ts=pd.read_csv(OUT/'metrics/teacher_summary.csv');ts=ts[ts.scope=='COMMON_TRAIN835']
    sf=pd.read_csv(OUT/'metrics/scene_metrics.csv');comp=pd.read_csv(OUT/'metrics/paired_comparisons.csv')
    def ci(label,metric):
        r=comp[(comp.comparison==label)&(comp.metric==metric)].iloc[0]
        return f"{r.mean_difference:+.6f}，95%CI[{r.ci_low:.6f}, {r.ci_high:.6f}]，n={int(r.n)}"
    core=sm[sm.model.isin(CFG['primary_models'])&sm.protocol.isin(CFG['protocols'])].copy()
    core['feasible_rate']*=100
    own=ts[(ts.policy==ts.origin)&(ts.kind=='NON_GT')].copy()
    for k in ['weighted_hit16_0p5','weighted_hit64_0p5','weighted_hit64_1p0']:own[k]*=100
    fit=ts[(ts.protocol=='eval')&(ts.origin.isin(['official_il','psi_sft','a5_sft','v6_sft']))&(ts.kind.isin(['NON_GT','GT']))].copy()
    fit=fit[((fit.origin=='official_il')&(fit.kind=='GT'))|((fit.origin!='official_il')&(fit.kind=='NON_GT'))]
    fit=fit[['policy','origin','scenes','weighted_epsilon_mse','weighted_hit64_0p5']]
    native=sm[(sm.protocol=='native_grpo')].copy();native.feasible_rate*=100
    fac_tokens=set(read(OUT/'manifests/scenes.json')['factorial_tokens'])
    fac=sf[sf.model.isin(CFG['primary_models'])&sf.token.isin(fac_tokens)].groupby(['model','protocol']).mean(numeric_only=True).reset_index()
    credits=[]
    for p in (OUT/'metrics').glob('native_credit_*.csv'):
        if p.name=='native_credit_summary.csv':continue
        f=pd.read_csv(p)
        for (m,pr),g in f.groupby(['model','protocol']):
            h=g.groupby('token').mean(numeric_only=True)
            credits.append(dict(model=m,protocol=pr,scenes=len(h),positive_count=h.positive_count.mean(),no_positive_group=h.no_positive_group.mean(),all_zero=h.all_zero.mean(),native_PIA=(g.positive_infeasible_count.sum()/g.infeasible_count.sum()) if g.infeasible_count.sum() else np.nan,positive_above_reference_count=h.positive_above_reference_count.mean()))
    credit=pd.DataFrame(credits);csv('native_credit_summary.csv',credit)
    teacher=pd.read_csv(OUT/'metrics/teachers.csv');teacher=teacher[teacher.common_train]
    teacher_composition=[]
    for m,g in teacher.groupby('origin'):
        weights=g.groupby('token').apply(lambda h:h.loc[h.is_gt,'weight'].sum()/h.weight.sum(),include_groups=False)
        teacher_composition.append(dict(model=m,scenes=g.token.nunique(),teacher_rows=len(g),teachers_per_scene=len(g)/g.token.nunique(),reconstructed_GT_mass_pre_budget=weights.mean()))
    csv('teacher_composition_common_train.csv',teacher_composition)
    parity=pd.read_csv(OUT/'metrics/factorial_parity.csv')
    native_trajectory_err=parity.native_vs_floor_and_clip_max_abs.max()
    text=f'''# 历史SFT监督覆盖与真实GRPO组内分布：机制核查

核查日期：2026-09-15。分支`analysis/historical-sft-grpo-support-mechanism-20260915`，parent commit `4716261be46b2bc3283a80f89377bb6eaf7dc3fa`。本轮只读取历史checkpoint，**没有SFT/GRPO权重更新，也没有使用9月快训权重代替历史权重**。

## 1. 这次回答的核心问题

实测确认：**PSI在GT-IL基础上微调后，普通推理分布比官方IL更宽；A5/V6从随机action head训练后更窄。真实GRPO的组内分布还会受到采样噪声与trajectory representation的强烈影响。** 但是，“没采到teacher导致GRPO失败”这个单一解释不成立：A5在0.5米标准下覆盖自己的非GT teacher，比成功链路的PSI还高；PSI在真实GRPO采样下64次不命中该小邻域，仍有历史GRPO提升记录。

因此目前能确定的是几层不同机制：监督权重与拟合程度不同；采样器的物理探索尺度不同；GRPO可利用的组内质量差异不同；后续advantage与reference约束也不同。**不能把历史差异唯一因果归结为“随机初始化 vs IL初始化”。** 以下不靠最终PDMS代替分布分析。

## 2. 实验对象、场景与分母

- 四个SFT起点：official GT-IL、PSI候选微调epoch11、A5 epoch155（历史多候选87.51标签）、V6 epoch165（历史多候选86.92标签）。这里“从头”指action head随机初始化，VLM观察编码被冻结。
- 同一1000个固定NAVTRAIN场景，4个独立group/scene，每group16条。每权重每sampler总计64,000条；每场景64条用于teacher覆盖。四模型各用普通`get_action`与实际`forward_grpo/forward_lfp_grpo`采样分支，同场景、同初始/逐步随机流。
- 另外读取original GRPO step11970、A5 step300/4842、V6 step300/3300，共9个checkpoint。两种主采样共1,152,000条；256场景采样参数消融另196,608条。
- 1000场景中835个是PSI/A5/V6的共同真实训练场景，另外165个是PSI validation。**监督学习主分析使用共同835；分布保留Full1000。这里不是untouched Navtest泛化评测，不能直接与checkpoint标签分数比较。**
- PSI的历史SR-PGRPO权重实体仍MISSING，不能实采它训练后的组分布；其长训评分和TensorBoard可查。没有用APR、其他GRPO或近期快训权重替代。

配置与数据身份：`configs/historical_sft_grpo_support/primary.yaml`、`manifests/protocol.json`、`models.json`、`observations.json`、`scenes.json`。主参数在生成结果前冻结，未按结果调整。正文路径除另注明外均相对`outputs/historical_sft_grpo_support/`。

## 3. SFT后分布与每组16条真实统计

下面每项先在16条组内计算，再平均4组，最后1000场景等权平均。**max/min列是“每组最大/最小值的平均”，不是整个数据集的极值。** `eval`与`native_grpo`行的权重完全相同；行间PDMS下降不是训练把权重训坏，而是切换了采样器。

{table(core[['model','protocol','mean_PDMS','max_PDMS','min_PDMS','range_PDMS','feasible_rate','pairwise_ADE','Spread_AUC','safe_count']])}

PDMS为0–100point，feasible_rate此表为%；保守可行定义NC=DAC=TTC=DDC=1。pairwise ADE是16条轨迹两两8未来点XY平均距离，单位米。Spread-AUC沿用旧报告的8时刻XY标准差模长平均；不是语义mode数量。每个模型/协议4000组、64,000条。原始逐组表`metrics/group16_metrics.parquet`；逐场景表`metrics/scene_metrics.parquet`；汇总`group16_summary.csv`。

![16条组内分布](../outputs/historical_sft_grpo_support/figures/Fig-M2_native_group16.png)

SFT普通推理pairwise ADE相对官方IL的scene-paired差值：

- PSI：{ci('psi_sft-minus-official_il:eval','pairwise_ADE')}。
- A5：{ci('a5_sft-minus-official_il:eval','pairwise_ADE')}。
- V6：{ci('v6_sft-minus-official_il:eval','pairwise_ADE')}。

这支持“训练后的分布形状不同”，并不证明多轨迹SFT必然压缩：PSI本身也是候选SFT，却向相反方向变化。A5/V6还同时换成PTA与FS逐步delta representation，不能把所有宽度差都算作数据的独立效应。

## 4. 各自的监督轨迹究竟有没有学好

### 4.1 恢复的真实监督，而非诊断候选库

A5/V6从原始token级support archive恢复，PSI从实际训练的`stage2_pareto_support_clean_full.pt`恢复。保留raw_index、source、文件hash、轨迹hash、训练身份；所有模型都评价相同teacher并集。A5/V6表中weight是**checkpoint时期、residual budget之前的抽样/权重期望**，不是声称恢复了历史每一步梯度。

{table(pd.DataFrame(teacher_composition))}

这个表最后一列不能与post-budget训练日志混用。实际完整epoch日志显示A5的GT权重约87.28%，V6约82.50%；V6非GT权重从budget前46.81%降到17.50%，非GT/GT loss比约4.67。PSI则每次按support权重抽一条目标，没有强制每次附加GT损失。在这835场景的实际support中，GT期望抽样质量约29.94%。**它们不是“同一组轨迹、同一权重，只换初始化”的实验。** 来源`metrics/historical_sft_learning.csv`与`teacher_composition_common_train.csv`；对应完整日志和代码的hash在`audits/historical_log_sources.json`。

### 4.2 直接检查teacher是否落入生成支持

命中定义为8同步未来点XY ADE≤0.5米。每scene先按自己的非GT teacher权重归一化，再平均scene；只统计实际有非GT监督的scene，并公开分母。GT-only scene没有伪记为“teacher全学会”。

{table(own[['policy','protocol','scenes','teachers','weighted_hit16_0p5','weighted_hit64_0p5','weighted_hit64_1p0','weighted_nearest_rollout_ADE']])}

hit列为百分比；16次是4个独立G16的平均命中率，64次是四组合计是否命中。1.0米是预先固定的敏感性，不替换主0.5米。逐teacher文件`metrics/teacher_coverage.parquet`、逐scene `metrics/teacher_scene_metrics.parquet`。**0/64是未观测到，不是概率严格为0，也不是不可学。**

![自己的teacher覆盖](../outputs/historical_sft_grpo_support/figures/Fig-M1_teacher_coverage.png)

### 4.3 去噪loss与采样覆盖不是一回事

原生epsilon目标为：`E_(t,epsilon) mean[(epsilon_theta(alpha_t * Normalize(teacher) + sigma_t * epsilon, t, observation) - epsilon)^2]`。本轮直接调用各自native forward，使用真实归一化、真实alpha/sigma与原训练均匀timestep分布；每scene4组共同epsilon/timestep，所有teacher与模型使用CRN。没有把loss转成概率。

以下在相同teacher bank上比较，mean为场景等权、scene内部teacher加权。legacy与FS的原始loss量纲受representation影响，**跨legacy/FS的loss绝对值不是统一的“谁更可学”排名**；PSI与official同legacy，可做相同目标的配对比较。

{table(fit)}

PSI相对official，在PSI非GT teacher上的加权epsilon loss差：{ci('teacher:psi_sft-minus-official:psi_sft:NON_GT:eval','weighted_epsilon_mse')}；在GT上的差：{ci('teacher:psi_sft-minus-official:official_il:GT:eval','weighted_epsilon_mse')}。必须同时看mean、median和正/负差scene比例，不能把平均改善说成每个teacher都改善。`paired_comparisons.csv`的历史字段名`win_fraction`仅表示差值>0的scene比例；对loss更小才是改善，对spread没有预定越大越好方向，不能直接把该字段解读为性能胜率。

具体来说，PSI非GT监督的平均loss从0.036265降到0.022529，约下降37.88%；但仅48.24%的767个scene发生loss下降，配对差的中位数略大于0。平均收益集中在部分高误差scene，**不是普遍逐teacher改善**。GT参考loss从0.001845升到0.005564，835个scene中约95.69%上升。PSI的候选吸收伴随GT拟合取舍，也不能说它完全保留了原IL的GT精确拟合。

模型学的是条件去噪函数，不是一个把训练trajectory逐条背下来并按给定权重输出的检索器。平均loss下降可以与少量teacher覆盖、对GT的拟合损失并存；有限5-step生成与额外采样噪声又会改变命中情况。**这些数据支持部分吸收监督，不能声称每条监督都被充分学会。**

![相同teacher库跨模型比较](../outputs/historical_sft_grpo_support/figures/Fig-M1b_cross_teacher_coverage.png)

**固定相同teacher库，比直接比较“各自teacher命中率”更可靠。** 普通推理、64次、0.5米标准下：A5自己的非GT teacher，official覆盖22.758%，A5覆盖23.321%；V6自己的teacher，official覆盖6.138%，V6覆盖6.649%。A5的配对增幅{ci('teacher:a5_sft-minus-official:a5_sft:NON_GT:eval','weighted_hit64_0p5')}；V6为{ci('teacher:v6_sft-minus-official:v6_sft:NON_GT:eval','weighted_hit64_0p5')}，两者CI都包含0。

PSI自己的teacher则从official的8.070%到PSI的10.039%，增幅{ci('teacher:psi_sft-minus-official:psi_sft:NON_GT:eval','weighted_hit64_0p5')}，CI不含0。PSI还覆盖A5 teacher的28.139%、V6 teacher的22.390%，均高于对应从头SFT模型；共享并集消除了“各自teacher不同”这一直接比较的混杂，但不能消除训练架构与representation差异。

与此相对，A5/V6对自己实际监督GT的64次覆盖为100%/99.88%（分母830/835scene）；说明它们并非普遍无法拟合轨迹，而是**候选训练没有像GT拟合那样转化成明显新增的生成覆盖**。这个结论限定于测得的采样器、半径和预算，不等于完全没有吸收候选知识。

## 5. 为什么GRPO组采样不同于SFT权重多次推理

这里至少有三种要分开的量：训练时给teacher加噪计算epsilon loss；SFT权重调用普通evaluation sampler；同一权重进入GRPO training sampler。第一种输入teacher作为loss目标，后两种只条件于当前观察/允许历史；后两种都从独立高斯初始噪声生成。

| 设置 | 普通eval | official/PSI的native GRPO | A5/V6 native LFP |
|---|---|---|---|
| DDIM步数 / eta | 5 / 1 | 5 / 1 | 5 / 1 |
| 逐步std floor | 0.0001 | 0.04 | 0.04 |
| 每步标准高斯噪声截断 | ±1 | ±5 | 不截断 |
| logprob std floor | 不用于采样 | 0.1 | 0.04 |
| trajectory representation | 各自representation | legacy绝对点 | FS逐步delta统计归一化 |
| conditioning | observation-only | observation-only | native PTA context，training=False，不注入target |

sampling std与logprob std分开；0.1不是额外采样噪声。GRPO调用`train()`但当前活动dropout的p均为0；没有活动BN，权重和buffer采样前后hash一致。A5/V6还必须传完整action input和native预计算context，不能只调用一个缺ego/command的sample_chain。

**同样的0.04在不同representation下不是同样的米级探索。** legacy仅最后一步的独立新噪声，就对应每未来点X std约1.3348米、Y约0.8400米；不是整条轨迹共同平移。FS通过逐步delta累加，同一floor的终点X/Y std约A5 0.1846/0.0783米、V6 0.1631/0.0718米。它们是表示层尺度计算，不代替实测完整rollout spread；来源旧审计`outputs/historical_sft_grpo_longchain/metrics/normalization_scale.csv`。

### 固定256场景的参数干预

只切换eval floor与noise clip，保持checkpoint、初始及每步random numbers不变。`floor_only`、`clip_only`、`floor_and_clip`都事先冻结。both与native的逐轨迹最大绝对差为 **{native_trajectory_err:.9g}** 米/弧度（`metrics/factorial_parity.csv`）。这在本256场景直接检验了采样参数的作用，而不是从宽度相关性猜原因。

{table(fac[['model','protocol','mean_PDMS','pairwise_ADE','feasible_rate']])}

![采样参数干预](../outputs/historical_sft_grpo_support/figures/Fig-M3_sampler_factorial.png)

**把同一采样器的独立样本组织成16条group，本身不改变单条policy分布；它改变的是best/min统计与组内advantage参照。** 原版正式GRPO实际G8，本轮G16用于统一比较；不能把本轮G16 advantage说成历史G8原样重放。若SFT权重也使用完全相同native sampler、网络状态、噪声协议，那么在权重更新前，其多次采样与GRPO rollout的单样本分布应一致。

## 6. GRPO到底改变了什么：读取真实历史后续权重

同一1000场景、相同随机流。下面仍是native G16；是内部NAVTRAIN采样统计，**不能与历史Navtest绝对分数混在一张增益曲线里**。

{table(native[['model','mean_PDMS','max_PDMS','min_PDMS','feasible_rate','pairwise_ADE','center_shift_IL_eval','center_shift_SFT_parent']])}

![历史checkpoint分布演化](../outputs/historical_sft_grpo_support/figures/Fig-M4_historical_distribution_evolution.png)

完整场景配对增益、中心位移、组内宽度、可行率见`metrics/paired_comparisons.csv`（3000次scene bootstrap、另有log-cluster CI）和`metrics/scene_metrics.parquet`。对PSI训练后的分布仍为UNTESTED，因为历史SR-PGRPO权重实体缺失。

这里有比“变宽/变窄”更关键的实测结果：

- **original GRPO成功并不依赖native分布变窄。** native组均PDMS提升{ci('original_grpo_11970-minus-official_il:native_grpo','mean_PDMS')}；pairwise ADE变化{ci('original_grpo_11970-minus-official_il:native_grpo','pairwise_ADE')}，几乎不变。分布中心相对SFT移动约0.498米，可行率提升7.9625pp。相同的末步噪声尺度仍在，但中心/质量分配改善了。
- **A5/V6并非在训练分布上没有学到东西。** A5 step4842 native组均分提升{ci('a5_grpo_4842-minus-a5_sft:native_grpo','mean_PDMS')}，V6 step3300为{ci('v6_grpo_3300-minus-v6_sft:native_grpo','mean_PDMS')}。同时保守可行率分别下降0.6828pp/2.5141pp，中心移动约2.016米/1.063米，组内宽度增大。这支持“变得更积极，但安全保留不够”，不支持“完全没有梯度、策略不动”。
- SFT起点native组训练reward完全无差异的比例，official约0.35%、PSI约0.425%、A5约33.625%、V6约31.85%。因此A5/V6确有更多饱和组，但仍有大量非饱和组；不能说所有组都没信号。无方差的raw reward也不能直接等同所有shaped advantage精确为0。

### 组内有差异，不代表没有梯度或一定有用

额外调用原生LFP advantage函数，使用真实历史reference cache，按固定token-hash组织64scene批次，每scene G16，最后batch40。这个表是新样本上的规则诊断，**不是历史minibatch/curriculum重放**。native_PIA以未同时满足NC=DAC=TTC=DDC=1的rollout数为分母，不能与LFP自己的相对DDC feasibility混用。

{table(credit[credit.protocol=='native_grpo'])}

`metrics/group16_metrics.parquet`中的`vanilla_proxy_*`只是所有模型共用的原始zscore对照，**不是**PSI SR-PGRPO或LFP实际advantage。真实历史日志（`historical_grpo_credit.csv`）显示A5首/末完整epoch约36.43%/41.54% rollout有正advantage，V6约29.42%/32.49%；不能解释成“组内全一样所以没有梯度”。PSI历史平均occupied support bucket约2.46；这是其标准化descriptor分桶，不等于0.5米teacher命中，也不等于语义模式数。

## 7. 为什么历史GRPO结果差那么多：证据支持到哪里

**SUPPORTED：SFT的有效监督不同。** A5/V6中GT占大多数有效权重，V6非GT误差显著更高；PSI将更多监督权重放在候选上，且同legacy配对loss确有候选拟合改善和GT损失。不能把“候选在文件中”当作充分吸收。

**PARTIALLY SUPPORTED：从头多轨迹SFT偏向GT、未充分扩展候选覆盖。** A5/V6接近完整覆盖监督GT，在相同非GT teacher库上相对official的覆盖增幅却不显著；PSI更宽且有可检测的增幅。它支持有限的“GT集中、候选覆盖增量有限”机制，不支持把原始候选条数直接等同于学到的mode数。

**SUPPORTED：实际探索分布显著不同，且有采样参数干预证据。** legacy sampler将同一SFT权重的组分布显著打散；FS sampler的物理尺度小得多。A5/V6的高组均分与小range，和official/PSI同时包含好坏轨迹的大range，是两种优化起点。最高分相近不意味着同样的改进空间，更不意味着reward-tail相同。

**NOT SUPPORTED：只要teacher采不到，就会导致GRPO退化。** PSI在主0.5米native采样下未命中仍有成功长训；A5对自身teacher覆盖并不比PSI差。几何小球覆盖、去噪拟合、可行高分行为、训练可迁移性必须分开。

**PARTIALLY SUPPORTED：保留约束与策略位移是关键候选原因。** 成功的original GRPO有reference-generated BC，PSI SR-PGRPO有BC（0.1→0.05）和KL；A5/V6 LFP的BC=0，只保留不同定义的transition KL。LFP还使用不同的reference/Pareto/进度/TTC credit gates与frontier curriculum。观察到的宽度/中心/安全变化与这一解释相容，但没有在同一initialization上做BC开关干预，不能把它写成已确定的单独因果原因。

**SUPPORTED：历史失败不能只归因于训练rollout没有变好。** 旧正式长训审计已有A5/V6训练目标改善而Navtest安全尾部恶化的证据。A5/V6的Navtest增益被NC/TTC退化抵消；这与本轮NAVTRAIN分布必须分开读。详见[历史长训报告](HISTORICAL_SFT_GRPO_LONGCHAIN_AUDIT_20260914.md)。A5末段相对SFT并非统计显著负增益，更准确是前期明显下降、后续未得到稳定提升；V6也不能概括为持续确定下降。

**UNTESTED：仅改变SFT初始化就会改变后续GRPO成败。** 历史几条链同时改变representation、监督库/权重、GRPO算法、LR、reference约束、scene sampler和预算。缺少这些因素固定的matched训练干预，且PSI GRPO实体缺失。最保守解释是：几种SFT产物形成了不同的目标拟合和支持结构，又被不同物理采样尺度与不同更新约束放大；当前证据不能选择其中一个因素作为唯一原因。

## 8. 复现、边界与交付

所有新增代码在`tools/analysis/historical_sft_grpo_support/`，冻结配置在对应`configs/`。原始rollout、teacher tensor、逐目标4噪声loss与真实评分留在`outputs/historical_sft_grpo_support/cache/`，不提交大缓存；可追溯索引及hash在`manifests/`。逐组、逐scene、逐teacher表保留，未只保留成功场景。

体积较大的CSV另保存同名Parquet并提交，CSV仍留在本地；转换与文件hash见`audits/published_tables.json`。官方IL的GT行作为统一GT拟合参考，不据此声称已恢复released IL的逐token训练清单；共同835身份严格指PSI/A5/V6训练交集。

历史源码恢复存在边界：PSI是可恢复的后续commit；A5未提交改动是否完全缺席不能证明；V6应用了原run保存patch，并从V6 worktree恢复缺失的未跟踪anchor依赖（该模型anchor功能关闭）。记录了实际import路径及文件hash，strict state schema加载；parity是相对这些已恢复native入口，不冒称拥有不可恢复的原始运行二进制。

工程修复记录在`audits/engineering_changes.json`：恢复缺失依赖；CPU扫描忽略atomic临时文件；训练reward派生公式在真实evaluator审计中发现DDC不应乘入，已在主统计前纠正并通过scalar/batch parity；独立group批处理FP32误差量化。没有调整主采样参数、teacher集合或scene身份去追求结论。

所有核心图有PNG/PDF/SVG与绘图CSV。固定例子只按共同训练token hash选一个场景，不按结果选胜者：

![固定场景真实teacher与16次采样](../outputs/historical_sft_grpo_support/figures/Fig-M5_fixed_scene_trajectories.png)

这轮足以否定“只看SFT PDMS”“普通采样等于GRPO采样”“没精确采到teacher就必然GRPO失败”的解释；足以确认部分监督吸收与显著的采样尺度差异。**要唯一定位历史GRPO成败，还需要同representation、同recipe、同reference约束的matched训练对照；本轮没有进行这种权重更新。**
'''
    path=ROOT/'reports/HISTORICAL_SFT_GRPO_SUPPORT_MECHANISM_20260915.md';path.write_text(text)
    save(OUT/'manifests/run_summary.json',dict(status='COMPLETE_READ_ONLY_DIAGNOSTIC',scene_count=1000,common_train=835,checkpoint_count=9,group_size=16,groups_per_scene=4,precision='fp32',primary_rollouts=1152000,factorial_rollouts=196608,optimizer_updates=0,PSI_GRPO_entities='MISSING',causal_isolation='UNTESTED',report=str(path),config_sha256=identity()))
    print('REPORT',path,flush=True)

if __name__=='__main__':main()
