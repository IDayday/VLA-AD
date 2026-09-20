"""Render factual results; causal scope is limited to this matched SFT budget."""
from common_matched import *
def md(f):
    cols=list(f.columns);lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in f.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('NA' if pd.isna(x) else f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)
def main():
    audit=read(OUT/'audits/final.json');assert audit['status']=='PASS'
    summary=pd.read_csv(OUT/'metrics/summary.csv');hold=summary[summary.split=='holdout'].copy();hold['feasible_%']=hold.feasible_rate*100
    comp=pd.read_csv(OUT/'metrics/paired_comparisons.csv');pair=comp[(comp.split=='holdout')&(comp.left=='psi')&comp.right.isin(['il_sft','score','pareto'])&comp.metric.isin(['mean_PDMS','feasible_rate','centroid_displacement','pairwise_ADE64'])].copy()
    composition=pd.read_csv(OUT/'metrics/supervision_composition.csv');targets=composition.groupby(['split','method'])[['count','weighted_PDMS','weighted_feasible','GT_weight','target_pairwise_ADE','fallback']].mean().reset_index()
    teacher=pd.read_parquet(OUT/'metrics/teacher_scene_metrics.parquet');teacher=teacher[(teacher.protocol=='eval')&((teacher.step==512)|(teacher.method=='official_il'))&(teacher.teacher_origin=='psi')&(teacher.kind=='NON_GT')]
    teacher=teacher.groupby(['split','method','token'],as_index=False)[['native_epsilon_loss','Hit64_0p5','Mass64_0p5','nearest_teacher_ADE']].mean()
    ts=teacher.groupby(['split','method']).agg(scenes=('token','size'),loss_scenes=('native_epsilon_loss','count'),epsilon_MSE=('native_epsilon_loss','mean'),Hit64=('Hit64_0p5','mean'),mass64=('Mass64_0p5','mean'),nearest_ADE=('nearest_teacher_ADE','mean')).reset_index()
    allteacher=pd.read_parquet(OUT/'metrics/teacher_scene_metrics.parquet')
    gt=allteacher[(allteacher.protocol=='eval')&((allteacher.step==512)|(allteacher.method=='official_il'))&(allteacher.teacher_origin=='il_sft')&(allteacher.kind=='GT')]
    gt=gt.groupby(['split','method','token'],as_index=False)[['native_epsilon_loss','Hit64_0p5','nearest_teacher_ADE']].mean()
    gt=gt.groupby(['split','method']).agg(loss_scenes=('native_epsilon_loss','count'),GT_epsilon_MSE=('native_epsilon_loss','mean'),GT_Hit64=('Hit64_0p5','mean'),GT_nearest_ADE=('nearest_teacher_ADE','mean')).reset_index()
    seeds=pd.read_csv(OUT/'metrics/summary_by_seed.csv');seeds=seeds[(seeds.split=='holdout')&(seeds.step==512)][['method','seed','protocol','mean_PDMS_mean','feasible_rate_mean','centroid_displacement_mean','pairwise_ADE64_mean']]
    text=f'''# PSI、Score、Pareto与GT-only的等预算IL续训实验

状态：实际训练、采样、NAVSIM评分及审计完成。结论仅针对本次预算与共享候选库；不代表完整训练性能，也不是 downstream GRPO 更新实验。

## 1. 这次真正控制了什么

- 初始化：同一个官方GT-IL checkpoint，SHA256 `{read(OUT/'manifests/protocol.json')['initial_checkpoint']}`。
- 训练3072场景，评估沿用原随机5000场景，两个集合的log交集为0。原5000场景覆盖991个log；排除后可用3268场景，按预定哈希取3072。这个可用性调整发生在protocol冻结和任何评分/训练之前。
- 四方法各2个seed（1701/2903），512次AdamW更新；batch128=microbatch16×accumulation8；每run65536次场景监督，约21.33个训练集遍历。相同seed下场景顺序、noise/timestep随机数完全对应。
- 全部训练同一34329219个原生action-head参数；VLM不更新。FP32；AdamW betas=(0.9,0.95)，weight decay=1e-4，clip=1；峰值LR5e-5、warmup96步、固定cosine降至1e-6。沿用历史优化器关键设置，缩短并预先冻结了schedule；不是历史60-epoch重放。
- 只改变目标选择。都使用原生trajectory normalization、uniform t∈[0,99]、epsilon prediction MSE。每场景每次按目标权重抽1条；没有额外GT retention loss。IL-SFT始终抽GT。
- 全量主结果使用step512；step128只作预先定义1000场景上的学习曲线，不按得分选最好checkpoint。
- 5000场景每个模型分别普通eval和真实forward_grpo采样：每种4个独立G16，共64条。官方IL原始缓存按hash复用；新checkpoint全部实采实评。输入仅观察特征，GT/teacher仅作为训练标签或离线评估对象。
- 这些是Navtrain内部、相对本轮微调log隔离的诊断场景；初始公开权重和历史teacher未必未见过它们。不得称为untouched Navtest。

冻结配置SHA256：`{identity()}`。配置：`configs/psi_matched_sft/primary.yaml`。manifest：`outputs/psi_matched_sft/manifests/`。

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

三种候选方法统一使用原PSI权重函数：best质量0.5、被选GT质量0.2、其他目标共享0.3；GT未被选则其0.2给best；无其他目标时0.3也给best；单目标权重1。不同选集会产生不同实际GT质量，这是selection bundle的组成部分，不能声称实际GT比例完全相同。目标描述符均值/方差只从3072训练场景计算。

**数据恢复限制**：历史PSI完整AWAC raw库已缺失，只有历史最终support index仍在。因此本次是原PSI选择机制在共享可恢复raw库上的前瞻受控对照，不能冒称重新训练了原始历史PSI数据。

共享库来自V6 archive的selection之前原始轨迹：GT、结构化扰动、8条A5策略候选，以及实际DDV2/DrivOR输出。此处`policy`来源是A5 checkpoint，并非当前official IL。所有候选重新以同一NAVSIM-v1 evaluator评分，跨来源完全重复合并保留标签。`raw_source_summary.csv`公开真实来源数量，`selected_targets.parquet`保存可恢复轨迹、archive hash和原索引。

## 3. 监督目标差异

{md(targets)}

上表场景等权；PDMS为0–100 points，可行率与GT_weight为0–1。Score/Pareto训练集完全相同选集比例为{composition.query('split=="train"').score_pareto_identical.mean():.4%}。因此它们接近可能来自实际目标退化，不能临时换Pareto目标制造差异。

## 4. Full-5000最终分布与质量

{md(hold[['method','protocol','mean_PDMS','feasible_%','pairwise_ADE64','centroid_displacement','Spread_AUC','min_PDMS','max_PDMS']])}

先在场景内平均64次采样，再对5000场景等权；最后平均两训练seed。Pairwise ADE64为64条两两XY-ADE；centroid displacement为同sampler下相对初始IL的平均轨迹中心ADE。min/max PDMS是每真实G16内min/max后平均，不是全数据极值。联合可行定义为NC/DAC/TTC/DDC全1。

`metrics/summary.csv`与`summary_denominators.csv`；原始场景/组表为`scene_metrics.parquet`与`group16_metrics.parquet`。

## 5. PSI相对三个微调baseline的场景配对差

{md(pair[['protocol','right','metric','n','mean_difference','median_difference','ci_low','ci_high','log_cluster_ci_low','log_cluster_ci_high','win_fraction']])}

这里差值=PSI-baseline；feasible_rate差值单位是比例，乘100得到百分点。win_fraction仅表示差值>0，对loss/center等不等同于性能胜率。3000次scene-paired bootstrap，同时报告log-cluster sensitivity。先在每scene平均两个seed，未把10000个scene-seed当10000独立场景；CI条件于这两个已训练模型，不能代表充分的训练seed不确定性。

## 6. 两seed是否一致

{md(seeds)}

必须同时看两个seed，不能只挑PSI较好的一次。完整配对表`paired_by_seed.csv`。step0/128/512使用相同1000场景的曲线见`matched1000_evolution.csv`。

## 7. 有无真正吸收相同监督信号

所有模型评价同一PSI non-GT target，避免“每个方法只看自己的teacher”造成直接难度混杂：

{md(ts)}

相同GT目标上的保留情况（所有方法都评价同一个GT，避免GT仅在部分PSI选集中出现造成分母混淆）：

{md(gt)}

覆盖指标在全部5000 holdout及固定512个train probe计算；native epsilon loss在固定1000 holdout及512 train probe上，16个共同noise/timestep draws。无non-GT teacher的scene不进入该teacher分母，表中公开scenes/loss_scenes。Hit64是64次中至少一条ADE≤0.5m，不是概率或可学习性的充分判据；mass64是64次中的附近频率。`teacher_summary.csv`同时包含各来源自己的teacher、GT与共同跨方法teacher比较。

训练loss降低、输出覆盖增加、可行高质量样本增加是三个不同检验。即使noise-MSE更低，64次没采到也只能说明本采样预算和邻域下低覆盖，不能断言永远生成不了。`target_presentations.parquet`记录目标究竟被实际抽到训练多少次，零次不伪称学过。

## 8. 为什么必须分开普通eval和GRPO采样

两者同一权重，但原实现采样floor/clip不同：eval floor0.0001、noise clip±1；native GRPO floor0.04、clip±5；logprob floor0.1是独立参数。在当前legacy归一化下，native最后一步0.04的噪声floor对应约1.3348m的X标准差、0.84m的Y标准差（裁剪前）；这也解释了为何0.5m/8点ADE邻域中的有限次命中可能极低。GRPO组内宽度与尾部风险因此不能从普通eval宽度直接外推。这里只调用真实forward_grpo采样、在reward/advantage/optimizer前截取；没有GRPO权重更新。G16是统一诊断group，历史成功训练原本用G8，不冒称重放历史G8更新。

## 9. 证据范围与暂不能成立的因果表述

- SUPPORTED：本轮直接干预了相同IL初始化之后的监督选择，预算、loss、网络和随机流一致；因此可讨论本次selection bundle对SFT输出的影响。
- 不得把结果分解成“单独diversity门槛”“单独quality band”“单独目标数量”的因果贡献：这些在PSI bundle中同时不同；需后续消融才能拆分。
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
'''
    p=WORK/'reports/PSI_MATCHED_SFT_20260920.md';p.write_text(text);print(p)
if __name__=='__main__':main()
