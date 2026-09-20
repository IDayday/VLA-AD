"""Render measured results and limitations; never insert expected outcomes."""
from common_5000 import *
import time

NAMES={'official_il':'Initial GT-IL','original_grpo_11970':'Original GRPO (90.41 checkpoint)','psi_sft':'PSI candidate-SFT'}

def main():
    audit=read(OUT/'audits/final.json');assert audit['status']=='PASS'
    start=time.time()
    while not (OUT/'audits/teacher_coverage.json').exists():
        if time.time()-start>3600:raise TimeoutError('Supplemental teacher coverage incomplete')
        time.sleep(10)
    teacher_audit=read(OUT/'audits/teacher_coverage.json');assert teacher_audit['status']=='PASS'
    summary=pd.read_csv(OUT/'metrics/summary.csv');cmp=pd.read_csv(OUT/'metrics/paired_comparisons.csv')
    full=summary[summary.scope=='FULL5000'];manifest=read(OUT/'manifests/scenes.json');protocol=read(OUT/'manifests/protocol.json')
    lines=['# IL → GRPO 与 PSI-SFT：5000场景轨迹分布复核（2026-09-20）','',
        '本报告使用真实缓存与新增采样，不按期望结论选择场景。三个checkpoint全程冻结，无SFT或GRPO权重更新。',
        '',f'完成 **5000/5000场景、120000组G16、1920000条轨迹**；新增4000场景、1536000条轨迹，其余复用经SHA256核验的旧缓存。缺失场景0、未解决评分异常0。',
        '', '## 设计和解释范围','',
        f"母体为canonical Navtrain的{manifest['population']}个token。保留先前随机1000场景，按指令类别分层、token hash随机增加4000；合计直行3170、左转1257、右转573。旧1000、新4000、全5000分别公开。新4000与旧1000 token不重叠，但可以来自相同驾驶日志，因此额外给出log-cluster区间。",
        '', '每个checkpoint、每个场景分别使用普通evaluation sampler与真实forward_grpo采样入口，各采4组×16=64条。模型间和采样协议间使用common random numbers；不同场景／组独立。使用FP32，同一观察特征、统一真实NAVSIM-v1评分。G16是本次标准化诊断；历史正式original GRPO用G8，本报告没有将G16的统计冒充历史G8训练日志。',
        '', '这批场景来自Navtrain，可能参与过历史训练。它们是内部机制分析，不是untouched Navtest泛化结果；表中采样均分不能与checkpoint名称中的90.41等历史单条评测分数直接比较。',
        '', '比较范围是Official IL→original GRPO与Official IL→PSI候选SFT。它不是MTS-86.92／87.51从头联合训练的5000场景复现，也不是PSI-SFT→GRPO因果链：后者历史GRPO权重仍未找到，不以APR替代。',
        '', '## 主要实测结果','',
        'Pairwise ADE64：全部64条轨迹之间的平均XY ADE；不含heading。Centroid displacement：64条均值轨迹相对同采样协议Official IL均值轨迹的ADE。Feasible：NC、DAC、TTC、DDC全部为1；PDMS以0–100 points显示。每个指标先在场景内算，再对5000场景等权平均。']
    for sampler in CFG['protocols']:
        d=full[full.protocol==sampler].copy();table=[]
        for _,r in d.iterrows():
            table.append({'Policy':NAMES[r.model],'Pairwise ADE64 (m)':r.pairwise_ADE64,'Centroid displacement (m)':r.centroid_displacement,'Feasible (%)':r.feasible_rate*100,'Mean PDMS':r.mean_PDMS})
        lines+=['',f'### {sampler}','',pd.DataFrame(table).to_markdown(index=False,floatfmt='.4f')]
    lines+=['','来源：`metrics/summary.csv`，scope=FULL5000，各行分母5000场景／320000条轨迹。组内16条Pairwise ADE另存`pairwise_ADE`字段；主表使用更充分的64条估计。','', '## 配对差值与置信区间','',
        '差值方向为当前checkpoint−Official IL。95% CI来自3000次scene-paired bootstrap；log-cluster CI对整段日志重采样。下面的概率相关差值以百分点呈现。']
    for sampler in CFG['protocols']:
        selected=cmp[(cmp.scope=='FULL5000')&(cmp.protocol==sampler)&(cmp.reference=='official_il')]
        table=[]
        for _,r in selected[selected.metric.isin(['pairwise_ADE64','centroid_displacement','feasible_rate','mean_PDMS','min_PDMS','max_PDMS'])].iterrows():
            scale=100 if r.metric=='feasible_rate' else 1
            table.append({'Checkpoint':NAMES[r.model],'Metric':r.metric,'Mean Δ':r.mean_difference*scale,'Median Δ':r.median_difference*scale,'95% CI':f'[{r.ci_low*scale:.4f}, {r.ci_high*scale:.4f}]','Log-cluster CI':f'[{r.log_cluster_ci_low*scale:.4f}, {r.log_cluster_ci_high*scale:.4f}]','Scene Δ>0 (%)':r.win_fraction*100,'N':r.n})
        lines+=['',f'### {sampler} 配对比较','',pd.DataFrame(table).to_markdown(index=False,floatfmt='.4f')]
    lines+=['','来源：`metrics/paired_comparisons.csv`。Δ>0仅表示数值增加，并非所有指标上都意味着更好；几何宽度与中心位移尤其不能直接按大小判优。','', '## GRPO组内的质量分布','',
        '下表每个最小值／最大值均先在真实16条组内求，再对4组和5000场景平均。它们不是整批数据的单个极值。']
    tail=full[['model','protocol','min_PDMS','mean_PDMS','max_PDMS','std_PDMS','CVaR25_PDMS','no_safe_group']].copy();tail['no_safe_group']*=100
    tail.rename(columns={'no_safe_group':'No-safe group (%)'},inplace=True)
    lines+=['',tail.to_markdown(index=False,floatfmt='.4f'),'', '来源：`metrics/group16_metrics.parquet`（120000组）与`metrics/summary.csv`。组内没有安全轨迹时best-safe PDMS记NA，并报告该类组比例。', '', '## 新增4000场景是否复现原结论','']
    rep=cmp[(cmp.reference=='official_il')&cmp.metric.isin(['pairwise_ADE64','feasible_rate','mean_PDMS'])].copy()
    lines+=[rep[['scope','protocol','model','metric','mean_difference','ci_low','ci_high','n']].to_markdown(index=False,floatfmt='.5f'),'',
        '这一表中的feasible_rate仍为0–1单位，乘100即百分点。NEW4000来自事前抽样，不因模型输赢而筛选。FULL5000包含旧1000，不能把它当作完全独立复现；独立token复核应看NEW4000。']
    def get(model,pr,metric):
        return cmp[(cmp.scope=='FULL5000')&(cmp.model==model)&(cmp.protocol==pr)&(cmp.reference=='official_il')&(cmp.metric==metric)].iloc[0]
    teacher_summary=pd.read_csv(OUT/'metrics/psi_teacher_summary.csv')
    teacher_cmp=pd.read_csv(OUT/'metrics/psi_teacher_comparisons.csv')
    tt=teacher_summary[(teacher_summary.scope=='FULL5000')&(teacher_summary.membership=='PSI_TRAIN')&(teacher_summary.kind=='NON_GT')]
    tc=teacher_cmp[(teacher_cmp.scope=='FULL5000')&(teacher_cmp.membership=='PSI_TRAIN')&(teacher_cmp.kind=='NON_GT')&(teacher_cmp.metric=='hit64_0p5')]
    lines+=['','## 真实PSI监督轨迹的采样覆盖（5000场景补充复核）','',
        f"依据原train_args中的train/val log名单核验：{teacher_audit['training_scenes']}个PSI训练场景、{teacher_audit['validation_scenes']}个验证场景；其中{teacher_audit['training_scenes_with_non_gt']}个训练场景具有正权重非GT教师。教师张量与权重直接读取原support index，其SHA256与历史审计一致。",
        '', '下表只统计实际训练场景内的正权重非GT监督。对每条教师，64次rollout中至少一条XY ADE≤0.5m即命中；先在场景内平均教师命中，再对场景等权平均。表中命中率／mass为0–1单位；这不是native diffusion loss，也不是精确轨迹生成概率。',
        '',tt[['model','protocol','scenes','teachers','hit64_0p5','hit16_0p5','mass64_0p5','nearest_ADE','weighted_hit64_0p5']].to_markdown(index=False,floatfmt='.5f'),
        '', '相对IL的Hit@64配对差值：', '',tc[['model','protocol','mean_difference','ci_low','ci_high','log_cluster_ci_low','log_cluster_ci_high','n']].to_markdown(index=False,floatfmt='.5f'),
        '', '完整结果含GT教师、验证场景、0.25/1.0m敏感性、旧1000／新4000拆分，见`psi_teacher_summary.csv`、`psi_teacher_comparisons.csv`。64次未命中仅表示有限采样下没有观测到，不能推断概率为零、严格不可达或不可学。',
        '','## 可以支持什么，不能支持什么','']
    for model in CFG['primary_models'][1:]:
        for pr in CFG['protocols']:
            q=get(model,pr,'mean_PDMS');w=get(model,pr,'pairwise_ADE64');f=get(model,pr,'feasible_rate');lo=get(model,pr,'min_PDMS');hi=get(model,pr,'max_PDMS')
            lines+=[f"- {NAMES[model]} / {pr}：相对IL，均分{q.mean_difference:+.3f}，可行率{100*f.mean_difference:+.3f}个百分点，宽度{w.mean_difference:+.4f}m；组内最低分{lo.mean_difference:+.3f}、最高分{hi.mean_difference:+.3f}。"]
    lines+=['',
        '解释应区分三个维度：几何宽度、分布中心和质量尾部。组内最低分改善而宽度近似不变，符合“相同探索噪声下提高采样质量”的观察；它不自动证明策略只在原有语义mode内微调。宽度增大且可行率下降，则意味着扩大的采样范围同时包含更多失败行为，不能称为新增feasible modes。中心位移必须在同一sampler下比较；0.5m左右的位移也不能直接称为无变化。',
        '', '普通推理与GRPO采样不是同一个分布。已审计的runtime中，evaluation noise floor=0.0001，native sampling floor=0.04；evaluation noise clip=±1，native=±5，logprob std floor=0.1。0.1用于密度计算，不能当成实际采样标准差。具体步骤、模块train/eval状态和实际import路径见`audits/sampling_*.json`。前次1000场景factorial诊断保留用于解释噪声机制，本轮没有再修改sampling floor或clip。',
        '', '本轮把实际教师的几何采样覆盖扩展到了5000场景，但没有重算5000场景的native epsilon loss；历史loss分析仍只有原1000场景。几何命中和训练loss衡量不同问题，即使平均loss下降或教师命中增加，也不能声称所有候选teacher都已被学会。',
        '', 'SUPPORTED / NOT_SUPPORTED的判定需结合实测方向与区间：只有均分上升、可行率也上升、且扩展独立复现时，才能描述为质量与安全同时改善。宽度扩大本身不是价值。无新权重更新、无matched训练消融，本轮仍不能给出“特定SFT损失导致后续GRPO退化”的唯一因果解释。',
        '', '## Provenance 与审计','',
        f"- 新分支：`analysis/il-rl-psi-distribution-5000-20260920`；parent `{protocol['parent_commit']}`。",
        f"- 冻结配置SHA256：`{protocol['config_sha256']}`。",
        f"- Scene manifest SHA256：`{protocol['scene_manifest_sha256']}`。",
        '- 三个checkpoint路径／SHA256、实际归档runtime路径／SHA256见`manifests/models.json`；旧缓存原protocol metadata不改写。',
        '- 原6000份rollout／score文件哈希一致；4scene FP32特征重建与原缓存完全一致；scalar-vs-batch NAVSIM评分完全一致。',
        '- 每个checkpoint重新运行native采样入口parity和独立组批处理parity；新采样与旧缓存误差阈值1e-4m/rad；全部rank完成后参数与buffer SHA一致。',
        '- 输入仅含4帧历史观察；未来GT／metric cache仅离线评分使用；没有optimizer step。',
        '- PSI checkpoint真实存在，但其原始当日源码快照未完整恢复；使用经严格state-dict与采样一致性核验的可恢复归档runtime。这一历史provenance限制并未因样本扩增而消失。',
        '', '## 文件索引','',
        '- 配置：`configs/il_rl_psi_distribution_5000/primary.yaml`。',
        '- 原始分场景与分组统计：`outputs/il_rl_psi_distribution_5000/metrics/scene_metrics.parquet`、`group16_metrics.parquet`。',
        '- 汇总／配对CI：`metrics/summary.csv`、`paired_comparisons.csv`；分驾驶指令：`command_strata.csv`。',
        '- 图1：`figures/Fig1_distribution_and_quality.{png,pdf,svg,csv}`。',
        '- 图2：`figures/Fig2_group16_quality_tails.{png,pdf,svg,csv}`。',
        '- 图3：`figures/Fig3_independent_replication.{png,pdf,svg,csv}`。',
        '- 图4：`figures/Fig4_scene_distribution_ECDF.{png,pdf,svg,csv}`；显示完整场景分布范围，不截尾。',
        '- 最终完整性：`audits/final.json`；全部输入／输出hash：`manifests/observations.json`、`cache_hashes.json`。',
        '', '统计和图形只读取实际缓存，没有生成理想结果，没有按checkpoint表现更换场景，没有修改旧实验数据。','']
    report=WORK/'reports/IL_RL_PSI_DISTRIBUTION_5000_20260920.md';report.write_text('\n'.join(lines));print(report)

if __name__=='__main__':main()
