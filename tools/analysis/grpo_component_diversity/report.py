from shared import *
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    s=pd.read_csv(OUT/'metrics/method_summary.csv');s=s[s.grouping=='prefix'];c=pd.read_csv(OUT/'metrics/paired_comparisons.csv');a=read(OUT/'audits/final.json')
    labels={'official_il':'Initial IL','original_grpo_11970':'Original GRPO-90.41'}
    lines=['# IL → original GRPO：分项结果多样性与组内改进信号（5000场景）','',
    '本报告是冻结策略的只读机制诊断。训练更新为0；不是新的SFT/GRPO训练实验，也不是untouched Navtest泛化评测。历史checkpoint名称中的90.41不能与这里随机采样均分直接比较。','',
    '## 1. 数据与固定比较','',
    f"完成 {a['scene_count']} 场景、{a['log_count']} logs，缺失 {a['missing_scenes']}；两策略各128条/场景，共{a['rollouts']:,}条。复用旧64条，额外真实采样64条。没有有放回抽样伪造128条，也没有按结果选择场景。",
    '同场景、同逐步Gaussian随机流，分别使用原生 `forward_grpo`。G=8/16/32/64/128主结果取前G条，补充结果平均所有互不重叠G组。不同G本身的有限样本量不同；更大的best-of-G或更多观察到的组合，不表示policy变强。历史训练本来使用G8，这里不是历史训练batch重放。',
    '原始5000场景及其旧均分已经被观察过；本轮规则在新分项结果分析前冻结，不称为盲化预注册研究。',
    f"主配置SHA256：`{identity()}`。场景、checkpoint、代码和缓存身份见 `manifests/`。",
    '', '## 2. Pairwise ADE回答什么，不能回答什么','',
    'Pairwise ADE是两次输出的平均几何距离，适合描述分散程度；它不是概率密度宽度、驾驶语义模式数量或有用探索的充分指标。共同采样噪声、速度/行程差异、少量离群轨迹都能增大它。替代/补充包括pairwise中位数/P90、medoid距离P50/P90和XY协方差effective rank；这些仍是几何指标。effective rank在G8最多7，不能将其随G增长直接解释为出现更多行为模式。',
    '本报告重点采用结果空间：NC/DAC/TTC/DDC真实离散取值的联合模式、分项方差和饱和率、安全子集的EP提升空间、以及这些差异是否被真实训练reward/advantage识别。结果组合不同不等于驾驶语义一定不同；同一组合仍可能包含不同刹车、让行或线路选择。',
    '', '## 3. G16核心比较','']
    q=s[s.G==16].set_index('model');metrics=[('mean_PDMS','采样PDMS均值',1),('feasible_rate','NC/DAC/TTC/DDC均满分 (%)',100),('mean_pairwise_ADE','Pairwise ADE (m)',1),('medoid_radius_P90','Medoid radius P90 (m)',1),('safety_pattern_disagreement','随机样本对安全结果不同 (%)',100),('effective_safety_patterns','观察到的有效安全结果组合数',1),('safe_EP_headroom','安全子集EP best-minus-mean (0–100点)',100),('safe_EP_opportunity','存在≥5 EP点安全进度差异的组 (%)',100),('no_safe_group','没有安全成员的组 (%)',100),('zero_advantage_group','训练reward无差异的组 (%)',100),('positive_infeasible_rate','P(A>0 | conservative-infeasible) (%)',100),('unsafe_reward_winner_with_safe_alternative','有安全替代但reward winner不安全的组 (%)',100)]
    il=q.loc[CFG['models'][0]];rl=q.loc[CFG['models'][1]]
    findings=['## 主要发现','',
        f"**SUPPORTED：质量改善并不需要几何变宽或失败模式增多。** G16的采样PDMS {il.mean_PDMS:.2f}→{rl.mean_PDMS:.2f}，可行率{il.feasible_rate*100:.2f}%→{rl.feasible_rate*100:.2f}%；但pairwise ADE {il.mean_pairwise_ADE:.5f}→{rl.mean_pairwise_ADE:.5f}m，其配对CI跨0。安全结果对的不一致率{il.safety_pattern_disagreement*100:.2f}%→{rl.safety_pattern_disagreement*100:.2f}%，是更少失败结果差异，与质量改善同时发生。",
        f"**NOT SUPPORTED：GRPO一致改善所有安全子项。** DAC {il.DAC_mean*100:.2f}→{rl.DAC_mean*100:.2f}，TTC {il.TTC_mean*100:.2f}→{rl.TTC_mean*100:.2f}，但DDC {il.DDC_mean*100:.2f}→{rl.DDC_mean*100:.2f}。存在安全替代、但固定first-argmax的最高reward样本不安全的组占全部5000场景的{il.unsafe_reward_winner_with_safe_alternative*100:.2f}%→{rl.unsafe_reward_winner_with_safe_alternative*100:.2f}%。这些方向在all-blocks敏感性与log-cluster CI中也保留。这里不是GRPO执行了winner选择；它只是对组内reward排名的诊断。",
        f"**SUPPORTED：GRPO之后仍有可区分的改进方向，但不是越多越好。** {rl.safe_EP_opportunity*100:.2f}%的G16组包含EP相差至少5点的安全样本；安全子集best-minus-mean EP为{rl.safe_EP_headroom*100:.2f}点。DAC/TTC/Comfort在{rl.DAC_mixed*100:.2f}%/{rl.TTC_mixed*100:.2f}%/{rl.Comfort_mixed*100:.2f}%的组内同时出现满分与非满分，可提供不同的改进对照。Headroom减少也可能是平均质量已提高，并非必然代表探索能力变差。",
        '**PARTIALLY SUPPORTED：现有reward能够区分大部分安全/进度差异，但并未完整覆盖评价目标。** DDC是被忽略的目标；EP与TTC之间允许补偿。Cov(A,x)和正advantage分配揭示这些局限，但没有梯度或训练干预证据可以把全部DDC下降归因于某一项权重。','']
    lines[4:4]=findings
    rows=[]
    for col,label,scale in metrics:
        cc=c[(c.grouping=='prefix')&(c.G==16)&(c.metric==col)].iloc[0]
        rows.append([label,f'{q.loc[CFG["models"][0],col]*scale:.4f}',f'{q.loc[CFG["models"][1],col]*scale:.4f}',f'{cc.mean_difference*scale:+.4f} [{cc.ci_low*scale:+.4f}, {cc.ci_high*scale:+.4f}]',str(int(cc.scenes))])
    lines += [table(['指标','IL','GRPO','配对差及scene-bootstrap 95% CI','共同定义场景'],rows),'',
    '表来自 `metrics/method_summary.csv`（grouping=prefix,G=16）和 `paired_comparisons.csv`。条件指标各自均值的分母可能不同，因此差值使用共同有定义场景，不能直接拿两个条件均值相减当作同一总体因果效应。所有主结果场景覆盖仍是5000；缺少安全成员的组不被删除。各指标有 `_n` 字段，完整中位数、log-cluster CI、scene win fraction及主检验族Holm调整见CSV。','',
    '### 各分项','']
    ties=pd.read_csv(OUT/'metrics/reward_winner_tie_audit.csv').groupby('model').mean(numeric_only=True)
    lines[-2:-2]=[f"补充并列审计：最高分统计使用固定first-argmax。去掉并列，只计 `max_unsafe_reward > max_safe_reward + 1e-8`，IL为{ties.loc[CFG['models'][0],'strict_unsafe_winner']*100:.2f}%，GRPO为{ties.loc[CFG['models'][1],'strict_unsafe_winner']*100:.2f}%（各5000场景）。安全与不安全样本并列最高的场景另列，分别为{ties.loc[CFG['models'][0],'safe_unsafe_tie_at_max']*100:.2f}%/{ties.loc[CFG['models'][1],'safe_unsafe_tie_at_max']*100:.2f}%。见 `reward_winner_tie_audit.csv`；未修改预先固定的主指标。",'']
    rows=[]
    for n in ['NC','DAC','EP','TTC','DDC','Comfort']:
        rows.append([n]+[f'{q.loc[m,n+"_mean"]*100:.3f}' for m in CFG['models']]+[f'{q.loc[m,n+"_mixed"]*100:.3f}' for m in CFG['models']]+[f'{q.loc[m,"covariance_adv_"+n]:+.5f}' for m in CFG['models']])
    lines += [table(['分项','IL均值(0–100)','GRPO均值','IL混合组%','GRPO混合组%','IL Cov(A,x)','GRPO Cov(A,x)'],rows),'',
    '“混合”是同组既有满分又有非满分；对于连续EP，它是满分饱和的混合，不是EP模式数。完整std/P10/P50/P90/min/max、全满分/全非满分，以及真实0/0.5/1等取值分布均保存。','',
    '## 4. G=8/16/32/64/128：增加样本究竟得到什么','']
    rows=[]
    for _,r in s.sort_values(['G','model']).iterrows():
        rows.append([labels[r.model],int(r.G),f'{r.mean_PDMS:.3f}',f'{r.min_PDMS:.3f}',f'{r.max_PDMS:.3f}',f'{r.feasible_rate*100:.2f}',f'{r.no_safe_group*100:.2f}',f'{r.safe_EP_opportunity*100:.2f}',f'{r.zero_advantage_group*100:.2f}'])
    lines += [table(['Policy','G','均值PDMS','组内最小PDMS均值','组内最大PDMS均值','安全样本%','无安全成员组%','安全EP机会组%','零reward差异组%'],rows),'',
    '最小/最大值是每场景组内统计后对5000场景等权平均，不是整个数据集的一个极值。主表nested prefix；`grouping=all_blocks`提供全部独立子组平均的稳健性检查。各G之间共享样本，不能当作独立实验。','',
    '## 5. 为什么有分项多样性，GRPO却未必获得正确方向','',
    '真实运行代码的评测PDMS与训练reward分别为：',
    '```text\nPDMS = NC × DAC × (5 EP + 5 TTC + 2 Comfort) / 12\nR_train = NC × DAC × (10 EP + 5 TTC + 2 Comfort) / 17\nA_i = (R_i - mean_group(R)) / (sample_std_group(R) + 1e-8)\n```',
    'DDC权重为0。NC/DAC为乘法项，0分会屏蔽其他分项的reward差异；NC=0.5仍是半权重，未人为二值化。NC、DAC均1且Comfort相同的两条轨迹中，EP多0.5就足以抵消训练reward中整个TTC项的损失。因此，正advantage表示相对组均值更优，不保证安全，也不保证相对IL或GT改进。',
    '优势计算直接提取归档forward_grpo中的原语句，在FP32 CUDA执行，另用真实归档scorer核验训练reward和评测PDMS。不是将普通PDMS z-score假装native advantage。后续实际loss还有denoising折扣、logprob截断和BC；这里没有计算参数梯度，因此Cov(A,component)只描述奖励系数与分项结果的关联，不能保证实际梯度更新改善该分项。',
    '本地实现的policy loss也不应仅因函数叫GRPO就假定与论文中完整PPO ratio/clipping一致：归档代码使用截断logprob的加权均值和BC。参见本地runtime hash与 `audits/native_advantage_excerpt.py.txt`。',
    '实际sampling std floor=0.04，而logprob std floor=0.1；两者并不相同，更不能将几何距离直接当作policy likelihood或优化梯度。',
    '数值审计也发现：若将这些advantage原语句改在CPU运行，个别相同reward的小组会因FP32 mean/std舍入出现虚假非零advantage。主结果使用CUDA；所有恒定reward组在本次CUDA上均为零，抽查单组/批量CUDA一致。见 `advantage_numerical_audit.csv`。不能据CPU近似结果声称原训练出现了这个数值错误。',
    '', '## 6. 末步噪声分解（固定64场景补充诊断）','']
    rows=[]
    for m in CFG['models']:
        n=pd.read_csv(OUT/'metrics'/f'noise_decomposition_{m}.csv')
        rows.append([labels[m],len(n),f'{n.pairwise_ADE.mean():.4f}',f'{n.pre_final_noise_mean_pairwise_ADE.mean():.4f}',f'{n.residual_pairwise_ADE.mean():.4f}'])
    lines += [table(['Policy','场景','实际输出pairwise ADE','末步加噪前mean的ADE','实际输出-minus-mean残差的ADE'],rows),'',
    '原生训练sampler最后一步仍有normalized std floor=0.04。按norm_odo缩放，未裁剪前XY单点std分别约1.3348m和0.84m。末步之后还有[-1,1]裁剪，因此不是所有轨迹的硬性几何宽度下限。这里直接hook最后一次p_mean_variance，保持全部随机抽样和最终输出不变，并与128条缓存核对。三列ADE不能相加，也不能当作严格方差贡献比例。末步条件均值不是可部署的eval policy，其PDMS未被测量。','',
    '## 7. 输出与结论边界','',
    '- `metrics/group_signal_regimes.csv`：没有安全样本、混合安全/失败、全安全且有EP空间、全安全且EP接近饱和；另分解零reward差异发生在哪类组。',
    '- `metrics/CRN_outcome_transitions.csv`：匹配随机流下IL失败→GRPO通过与反向转换；不是驾驶语义模式的因果对应。',
    '- `metrics/joint_safety_outcome_distribution.csv`：完整安全结果组合分布。',
    '- `metrics/scene_prefix_metrics.parquet`：5000×2×5个主分析场景行；完整310000个独立子组行在本地 `group_metrics.parquet`。',
    '- `metrics/paired_comparisons.csv`：3000次scene-paired及log-cluster bootstrap。主检验族为G16下8项；其余为描述性分析。',
    '- `figures/Fig1_*`：组大小、质量与安全EP机会；`Fig2_*`：分项分布与credit；`Fig3_*`：结果差异和reward局限；`Fig4_*`：几何宽度；`Fig5_*`：末步噪声。全部PNG/PDF/SVG并附CSV。',
    '',
    'SUPPORTED：这些数据可用于比较冻结IL和历史GRPO在相同真实sampler下的分项结果分布、组内改进机会及奖励识别限制；具体方向与显著性以本报告表格和CI为准。',
    'UNTESTED：结果模式是否对应不同交通交互语义；每一种reward对照是否会实际改善梯度方向；用这些诊断设计PSI/SFT是否优于等量baseline；更大训练G是否改善最终性能。这些都不能由只读rollout统计证明。',
    '推荐后续将“安全样本是否存在→安全子集是否存在真实进度空间→reward/advantage是否正确区分”作为组内诊断主线，几何宽度为辅助。不能以扩大失败结果熵、制造更多reward方差或提高单个oracle样本分数作为训练方法有效性的证据。',
    '', '参考：组相对优势的通用背景见 [DeepSeekMath原论文](https://arxiv.org/abs/2402.03300)；指标和非反应式仿真背景见 [NAVSIM原论文](https://www.cvlibs.net/publications/Dauner2024NEURIPS.pdf)。本实验的具体权重、sampler和loss以核验过的本地归档代码为准。',
    '', f"审计：`audits/final.json` = {a['status']}；旧缓存核验{a['old_cache_files_verified']}文件，V1/V2/V3等受保护文件{a['old_protected_files_unchanged']}个未改变。"]
    dest=ROOT/'reports/GRPO_COMPONENT_DIVERSITY_5000_20260921.md';dest.parent.mkdir(exist_ok=True);dest.write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
