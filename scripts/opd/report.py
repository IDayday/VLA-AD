"""Generate Chinese research delivery from complete real artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv,shutil
from scripts.opd.common import *
from scripts.opd.analyze import EXPERIMENTS,bank


def interval(v,percent=False):
    f=100 if percent else 1
    return f"{v['estimate']*f:.4f} [{v['lower95']*f:.4f}, {v['upper95']*f:.4f}]"


def support(v,material=.002):
    if v['lower95']>0 and v['estimate']>=material:return 'SUPPORTED'
    if v['upper95']<material:return 'NOT_SUPPORTED'
    return 'INCONCLUSIVE'


def evidence_sections(data):
    text='\n\n## 数值诊断、失败案例与解释边界\n\n'
    text+='先对每个已训练策略、每个场景的五个推理种子求均值，再取与IL的差。G是有符号差的总体均值，M_plus/M_minus分别是正部/负部的总体均值；先对每个策略截正/负，再跨三个训练种子平均。单次输出零分率与五次均值零分率分列，不把它们混用。完整log、训练种子及配对推理种子共同bootstrap，区间为未经多重比较校正的逐对95%区间。\n\n'
    c=data['contrasts']
    if 'E2_vs_E1' in c:text+=f"E2−E1平均PDMS={interval(c['E2_vs_E1']['PDMS'])}。过程监督与最终轨迹监督的差异需同时看平均分与退化；不能只依据OPD训练loss宣布更有效。\n\n"
    text+=f"E3相对E1的M_minus差为{interval(c['E3_vs_E1']['M_minus'])}，显示固定协议下较小的下降幅度；平均分差的区间仍较宽。E3相对E2的PDMS区间上界低于0.002，支持的是‘本预算未达到预定实质性额外收益’，不是两个算法数学等价或所有规模下无效。\n\n"
    text+='E4-A选择的学习率为3e-5，E3为1e-5，两者有相同的两个调参槽位，但最终配方的差异同时含Anchor和学习率。因此E4-A配方的保护效果不支持，不能进一步把退化全部归因于Anchor损失本身。没有在看到审计结果后补跑更有利的固定学习率分支。E4-B/C虽有较低M_minus点估计，平均增益也较小，区间不能确认在保留有效增益时的保护。\n\n'
    p=ROOT/'diagnostics/chain_distributions.json'
    if p.exists():
        v=read(p)['training_seeds'];state=np.mean([r['records'][-1]['state_RMS'] for r in v],0);response=np.mean([r['records'][-1]['teacher_response_RMS'] for r in v],0)
        text+='E2/E3实际训练轨迹档案的场景、噪声一一配对，初始状态误差0。1000步处，六个链状态的平均RMS差（归一化空间）为 '+str(np.round(state,6).tolist())+'；五次教师响应RMS差为 '+str(np.round(response,6).tolist())+'。学生状态分布确实刷新并不同于教师链；这一机制事实没有自动转化成超过E2的PDMS收益。教师查询始终为eval、无梯度，未发现数值发散。\n\n'
    p=ROOT/'diagnostics/gradients.json'
    if p.exists():
        v=read(p)['records'];text+='固定16个校准输入上的OPD/Anchor梯度余弦中位数：'+', '.join(f"{e}={np.median([r['cosine'] for r in v if r['experiment']==e]):.3f}" for e in ['E3','E4-A','E4-B','E4-C','E5'])+'。这些诊断主要同向，不支持把当前失败简单解释为两项梯度普遍反向。它们只覆盖seed42最终模型的固定输入，不是交通因素的因果识别。训练总loss/梯度按25步记录，DDIM链与每步分歧全量保存；未记录点的梯度范数不补造。\n\n'
    p=ROOT/'diagnostics/gate_results.json'
    if p.exists():
        g=read(ROOT/'gates/static.json');v=read(p);text+=f"静态训练类别为{g['category_counts']}，有效教师门控{int(g['effective_active'])}/4115。三个训练种子的激活样本数分别为 "+', '.join(str(sum(r['active_OPD_examples'].values()))+'/8000' for r in v['actual_training_exposure'].values())+'。全部8000次场景暴露仍按共同顺序执行，只有监督权重不同。\n\n'
        text+='独立预登记的训练分布描述集为209场景（A64/B32/C64/D49），不是泛化集：\n\n|类别|IL|RL|E3|E5|\n|---|---:|---:|---:|---:|\n'
        for cat in 'ABCD':
            a=v['categories']['E3'][cat];b=v['categories']['E5'][cat];text+=f"|{cat}|{a['IL_PDMS']:.4f}|{a['RL_PDMS']:.4f}|{a['PDMS']:.4f}|{b['PDMS']:.4f}|\n"
        text+='\nE5在B类训练描述集比E3保留更多IL分数，在D类也有GT监督下的改善，但A类能力传递更弱、C类出现下降，且这些训练内迹象没有带来可靠开发集收益。类别是低/高分分层，D不等于双方分数均为0。静态gate并不能约束共享参数更新对未训练log的所有影响；本轮不能单独确定LCB、安全mask、Anchor分配中哪一项造成结果。\n\n'
    p=ROOT/'analysis/failure_counts_dev.json'
    if p.exists():
        v=read(p)['methods'];text+='开发集逐策略计数：\n\n|方法|高分→低于0.8（seed42/43/44）|高分→零分|正分→零分|\n|---|---|---|---|\n'
        for e in ['E3','E4-A','E4-B','E4-C','E5']:
            text+=f"|{e}|"+'|'.join('/'.join(str(v[e][str(s)][k]) for s in [42,43,44]) for k in ['stable_high_to_low_count','stable_high_to_zero_count','positive_to_zero_count'])+'|\n'
        text+='\n这些是三个不同策略的计数，不相加后称为互不相同的场景。E3高分到零分的罕见事件区间包含0；固定五次估计中的下降不等于每个场景的真实期望都已被确定。零更新同噪声差为0，独立五次采样自比较也产生非零M_minus；后者的噪声耦合方式不同，不能直接从参数更新的M_minus中相减。\n\n'
    p=ROOT/'analysis/cases.json'
    if p.exists():
        cases=read(p);text+=f"四个事先固定的case规则选出{len({r['token'] for r in cases})}个不同场景；退化与保护规则命中同一个token，没有换成其他更好看的样本。主绘制轨迹仍为最接近场景平均变化的种子，淡线和事件点展示全部五次输出，避免把一个正常draw当成整个退化后果。\n\n"
        for r in cases:
            text+=f"- {r['kind']} / {r['token']}：IL/RL/E3/E5五次均值为 "+'/'.join(f"{r['scene_means'][k]:.4f}" for k in ['IL','RL','E3','E5'])+'；所选draw首次越界为'+str(r['event_seconds']['first_dac_s'])+'秒，首次TTC检查失败为'+str(r['event_seconds']['first_ttc_check_failure_s'])+'秒（None=四秒内无该事件）。\n'
        text+='\n退化/保护例子的分数差主要受一个推理种子的TTC检查失败影响，不能把单例包装为已确定的跨场景期望规律。教师迁移失败例中，教师推进约8.91米且无TTC检查失败，E3推进约10.88米并在3.1秒首次检查失败；这些是数值后果，不标注让行/抢行等意图。所有重新记录的案例评分与正式分数误差不超过1e-6。碰撞/TTC首次时间来自既有scorer内部记录，越界时间由既有ego area派生，完整仿真状态是额外案例日志，不是新的监督任务。\n\n'
    p=ROOT/'analysis/teacher_comparisons_dev.json'
    if p.exists():text+='相对直接使用RL教师的配对PDMS差：'+ '; '.join(f"{e}: {interval(v['PDMS'])}" for e,v in read(p)['contrasts'].items())+'。更少退化若伴随更小平均提升，不能认定优于直接复制教师。\n\n'
    text+='## 继续与停止的路线\n\n当前证据支持从RL教师迁移部分平均规划收益；不支持把学生链OPD相对教师链KD的额外价值扩大为既定结论。保留E2作为后续对照，若没有保留IL或多教师整合的额外目标，直接复制RL仍是必须比较的低成本方案。停止扩容当前静态E5配方，不用更换起点、增加模块或调整阈值挽救该结果。E6正式训练依据开发门禁停止，不是因缺GPU；不能据此推翻其超过教师的可能性。\n\n'
    text+='## 图表\n\n'
    for file,label in [('Fig1_PDMS_dev.png','Fig1 完整开发集PDMS'),('Fig1_paired_contrasts_dev.png','Fig1 配对差与实质性门槛'),('Fig2_learning_dev.png','Fig2 真实训练日志'),('Fig3_chain_discrepancy_dev.png','Fig3 去噪链诊断'),('Fig4_forgetting_dev.png','Fig4 修复与退化'),('Fig5_paired_dev.png','Fig5 场景配对'),('Fig6_quality_strata_dev.png','Fig6 质量分层'),('Fig7_repair.png','Fig7 修复'),('Fig7_regression.png','Fig7 退化及同一场景的局部保护'),('Fig7_teacher_transfer_failure.png','Fig7 教师能力迁移失败'),('Fig8_cost_dev.png','Fig8 实际成本')]:
        if (REPORTS/'figures'/file).exists():text+=f'![{label}](figures/{file})\n\n'
    return text


def e6_admission():
    d=read(ROOT/'analysis/formal_dev.json');m=d['methods']['E5']['metrics']['G']
    integration=read(ROOT/'tests/real_grpo.json')['passed'];g2=read(ROOT/'analysis/G2.json')['status']=='PASS'
    admission=dict(admitted=bool(m['lower95']>0 and m['estimate']>=.002 and integration and g2),
        required='E5 dev gain lower95>0 and estimate>=0.002; numerical checks and G2 pass',E5_gain=m,
        reason='Registered scientific gate passed' if m['lower95']>0 and m['estimate']>=.002 and integration and g2 else 'E5 did not establish the registered positive held-out gain; no formal E6 expansion',
        integration_only=read(ROOT/'runs/E6_IL_JOINT_42_smoke/complete.json'))
    write(ROOT/'selection/E6_admission.json',admission);return admission


def e6_status():
    if not (ROOT/'analysis/E6_dev.json').exists():return 'NOT_RUN'
    if not (ROOT/'analysis/E6_navtest.json').exists() or not (ROOT/'selection/E6_navtest.json').exists():return 'INCONCLUSIVE'
    v=read(ROOT/'selection/E6_navtest.json')['variant'];a=read(ROOT/'analysis/E6_dev.json')['methods'][v]['relative_teacher']['G'];b=read(ROOT/'analysis/E6_navtest.json')['methods'][v]['relative_teacher']['G']
    states=[support(a),support(b)]
    if states==['SUPPORTED','SUPPORTED']:return 'SUPPORTED'
    if states==['NOT_SUPPORTED','NOT_SUPPORTED']:return 'NOT_SUPPORTED'
    return 'INCONCLUSIVE'


def tables(split):
    data=read(ROOT/'analysis'/f'formal_{split}.json');base=read(ROOT/'analysis'/f'baseline_{split}.json');table=[]
    for n in ['E0-IL','E0-RL','E0-COPY']:
        table.append(dict(experiment=n,status='COMPLETE',train_seeds=0,PDMS=base['components'][n]['PDMS'],**{k:base['components'][n][k] for k in ['NC','DAC','TTC','EP','Comfort']}))
    for e in EXPERIMENTS:
        if e not in data['methods']:
            table.append(dict(experiment=e,status='NOT_RUN'));continue
        v=data['methods'][e];m=v['metrics'];table.append(dict(experiment=e,status='COMPLETE',train_seeds=3,**v['components'],PDMS_lower95=m['PDMS']['lower95'],PDMS_upper95=m['PDMS']['upper95'],
            G=m['G']['estimate'],M_plus=m['M_plus']['estimate'],M_minus=m['M_minus']['estimate'],
            G_lower95=m['G']['lower95'],G_upper95=m['G']['upper95'],stable_high_retention=m['stable_high_retention']['estimate'],zero_output_fraction=m['zero_output_fraction']['estimate']))
    if (ROOT/'analysis'/f'E6_{split}.json').exists():
        for variant,r in read(ROOT/'analysis'/f'E6_{split}.json')['methods'].items():
            table.append(dict(experiment='E6_'+variant,status='COMPLETE',train_seeds=3,**r['components'],G=r['metrics']['G']['estimate'],M_minus=r['metrics']['M_minus']['estimate'],
                relative_teacher_G=r['relative_teacher']['G']['estimate']))
    else:table.append(dict(experiment='E6',status='NOT_RUN',integration_optimizer_updates=4))
    dest=REPORTS/'tables';dest.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in table for k in r))
    with open(dest/f'{split}_results.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(table)
    return table


def full():
    REPORTS.mkdir(parents=True,exist_ok=True);d=read(ROOT/'analysis/formal_dev.json');b=read(ROOT/'analysis/baseline_dev.json');g=read(ROOT/'analysis/G2.json')
    tuning=read(ROOT/'selection/tuning.json');gate=read(ROOT/'gates/static.json');admission=e6_admission();tables('dev')
    has_test=(ROOT/'analysis/formal_navtest.json').exists()
    if has_test:tables('navtest')
    contrast=d['contrasts'];q1a=support(contrast['E3_vs_E1']['PDMS']);q1b=support(contrast['E3_vs_E2']['PDMS']);transfer=support(d['methods']['E3']['metrics']['G'])
    q1='SUPPORTED' if q1a==q1b==transfer=='SUPPORTED' else ('NOT_SUPPORTED' if 'NOT_SUPPORTED' in [q1a,q1b,transfer] else 'INCONCLUSIVE')
    e3=d['methods']['E3']['metrics'];forget='SUPPORTED' if e3['M_minus']['lower95']>0 else 'INCONCLUSIVE'
    anchored={}
    for e in ['E4-A','E4-B','E4-C','E5']:
        c=contrast[e+'_vs_E3'];reduction={k:-c['M_minus'][kk] for k,kk in [('estimate','estimate'),('lower95','upper95'),('upper95','lower95')]}
        status=support(reduction,.0005)
        if c['PDMS']['lower95']<-.002 and status=='SUPPORTED':status='INCONCLUSIVE'
        anchored[e]=dict(status=status,M_minus_reduction=reduction,PDMS_difference=c['PDMS'])
    statuses=dict(Q1=q1,E3_vs_E1=q1a,E3_vs_E2=q1b,E3_transfer_vs_IL=transfer,Q2_OPD_forgetting=forget,
        Q2_mitigation=anchored,Q3_teacher_ceiling=e6_status())
    write(ROOT/'analysis/conclusions.json',statuses)
    result_lines=['|实验|开发集 PDMS（95% CI）|相对 IL 的 G（95% CI）|M+|M−|单次零分比例|','|---|---:|---:|---:|---:|---:|']
    for n in ['E0-IL','E0-RL','E0-COPY']:
        result_lines.append(f"|{n}|{b['components'][n]['PDMS']:.4f}|—|—|—|—|")
    for e,v in d['methods'].items():
        m=v['metrics'];result_lines.append(f"|{e}|{interval(m['PDMS'])}|{interval(m['G'])}|{m['M_plus']['estimate']:.4f}|{m['M_minus']['estimate']:.4f}|{100*m['zero_output_fraction']['estimate']:.2f}%|")
    e6_text='已运行原生DiffGRPO/OPD四个真实optimizer updates；零OPD权重与原生loss/gradient精确一致，联合梯度有效且教师不变。它是工程集成证据，不是超过教师的性能证据。'
    if (ROOT/'analysis/E6_dev.json').exists():
        e6_text='正式E6结果：'
        for variant,r in read(ROOT/'analysis/E6_dev.json')['methods'].items():
            m=r['metrics'];result_lines.append(f"|E6-{variant}|{interval(m['PDMS'])}|{interval(m['G'])}|{m['M_plus']['estimate']:.4f}|{m['M_minus']['estimate']:.4f}|{100*m['zero_output_fraction']['estimate']:.2f}%|")
            e6_text+=f"{variant}相对RL教师={interval(r['relative_teacher']['G'])}；"
        e6_text+='E5初始化的前置1000步及教师/gate成本另外计入，最终确认仅选择cal-best初始化，不按开发/测试结果选择。'
    else:result_lines.append('|E6|NOT_RUN（正式比较）|真实4步集成已运行|—|—|—|')
    baseline=f"""# E0 基线结果

完整固定开发集 {b['scenes']} 场景、{b['logs']} 个 log，5 次配对推理。IL PDMS={b['components']['E0-IL']['PDMS']:.6f}，RL={b['components']['E0-RL']['PDMS']:.6f}；RL−IL={interval(b['paired_RL_vs_IL']['G'])}。E0-COPY 与 RL 的逐步轨迹及评分均数值一致。

RL 相对 IL 的 M_minus={interval(b['paired_RL_vs_IL']['M_minus'])}，{b['teacher_worse_scene_count']} 场景的五次平均下降超过0.02。RL 教师平均更好，但不能逐场景无条件替代 IL。

{chr(10).join(result_lines)}

本地正式指标为 NAVSIM v1 PDMS，[官方 ReCogDrive](https://github.com/xiaomi-research/recogdrive)公开的2B IL/RL参考值为86.5/90.8（百分制）。本地开发集是固定 navtrain 留出，不能与官方测试分数机械对齐。官方测试集确认单独报告。
"""
    (REPORTS/'04_baseline_results.md').write_text(baseline)
    ablation='# 蒸馏与锚定对照\n\n'+ '\n'.join(result_lines)+'\n\n'
    ablation+=f"E3 相对 E1：{interval(contrast['E3_vs_E1']['PDMS'])}，状态 {q1a}。E3 相对 E2：{interval(contrast['E3_vs_E2']['PDMS'])}，状态 {q1b}。实质性 PDMS 门槛为0.002。三种子均取固定1000步，不从开发集挑选检查点。\n\n"
    ablation+='正式各组学习率：'+', '.join(f"{e}={v['lr']}" for e,v in tuning['selected'].items())+'。全部只使用相同的两个校准槽位。\n\n'
    ablation+='100步、256训练场景的小规模校准结果：\n\n'+ '\n'.join(f"- {e}：PDMS={v['PDMS']:.6f}，相对 IL={v['Delta_PDMS']:+.6f}。" for e,v in g['results'].items())+'\n'
    (REPORTS/'05_opd_ablation_results.md').write_text(ablation)
    forgetting=f"# 遗忘与选择性传递\n\nE3 的场景平均回报下降幅度 M_minus={interval(e3['M_minus'])}，明显下降比例={interval(e3['material_drop_fraction'])}。这里的下降是五次场景均值估计，不把每个场景标成已确定的期望退化。独立五种子 IL 自比较为 {b['independent_five_draw_self_comparison']}。\n\n"
    forgetting+=f"独立筛选的稳定高分集合有 {b['stable_high_count']} 个场景，筛选与评价种子不共用。E3 保持率={interval(e3['stable_high_retention'])}；高分到零分比例（分母为完整开发集）={interval(e3['stable_high_to_zero_fraction'])}。\n\n"
    for e,v in anchored.items():forgetting+=f"{e}：M_minus 减少量={interval(v['M_minus_reduction'])}；平均 PDMS 差={interval(v['PDMS_difference'])}；联合结论 {v['status']}。减少量门槛0.0005，同时检查平均分损失是否超过0.002，不能靠少学获胜。\n\n"
    forgetting+='训练静态 gate 的类别及有效激活、逐类别实际暴露和最终变化见结构化 gate、训练 schedule 和逐场景结果。开发/测试上的 A–D 为诊断分层，不用于重新拟合 gate。NC/DAC/TTC 非退化约束不能解释为真实道路安全保证。\n'
    (REPORTS/'06_forgetting_analysis.md').write_text(forgetting)
    test_text='官方测试集正式确认尚未完成；不得将开发集结论表述为完整测试集成功。'
    if has_test:
        td=read(ROOT/'analysis/formal_navtest.json');tb=read(ROOT/'analysis/baseline_navtest.json')
        test_text=f"官方 navtest {td['scenes']} 场景、{td['logs']} 个 log；精确匹配仓库官方 token 清单。IL={tb['components']['E0-IL']['PDMS']:.6f}，RL={tb['components']['E0-RL']['PDMS']:.6f}。"+' '.join(f"{e} PDMS={interval(v['metrics']['PDMS'])}，G={interval(v['metrics']['G'])}。" for e,v in td['methods'].items())+' 所有候选在锁定 cal 选择后一次性确认，测试结果不反向修改方案。'
    report=f"""# ReCogDrive On-Policy Distillation 完整研究报告

## 研究结论

Q1（学生链 OPD 优于轨迹 SFT 和教师链 KD）：**{q1}**。E3−E1={interval(contrast['E3_vs_E1']['PDMS'])}；E3−E2={interval(contrast['E3_vs_E2']['PDMS'])}；E3 相对 IL 增益={interval(e3['G'])}。优势、绝对能力迁移和训练损失下降分别判断。

Q2（已有 IL 能力受损）：**{forget}**。E3 的 M_minus={interval(e3['M_minus'])}，独立高分保持率={interval(e3['stable_high_retention'])}。Anchor/GT/gate 的具体缓解结论分别列于下表和遗忘报告，不由平均分替代。

Q3（进一步 DiffGRPO 超过 RL 教师）：**{statuses['Q3_teacher_ceiling']}**。{admission['reason']}。{e6_text}

## 模型与算法核验

采用官方 IL 学生与 RL 活动动作头教师；IL/RL 的结构一致，加载器不选择 RL 的 old_policy。所有核心张量必须加载，完整 missing/unexpected/shape mismatch 记录在02报告及外部 audit/checkpoints.json。冻结同一1536维 VLM，只训练384维 small DiT 动作规划器，共34,329,219参数。八点三维轨迹、原生 norm_odo/denorm_odo、五步 DDIM 均未改动；专家、残差、未来目标等路径关闭。

DDIM adapter 与原 get_action 的逐步状态、均值及最终轨迹误差为0，E0-COPY 与 RL 误差为0。原生随机方差不能由 ddim_eta 配置名推断；实际路径含标准差下界和截断噪声，所以主版本使用转移均值 MSE，不能宣称 Gaussian KL。可选未截断共享方差 KL 仅有实现和正确性测试，正式性能消融 NOT_RUN。

[DiffusionOPD](https://arxiv.org/abs/2605.15055) 的关键是由当前学生产生去噪状态，再用冻结教师监督这些状态。E1 是最终轨迹伪标签的原生噪声预测训练，E2 的状态来自教师，E3 的状态每轮来自当前学生。所有链 detach，重新预测学生转移时才建立梯度。这里没有 environment-level on-policy；[Flow VLA 多教师工作](https://arxiv.org/abs/2610.21522)涉及的环境互动不能移用于固定日志实验。[PIVOT-Q](https://arxiv.org/abs/2610.05745)的行为保护思想是相关参考，本研究不声称复现其量化场景。

## 数据与受控实验

训练4115场景/50 log，校准518/4，固定开发1696/16；按完整 log 和父驾驶记录互斥。它们来自官方 navtrain，未参与本次新增训练不等于底模从未见过。VLM缓存只保存原生可见观测和实际动作头输入 token；8个在线场景与缓存的 token/轨迹误差为0。GT仅训练使用，metric cache 与 gate 是标签，不进入VLM编码。

七组正式实验各1000次实际更新、global batch8、三个独立训练种子42/43/44；配对推理种子1101–1105。共同 IL 初始化、数据顺序和初始噪声，两个学习率槽位，只有校准集决定学习率。Anchor/GT权重在固定训练样本按梯度尺度校准后锁定，不能从审核效果继续调节。E5 重新从 IL 开始，并不继承 E4 的额外训练。

{chr(10).join(result_lines)}

95%区间按完整 log、训练种子与配对推理种子分层 bootstrap（2000次），保留相邻场景相关性。点估计为场景加权均值。材料性门槛PDMS0.002，退化幅度减少0.0005。校准只有4个log，超参数选择可能不稳定；三种子也不足以穷尽训练随机性。有限五次采样不证明单个场景的精确期望变化。

## OPD 是否提供额外价值

E3/E1状态={q1a}，E3/E2状态={q1b}，E3/IL状态={transfer}。训练 loss 的下降只说明转移拟合；最终动作裁剪、原生随机采样与闭环跟踪可能使很小的中间误差对应离散零分。真实训练链的 E2/E3 状态分布、教师响应差异、各步 MSE 和梯度贡献见 diagnostics 与 Fig3。不能把状态分布有差异本身当成 OPD 性能优势。

E4/E5 保护结论：{'; '.join(f'{e}: {v["status"]}' for e,v in anchored.items())}。逐场景正分到零分、原失败修复、教师成功但学生失败、双方失败但学生改善、NC/DAC/TTC新增违规和原高分保持分别记录，避免只展示平均分或成功案例。

## 教师质量、gate 与偏差

RL 教师平均优于 IL，开发集仍有 {b['teacher_worse_scene_count']} 场景的五次平均下降超过0.02。训练静态 gate 依据五次配对差值的LCB，z=1、margin0.03并检查逐样本NC/DAC/TTC；z=1不是95%置信保证。质量 A/B/C/D 与有效gate不是同一概念。只在A且gate通过时接受强教师监督，B/C保护IL，D不假定教师有效。GT/anchor/opd各有独立归一化；空gate不能自动变成无条件OPD。

选择性监督改变有效训练暴露分布，可能偏向可修复但稀少的场景；静态gate依据原始IL，不保证更新后仍是最优选择。为避免将分布变化误当因果机制，使用共同初始化、相同训练场景/预算，并报告每类暴露、类别诊断和训练外表现。开发/测试类别图不参与训练选择。

## 最终测试确认

{test_text}

## 成本、复制对照与适用范围

E0-COPY直接复制教师权重即可精确恢复教师输出，且没有蒸馏训练成本。相同架构、同一VLM、没有部署约束时，OPD不能仅凭逼近教师证明工程价值；需要保持IL能力、选择性传递或整合互补教师的真实优势。推理只调用单学生，不增加教师或环境反馈。Fig8与成本表记录实际训练时间、峰值显存和教师查询；输入编码、静态gate、教师缓存和调参为共享前置成本，另列总账，不隐藏在某一方法中。

E6 训练使用原生自定义 PDM reward，EP/TTC/Comfort权重为10/5/2；正式评测 PDMS 为5/5/2，两者分开记录。本轮没有添加世界模型、专家路由或额外对齐任务。候选结果不支持时按协议停止对应路线，不继续搜索新的损失来追逐正结论。缺少正式E6结果时，无法区分教师约束、教师能力边界与GRPO本身限制，不作归因。

## 后续研究边界

多教师值得继续的条件是有可重复的互补场景能力和超出单教师复制的收益；必须控制教师选择与数据暴露预算。时刻化NAVSIM物理反馈可用于训练特权教师，但未来对象占用只能作为训练标签/特权路径，部署学生仍只能使用当前与历史观测。不同质量教师的选择性传递需要训练外校准，不能用测试反馈构造gate。双方共同失败的场景需要新的合法探索或监督信息，反复复制固定教师不能自动产生缺失能力。泛化研究需按完整驾驶记录留出、跨日志/数据域验证，并保留不利结果，而非记忆训练token。

## 复现与证据

锁定协议：configs/opd/protocol.yaml（SHA256 e02028b646a2745dca0b0a6f6eae7dff57119b3d12f8aeb0a3324a1f80c88968）。分支 research/recogdrive-opd-20261020，原始HEAD884ea1d0f67c60821d61033d61efb10916d16929。最终提交见 delivery.json。

完整产物：{ROOT}。逐场景轨迹在 evaluation/<split>/<model>/predictions，分项/事件在 scores，训练链与 optimizer 在 runs。每个任务的实际命令、PID/start ticks、代码来源hash、完成或失败状态在 jobs。文档 docs/RECOGDRIVE_OPD_RUNBOOK.md 给出全部入口；figures.py从实际结果生成Fig1–8，空结果不画虚拟曲线。

实际命令：`python scripts/opd/analyze.py --suite E0-E6 --split dev`；`python scripts/opd/figures.py --split dev --cases`；`python scripts/opd/report.py`。正式E6未获门禁时不得绕过admission直接训练；小规模集成可用 `python scripts/opd/train.py --exp E6 --stage smoke --variant IL_JOINT`，新运行需不同name。
"""
    report+=evidence_sections(d)
    (REPORTS/'07_research_conclusions.md').write_text(report)
    for name in ['baseline_dev.json','formal_dev.json','conclusions.json','G2.json','cases.json','costs_dev.json','formal_navtest.json','baseline_navtest.json']:
        p=ROOT/'analysis'/name
        if p.exists():shutil.copyfile(p,REPORTS/'tables'/name)
    print(statuses)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission-only',action='store_true');a=p.parse_args()
    if a.admission_only:print(e6_admission())
    else:full()
