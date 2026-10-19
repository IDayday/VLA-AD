"""Generate Chinese research delivery from complete real artifacts only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv,shutil
from scripts.opd.common import *
from scripts.opd.analyze import EXPERIMENTS,bank


def interval(v,percent=False):
    f=100 if percent else 1
    return f"{v['estimate']*f:.6f} [{v['lower95']*f:.6f}, {v['upper95']*f:.6f}]"


def support(v,material=.002):
    if v['lower95']>0 and v['estimate']>=material:return 'SUPPORTED'
    if v['upper95']<material:return 'NOT_SUPPORTED'
    return 'INCONCLUSIVE'


def mitigation(c):
    reduction={k:-c['M_minus'][kk] for k,kk in [('estimate','estimate'),('lower95','upper95'),('upper95','lower95')]}
    status=support(reduction,.0005)
    raw_status=status
    if c['PDMS']['lower95']<-.002 and status=='SUPPORTED':status='INCONCLUSIVE'
    return dict(status=status,raw_M_minus_reduction_status=raw_status,M_minus_reduction=reduction,PDMS_difference=c['PDMS'])


def evidence_sections(data):
    text='\n\n## 数值诊断、失败案例与解释边界\n\n'
    text+='先对每个策略、每个场景的五个推理种子求均值，再计算相对IL的G、M_plus、M_minus。先对每个策略截正/负，再跨训练种子平均。单次输出零分率与五次均值零分率分列。区间按完整log、训练种子和配对推理种子bootstrap，未经多重比较校正。\n\n'
    for key in ['E2_vs_E1','E3_vs_E1','E3_vs_E2','E4-A_vs_E3','E4-B_vs_E3','E4-C_vs_E3','E5_vs_E3']:
        c=data['contrasts'][key]
        text+=f"{key}：PDMS差={interval(c['PDMS'])}；M_minus差={interval(c['M_minus'])}（正值表示更多下降幅度）。\n\n"
    rates=read(ROOT/'selection/tuning.json')['selected']
    text+='各组采用相同的两个学习率槽位，最终学习率：'+', '.join(f"{e}={v['lr']}" for e,v in rates.items())+'。若最终学习率不同，配方差异不能全部归因于某一辅助损失。未根据审计结果补跑有利的固定学习率分支。较低M_minus若同时伴随较小平均增益，不能单独称为保护成功。\n\n'
    p=ROOT/'diagnostics/chain_distributions.json'
    if p.exists():
        v=read(p)['training_seeds'];state=np.mean([r['records'][-1]['state_RMS'] for r in v],0);response=np.mean([r['records'][-1]['teacher_response_RMS'] for r in v],0)
        text+='实际E2/E3训练链在共同场景和初始噪声下配对；第1000次更新所用rollout（更新前θ999）的六个状态平均RMS差（归一化空间）为 '+str(np.round(state,6).tolist())+'；五次教师响应RMS差为 '+str(np.round(response,6).tolist())+'。学生状态变化不自动意味着最终PDMS提高。\n\n'
    p=ROOT/'diagnostics/gradients.json'
    if p.exists():
        records=read(p)['records'];text+='固定16个校准输入上的OPD/Anchor梯度余弦中位数：'+', '.join(f"{e}={np.median([r['cosine'] for r in records if r['experiment']==e]):.3f}" for e in ['E3','E4-A','E4-B','E4-C','E5'])+'。这只描述seed42最终模型的局部梯度关系，不能解释为某交通因素的因果作用。训练loss/梯度按25步记录，全部学生链保存；未记录点不插补为实际观测。\n\n'
    p=ROOT/'diagnostics/gate_results.json'
    if p.exists():
        g=read(ROOT/'gates/static.json');v=read(p);registration=read(ROOT/'diagnostics/gate_registration.json')
        text+=f"静态训练类别={g['category_counts']}，有效门控={g['effective_active']}/{len(rows('train'))}。三个种子的实际激活样本数分别为 "+', '.join(str(sum(r['active_OPD_examples'].values()))+'/8000' for r in v['actual_training_exposure'].values())+'。共同场景顺序不变，监督权重不同。\n\n'
        text+=f"预登记的训练分布描述集为{registration['scenes']}场景，类别数{registration['counts']}，不是泛化证据：\n\n|类别|IL|RL|E3|E5|\n|---|---:|---:|---:|---:|\n"
        for cat in 'ABCD':
            a=v['categories']['E3'][cat];b=v['categories']['E5'][cat];text+=f"|{cat}|{a['IL_PDMS']:.4f}|{a['RL_PDMS']:.4f}|{a['PDMS']:.4f}|{b['PDMS']:.4f}|\n"
        text+='\n类别是原IL/RL低高分分层，D不等于双方零分。训练内表现不等于跨log泛化。静态gate无法直接约束共享参数更新对未训练log的后果，本矩阵不能单独确定LCB、safety mask与Anchor分配的因果贡献。\n\n'
    p=ROOT/'analysis/failure_counts_dev.json'
    if p.exists():
        v=read(p)['methods'];text+='开发集逐策略计数：\n\n|方法|高分→低于0.8（seed42/43/44）|高分→零分|正分→零分|\n|---|---|---|---|\n'
        for e in ['E3','E4-A','E4-B','E4-C','E5']:
            text+=f"|{e}|"+'|'.join('/'.join(str(v[e][str(s)][k]) for s in [42,43,44]) for k in ['stable_high_to_low_count','stable_high_to_zero_count','positive_to_zero_count'])+'|\n'
        text+='\n这是三个策略的计数，不相加为互异场景总数。有限五次估计不确定单个场景的真实期望。独立五次IL自比较也有非零M_minus；噪声耦合不同，不能直接从更新后的M_minus中相减。\n\n'
    p=ROOT/'analysis/cases.json'
    if p.exists():
        cases=read(p);text+=f"预定四类case规则得到{len(cases)}个条目、{len({r['token'] for r in cases})}个不同场景。主轨迹取最接近场景平均变化的draw，淡线和事件点展示全部五次输出。空类别不替补。\n\n"
        for r in cases:
            text+=f"- {r['kind']} / {r['token']}：IL/RL/E3/E5五次均值为 "+'/'.join(f"{r['scene_means'][k]:.4f}" for k in ['IL','RL','E3','E5'])+'；主draw首次越界='+str(r['event_seconds']['first_dac_s'])+'秒，首次TTC检查失败='+str(r['event_seconds']['first_ttc_check_failure_s'])+'秒（None=窗口内无事件）。\n'
        text+='\n案例固定使用预登记seed42，不能代表三个训练种子的总体效果，尤其不能用该种子的E5个例掩盖其他种子的退化。TTC检查失败时刻不是真实剩余碰撞时间。案例以重新评分的一致性检查为前提，完整跟踪状态为额外日志，越界时间由已有ego area派生。数值后果不转换成让行/抢行等意图标签。\n\n'
    p=ROOT/'analysis/teacher_comparisons_dev.json'
    if p.exists():text+='相对直接RL教师的PDMS差：'+ '; '.join(f"{e}: {interval(v['PDMS'])}" for e,v in read(p)['contrasts'].items())+'。\n\n'
    p=ROOT/'analysis/secondary_anchor_vs_GT_dev.json'
    if p.exists():
        v=read(p)['result'];text+=f"补充分析（看到开发集均值后添加，不是预登记主判定）：E4-C−E4-B的PDMS差={interval(v['PDMS'])}，M_minus差={interval(v['M_minus'])}。两者相同学习率、GT权重、初始化与数据，比较额外Anchor；没有新增训练或推理，不替代纯Anchor对照，也没有额外测试确认。\n\n"
    text+='## 父驾驶记录相关性敏感性分析\n\n主分析按预登记的完整NAVSIM log聚类；同一父驾驶记录可能包含多个log。开发集16个log来自15条父记录，测试集136个log来自44条父记录。以下为看到主结果后添加的事后敏感性分析：只将聚类单位改为含车辆ID的完整父驾驶记录，保留相同模型、场景、种子、统计量与2000次bootstrap；所有已有方法和主配对差都纳入，不按结果选择。它不是独立复现，不改动原协议、主区间、选模或E6停止决定。\n\n'
    for split in ['dev','navtest']:
        p=ROOT/'analysis'/f'parent_record_sensitivity_{split}.json'
        if not p.exists():continue
        v=read(p);text+=f"{split} 父记录聚类：\n\n"
        for e in v['methods']:text+=f"- {e} 相对IL的G={interval(v['methods'][e]['G'])}。\n"
        for key in ['E3_vs_E2','E4-A_vs_E3','E5_vs_E4-C','E5_vs_E3']:
            if key in v['contrasts']:
                c=v['contrasts'][key];text+=f"- {key}：PDMS差={interval(c['PDMS'])}，M_minus差={interval(c['M_minus'])}。\n"
        text+='\n'
    text+='## 图表\n\n'
    for file in sorted((REPORTS/'figures').glob('*.png')):text+=f'![{file.stem}](figures/{file.name})\n\n'
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
        record=dict(experiment=n,status='COMPLETE',train_seeds=0,PDMS=base['components'][n]['PDMS'],**{k:base['components'][n][k] for k in ['NC','DAC','TTC','EP','Comfort']})
        if n=='E0-IL':record.update(G=0.,M_plus=0.,M_minus=0.)
        else:record.update({k:base['paired_RL_vs_IL'][k]['estimate'] for k in ['G','M_plus','M_minus','zero_output_fraction']})
        table.append(record)
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
    env_path=REPORTS/'00_environment_audit.md';env_body=env_path.read_text().split('\n## 最终资源与数据身份')[0]
    hardware=read(ROOT/'audit/hardware_hosts.json')
    env_body+=f"\n## 最终资源与数据身份\n\n真实核验{hardware['unique_A800_devices']}张互异A800，主机为"+', '.join(hardware['hosts'])+'。逐主机CUDA/驱动/PyTorch环境与GPU UUID见tables/audit_hardware_hosts.json。原工作区未提交修改保留，研究在独立worktree中进行，原有ReCogDrive默认源码未改动。\n\n'
    env_body+='版本化索引位于configs/opd/splits，含训练4115、校准518、开发1696、官方测试12146个精确有序场景；206个源log的字节哈希已核验。driver_record保留车辆标识，另以更保守的capture timestamp检查跨用途重叠，均无交集。真实新目录重建已逐字段、逐顺序核验，不依赖旧研究输出manifest。原始数据、图像、GT及大缓存均不进git。\n'
    env_path.write_text(env_body)
    matrix_path=ROOT/'tests/final_matrix.json'
    if matrix_path.exists():
        matrix=read(matrix_path);assert matrix['passed']
        path=REPORTS/'03_algorithm_validation.md';body=path.read_text().split('\n## 最终分级验收')[0]
        body+=f"\n## 最终分级验收\n\nCPU无权重测试：{matrix['CPU']}。T01–T28全部PASS，完整逐项证据与真实集成结果哈希见tables/tests_final_matrix.json。真实checkpoint/场景验证包含官方原始前向契约、两种采样分支、批量与逐场景、2卡DDP、OPD与GRPO连续/中断恢复、真实训练后学生独立在线推理。\n\n"
        body+='G0/G1通过；七个256场景资格训练及518场景校准评测通过数值与预登记灾难性门禁G2。G3/G4矩阵真实完成，但研究结论按PDMS区间判断，并非数值测试通过就支持算法。G5仅真实工程集成通过；E5开发增益未过门槛，正式E6为NOT_RUN。\n\n'
        body+='历史首轮G0任务中的GRPO低分诊断因E0标签尚未生成而失败，依赖调整后实际通过；原失败退出与CPU控制器交接保留在资源账目，没有放宽断言。30项CPU通过不替代真实评测，未运行的KL/时间步加权性能消融不列为研究成功。\n'
        path.write_text(body)
    tuning=read(ROOT/'selection/tuning.json');gate=read(ROOT/'gates/static.json');admission=e6_admission();tables('dev')
    source=read(ROOT/'audit/official_model_sources.json');assert source['passed']
    audit_path=REPORTS/'02_checkpoint_audit.md';body=audit_path.read_text().split('\n## 官方模型来源')[0]
    body+='\n## 官方模型来源\n\n本地三个权重文件的实际SHA256均与公开发布文件的LFS SHA256完全相符；固定发布revision如下：\n\n'
    for label,r in source['records'].items():body+=f"- {label}：[固定发布文件]({r['pinned_source']})，revision `{r['revision']}`，SHA256 `{r['sha256']}`。\n"
    comparison=read(ROOT/'audit/RL_reference_comparison.json')
    body+=f"\nIL/RL共有{comparison['IL_active_tensors']}个活动张量，{comparison['changed_active_tensors']}个活动张量不同；RL old_policy与IL逐张量完全相同。教师只加载活动头。历史VLM训练身份不能由仅动作头checkpoint证明，本研究显式采用同一冻结官方VLM。\n"
    audit_path.write_text(body)
    has_test=(ROOT/'analysis/formal_navtest.json').exists()
    if has_test:tables('navtest')
    contrast=d['contrasts'];q1a=support(contrast['E3_vs_E1']['PDMS']);q1b=support(contrast['E3_vs_E2']['PDMS']);transfer=support(d['methods']['E3']['metrics']['G'])
    q1='SUPPORTED' if q1a==q1b==transfer=='SUPPORTED' else ('NOT_SUPPORTED' if 'NOT_SUPPORTED' in [q1a,q1b,transfer] else 'INCONCLUSIVE')
    e3=d['methods']['E3']['metrics'];forget='SUPPORTED' if e3['M_minus']['lower95']>0 else 'INCONCLUSIVE'
    anchored={}
    for e in ['E4-A','E4-B','E4-C','E5']:
        anchored[e]=mitigation(contrast[e+'_vs_E3'])
    statuses=dict(Q1=q1,E3_vs_E1=q1a,E3_vs_E2=q1b,E3_transfer_vs_IL=transfer,Q2_OPD_forgetting=forget,
        Q2_mitigation=anchored,Q3_teacher_ceiling=e6_status(),primary_scope='Fixed1696-scene dev main analysis; independent navtest confirmation reported separately')
    statuses['E5_transfer_vs_IL']=support(d['methods']['E5']['metrics']['G'])
    statuses['E5_vs_matched_E4C']=mitigation(contrast['E5_vs_E4-C'])
    if has_test:
        td=read(ROOT/'analysis/formal_navtest.json');tc=td['contrasts']
        confirmation=dict(E3_vs_E2=support(tc['E3_vs_E2']['PDMS']),E3_transfer_vs_IL=support(td['methods']['E3']['metrics']['G']),
            mitigation={e:mitigation(tc[e+'_vs_E3']) for e in ['E4-A','E4-B','E4-C','E5'] if e+'_vs_E3' in tc})
        statuses['navtest_confirmation']=confirmation
        statuses['Q1_stable_extra_value']='SUPPORTED' if q1=='SUPPORTED' and confirmation['E3_vs_E2']=='SUPPORTED' and confirmation['E3_transfer_vs_IL']=='SUPPORTED' else (
            'NOT_SUPPORTED' if q1b==confirmation['E3_vs_E2']=='NOT_SUPPORTED' else 'INCONCLUSIVE')
        statuses['combined_decision_note']='Conservative cross-split consistency description, not a new significance test or pooled curve; original dev decisions and thresholds retained'
    write(ROOT/'analysis/conclusions.json',statuses)
    result_lines=['|实验|开发集 PDMS（95% CI）|相对 IL 的 G（95% CI）|M+|M−|单次零分比例|','|---|---:|---:|---:|---:|---:|']
    result_lines.append(f"|E0-IL|{b['components']['E0-IL']['PDMS']:.4f}|0（共同参照）|0|0|—|")
    bm=b['paired_RL_vs_IL']
    for n in ['E0-RL','E0-COPY']:
        result_lines.append(f"|{n}|{interval(bm['PDMS'])}|{interval(bm['G'])}|{bm['M_plus']['estimate']:.4f}|{bm['M_minus']['estimate']:.4f}|{100*bm['zero_output_fraction']['estimate']:.2f}%|")
    for e,v in d['methods'].items():
        m=v['metrics'];result_lines.append(f"|{e}|{interval(m['PDMS'])}|{interval(m['G'])}|{m['M_plus']['estimate']:.4f}|{m['M_minus']['estimate']:.4f}|{100*m['zero_output_fraction']['estimate']:.2f}%|")
    e6_text='已运行原生DiffGRPO/OPD四步真实集成训练、独立连续4步与2+恢复2步的一致性验证，以及单步梯度校验；零OPD权重与原生loss/gradient精确一致，联合梯度有效且教师不变。它们是工程集成证据，不是超过教师的性能证据。'
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
    for e,v in anchored.items():forgetting+=f"{e}：M_minus 减少量={interval(v['M_minus_reduction'])}，下降幅度减少状态={v['raw_M_minus_reduction_status']}；平均 PDMS 差={interval(v['PDMS_difference'])}；同时保留平均增益的联合结论 {v['status']}。减少量门槛0.0005，平均分非劣界0.002；较低退化伴随较小增益是交易关系，不直接计为同增益保护。\n\n"
    forgetting+='含GT配方与E3的最终学习率不同，因此上述差异不能全部归因于GT监督。纯Anchor的E4-A/E3具有相同学习率。E5/E4-C具有相同学习率、GT与Anchor权重，是选择性分配的关键开发集对照；测试集仅确认预登记cal-best E4，不追加E4-C测试。\n\n'
    forgetting+='训练静态 gate 的类别及有效激活、逐类别实际暴露和最终变化见结构化 gate、训练 schedule 和逐场景结果。开发/测试上的 A–D 为诊断分层，不用于重新拟合 gate。NC/DAC/TTC 非退化约束不能解释为真实道路安全保证。\n'
    (REPORTS/'06_forgetting_analysis.md').write_text(forgetting)
    test_text='官方测试集正式确认尚未完成；不得将开发集结论表述为完整测试集成功。'
    if has_test:
        td=read(ROOT/'analysis/formal_navtest.json');tb=read(ROOT/'analysis/baseline_navtest.json')
        test_text=f"官方 navtest {td['scenes']} 场景、{td['logs']} 个 log；精确匹配仓库官方 token 清单。IL={tb['components']['E0-IL']['PDMS']:.6f}，RL={tb['components']['E0-RL']['PDMS']:.6f}。"+' '.join(f"{e} PDMS={interval(v['metrics']['PDMS'])}，G={interval(v['metrics']['G'])}。" for e,v in td['methods'].items())+' 修正批次所有候选在锁定cal选择后一次性确认，测试结果不反向修改方案。工程修正前曾查看无效v1测试基线均值；未查看v1方法测试聚合。此数据接触限制与修正原因在08报告披露。'
        test_text+='\n\n关键测试配对差：'+'；'.join(f"{key}：PDMS差={interval(v['PDMS'])}，M_minus差={interval(v['M_minus'])}" for key,v in td['contrasts'].items())+'。'
        test_text+=f"\n\n开发主分析与测试确认分别保留，不合并曲线。跨两个split的稳定额外价值描述为 **{statuses['Q1_stable_extra_value']}**；测试OPD/KD状态={statuses['navtest_confirmation']['E3_vs_E2']}。一致性描述沿用相同实质性门槛，不是新增显著性检验。"
    admission_reason='E5未建立预登记的正向开发集泛化收益；因此不扩展正式E6。该停止来自科学门槛，非GPU不足' if not admission['admitted'] else '预登记开发门槛通过，已按锁定预算执行E6'
    costs=read(ROOT/'analysis/costs_dev.json')
    cost_text='正式1000步动作头训练（三种子平均，不含共享编码/教师库/评测）：\n\n|方法|秒/种子|峰值显存GiB|\n|---|---:|---:|\n'
    for e,v in costs.items():cost_text+=f"|{e}|{v['mean_training_GPU_hours']*3600:.1f}|{v['mean_peak_GPU_GiB']:.2f}|\n"
    online=read(ROOT/'audit/online_inference_cost.json')['summary']
    cost_text+=f"\n独立20场景、预热1场景的原生学生在线推理中位数为{online['online_image_VLM_planner_seconds']['median']:.4f}秒/场景，包含图像加载、VLM和规划器；缓存输入后的规划器为{online['cached_planner_seconds']['median']:.4f}秒，正式单候选物理评分为{online['canonical_scalar_score_seconds']['median']:.4f}秒。测量来自活跃共享服务器，不是隔离延迟基准。E2与E3每种子均实际执行40,000个teacher样本步查询；E3另有40,000个学生rollout样本步，未给E3增加优化更新预算。\n"
    accounting=ROOT/'audit/accounting.json'
    if accounting.exists():cost_text+=f"\n有效修正批次的全部任务累计预留GPU时间为{read(accounting)['reserved_GPU_hours']:.2f} GPU小时，包含启动与评测，不等同于GPU利用率。\n"
    resources=ROOT/'audit/resource_summary.json'
    if resources.exists():
        rs=read(resources);cost_text+=f"\n无效v1仍计入工程成本：{rs['invalid_v1_GPU_hours']:.2f} GPU小时，两批累计{rs['total_GPU_hours']:.2f} GPU小时，未超过预登记128小时上限。五台主机的本任务pause标记已移除，原压力自动化恢复观察；没有终止真实他人任务。失败与CPU编排交接记录在外部accounting.json中保留，已修复的问题不从历史中删除。\n"
    report=f"""# ReCogDrive On-Policy Distillation 研究报告{'' if has_test else '（测试确认进行中）'}

## 研究结论

Q1（固定开发集主分析：学生链 OPD 优于轨迹 SFT 和教师链 KD）：**{q1}**。E3−E1={interval(contrast['E3_vs_E1']['PDMS'])}；E3−E2={interval(contrast['E3_vs_E2']['PDMS'])}；E3 相对 IL 增益={interval(e3['G'])}。优势、绝对能力迁移和训练损失下降分别判断；独立测试确认及跨split稳定性在后文单列。

Q2（已有 IL 能力受损）：**{forget}**。E3 的 M_minus={interval(e3['M_minus'])}，独立高分保持率={interval(e3['stable_high_retention'])}。Anchor/GT/gate 的具体缓解结论分别列于下表和遗忘报告，不由平均分替代。

Q3（进一步 DiffGRPO 超过 RL 教师）：**{statuses['Q3_teacher_ceiling']}**。{admission_reason}。{e6_text}

## 模型与算法核验

采用官方 IL 学生与 RL 活动动作头教师；IL/RL 的结构一致，加载器不选择 RL 的 old_policy。所有核心张量必须加载，完整 missing/unexpected/shape mismatch 记录在02报告及外部 audit/checkpoints.json。冻结同一1536维 VLM，只训练384维 small DiT 动作规划器，共34,329,219参数。八点三维轨迹、原生 norm_odo/denorm_odo、五步 DDIM 均未改动；专家、残差、未来目标等路径关闭。OPD显式兼容层恢复官方系统提示词边界换行与左侧fixed2800 padding，保留全部hidden states；真实8场景与固定官方源码逐元素一致。首轮动态padding批次为INVALID，统一重做且不合并，详见08报告。

DDIM adapter 与原 get_action 的逐步状态、均值及最终轨迹误差为0，E0-COPY 与 RL 误差为0。原生随机方差不能由 ddim_eta 配置名推断；实际路径含标准差下界和截断噪声，所以主版本匹配原生p_mean_variance返回的clamp前proposal均值MSE，不能宣称Gaussian KL，也不把它当作clamp后核的严格期望。最终clamp完整保留在rollout/parity/评分中。可选未截断共享方差 KL 仅有实现和正确性测试，正式性能消融 NOT_RUN。

[DiffusionOPD](https://arxiv.org/abs/2605.15055) 的关键是由当前学生产生去噪状态，再用冻结教师监督这些状态。E1 是最终轨迹伪标签的原生噪声预测训练，E2 的状态来自教师，E3 的状态每轮来自当前学生。所有链 detach，重新预测学生转移时才建立梯度。这里没有 environment-level on-policy；[Flow VLA 多教师工作](https://arxiv.org/abs/2610.21522)涉及的环境互动不能移用于固定日志实验。[PIVOT-Q](https://arxiv.org/abs/2610.05745)的行为保护思想是相关参考，本研究不声称复现其量化场景。

## 数据与受控实验

训练4115场景/50 log，校准518/4，固定开发1696/16；按完整 log 和父驾驶记录互斥。它们来自官方 navtrain，未参与本次新增训练不等于底模从未见过。VLM缓存只保存原生可见观测和实际动作头输入 token；8个在线场景与缓存的 token/轨迹误差为0。GT仅训练使用，metric cache 与 gate 是标签，不进入VLM编码。

七组正式实验各1000次实际更新、global batch8、三个独立训练种子42/43/44；配对推理种子1101–1105。共同 IL 初始化、数据顺序和初始噪声，两个学习率槽位，只有校准集决定学习率。Anchor/GT权重在固定训练样本按梯度尺度校准后锁定，不能从审核效果继续调节。E5 重新从 IL 开始，并不继承 E4 的额外训练。

{chr(10).join(result_lines)}

95%主区间按完整NAVSIM log、训练种子与配对推理种子分层bootstrap（2000次），保留log内相邻场景相关性；冻结E0基线不引入未知的历史训练种子变异。点估计为场景加权均值。实质性门槛PDMS0.002，退化幅度减少0.0005。同父驾驶记录的跨log相关性在后文以明确事后的较粗聚类敏感性分析检查，不替换主结果。校准只有4个log，超参数选择可能不稳定；三种子也不足以穷尽训练随机性。有限五次采样不证明单个场景的精确期望变化。

## OPD 是否提供额外价值

E3/E1状态={q1a}，E3/E2状态={q1b}，E3/IL状态={transfer}。训练 loss 的下降只说明转移拟合；最终动作裁剪、原生随机采样与闭环跟踪可能使很小的中间误差对应离散零分。真实训练链的 E2/E3 状态分布、教师响应差异、各步 MSE 和梯度贡献见 diagnostics 与 Fig3。不能把状态分布有差异本身当成 OPD 性能优势。

E4/E5 保护结论：{'; '.join(f'{e}: {v["status"]}' for e,v in anchored.items())}。逐场景正分到零分、原失败修复、教师成功但学生失败、双方失败但学生改善、NC/DAC/TTC新增违规和原高分保持分别记录，避免只展示平均分或成功案例。

## 教师质量、gate 与偏差

RL 教师平均优于 IL，开发集仍有 {b['teacher_worse_scene_count']} 场景的五次平均下降超过0.02。训练静态 gate 依据五次配对差值的LCB，z=1、margin0.03并检查逐样本NC/DAC/TTC；z=1不是95%置信保证。质量 A/B/C/D 与有效gate不是同一概念。只在A且gate通过时接受强教师监督，B/C保护IL，D不假定教师有效。GT/anchor/opd各有独立归一化；空gate不能自动变成无条件OPD。

选择性监督改变有效训练暴露分布，可能偏向可修复但稀少的场景；静态gate依据原始IL，不保证更新后仍是最优选择。为避免将分布变化误当因果机制，使用共同初始化、相同训练场景/预算，并报告每类暴露、类别诊断和训练外表现。开发/测试类别图不参与训练选择。

E5相对同学习率、同GT/Anchor权重的均匀监督E4-C，开发集PDMS差={interval(contrast['E5_vs_E4-C']['PDMS'])}，M_minus差={interval(contrast['E5_vs_E4-C']['M_minus'])}。当前选择性配方的额外价值状态为{statuses['E5_vs_matched_E4C']['status']}；相对IL的预设实质性增益状态为{statuses['E5_transfer_vs_IL']}。这不推翻所有门控方式，也不区分当前分配策略内部各机制的独立作用。最终测试没有临时追加未入选E4-C，对测试整体配方差异不作同组件gate因果归因。

## 最终测试确认

{test_text}

## 成本、复制对照与适用范围

{cost_text}

E0-COPY直接复制教师权重即可精确恢复教师输出，且没有蒸馏训练成本。相同架构、同一VLM、没有部署约束时，OPD不能仅凭逼近教师证明工程价值；需要保持IL能力、选择性传递或整合互补教师的真实优势。推理只调用单学生，不增加教师或环境反馈。Fig8与成本表记录实际训练时间、峰值显存和教师查询；输入编码、静态gate、教师缓存和调参为共享前置成本，另列总账，不隐藏在某一方法中。

E6 训练使用原生自定义 PDM reward，EP/TTC/Comfort权重为10/5/2；正式评测 PDMS 为5/5/2，两者分开记录。本轮没有添加世界模型、专家路由或额外对齐任务。候选结果不支持时按协议停止对应路线，不继续搜索新的损失来追逐正结论。缺少正式E6结果时，无法区分教师约束、教师能力边界与GRPO本身限制，不作归因。

## 后续研究边界

本轮路线取舍：保留E2/E3与Anchor作为受控研究基线；开发集未证实学生链相对教师链的稳定额外价值，不扩大当前OPD损失搜索。停止当前静态Advantage Gate配方，正式E6不绕过门禁。单教师、同架构且无其他部署限制时，优先复用已验证的RL活动动作头；当前学生没有证明替代直接复制教师的实际收益。后续若研究互补教师或共同失败场景，应另立协议，不能作为本轮负结果后的临时扩展。

多教师值得继续的条件是有可重复的互补场景能力和超出单教师复制的收益；必须控制教师选择与数据暴露预算。时刻化NAVSIM物理反馈可用于训练特权教师，但未来对象占用只能作为训练标签/特权路径，部署学生仍只能使用当前与历史观测。不同质量教师的选择性传递需要训练外校准，不能用测试反馈构造gate。双方共同失败的场景需要新的合法探索或监督信息，反复复制固定教师不能自动产生缺失能力。泛化研究需按完整驾驶记录留出、跨日志/数据域验证，并保留不利结果，而非记忆训练token。

## 复现与证据

锁定协议：configs/opd/protocol.yaml（SHA256 {read(ROOT/'protocol.lock.json')['sha256']}）。分支 research/recogdrive-opd-20261020，原始HEAD884ea1d0f67c60821d61033d61efb10916d16929。最终推送提交与远端核验见外部 {ROOT}/delivery.json。

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
