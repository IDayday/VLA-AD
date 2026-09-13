"""Publish completed A–E evidence while F/G continue, without implying completion."""
from common_v3 import *
from report_tables import markdown_table

def table(frame, columns):
    return markdown_table(frame[columns])

def main():
    for phase in ['A','B','C','D','E','F_recipe','native','immutability']:
        assert (OUT/'manifests'/f'audit_{phase}.json').exists(), phase
    timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    a=pd.read_csv(OUT/'metrics/A_pool_summary.csv')
    ap=pd.read_csv(OUT/'metrics/A_scene_paired_bootstrap.csv')
    b=pd.read_csv(OUT/'metrics/B_key_results.csv')
    c=pd.read_csv(OUT/'metrics/C_paired_bootstrap.csv')
    c=c[(c.protocol=='primary')&(c.near_region=='all')]
    fe=pd.read_csv(OUT/'metrics/C_fixed_effects.csv')
    d=pd.read_csv(OUT/'metrics/D_matched_bootstrap.csv')
    e=pd.read_csv(OUT/'metrics/E_seed_means.csv')
    ep=pd.read_csv(OUT/'metrics/E_final_paired.csv')
    n=len(list((OUT/'cache/rollouts/grpo').rglob('*.npz')))
    p=read(OUT/'manifests/audit_split_provenance.json')
    lines=[
        '# PC-MTS V3 阶段性结果：A–E 完成，F/G 继续运行',
        f'整理时间：{timestamp}。这是用户要求的首次阶段性推送，**不是最终完成报告**。',
        f'分支：`analysis/pc-mts-policy-diagnostics-v3-20260913`；base：`{CFG["base_commit"]}`；primary YAML SHA256：`{sha(CONFIG_PATH)}`。',
        'V1/V2 的代码、结果、缓存、报告与图表保留。所有新增产物位于 V3 namespace；192-candidate reservoir 是受控诊断候选空间，不是历史训练数据。',
        '## 已完成与待完成',
        'A selection audit、B quality–compatibility frontier、C source/quality-matched native learnability、D exact gradient diagnostic、E controlled Micro-SFT 均已完成。12 个 Micro-SFT run、12 个短程 original-GRPO run 的训练均完成。',
        f'F 的快照评测正在运行；本次整理时完成 {n}/10800 个非零 GRPO 快照×场景缓存。每个快照固定300个holdout scenes、每场景64条CRN轨迹。训练成功不等于下游有效，**本次不报告尚未完整评分的GRPO收益**。',
        '待完成：F 全部快照评分、gain与safety统计；G current-policy独立R128/Q128重新校准及unchanged-IL重采样对照；最终审计、Fig6/7与完整报告。后台实验持续运行，完成后第二次commit/push。',
        '## 预先固定的实验与边界',
        '1000 scenes×192 unique raw candidates；token hash固定700 train/300 holdout；seeds=1701,2903。SFT 200 updates，snapshots=0/20/100/200；GRPO 100 updates，snapshots=0/10/50/100。优化器、LR、sampling budget、GT retention及unique-parent weighting对所有方法一致。',
        f'新optimizer更新未使用300个holdout tokens。但其中{p["holdout_scenes_sharing_a_train_log"]}个scene与训练共享log，且历史checkpoint可能已见过这些Navtrain scenes。冻结V1 GT-distance阈值利用了原1000场景校准。因此这是内部机制诊断，不能称为完全未接触的独立测试集。',
        '核心比较使用3000次bootstrap；配对候选同时报告pair-level和scene-cluster区间。训练结果先在同scene内平均两个seeds，再进行scene-paired比较；同时保留每个seed。q是Q→R距离经验秩，不是概率/likelihood。',
        '## A：Conditional selector与gate order',
        '实际顺序：hard safety NC=DAC=1 → q<95 → PDMS≥reference → selection-local≥0.75 → compatible集合内Pareto → Boundary优先 → diversity；strict不足时secondary才允许reference−1。Core按靠近Boundary优先。只计unique raw parents，禁止tiny/duplicate构成新模式。训练重复slot按1/(unique count×parent multiplicity)赋权，空pool使用显式GT fallback。',
        table(a,['method','strict_candidate_count','unique_candidate_count','zero_strict_scene','scene_has_4','scene_has_8','scene_has_16','PDMS']),
        'Conditional平均strict selected unique数15.113，Old-PC按同一strict定义为6.303；Conditional有40个零strict scene、38个空pool，全部保留在Full-1000 coverage中。候选PDMS对空pool未定义，因此不能拿Conditional的962-scene均值直接对比其他方法1000-scene均值。',
        '共1135条rescued candidates，涉及582/1000 scenes；far-dominator poisoning为100%。保持V2其他eligibility不变，仅改变Pareto排名集合时，平均Front≤2可用数由12.655变为13.798。完整新selector还修改了其他gate，不能把全部覆盖收益归因于gate order。',
        '同一962个可比较scene上的Conditional-minus-baseline候选PDMS：',
        table(ap[ap.metric=='PDMS'],['method','n','mean','median','ci_low','ci_high','win_fraction']),
        '**Pareto degeneration：** strict eligible集合内TTC/DDC恒为1，Pareto几乎退化为EP排序；没有事后换目标。覆盖提高不代表没有质量代价。',
        '## B：质量与兼容性frontier',
        table(b[b.feasibility=='conservative'],['metric','n','mean','median','ci_low','ci_high']),
        'Full-1000中，q<95可行oracle距global feasible oracle不超过0.25/0.5/1 point的场景比例为42.0%/44.5%/49.8%。Boundary相对Core有正headroom的场景为50.3%，相对IL reference为62.3%。有限的高质量frontier存在，但compatibility tax也真实存在。',
        '## C：source/quality-matched learnability',
        'primary gap≤0.25 point，共3274对、890 scenes；3209对Boundary-vs-Far、65对Core-vs-Far。同scene、exact source、NC/DAC class、一对一无replacement，pair内完全相同timestep/epsilon。原生epsilon-prediction MSE，未构造不存在的x0预测头。',
        table(c,['metric','near_mean','far_mean','mean','cluster_ci_low','cluster_ci_high','scene_win_fraction']),
        '控制scene/source固定效应、PDMS及d_GT后的r_knn系数：',
        table(fe[fe.predictor=='r_knn'],['outcome','coefficient','ci_low','ci_high','n','scenes']),
        '**SUPPORTED：** 离线quality、source与GT-distance不足以替代policy compatibility对原生拟合难度的描述；仍不等于距离本身的完整因果效应。',
        '## D：gradient compatibility',
        '固定256 scenes，最后DiT block+action_decoder共2453219个参数，exact gradients，没有sketch。matched gradient analysis使用393对候选。',
        table(d,['metric','near_mean','far_mean','mean','cluster_ci_low','cluster_ci_high']),
        '**SUPPORTED：** Far梯度范数更大。**NOT SUPPORTED：** Far具有更强gradient conflict/reference interference；相关区间包含0。不能把更大范数等同于有害冲突。',
        '## E：Controlled Micro-SFT',
        '所有方法从相同official IL action head开始，原生diffusion MSE+0.25 GT retention；200 updates、effective scene batch8、16 candidate slots。以下是final的两个seed结果：',
        table(e[e.step==200],['method','seed','PDMS','feasible_rate','Spread_AUC','center_shift','IL_retention_loss']),
        'Conditional-minus-baseline的scene-paired差值：',
        table(ep[ep.metric.isin(['PDMS','feasible_rate','center_shift','IL_retention_loss'])],['method','metric','mean','median','ci_low','ci_high','win_fraction']),
        '**PARTIALLY SUPPORTED：** Conditional位移和retention loss更小，相对Score/Pareto安全率更高；但相对GT-only PDMS仅+0.0141 point，CI包含0，没有证明额外quality headroom。Score/Pareto-MTS本轮分布更宽，不能将历史“Multi-SFT更窄”当成所有多候选SFT的必然性质。',
        '## 当前可以与不能得出的结论',
        'SUPPORTED：global gate会过滤compatible candidates；存在有限Boundary headroom；匹配与固定效应支持support distance和native fitting difficulty相关。',
        'PARTIALLY SUPPORTED：strict coverage和策略保留改善；但质量存在取舍，实际pool仍混合source、quality、support属性，不能唯一归因于compatibility。',
        'NOT SUPPORTED：更强gradient conflict、相对GT-only明确的SFT质量headroom，以及“多候选SFT必然压缩support”的普遍表述。NOT SUPPORTED表示当前未获支持，不是证明普遍无效。',
        'UNTESTED/评测未完成：Conditional-PC是否改善GT-SFT→GRPO的gain/safety tradeoff，以及更强policy是否促进Far→compatible。尚不足以将PC-MTS写成已验证的核心下游性能贡献。',
        '## 图表与审计',
        '全部表位于`outputs/pc_mts_diagnostics_v3/metrics/`，配置位于`configs/pc_mts_diagnostics_v3/`。数值无改动的C字符串序列化修复及A计数范围澄清均有独立审计；未修改阈值、seed、split或rollout数值。大型缓存/checkpoint只保留服务器。',
    ]
    for i in range(1,6):
        path=next((OUT/'figures').glob(f'Fig-V3-{i}_*.png'));stem=path.stem
        lines.append(f'Fig-V3-{i}: [PNG](../outputs/pc_mts_diagnostics_v3/figures/{stem}.png) / [PDF](../outputs/pc_mts_diagnostics_v3/figures/{stem}.pdf)。')
    report=ROOT/'reports/PC_MTS_POLICY_DIAGNOSTICS_V3_20260913.md'
    report.write_text('\n\n'.join(lines)+'\n')
    archive=OUT/'report/INTERIM_RESULTS_20260913.md';archive.write_text(report.read_text().replace('../outputs/pc_mts_diagnostics_v3/','../'))
    checkpoints=[]
    for path in (OUT/'checkpoints').rglob('*.json'):
        r=read(path);checkpoints.append(dict(path=r['path'],sha256=r['sha256'],step=r['step'],method=r['method'],seed=r['seed']))
    save(OUT/'manifests/INTERIM_CHECKPOINT_INDEX.json',dict(identity=identity(),checkpoints=checkpoints))
    save(OUT/'manifests/interim_report_identity.json',dict(identity=identity(),timestamp=timestamp,sha256=sha(report),completed_phases=['A','B','C','D','E'],F_training_complete=True,F_evaluated_scene_caches=n,F_required_scene_caches=10800,G_complete=False,final_report_pending=True))
    print(report,flush=True)

if __name__=='__main__':main()
