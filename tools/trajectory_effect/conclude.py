"""Evidence-based stage admission and separately scoped H1/H2/H3 decisions."""
from __future__ import annotations
from .common import *


def main():
    cfg,identity=protocol();h1=read(ROOT/'analysis/h1.json');h2=read(ROOT/'analysis/probes_formal_D_audit.json')
    severe=read(ROOT/'analysis/severe_confirmation.json');c=h2['comparisons']
    confirmed={f['family'] for f in severe['families'] if f['scenes'] and f['metrics']['upper95'][1][0]<0}
    positive=[r for r in h1['families'] if r['split']=='audit' and r['step']==4 and r['metrics']['G']['lower95']>0 and r['family'] in confirmed]
    status1='SUPPORTED' if positive else ('PARTIALLY_SUPPORTED' if confirmed else 'INCONCLUSIVE')
    reason1=(('同一真实更新族 '+','.join(r['family'] for r in positive)+' 的总体增益区间高于0，同时独立25种子确认的严重下降子集仍为负；支持跨场景得失共存。') if positive else
             ('独立采样确认了真实局部损害，但同一更新同时提高其他场景期望回报的证据仍有限。' if confirmed else '当前样本不能稳定区分真实期望变化与采样波动。'))
    representations=[c['JQ-vs-FQ'],c['JE-vs-FE']]
    if any(v['lower95']>=.1 for v in representations):status2='SUPPORTED'
    elif any(v['lower95']>0 for v in representations):status2='PARTIALLY_SUPPORTED'
    elif all(v['upper95']<.1 for v in representations):status2='NOT_SUPPORTED'
    else:status2='INCONCLUSIVE'
    effect=('SUPPORTED' if c['JE-vs-JQ']['lower95']>=.1 else ('NOT_SUPPORTED' if c['JE-vs-JQ']['upper95']<.1 else 'INCONCLUSIVE'))
    reason2=(f"JQ/FQ的RMSE相对降低{100*c['JQ-vs-FQ']['estimate']:.3f}%，JE/FE为{100*c['JE-vs-FE']['estimate']:.3f}%；"
             f"JE/JQ为{100*c['JE-vs-JQ']['estimate']:.3f}%。差值监督的10%实质收益判断为{effect}。")
    detail2=(reason2+f" 对应95%区间分别为[{100*c['JQ-vs-FQ']['lower95']:.3f},{100*c['JQ-vs-FQ']['upper95']:.3f}]%、"
        f"[{100*c['JE-vs-FE']['lower95']:.3f},{100*c['JE-vs-FE']['upper95']:.3f}]%、"
        f"[{100*c['JE-vs-JQ']['lower95']:.3f},{100*c['JE-vs-JQ']['upper95']:.3f}]%。"
        '这些判断限定于本次观测、读出头、LoRA范围与训练预算；不证明所有VLM训练方式都无效，也不把非显著差异本身视为反证。各组与零差值预测基线几乎相同，不支持冻结表征已充分的结论。普通Q与差值损失没有获得不同的真值信息。')
    admitted=h1['gate_B']=='PASS' and h2['gate_D']=='PASS'
    stage_E='PASS' if admitted else 'NOT_RUN_GATE'
    reason3=('阶段E尚需实际RL-Q/RL-E结果，后果预测精度不能替代策略优化证据。' if admitted else
             f"阶段B={h1['gate_B']}，阶段D={h2['gate_D']}；按锁定停止规则未启动500步RL对照，H3未被检验。")
    status3='INCONCLUSIVE';detail3=reason3+' 本轮原生GRPO短分支验证了真实更新后果，但不构成同数据RL-Q/RL-E的同预算或匹配增益比较。没有把较小更新或没有正向增益的低退化率计作方法优势。'
    if admitted and (ROOT/'analysis/rl.json').exists():
        rl=read(ROOT/'analysis/rl.json');matched=rl['matched_gain']
        if matched and matched['both_audit_gains_positive']:
            v=matched['comparison'];threshold=matched['material_M_minus_threshold']
            status3='SUPPORTED' if v['lower95'][0][2]>=threshold else ('NOT_SUPPORTED' if v['upper95'][0][2]<threshold else 'INCONCLUSIVE')
            reason3=f'匹配正向增益下M_minus减少{v["estimate"][0][2]:.6f}，实质门槛{threshold:.6f}。'
            detail3=reason3+' 该判断仅覆盖当前开发集；官方测试集确认必须独立完成。'
    if effect=='NOT_SUPPORTED':recommendation='停止继续扩大当前差值辅助监督路线；保留真实退化和后果数据作为诊断资产。未验证的RL效果不能写成已被改善。'
    elif not admitted:recommendation='停止本轮扩容。现有证据不足以进入有效增益下的RL退化比较；如开展后续研究，应另行预注册。'
    else:recommendation='按锁定协议进入实际RL对照，完成后再决定是否进行官方测试集确认。'
    result=dict(protocol_sha256=identity,H1=dict(status=status1,reason=reason1,supporting_families=[r['family'] for r in positive]),
        H2=dict(status=status2,reason=reason2,detailed_interpretation=detail2,difference_supervision_status=effect),
        H3=dict(status=status3,reason=reason3,detailed_interpretation=detail3),stage_E=stage_E,
        gate_reason_en=f"B: {h1['gate_B']}; D: {h2['gate_D']}",recommendation=recommendation,
        navtest_statement='本轮未通过进入最终确认的开发门槛，未执行官方测试集确认。' if not admitted else '官方测试集确认状态见实际运行记录；开发结果不能替代最终确认。')
    atomic_json(ROOT/'analysis/decisions.json',result)
    atomic_json(ROOT/'analysis/stage_admission.json',dict(protocol_sha256=identity,stage_B=h1['gate_B'],stage_D=h2['gate_D'],stage_E=stage_E,
        confirmation_uses=1,source_results={str(p):sha(p) for p in [ROOT/'analysis/h1.json',ROOT/'analysis/probes_formal_D_audit.json']}))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
