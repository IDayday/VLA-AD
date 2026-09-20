"""Outcome-selected examples, explicitly not evidence for population effects."""
from common_support import *
from scipy.optimize import milp, Bounds, LinearConstraint
from datetime import datetime, timezone

PLAN=ROOT/'configs/historical_sft_grpo_support/posthoc_cases_20260920.yaml'
P=yaml.safe_load(PLAN.read_text())
DEST=OUT/'posthoc_case_selection_20260920'
REPORT=ROOT/'reports/POSTHOC_IL_RL_PSI_CASES_20260920.md'
METRICS=['pairwise_ADE','center_shift_m','feasible_rate','mean_PDMS']


def matrix(block,rule):
    items={m:block[block.model==m].set_index('token').sort_index() for m in P['models']}
    i,r,m=[items[k] for k in P['models']]
    assert i.index.equals(r.index) and i.index.equals(m.index)
    e=P['directional_numerical_guard']
    if rule=='directional':
        d={
            'RL_PDMS_up':r.mean_PDMS-i.mean_PDMS-e['pdms_points'],
            'PSI_PDMS_up':m.mean_PDMS-i.mean_PDMS-e['pdms_points'],
            'RL_PDMS_gain_larger':r.mean_PDMS-m.mean_PDMS-e['pdms_points'],
            'RL_feasible_up':r.feasible_rate-i.feasible_rate-e['feasible_fraction'],
            'PSI_feasible_gain_larger':m.feasible_rate-r.feasible_rate-e['feasible_fraction'],
            'RL_ADE_not_up':i.pairwise_ADE-r.pairwise_ADE,
            'PSI_ADE_up':m.pairwise_ADE-i.pairwise_ADE-e['displacement_m'],
            'PSI_center_shift_larger':m.center_shift_m-r.center_shift_m-e['displacement_m'],
        }
    else:
        t=P['practical_illustration_conditions']
        d={
            'RL_PDMS_gain_min':r.mean_PDMS-i.mean_PDMS-t['RL_PDMS_gain_min_points'],
            'PSI_PDMS_gain_min':m.mean_PDMS-i.mean_PDMS-t['PSI_PDMS_gain_min_points'],
            'PSI_PDMS_gain_max':t['PSI_PDMS_gain_max_points']-(m.mean_PDMS-i.mean_PDMS),
            'RL_feasible_gain_min':r.feasible_rate-i.feasible_rate-t['RL_feasible_gain_min_pp']/100,
            'RL_feasible_gain_max':t['RL_feasible_gain_max_pp']/100-(r.feasible_rate-i.feasible_rate),
            'PSI_feasible_gain_min':m.feasible_rate-i.feasible_rate-t['PSI_feasible_gain_min_pp']/100,
            'PSI_feasible_gain_larger':m.feasible_rate-r.feasible_rate-t['PSI_minus_RL_feasible_min_pp']/100,
            'RL_center_small':t['RL_centroid_max_m']-r.center_shift_m,
            'PSI_center_larger':m.center_shift_m-r.center_shift_m-t['PSI_minus_RL_centroid_min_m'],
            'RL_ADE_not_up':t['RL_ADE_change_max_m']-(r.pairwise_ADE-i.pairwise_ADE),
            'PSI_ADE_up':m.pairwise_ADE-i.pairwise_ADE-t['PSI_ADE_gain_min_m'],
        }
    return pd.DataFrame(d),items


def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in df.itertuples(index=False,name=None)])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    source=ROOT/P['source'];source_hash=sha(source);df=pd.read_parquet(source)
    assert df.token.nunique()==1000
    manifest=dict(purpose=P['purpose'],not_population_validation=True,selection_uses_outcomes=True,
                  timestamp_utc=datetime.now(timezone.utc).isoformat(),source=str(source),source_sha256=source_hash,
                  config_sha256=sha(PLAN),requested_selected_count=1000,available_comparable_scenes=1000,
                  new_inference=False,optimizer_updates=0,cohorts=[])
    selections=[];summaries=[];conditions=[];counts=[]
    for protocol in P['protocols']:
        base=df[(df.protocol==protocol)&(df.sample_partition=='FULL64')]
        for rule in ['directional','practical']:
            margins,items=matrix(base,rule);tokens=margins.index.to_numpy();n=len(tokens)
            each=margins.ge(0).all(axis=1)
            # Scaling helps MILP conditioning; it does not change a constraint.
            scale=np.maximum(np.abs(margins.to_numpy()).max(axis=0),1e-6)
            a=margins.to_numpy().T/scale[:,None]
            result=milp(-np.ones(n),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                        constraints=LinearConstraint(a,0,np.inf),
                        options={'time_limit':P['solver_time_limit_seconds'],'mip_rel_gap':0.0})
            chosen=np.zeros(n,dtype=bool) if result.x is None else result.x>.5
            means=margins.loc[chosen].mean() if chosen.any() else pd.Series(np.nan,index=margins.columns)
            assert not chosen.any() or means.min()>=-1e-7,(protocol,rule,means)
            status='OPTIMAL' if result.success else 'FEASIBLE_INCUMBENT_NOT_PROVEN_MAXIMAL' if chosen.any() else 'NO_NONEMPTY_SET_FOUND'
            rec=dict(protocol=protocol,rule=rule,available_scenes=n,individual_all_conditions_count=int(each.sum()),
                     selected_count=int(chosen.sum()),selected_fraction=float(chosen.mean()),solver_status=status,
                     solver_message=result.message,mip_gap=float(result.mip_gap) if getattr(result,'mip_gap',None) is not None else None,
                     outcome_selected=True,not_population_validation=True)
            manifest['cohorts'].append(rec);counts.append(rec)
            for j,token in enumerate(tokens):
                selections.append(dict(protocol=protocol,rule=rule,token=token,selected=bool(chosen[j]),
                                       individually_qualified=bool(each.iloc[j]),outcome_selected=True,source_scope='available1000'))
            for partition in P['partitions']:
                block=df[(df.protocol==protocol)&(df.sample_partition==partition)]
                picked=block[block.token.isin(tokens[chosen])]
                for scope,rows in [('selected_posthoc',picked),('all_available1000',block),('unselected',block[~block.token.isin(tokens[chosen])])]:
                    for model in P['models']:
                        g=rows[rows.model==model]
                        summaries.append(dict(protocol=protocol,rule=rule,partition=partition,scope=scope,model=model,scenes=len(g),
                                              outcome_selected=scope!='all_available1000',
                                              **{k:float(g[k].mean()) if len(g) else np.nan for k in METRICS}))
                if len(picked):
                    pm,_=matrix(picked,rule)
                    for name,value in pm.mean().items():
                        conditions.append(dict(protocol=protocol,rule=rule,partition=partition,condition=name,
                                               selected_scenes=int(chosen.sum()),mean_margin=float(value),passes=bool(value>=-1e-7),
                                               interpretation='descriptive_only_selection_used_FULL64_including_both_halves'))
            print(json.dumps(rec),flush=True)
    assert sha(source)==source_hash
    cf=pd.DataFrame(counts);sf=pd.DataFrame(summaries)
    pd.DataFrame(selections).to_csv(DEST/'scene_selection_with_exclusions.csv',index=False)
    sf.to_csv(DEST/'selected_full_and_excluded_statistics.csv',index=False)
    cf.to_csv(DEST/'selection_counts.csv',index=False)
    pd.DataFrame(conditions).to_csv(DEST/'condition_checks_full_A_B.csv',index=False)
    save(DEST/'manifest.json',manifest)
    c=cf[['protocol','rule','available_scenes','individual_all_conditions_count','selected_count','solver_status']].astype(str)
    show=sf[(sf.scope=='selected_posthoc')&(sf.partition=='FULL64')].copy()
    for k in METRICS:show[k]=show[k].map(lambda x:'NA' if not np.isfinite(x) else f'{x*(100 if k=="feasible_rate" else 1):.4f}')
    report='''# 事后按结果筛选的IL/RL/PSI案例集

**OUTCOME-SELECTED ILLUSTRATIONS — NOT POPULATION VALIDATION。** 这是应用户要求专门按期望结果选择的案例集。不能将此表作为随机1000场景结果、无偏评测、预注册验证或总体机制证据。原完整1000场景报告及负结果保持不变。

## 数据与筛选目标

可复用且具备三模型、同场景、多次采样、真实评分和同口径中心的完整缓存是1000场景。没有新增推理，没有把历史单输出评分当成64次分布，没有重复或加权复制场景来凑1000。

分别在eval、native_grpo中按相同scene ID选择，三模型使用同一选集且场景等权，禁止跨采样器拼指标。先报告每个场景单独满足条件的数量，再通过整数规划寻找子集均值满足条件的案例集。后者允许个别场景不满足趋势，由其他场景补偿；所有选中及排除ID完整公开。

directional只要求RL/PSI均分上升且RL增幅更大、两者可行率上升且PSI增幅更大、RL宽度不升、PSI宽度升、PSI中心位移更大。数值guard只避免浮点误差，不代表差异明显或显著。

practical把“小幅/明显”操作化为一套明确但并非唯一的事后示例阈值：RL均分至少+1分、PSI +0.1至+1分；RL可行率+0.1至+2个百分点、PSI至少+2个百分点且比RL高至少0.1个百分点；RL中心位移≤0.5m、PSI至少比RL多0.1m；RL ADE不增加、PSI至少增加0.01m。这些不是经用户确认或验证过的科学界限，不把它们包装成预注册标准。

目标为在上述条件下尽量保留更多unique scenes。求解时间上限20秒；只有solver_status=OPTIMAL才可称为该有限候选全集、该规则下的最大集合，其他只是找到的可行集合。

## 筛选数量

'''+md(c)+'''

## 选中案例的描述性统计

以下数字已经被筛选条件优化过。不能对这些条件性均值套普通bootstrap CI后声称验证了原假说。Feasible单位%，ADE/中心位移单位m，PDMS单位0–100。

'''+md(show[['protocol','rule','model','scenes']+METRICS].astype(str))+'''

A/B半组复查完整公开，但FULL64选择已经用了A和B，两者均不是未见验证数据。未选场景与完整1000场景同表提供，不能隐藏被排除的负结果。实际半组复查并非全部通过：practical-eval在A/B分别满足8/11、10/11条条件，practical-native分别满足10/11、8/11条；完整64次则都满足11/11条。这显示均值门槛附近的筛选结果对采样有敏感性，不能把FULL64满足门槛当作稳定泛化。

当前不能交付1000个不同且满足这些条件的案例：现有完整候选全集本身只有1000，选择全部就回到已经不满足期望趋势的总体。这里不声称在其他未采样场景中不存在这种案例；也没有扩大搜索直到得到期望结论后将其冒称随机数据。

完整原始对照：[1000场景假设核查](IL_RL_PSI_TREND_AUDIT_1000_20260920.md)。选集路径：outputs/historical_sft_grpo_support/posthoc_case_selection_20260920/scene_selection_with_exclusions.csv。结果用途仅为描述“在哪些已观测场景中能展示这种现象”。
'''
    REPORT.write_text(report)
    print(show[['protocol','rule','model','scenes']+METRICS].to_string(index=False));print('REPORT',REPORT)


if __name__=='__main__':main()
