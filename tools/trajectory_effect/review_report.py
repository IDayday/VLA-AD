"""Report the bounded numerical replay without changing confirmatory artifacts."""
from __future__ import annotations

import shutil
import numpy as np
from .common import *
from .review import DEST,ARMS


def main():
    reg=read(DEST/'registration.json');summary=read(DEST/'summary.json')
    offline=read(DEST/'offline.json');decision=read(ROOT/'analysis/decisions.json')
    complete=[];gradients=[];shared=[];jobs=[]
    for arm in ARMS:
        for seed in reg['training_seeds']:
            name=f'formal_{arm}_{seed}';c=read(DEST/name/'complete.json')
            assert c['status']=='PASS';complete.append(c)
            result=read(DEST/name/'results.json');gradients+=result['gradient_checks'];shared+=result['shared_feature_checks']
            job=ROOT/'jobs'/f'review_{arm}_{seed}';status=read(job/'complete.json')
            assert status['status']=='COMPLETE'
            jobs.append(dict(name=job.name,spec=read(job/'spec.json'),seconds=status['seconds'],
                source_commit=c['owner']['commit'],owner=c['owner']))
    gpu_hours=sum(j['seconds']/3600 for j in jobs)
    originals_changed=[p for p,h in reg['immutable_inputs'].items() if not Path(p).exists() or sha(p)!=h]
    assert not originals_changed
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    cost=dict(reserved_gpu_hours=gpu_hours,jobs=jobs,code_commit=code,
        definition='One reserved GPU per review worker times its wall time; includes startup and data loading',
        optimizer_updates=0,new_score_labels=0,audit_forward_passes=0,scene_model_pairs=sum(c['scene_count'] for c in complete))
    atomic_json(DEST/'accounting.json',cost)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));colors={'D_fit':'#457b9d','D_cal':'#e76f51'}
    for i,arm in enumerate(ARMS):
        for split,offset in [('D_fit',-.15),('D_cal',.15)]:
            values=[v for v in summary['rows'] if v['arm']==arm and v['split']==split]
            x=i+offset+np.array([-.035,0,.035])
            axes[0].scatter(x,[v['pair_RMSE']/v['zero_pair_RMSE'] for v in values],color=colors[split],s=22,label=split if i==0 else None)
            axes[1].scatter(x,[v['within_scene_std_ratio'] for v in values],color=colors[split],s=22)
    axes[0].axhline(1,color='gray',ls='--',lw=.8)
    axes[0].set(ylabel='Pair RMSE / zero-difference RMSE',title='Saved models, real candidate labels')
    axes[1].set(yscale='log',ylabel='Within-scene predicted / true score std',title='Response to changes in numerical trajectories')
    for ax in axes:ax.set_xticks(range(len(ARMS)),ARMS);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[0].legend();fig.tight_layout()
    dest=REPORT_DIR;dest.mkdir(parents=True,exist_ok=True);(dest/'figures').mkdir(exist_ok=True)
    fig.savefig(dest/'figures/review_candidate_sensitivity.png',dpi=180);plt.close(fig)
    table=[]
    for split in ('D_fit','D_cal'):
        for arm in ARMS:
            v=[r for r in summary['rows'] if r['split']==split and r['arm']==arm]
            def mean(k):return np.mean([r[k] for r in v])
            table.append(f"| {split} | {arm} | {mean('absolute_Q_RMSE'):.6f} / {mean('fit_constant_Q_RMSE'):.6f} | {mean('pair_RMSE'):.6f} / {mean('zero_pair_RMSE'):.6f} | {100*mean('within_scene_std_ratio'):.3f}% |")
    def span(values):return f'{min(values):.6g}–{max(values):.6g}'
    replay_error=max(c['saved_prediction_max_error'] for c in complete)
    formula_error=max(c['loss_formula_max_error'] for c in complete)
    query_norms=[g['groups']['query']['difference_gradient_norm'] for g in gradients]
    vlm_norms=[g['groups']['vlm']['difference_gradient_norm'] for g in gradients if 'vlm' in g['groups']]
    fit=offline['library_distributions']['D_fit'];cal=offline['library_distributions']['D_cal']
    text=f"""# 已保存后果预测器的继续复核

本次继续工作对已保存的15个正式模型执行真实前向与反向复核，没有新增优化器更新、评分真值或审计集前向。原[主报告](REPORT_zh.md)、锁定协议、正式统计结果和研究判断均通过内容哈希验证保持不变。

## 复核结果

- 完成5组×3种子×128场景，共{cost['scene_model_pairs']}个场景—模型组合；每个模型另对两个固定场景分别计算绝对值项与差值项梯度。15组全部通过。
- 原保存的基准Q和真实更新差值重放最大误差为 **{replay_error:.9g}**。从原逐轨迹分数重建的真实更新标签逐元素一致。
- 损失与独立NumPy公式的最大误差为 **{formula_error:.9g}**；将候选全部替换成同一数值轨迹后，候选间输出差异为0。
- 30个反向检查的轨迹查询分支差值梯度均非零，范数范围 **{span(query_norms)}**。联合组12次VLM差值梯度也均非零，范数范围 **{span(vlm_norms)}**。普通Q组的这次差值梯度仅作诊断，没有用于训练。
- 联合组12次零LoRA对照的原生特征与冻结缓存最大误差为0；加载已训练LoRA后，特征最大变化为 **{span([v['token_change_max'] for v in shared])}**，同噪声动作的xy RMS变化为 **{span([v['action_change_xy_RMS_m'] for v in shared])} 米**。因此训练确实改变了动作生成共享的VLM路径。
- 每个检查点文件哈希与头参数在复核前后均保持一致。复核用的是固定的已训练模型。

## 近零预测的含义

确认训练库共有{fit['candidates']:,}个fit候选、{cal['candidates']:,}个cal候选；同场景候选对中，真实分差绝对值超过0.02的比例分别为 **{100*fit['material_pair_fraction']:.3f}%** 和 **{100*cal['material_pair_fraction']:.3f}%**。标签并非全部相同。

下表使用复核前固定的64个fit场景和64个cal场景，对三个训练种子取描述性平均。“常数Q”只使用完整D_fit候选的平均标签；“零差值”恒预测0。响应比例先在每个场景内中心化候选分数，再计算预测与真值标准差之比。

| 用途 | 组别 | 绝对Q RMSE / 常数Q RMSE | 候选对差值RMSE / 零差值RMSE | 同场景响应比例 |
|---|---|---|---|---|
{chr(10).join(table)}

![候选后果响应复核](figures/review_candidate_sensitivity.png)

模型对数值轨迹变化的响应很弱，而标签具有明显的候选差异。非零梯度和参数变化说明计算路径连通；它们不能证明模型已学会辨识后果。本次没有发现足以要求作废原结果的标签、保存、损失或共享特征错误。也没有验证导致学习失败的唯一原因，容量、优化和监督分布的作用尚未被区分。

原H2的 `NOT_SUPPORTED` 仅表示本次固定读出头、LoRA范围、数据和预算没有取得预注册的实质改善。不能据此声称VLM中不存在所需信息，或所有后果学习路线都无效。

## 结论与执行边界

H1维持 `{decision['H1']['status']}`，H2维持 `{decision['H2']['status']}`，H3维持 `{decision['H3']['status']}`。阶段B/D停止条件没有改变，未启动RL-Q/RL-E训练或navtest。继续扩大当前差值损失路线缺少支持；本次复核完成后停止该路线。

这是查看正式结果后的数值复核，不是新增确认实验；表中没有置信区间，也未用于调参或选择检查点。场景集合按token哈希从既有D_fit/D_cal评价目标中各取64个；梯度检查按预先记录的规则在每个集合内选标签跨度最大的一个场景。未访问新的审计输入，未更换研究起点。

使用4台获准服务器的空闲GPU并行重放，新增占用 **{gpu_hours:.4f} GPU小时**。原研究成本记录保持原样；本次成本和各命令另存 [复核运行记录]({DEST}/accounting.json)。

## 记录与复现

- 复核前固定的[注册记录](review_registration.json)，SHA256 `{sha(DEST/'registration.json')}`。
- 原锁定协议SHA256 `{reg['protocol_sha256']}`；报告生成代码 `{code}`；各实际进程使用的代码版本见运行记录。
- [逐模型汇总](review_summary.json)；逐场景预测、标签、梯度和共享动作变化：`{DEST}`。
- 研究分支：`research/trajectory-effect-validation-20261019`。

以下是真实执行入口；register和replay拒绝覆盖已完成记录。完整的15个replay命令及其host/GPU分配保存在运行记录，重放只需相应空闲GPU。

```bash
cd {SOURCE}
{PYTHON} -m tools.trajectory_effect.review register
{PYTHON} -m tools.trajectory_effect.review offline
CUDA_VISIBLE_DEVICES=0 {PYTHON} -m tools.trajectory_effect.review replay --arm FQ --seed 42
# 其余arm=T/FQ/FE/JQ/JE，seed=42/43/44；每个组合执行一次。
{PYTHON} -m tools.trajectory_effect.review summarize
{PYTHON} -m tools.trajectory_effect.review_report
```
"""
    (dest/'FOLLOWUP_REVIEW_zh.md').write_text(text)
    for name,target in [('registration.json','review_registration.json'),('summary.json','review_summary.json')]:
        shutil.copyfile(DEST/name,dest/target)
    print(dest/'FOLLOWUP_REVIEW_zh.md')


if __name__=='__main__':main()
