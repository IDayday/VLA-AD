"""Retrospective test of the proposed IL/RL/PSI trends; never select scenes by outcome."""
from common_support import *
from analyze import safe
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEST = OUT/'trend_audit_1000_20260920'
PLAN = ROOT/'configs/historical_sft_grpo_support/trend_audit_20260920.yaml'
REPORT = ROOT/'reports/IL_RL_PSI_TREND_AUDIT_1000_20260920.md'
P = yaml.safe_load(PLAN.read_text())
METRICS = ['pairwise_ADE', 'center_shift_m', 'feasible_rate', 'mean_PDMS']
LABEL = {'official_il':'Initial GT-IL', 'original_grpo_11970':'After original GRPO', 'psi_sft':'After PSI candidate SFT'}


def estimate(values, logs, key):
    x=np.asarray(values,dtype=float);logs=np.asarray(logs)
    good=np.isfinite(x);x=x[good];logs=logs[good]
    if not len(x): return dict(n=0)
    rng=np.random.default_rng(int(digest([P['bootstrap_seed'],key])[:15],16))
    b=P['bootstrap_replicates'];draws=[];cluster=[]
    u,inv=np.unique(logs,return_inverse=True);sums=np.bincount(inv,weights=x);counts=np.bincount(inv)
    for start in range(0,b,100):
        size=min(100,b-start)
        draws.extend(x[rng.integers(len(x),size=(size,len(x)))].mean(1))
        ix=rng.integers(len(u),size=(size,len(u)))
        cluster.extend(sums[ix].sum(1)/counts[ix].sum(1))
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),
                log_cluster_ci_low=float(np.quantile(cluster,.025)),log_cluster_ci_high=float(np.quantile(cluster,.975)),
                positive_fraction=float((x>1e-10).mean()),negative_fraction=float((x< -1e-10).mean()),
                tie_fraction=float((abs(x)<=1e-10).mean()))


def md(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    identity();DEST.mkdir(parents=True,exist_ok=True)
    frozen=DEST/'analysis_plan_frozen.json'
    if frozen.exists(): assert read(frozen)['sha256']==sha(PLAN)
    else: save(frozen,dict(sha256=sha(PLAN),timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          prior_aggregate_results_already_known=True,not_a_prospective_preregistration=True))
    ss=scenes();tokens=[s['token'] for s in ss];assert len(tokens)==len(set(tokens))==1000
    source_paths=[OUT/'metrics/scene_metrics.parquet',OUT/'metrics/group16_metrics.parquet',
                  OUT/'center_shift_20260919/scene_center_shifts.parquet',OUT/'manifests/scenes.json',OUT/'manifests/models.json']
    identities={str(p):sha(p) for p in source_paths}
    sf=pd.read_parquet(source_paths[0]);gf=pd.read_parquet(source_paths[1]);cent=pd.read_parquet(source_paths[2])
    sf=sf[sf.model.isin(P['models']) & sf.protocol.isin(P['protocols'])].copy()
    gf=gf[gf.model.isin(P['models']) & gf.protocol.isin(P['protocols'])].copy()
    cent=cent[(cent.reference_kind=='official_IL_same_sampler') & cent.model.isin(P['models'])]
    sf=sf.merge(cent[['token','model','protocol','center_shift_m']],on=['token','model','protocol'],validate='one_to_one')
    assert len(sf)==6000 and len(gf)==24000
    for _,g in sf.groupby(['model','protocol']):assert set(g.token)==set(tokens)
    meta=pd.DataFrame([{k:s[k] for k in ['token','log','command','frame_start']} for s in ss])
    membership=pd.read_csv(OUT/'metrics/training_membership.csv')[['token','common_train']]
    meta=meta.merge(membership,on='token',validate='one_to_one')
    meta.to_csv(DEST/'scenes_1000.csv',index=False)
    full=sf.copy();full['sample_partition']='FULL64'
    halves=[];novel=[];cache_hashes={r['path']:r['sha256'] for r in read(OUT/'manifests/cache_hashes.json')}
    for i,s in enumerate(ss):
        token=s['token'];banks={};scores={}
        for model in P['models']:
            for kind,holder in [('rollouts',banks),('scores/rollouts',scores)]:
                path=OUT/'cache'/kind/model/f'{token}.npz'
                assert sha(path)==cache_hashes[str(path)],path
                with np.load(path) as f: holder[model]={pr:f[pr] for pr in P['protocols']}
        for pr in P['protocols']:
            for model in P['models']:
                traj=banks[model][pr];score=scores[model][pr]
                assert traj.shape==(4,16,8,3) and score.shape==(4,16,7)
                assert np.isfinite(traj).all() and np.isfinite(score).all()
                for name,ids in P['sample_halves'].items():
                    g=gf[(gf.token==token)&(gf.model==model)&(gf.protocol==pr)&gf.group.isin(ids)]
                    assert len(g)==2
                    c=traj[ids].reshape(32,8,3).mean(0)
                    ref=banks['official_il'][pr][ids].reshape(32,8,3).mean(0)
                    row=dict(token=token,log=s['log'],command=s['command'],common_train=bool(meta.set_index('token').loc[token,'common_train']),
                             model=model,protocol=pr,sample_partition=name,center_shift_m=float(distance(c[None],ref[None])[0,0]))
                    row.update({k:float(g[k].mean()) for k in ['pairwise_ADE','feasible_rate','mean_PDMS']})
                    halves.append(row)
                feasible=safe(score)
                near=[]
                for name,ids in P['sample_halves'].items():
                    opposite=P['sample_halves']['B' if name=='A' else 'A']
                    ref=banks['official_il'][pr][opposite].reshape(32,8,3)
                    target=traj[ids].reshape(32,8,3)
                    near.append(distance(target,ref).min(1))
                nearest=np.concatenate(near);f=feasible.reshape(64)
                assert nearest.shape==f.shape==(64,)
                for radius in [.25,.5,1.]:
                    novel.append(dict(token=token,log=s['log'],command=s['command'],model=model,protocol=pr,radius_m=radius,
                                      geometric_noncoverage_mass=float((nearest>radius).mean()),
                                      feasible_noncoverage_mass=float(((nearest>radius)&f).mean()),
                                      feasible_rate=float(f.mean()),nearest_initial_ADE_mean=float(nearest.mean())))
        if (i+1)%200==0:print('CACHE CHECK / CROSS-FIT',i+1,flush=True)
    hf=pd.DataFrame(halves);nf=pd.DataFrame(novel)
    # Ensure extraction agrees with the previously audited full64 means.
    reconstructed=hf.groupby(['token','model','protocol'])[['pairwise_ADE','feasible_rate','mean_PDMS']].mean()
    expected=sf.set_index(['token','model','protocol']).loc[reconstructed.index,reconstructed.columns]
    assert np.max(abs(reconstructed.to_numpy()-expected.to_numpy()))<1e-8
    dataset=pd.concat([full,hf],ignore_index=True)
    dataset['rollout_cache_path']=[str(OUT/'cache/rollouts'/m/f'{t}.npz') for m,t in zip(dataset.model,dataset.token)]
    dataset['rollout_cache_sha256']=[cache_hashes[p] for p in dataset.rollout_cache_path]
    dataset.to_parquet(DEST/'scene_dataset.parquet',index=False)
    gf.to_parquet(DEST/'group_dataset.parquet',index=False)
    nf.to_parquet(DEST/'crossfit_geometric_coverage.parquet',index=False)
    absolute=[];paired=[]
    for (partition,pr),block in dataset.groupby(['sample_partition','protocol']):
        for model,g in block.groupby('model'):
            for metric in METRICS:
                scale=100 if metric=='feasible_rate' else 1
                absolute.append(dict(partition=partition,protocol=pr,model=model,metric=metric,
                                     **estimate(g[metric]*scale,g.log,f'absolute:{partition}:{pr}:{model}:{metric}')))
        for model,ref in [('original_grpo_11970','official_il'),('psi_sft','official_il'),('psi_sft','original_grpo_11970')]:
            a=block[block.model==model].set_index('token').loc[tokens]
            b=block[block.model==ref].set_index('token').loc[tokens]
            for metric in METRICS:
                scale=100 if metric=='feasible_rate' else 1
                paired.append(dict(partition=partition,protocol=pr,model=model,reference=ref,metric=metric,
                                   unit='percentage_points' if metric=='feasible_rate' else 'PDMS_points' if metric=='mean_PDMS' else 'm',
                                   **estimate((a[metric]-b[metric])*scale,a.log,f'delta:{partition}:{pr}:{model}:{ref}:{metric}')))
    af=pd.DataFrame(absolute);pf=pd.DataFrame(paired)
    af.to_csv(DEST/'absolute_statistics.csv',index=False);pf.to_csv(DEST/'paired_deltas.csv',index=False)
    strata=[]
    for label,col in [('command','command'),('PSI_membership','common_train')]:
        for keys,g in sf.groupby([col,'model','protocol']):
            strata.append(dict(stratification=label,stratum=keys[0],model=keys[1],protocol=keys[2],scenes=len(g),
                               **{k:float(g[k].mean()*(100 if k=='feasible_rate' else 1)) for k in METRICS}))
    pd.DataFrame(strata).to_csv(DEST/'all_prespecified_strata.csv',index=False)
    novelty=[]
    for (pr,radius),b in nf.groupby(['protocol','radius_m']):
        ref=b[b.model=='official_il'].set_index('token').loc[tokens]
        for model,g in b.groupby('model'):
            a=g.set_index('token').loc[tokens]
            result=estimate(100*(a.feasible_noncoverage_mass-ref.feasible_noncoverage_mass),a.log,f'novel:{pr}:{radius}:{model}')
            novelty.append(dict(protocol=pr,radius_m=radius,model=model,
                                feasible_noncoverage_mass_percent=float(a.feasible_noncoverage_mass.mean()*100),
                                initial_finite_bank_null_percent=float(ref.feasible_noncoverage_mass.mean()*100),**result))
    ns=pd.DataFrame(novelty);ns.to_csv(DEST/'finite_bank_coverage_summary.csv',index=False)
    # Plot paired effects, with separate units; both changes from IL are shown.
    fig,axs=plt.subplots(2,4,figsize=(15,6.5))
    for row,pr in enumerate(P['protocols']):
        for col,metric in enumerate(METRICS):
            ax=axs[row,col];b=pf[(pf.partition=='FULL64')&(pf.protocol==pr)&(pf.metric==metric)&(pf.reference=='official_il')]
            for j,model in enumerate(['original_grpo_11970','psi_sft']):
                v=b[b.model==model].iloc[0]
                ax.errorbar(v['mean'],j,xerr=[[v['mean']-v.ci_low],[v.ci_high-v['mean']]],fmt='o',capsize=4)
            ax.axvline(0,color='gray',lw=1);ax.set_yticks([0,1],['RL - IL','PSI - IL']);ax.set_title(pr+'\n'+metric)
            ax.set_xlabel('percentage points' if metric=='feasible_rate' else 'PDMS points' if metric=='mean_PDMS' else 'm');ax.grid(alpha=.2)
    fig.suptitle('Full 1000 scenes: paired effects and 95% scene-bootstrap CI');fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(DEST/f'trend_effects.{ext}',dpi=180,bbox_inches='tight')
    svg=DEST/'trend_effects.svg';svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n');plt.close(fig)
    pf[pf.partition=='FULL64'].to_csv(DEST/'trend_effects_plot_data.csv',index=False)
    manifest=dict(status='COMPLETE_RETROSPECTIVE_CACHE_ANALYSIS',scenes=1000,logs=meta.log.nunique(),
                  scenes_replaced=0,scenes_excluded=0,score_errors=0,models={k:models()[k] for k in P['models']},
                  protocols=P['protocols'],rollouts_per_model_protocol=64000,total_distinct_cached_rollouts=384000,
                  verified_cache_files=6000,optimizer_updates=0,source_sha256=identities,config_sha256=sha(PLAN),
                  novel_semantic_modes='UNTESTED',PSI_after_GRPO='MISSING',
                  A_B='independent RNG groups, same scenes, previously analyzed aggregate; not an unseen new cohort')
    save(DEST/'dataset_manifest.json',manifest)
    tables=[]
    for pr in P['protocols']:
        b=af[(af.partition=='FULL64')&(af.protocol==pr)]
        records=[]
        for model in P['models']:
            v=b[b.model==model].set_index('metric');r={'Policy stage':LABEL[model]}
            for metric in METRICS:r[metric]=f"{v.loc[metric,'mean']:.4f}"
            records.append(r)
        tables.append(md(pd.DataFrame(records)))
    delta=pf[(pf.partition=='FULL64')&(pf.reference=='official_il')].copy()
    delta['effect_95CI']=delta.apply(lambda r:f"{r['mean']:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",axis=1)
    novel_primary=ns[ns.radius_m==.5][['protocol','model','feasible_noncoverage_mass_percent','initial_finite_bank_null_percent','mean','ci_low','ci_high']].round(4).astype(str)
    report='''# IL / RL / PSI：1000场景趋势假设核查

这是对已知总体结果的回顾性分析，不能称为预注册的新实验。固定原1000 NAVTRAIN场景，未根据结果替换或删除场景；没有重新训练或推理。每模型、每采样器64次（4组×16），共复用384,000条真实采样与统一NAVSIM评分。三个checkpoint对应IL→original GRPO、IL→PSI candidate SFT两个分支，不是三个连续阶段。PSI后续GRPO权重缺失。

## 数据集与口径

共有1000场景、512个log；直行634、左转251、右转115。835是PSI/A5/V6共同训练场景，165是PSI验证场景；不宣称untouched Navtest泛化。所有数据集都保留完整1000场景，命令/原训练身份只作公开分层。

Pairwise ADE按组内16条计算，再平均4组和场景。中心为64条XY均值轨迹，位移参照同一采样器下官方IL。Feasible要求NC=DAC=TTC=DDC=1，表中单位百分比，配对差为百分点。PDMS单位0–100。

## 普通推理（Full1000）

'''+tables[0]+'''

## 真实GRPO采样（Full1000）

'''+tables[1]+'''

## 场景配对差值

3000次scene-paired bootstrap；log-cluster区间及正/负/相等scene比例另存。没有把候选数当作独立场景数。以下CI是描述性区间，不从多指标中挑显著项定义整体成功。

'''+md(delta[['protocol','model','metric','effect_95CI']])+'''

## 独立采样组复核

固定group0/1为A、group2/3为B，每批每scene32条，二者随机流独立但场景相同。此前总体分析已包含这些样本，因此这只是采样稳定性检查，不是新盲测或独立场景验证。所有A/B绝对值及配对区间完整保存在absolute_statistics.csv和paired_deltas.csv；没有根据哪一半有利选择结果。本次PSI相对IL的可行率下降在两半均复现：eval A/B分别−2.150/−2.009个百分点，native A/B分别−6.919/−6.525个百分点。PSI的eval小幅均分增益在两半CI均包含0；native均分下降在两半CI均不包含0。这支持采样稳定性，不增加独立场景数。

## 是否增加有限IL参考库较少覆盖的可行轨迹

对目标A的32条轨迹，与IL B的32条参考比较，B对A同理。若到所有参考轨迹的最小ADE>0.5米且目标可行，则计入；分母始终64，包括不安全轨迹。IL自身两半互查给出有限参考库的非覆盖基线。0.25/1.0米敏感性全部公开。

'''+md(novel_primary)+'''

这是几何有限采样非覆盖，不是语义mode数量；高维强噪声下即使同一policy的两批样本也可能互不接近。必须对照IL自身基线，不能仅凭PSI数值大就宣称新模式。mean/CI列是目标相对该基线的百分点差。本次native在0.5米下，三模型几何非覆盖率均为100%，该指标已饱和，feasible_noncoverage_mass完全退化为可行率，不能提供额外的support扩张证据。eval下，PSI有21.814%的采样同时可行且不落在IL参考库的0.5米邻域内；original RL为26.648%，IL自身为0%。这不能证明新增语义mode，且这种几何变化也不是PSI独有。

## 对原假说的判定

- RL提高质量：SUPPORTED，两个采样协议均提高平均PDMS。
- RL可行率小幅上升：普通推理点估计小幅上升但CI包含0；native上升约7.96个百分点，不能把它描述成同一口径下的小幅变化。
- RL宽度持平或略降：native与该描述相容；普通推理ADE明确增加。不能混用普通推理的安全率与native的宽度拼成一行。
- PSI平均分小幅上升：普通推理点估计上升但CI包含0；native明确下降。PARTIALLY SUPPORTED至多适用于普通推理的描述性均值。
- PSI可行率明显上升：NOT SUPPORTED，两个协议均下降。
- PSI中心位移明显大于RL：NOT SUPPORTED，普通推理更小，native与RL接近。
- PSI宽度增加：SUPPORTED，两个协议都增加。
- RL仅在既有behavior modes内refinement、PSI创造新的feasible语义modes：UNTESTED。中心与宽度不能单独识别mode身份，有限参考库几何覆盖只提供补充证据。

现有数据不支持用户提出的完整对比叙事。不应根据目标趋势另选1000个成功场景或改可行性定义。如果进一步采集新的随机1000场景，应先冻结新manifest并完整报告结果；不能把结果最好的一批替代本报告。

数据文件：scenes_1000.csv、scene_dataset.parquet、group_dataset.parquet、crossfit_geometric_coverage.parquet、absolute_statistics.csv、paired_deltas.csv、all_prespecified_strata.csv、finite_bank_coverage_summary.csv。图trend_effects保存PNG/PDF/SVG及源CSV。源模型身份、轨迹缓存路径/hash和输入表hash可追溯；旧结果不变。
'''
    REPORT.write_text(report)
    print(delta[['protocol','model','metric','effect_95CI']].to_string(index=False))
    print('NOVELTY',novel_primary.to_string(index=False));print('REPORT',REPORT,flush=True)


if __name__=='__main__':main()
