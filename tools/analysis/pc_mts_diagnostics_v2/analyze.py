"""Scene-paired V2 summaries and independent validation; never modifies selection."""
import argparse,concurrent.futures
from v2_common import *
import pandas as pd
from scipy.stats import spearmanr
from raw_diagnostics import POS

def write(df,name):
    df.to_csv(OUT/'metrics'/f'{name}.csv',index=False);df.to_parquet(OUT/'metrics'/f'{name}.parquet',index=False)

def candidate_scene(scene):
    token=scene['token'];raw=np.load(OUT/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'][:,6]*100;cut=float(np.quantile(raw,.75));rows=[];poolrows=[];curves=[];denoise=OUT/'denoising'/f'{token}.npz';dn=np.load(denoise) if denoise.exists() else None
    for mi,method in enumerate(METHODS):
        a=np.load(OUT/'candidate_pools'/method/f'{token}.npz');trajs=a['trajectories'];scores=a['scores'];pdms=scores[:,6]*100;local=np.load(OUT/'evaluator/heldout_seeded'/method/f'{token}.npz')['trajectories'];lp=local[...,6]*100;lf=feasible(local);hf=hard_failure(local);drop=pdms[:,None]-lp;amp_mean=lp.reshape(16,4,6).mean(-1);amp_feas=lf.reshape(16,4,6).mean(-1);amp_drop=pdms[:,None]-amp_mean;passes=(amp_feas>=2/3)&(amp_drop<=2);passes=np.logical_and.accumulate(passes,axis=1);radius=(passes*np.asarray(CFG['heldout_amplitudes_m'])).max(1);auc=np.trapz(np.c_[np.zeros(16),amp_drop/100],x=[0]+CFG['heldout_amplitudes_m'],axis=1)/.5;dg=distance(trajs,np.asarray(scene['gt'])[None])[:,0];q=a['q_holdout'];regs=region(q)
        for j in range(16):
            row=dict(token=token,log=scene['log'],command=scene['command'],method=method,candidate=j,candidate_id=str(a['candidate_id'][j]),source=str(a['source'][j]),bridge_alpha=float(a['bridge_alpha'][j]),is_fill=bool(a['is_fill'][j]),qualified=bool(a['qualified'][j]),ood_filler=bool(a['ood_filler'][j]),fallback_level=int(a['fallback_level'][j]),raw_index=int(a['raw_index'][j]),pdms=pdms[j],feasible=bool(feasible(scores[j:j+1])[0]),d_GT=dg[j],region=regs[j],core=float(regs[j]=='core'),boundary=float(regs[j]=='boundary'),far=float(regs[j]=='far'),HQ_boundary=float(regs[j]=='boundary' and pdms[j]>=cut),HQ_far=float(regs[j]=='far' and pdms[j]>=cut),local_mean=float(lp[j].mean()),local_P10=float(np.percentile(lp[j],10)),local_CVaR20=float(np.sort(lp[j])[:5].mean()),local_feasible=float(lf[j].mean()),hard_failure=float(hf[j].mean()),robust_radius=radius[j],robustness_AUC=auc[j],worst_direction_drop=float(drop[j].reshape(4,3,2).mean(axis=(0,2)).max()),**{k:float(a[k][j]) for k in POS})
            if dn is not None:row.update(E_rec=float(dn['E_rec'][mi,j]),return_rate=float(dn['return_rate'][mi,j]),epsilon=float(dn['epsilon']),Q_E_rec=float(dn['Q_errors'].mean()),**{f'E_rec_{level}':float(dn['candidate_errors'][mi,j,level*2:level*2+2].mean()) for level in range(3)})
            rows.append(row)
        poolrows.append(dict(token=token,method=method,pool_diversity=pair_mean(trajs),unique_trajectories=len(set(digest(t.tolist()) for t in trajs)),unique_raw_count=int((a['raw_index']>=0).sum()),source_entropy=float(-sum(p*np.log(p) for p in pd.Series(a['source']).value_counts(normalize=True))),source_count=len(set(a['source']))))
        for k,amp in enumerate(CFG['heldout_amplitudes_m']):curves.append(dict(token=token,method=method,amplitude=amp,original_PDMS=float(pdms.mean()),local_PDMS=float(amp_mean[:,k].mean()),PDMS_drop=float(amp_drop[:,k].mean()),feasible_rate=float(amp_feas[:,k].mean()),failure_rate=float(1-amp_feas[:,k].mean()),hard_failure_rate=float(hf.reshape(16,4,6)[:,k].mean())))
    return rows,poolrows,curves

def paired(df,metrics,name):
    output=[];rng=np.random.default_rng(CFG['seed'])
    for metric in metrics:
        if metric not in df:continue
        pivot=df.pivot(index='token',columns='method',values=metric)
        for other in METHODS[:-1]:
            if 'pc_mts' not in pivot or other not in pivot:continue
            x=(pivot['pc_mts']-pivot[other]).dropna().to_numpy();n=len(x)
            if not n:continue
            boot=np.concatenate([x[rng.integers(0,n,size=(min(100,CFG['bootstrap_replicates']-i),n))].mean(1) for i in range(0,CFG['bootstrap_replicates'],100)])
            output.append(dict(metric=metric,comparison='pc_mts - '+other,n_scenes=n,delta=float(x.mean()),CI95_low=float(np.quantile(boot,.025)),CI95_high=float(np.quantile(boot,.975)),positive_scene_fraction=float((x>0).mean())))
    pd.DataFrame(output).to_csv(OUT/'metrics'/f'{name}.csv',index=False)

def pools():
    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as ex:results=list(ex.map(candidate_scene,scenes()['scenes'],chunksize=1))
    d=pd.DataFrame(sum([r[0] for r in results],[]));p=pd.DataFrame(sum([r[1] for r in results],[]));curve=pd.DataFrame(sum([r[2] for r in results],[]));contrast=set(read(OUT/'manifests/contrastive_scenes.json')['tokens']);d['contrastive']=d.token.isin(contrast);write(d,'candidates');write(p,'pool_geometry');write(curve,'robustness_curves_scene')
    selected_metrics=['pdms','d_GT','q_holdout','r_knn','maha_ratio','maha_percentile','E_rec','return_rate','local_mean','local_P10','local_CVaR20','local_feasible','hard_failure','robust_radius','robustness_AUC','worst_direction_drop','pool_diversity','HQ_boundary','HQ_far']
    for tag,tokens in [('full1000',set(d.token)),('contrastive',contrast)]:
        cut=d[d.token.isin(tokens)];scene=cut.groupby(['token','method']).mean(numeric_only=True).reset_index().merge(p,on=['token','method']);write(scene,f'pool_scene_{tag}');summary=scene.groupby('method').mean(numeric_only=True).reset_index();summary['n_scenes']=len(tokens);write(summary,f'pool_summary_{tag}');paired(scene,selected_metrics,f'paired_{tag}')
        c=curve[curve.token.isin(tokens)].groupby(['method','amplitude']).mean(numeric_only=True).reset_index();write(c,f'robustness_curves_{tag}')
        source=cut.groupby(['method','source']).agg(count=('candidate','size'),scenes=('token','nunique'),pdms=('pdms','mean'),r_knn=('r_knn','mean'),E_rec=('E_rec','mean'),robust_radius=('robust_radius','mean'),local_feasible=('local_feasible','mean')).reset_index();source['fraction']=source['count']/(len(tokens)*16);write(source,f'source_composition_{tag}')
        # Within-source, same-scene comparisons reduce source-mixture confounding. They are conditional associations.
        source_scene=cut.groupby(['token','method','source']).mean(numeric_only=True).reset_index()
        for source in sorted(cut.source.unique()):paired(source_scene[source_scene.source==source],['pdms','E_rec','return_rate','robust_radius','local_feasible'],f'paired_source_{tag}_{source}')
        unique=cut[~cut.is_fill].groupby(['token','method']).mean(numeric_only=True).reset_index();write(unique.groupby('method').mean(numeric_only=True).reset_index(),f'nonfill_summary_{tag}')
        parent_counts=cut[(cut.method=='pc_mts')&~cut.is_fill].groupby('token').size()
        matched=cut[((cut.method=='pc_mts')&~cut.is_fill)|((cut.method!='pc_mts')&(cut.candidate<cut.token.map(parent_counts)))].groupby(['token','method']).mean(numeric_only=True).reset_index()
        ms=matched.groupby('method').mean(numeric_only=True).reset_index();ms['n_scenes']=len(tokens);write(ms,f'matched_cardinality_summary_{tag}');paired(matched,selected_metrics,f'paired_matched_cardinality_{tag}')
        scorecheck=matched.pivot(index='token',columns='method',values='pdms');assert (scorecheck.pc_mts<=scorecheck.score+1e-8).all(),'Score top-n must dominate equal-count raw PC parents in mean PDMS'
        qualified_tokens=set(cut[(cut.method=='pc_mts')&cut.qualified].token)
        qualified=cut[cut.token.isin(qualified_tokens)].groupby(['token','method']).mean(numeric_only=True).reset_index();qs=qualified.groupby('method').mean(numeric_only=True).reset_index();qs['n_scenes']=len(qualified_tokens);write(qs,f'qualified_summary_{tag}');paired(qualified,selected_metrics,f'paired_qualified_{tag}')
        fallback=cut[~cut.token.isin(qualified_tokens)].groupby(['token','method']).mean(numeric_only=True).reset_index();fs=fallback.groupby('method').mean(numeric_only=True).reset_index();fs['n_scenes']=len(tokens-qualified_tokens);write(fs,f'unqualified_scene_summary_{tag}')
    corr=[]
    for name,frame in [('raw',pd.read_parquet(OUT/'metrics/raw_candidates_full.parquet')),('pools',d)]:
        for x,y in [('d_GT','r_knn'),('d_GT','maha_ratio'),('r_knn','maha_ratio'),('r_knn','E_rec'),('maha_ratio','E_rec'),('pdms','robust_radius')]:
            if y not in frame:continue
            clean=frame[[x,y]].dropna();value=spearmanr(clean[x],clean[y]);per_scene=[]
            for token,g in frame.groupby('token'):
                z=g[[x,y]].dropna()
                if len(z)>2 and z[x].nunique()>1 and z[y].nunique()>1:per_scene.append(spearmanr(z[x],z[y])[0])
            corr.append(dict(space=name,x=x,y=y,n=len(clean),pooled_spearman=float(value[0]),mean_within_scene_spearman=float(np.mean(per_scene)),scene_count=len(per_scene)))
    write(pd.DataFrame(corr),'correlations');save(OUT/'manifests/analysis_counts.json',dict(identity=ident(),candidate_rows=len(d),pool_rows=len(p),full_scenes=d.token.nunique(),contrastive_scenes=len(contrast),denoising_scenes=d[d.E_rec.notna()].token.nunique(),denoising_contrastive_scenes=d[d.E_rec.notna()&d.contrastive].token.nunique(),pc_unqualified_scene_count=d[(d.method=='pc_mts')&~d.qualified].token.nunique(),pc_fill_count=int(d[(d.method=='pc_mts')].is_fill.sum()),pc_OOD_filler_count=int(d.ood_filler.sum())))

def stratification():
    p=pd.read_csv(V1/'metrics/policy_distribution.csv');r=pd.read_csv(V1/'metrics/grpo_readiness.csv');cols=['token','checkpoint','hit8_1','feasible_rate','mean_pdms','p_plus_1'];d=p.merge(r[cols],on=['token','checkpoint']);il=d[d.checkpoint=='official_il'].set_index('token');quant=il.spread_auc.quantile([.25,.5,.75]).to_numpy();q=dict(zip(il.index,np.searchsorted(quant,il.spread_auc,side='right')+1));heading={s['token']:float(np.abs(np.diff(np.unwrap(np.r_[0,np.asarray(s['gt'])[:,2]]))).sum()) for s in scenes()['scenes']};bounds=np.quantile(list(heading.values()),[1/3,2/3]);d['IL_spread_quartile']=d.token.map(q).map(lambda i:'Q'+str(i));d['GT_heading_change']=d.token.map(heading);d['GT_heading_tertile']=np.asarray(['low','medium','high'])[np.searchsorted(bounds,d.GT_heading_change,side='right')];d['compressed']=d.spread_ratio_il<1;d['strongly_compressed']=d.spread_ratio_il<.5;d['hit8_delta_IL']=d.hit8_1-d.token.map(il.hit8_1);d['mean_pdms_delta_IL']=d.mean_pdms-d.token.map(il.mean_pdms);write(d,'policy_stratification_scene')
    save(OUT/'manifests/policy_strata.json',dict(identity=ident(),IL_spread_quartile_bounds=quant.tolist(),GT_heading_tertile_bounds=bounds.tolist(),difficulty_definition='Official IL stochastic Spread-AUC, not outcomes from APR',heading_definition='Total absolute unwrapped GT heading variation including origin heading 0'))
    measures=['pairwise_ade','spread_auc','spread_ratio_il','center_shift_from_il','D_positive','hit8_1','feasible_rate','mean_pdms','compressed','strongly_compressed','hit8_delta_IL','mean_pdms_delta_IL']
    for by in ['IL_spread_quartile','command','GT_heading_tertile']:
        summary=d.groupby([by,'checkpoint'])[measures].mean().reset_index();summary['n_scenes']=d.groupby([by,'checkpoint']).size().to_numpy();summary['median_SR']=d.groupby([by,'checkpoint']).spread_ratio_il.median().to_numpy();write(summary,'policy_by_'+by)
    summary=d.groupby('checkpoint')[measures].mean().reset_index();summary['median_SR']=d.groupby('checkpoint').spread_ratio_il.median().to_numpy();summary['D_positive_valid_scenes']=d.groupby('checkpoint').D_positive.count().to_numpy();write(summary,'policy_global')
    bins=[0,.25,.5,1,2,np.inf];d['SR_bin']=pd.cut(d.spread_ratio_il,bins,right=False).astype(str);summary=d[d.checkpoint.str.startswith('mts')].groupby(['checkpoint','SR_bin'])[measures].mean().reset_index();summary['n_scenes']=d[d.checkpoint.str.startswith('mts')].groupby(['checkpoint','SR_bin']).size().to_numpy();write(summary,'MTS_compression_bins')

def chains():
    from analyze_policy_distribution import geometry
    rows=[];scene_rows=[r for r in scenes()['scenes'] if r['token'] in set(subset('chain'))]
    for chain in read(OUT/'manifests/historical_chains.json')['chains']:
        for r in scene_rows:
            token=r['token'];il=np.load(V1/'rollouts/official_il'/f'{token}.npz')['trajectories'][:32];ilg,ilm=geometry(il);base=np.load(V1/'rollouts'/chain['baseline']/f'{token}.npz')['trajectories'][:32];baseg,basem=geometry(base);ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])*100
            for snap in chain['snapshots']:
                if snap['reuse_V1']:t=np.load(V1/'rollouts'/snap['reuse_V1']/f'{token}.npz')['trajectories'][:32];s=np.load(V1/'evaluator/rollouts'/snap['reuse_V1']/f'{token}.npz')['trajectories'][:32]
                else:t=np.load(OUT/'chains/rollouts'/snap['name']/f'{token}.npz')['trajectories'];s=np.load(OUT/'chains/scores'/snap['name']/f'{token}.npz')['trajectories']
                g,med=geometry(t);f=feasible(s);reward=s[:,6]*100;pos=f&(reward>ref)
                rows.append(dict(token=token,chain=chain['name'],algorithm=chain['algorithm'],step=snap['step'],role=snap['role'],**g,spread_ratio_IL=g['spread_auc']/max(ilg['spread_auc'],1e-12),spread_ratio_step0=g['spread_auc']/max(baseg['spread_auc'],1e-12),step0_SR=baseg['spread_auc']/max(ilg['spread_auc'],1e-12),center_shift_IL=float(distance(med[None],ilm[None])[0,0]),center_shift_step0=float(distance(med[None],basem[None])[0,0]),mean_pdms=float(reward.mean()),feasible_rate=float(f.mean()),D_positive=pair_mean(t[pos]),hit8_1=float((f&(reward>ref+1)).reshape(4,8).any(1).mean())))
    d=pd.DataFrame(rows);write(d,'chain_evolution_scene');write(d.groupby(['chain','algorithm','step','role']).mean(numeric_only=True).reset_index(),'chain_evolution_summary');gain=[]
    for (chain,token),g in d.groupby(['chain','token']):
        start=g[g.step==0].iloc[0];last=g[g.step==g.step.max()].iloc[0];gain.append(dict(chain=chain,token=token,step0_SR=start.step0_SR,pdms_gain=last.mean_pdms-start.mean_pdms,hit8_gain=last.hit8_1-start.hit8_1,spread_ratio_gain=last.spread_ratio_step0))
    write(pd.DataFrame(gain),'chain_gain_scene')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pools','stratification','chains']);a=p.parse_args();globals()[a.stage]()
