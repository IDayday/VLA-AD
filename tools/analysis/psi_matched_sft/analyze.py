"""Scene-paired, seed-averaged distributions and cross-target fitting."""
from common_matched import *
import concurrent.futures
setup_legacy();legacy.CFG['bootstrap_replicates']=3000
import importlib.util
spec=importlib.util.spec_from_file_location('historical_stats',MAIN/'tools/analysis/historical_sft_grpo_support/analyze.py');hist=importlib.util.module_from_spec(spec);sys.modules[spec.name]=hist;spec.loader.exec_module(hist)
METRICS=['mean_PDMS','feasible_rate','hard_failure_rate','pairwise_ADE64','centroid_displacement','Spread_AUC','min_PDMS','max_PDMS','std_PDMS','CVaR25_PDMS','Hit16','Hit8','no_safe_group','EP','NC','DAC','TTC','DDC','Comfort','GT_center_distance','CRN_policy_shift']

def loc(name,token,score=False):
    if name=='official_il':
        p=baseline(token,score)
        if p.exists():return p
    return OUT/('cache/scores/rollouts' if score else 'cache/rollouts')/name/f'{token}.npz'

def analyze_scene(s):
    t=s['token'];sr=[];gr=[];teachers=[]
    names=[('official_il','official_il',0,0)]+[(f'{m}_seed{r}_step{step:04d}',m,r,step) for m in CFG['methods'] for r in CFG['train_seeds'] for step in [128,512] if loc(f'{m}_seed{r}_step{step:04d}',t).exists()]
    if s['split']=='holdout':assert len([n for n in names if n[3]==512])==8
    with np.load(loc('official_il',t)) as f:ref={p:f[p] for p in CFG['evaluation']['protocols']}
    with np.load(loc('official_il',t,True)) as f:refs={p:f[p] for p in CFG['evaluation']['protocols']}
    threshold=refs['eval'][...,6].mean()*100+1
    raw=np.load(OUT/'cache/raw'/f'{t}.npz')['trajectories'];pools=read(OUT/'cache/selection'/f'{t}.json')['methods']
    for name,method,runseed,step in names:
        with np.load(loc(name,t)) as f:bank={p:f[p] for p in CFG['evaluation']['protocols']}
        with np.load(loc(name,t,True)) as f:sc={p:f[p] for p in CFG['evaluation']['protocols']}
        fitting=OUT/'cache/fitting'/name/f'{t}.npz';loss={}
        if fitting.exists():
            with np.load(fitting) as f:
                ids=json.loads(str(f['metadata']))['raw_indices'];vals=f['epsilon_mse'].mean(0);loss=dict(zip(ids,vals))
        for pr in CFG['evaluation']['protocols']:
            tr=bank[pr];scores=sc[pr];flat=tr.reshape(64,8,3);rflat=ref[pr].reshape(64,8,3)
            key=dict(token=t,log=s['log'],command=s['command'],split=s['split'],model=name,method=method,seed=runseed,step=step,protocol=pr)
            g=[dict(key,group=i,**hist.group_stats(tr[i],scores[i],threshold)) for i in range(4)];gr+=g
            keys=[k for k in g[0] if k not in key and k!='group']
            row=dict(key,**{k:hist.mean_or_nan([v[k] for v in g if np.isfinite(v[k])]) for k in keys})
            hq=(safe(scores)&(scores[...,6]*100>=threshold)).reshape(8,8)
            row.update(pairwise_ADE64=legacy.pair_mean(flat),centroid_displacement=float(legacy.distance(flat.mean(0)[None],rflat.mean(0)[None])[0,0]),CRN_policy_shift=float(np.linalg.norm(flat[...,:2]-rflat[...,:2],axis=-1).mean()),GT_center_distance=float(legacy.distance(flat.mean(0)[None],np.asarray(s['gt'])[None])[0,0]),Hit8=float(hq.any(1).mean()))
            sr.append(row)
            dist=legacy.distance(raw,flat)
            for origin,pool in pools.items():
                for kind in ['ALL','NON_GT','GT']:
                    ids=[i for i in pool['indices'] if kind=='ALL' or (kind=='GT' and i==0) or (kind=='NON_GT' and i!=0)]
                    if not ids:continue
                    weights=np.array([pool['weights'][pool['indices'].index(i)] for i in ids]);weights/=weights.sum()
                    target_center=np.einsum('i,ijk->jk',weights,raw[ids])
                    rec=dict(key,teacher_origin=origin,kind=kind,teacher_count=len(ids),native_epsilon_loss=float(np.array([loss[i] for i in ids])@weights) if all(i in loss for i in ids) else np.nan,nearest_teacher_ADE=float(dist[ids].min(1)@weights),teacher_center_ADE=float(legacy.distance(target_center[None],flat.mean(0)[None])[0,0]),rollout_nearest_target_ADE=float(dist[ids].min(0).mean()))
                    for radius in [.5,1.]:
                        suffix=str(radius).replace('.','p');hit=dist[ids]<=radius
                        rec['Hit64_'+suffix]=float(hit.any(1)@weights);rec['Mass64_'+suffix]=float(hit.mean(1)@weights)
                    teachers.append(rec)
    return sr,gr,teachers

def compare(frame,metrics,label):
    rows=[]
    for (split,pr),f in frame.groupby(['split','protocol']):
        # Average corresponding training seeds within each scene first.
        avg=f.groupby(['method','token','log'],as_index=False)[metrics].mean()
        for left,right in [('psi',m) for m in ['il_sft','score','pareto']]+[(m,'official_il') for m in CFG['methods']]:
            a=avg[avg.method==left].set_index('token');b=avg[avg.method==right].set_index('token');ix=a.index.intersection(b.index)
            for metric in metrics:
                v=hist.bootstrap_difference(a.loc[ix,metric]-b.loc[ix,metric],a.loc[ix,'log'],f'{label}/{split}/{pr}/{left}/{right}/{metric}')
                rows.append(dict(analysis=label,split=split,protocol=pr,left=left,right=right,metric=metric,**v))
    return rows

def main():
    assert (OUT/'audits/data_ready.json').exists();identity()
    records=scenes('holdout')+scenes('train')[:512];sr=[];gr=[];tr=[]
    with concurrent.futures.ProcessPoolExecutor(20) as pool:
        for i,(s,g,t) in enumerate(pool.map(analyze_scene,records,chunksize=4)):
            sr+=s;gr+=g;tr+=t
            if i%100==0:print('ANALYSIS',i,len(records),flush=True)
    sf=table('scene_metrics.parquet',sr);gf=table('group16_metrics.parquet',gr);tf=table('teacher_scene_metrics.parquet',tr)
    summary=sf.groupby(['split','method','seed','step','protocol'])[METRICS].agg(['mean','median','count']);summary.columns=['_'.join(c) for c in summary.columns];table('summary_by_seed.csv',summary.reset_index())
    final=sf[(sf.step==512)|(sf.method=='official_il')]
    avg=final.groupby(['split','protocol','method','token'],as_index=False)[METRICS].mean()
    summary=avg.groupby(['split','protocol','method'])[METRICS].mean().reset_index();table('summary.csv',summary)
    counts=avg.groupby(['split','protocol','method']).size().reset_index(name='scenes');table('summary_denominators.csv',counts)
    table('paired_comparisons.csv',compare(final,METRICS,'final512_seed_average'))
    seedrows=[]
    for runseed in CFG['train_seeds']:
        rr=compare(final[(final.seed==runseed)|(final.method=='official_il')],METRICS,f'final512_seed{runseed}');seedrows+=rr
    table('paired_by_seed.csv',seedrows)
    tsum=tf.groupby(['split','method','seed','step','protocol','teacher_origin','kind'])[['native_epsilon_loss','nearest_teacher_ADE','Hit64_0p5','Mass64_0p5','Hit64_1p0','Mass64_1p0']].agg(['mean','count']);tsum.columns=['_'.join(c) for c in tsum.columns];table('teacher_summary.csv',tsum.reset_index())
    tcomp=[]
    for (origin,kind),f in tf[(tf.step==512)|(tf.method=='official_il')].groupby(['teacher_origin','kind']):
        rr=compare(f,['native_epsilon_loss','nearest_teacher_ADE','Hit64_0p5','Mass64_0p5','Hit64_1p0','Mass64_1p0'],f'teacher:{origin}:{kind}');tcomp+=rr
    table('teacher_paired.csv',tcomp)
    table('command_strata.csv',final.groupby(['split','protocol','method','command'])[METRICS].mean().reset_index())
    earlytokens=set(sf[(sf.step==128)&(sf.split=='holdout')].token)
    evolution=sf[(sf.split=='holdout')&sf.token.isin(earlytokens)]
    table('matched1000_evolution.csv',evolution.groupby(['method','seed','step','protocol'])[METRICS].mean().reset_index())
    save(OUT/'audits/analysis_complete.json',dict(status='PASS',protocol_hash=identity(),scene_rows=len(sf),group_rows=len(gf),teacher_rows=len(tf),bootstrap_replicates=3000,seed_count=2,primary_denominator=5000))
    print(summary.query('split=="holdout"')[['method','protocol','mean_PDMS','feasible_rate','pairwise_ADE64','centroid_displacement']].to_string(index=False),flush=True)
if __name__=='__main__':main()
