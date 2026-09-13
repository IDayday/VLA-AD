"""Candidate position, pool sanity, ALL-16 vs UNIQUE-NON-FILL, and local tails."""
import argparse
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from common import *

def cvar20(v):
    a=np.sort(v,axis=-1);mass=a.shape[-1]*.2;n=int(np.floor(mass));frac=mass-n
    return (a[...,:n].sum(-1)+(a[...,n]*frac if frac else 0))/mass

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];records=[];pools=[];localrows=[];localrecords=[];overlaps=[]
    pending=pending_pc_tokens(out,cfg,sc) if args.matched_pc_only else set()
    if args.matched_pc_only:assert args.methods==METHODS, 'Conditional analysis must compare all four methods on identical scenes'
    for r in sc['scenes']:
        if r['token'] in pending:continue
        token=r['token'];gt=np.asarray(r['gt']);il=np.load(out/'rollouts/official_il'/f'{token}.npz')['trajectories'];raw_scores=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];raw_median=np.median(raw_scores[:,6])*100;method_sets={}
        for method in args.methods:
            f=np.load(out/'candidate_pools'/method/f'{token}.npz');a=f['trajectories'];s=f['scores'];fill=f['is_fill'];fingerprints=[digest(np.round(t,5).tolist()) for t in a]
            extra=(f['fallback_level']>=6) if method=='pc_mts' else np.zeros(len(a),bool)
            unique=[];seen=set()
            for i,(key,is_fill) in enumerate(zip(fingerprints,fill)):
                if not is_fill and key not in seen:unique.append(i);seen.add(key)
            method_sets[method]=set(fingerprints)
            subsets=[('all16',np.arange(len(a))),('unique_non_fill',np.asarray(unique,dtype=int)),('qualified_non_fill',np.asarray([i for i in unique if not extra[i]],dtype=int))]
            pos=policy_position(a,il,cfg['knn']) if len(a) else {key:np.array([]) for key in ['d_PC','q_policy','nearest_IL_distance','distance_to_IL_medoid']}
            dg=distance(a,gt[None])[:,0] if len(a) else np.array([]);q=pos['q_policy'];reward=s[:,6]*100
            for i in range(len(a)):
                records.append(dict(token=token,log=r['log'],command=r['command'],method=method,candidate_id=str(f['candidate_id'][i]),source=str(f['source'][i]),is_fill=bool(fill[i]),is_coverage_fallback=bool(extra[i]),fill_parent_id=str(f['fill_parent_id'][i]),fallback_level=int(f['fallback_level'][i]),pdms=float(reward[i]),d_GT=float(dg[i]),**{k:float(v[i]) for k,v in pos.items() if k!='self_distances'},region='core' if q[i]<50 else 'boundary' if q[i]<95 else 'ood',**{key:float(s[i,j]) for j,key in enumerate(FIELDS)}))
            for subset,indices in subsets:
                n=len(indices);base=dict(token=token,log=r['log'],command=r['command'],method=method,subset=subset,count=n,construction_success=len(a)==cfg['pool_size'])
                vals=dict(candidate_pdms=float(reward[indices].mean()) if n else np.nan,median_pdms=float(np.median(reward[indices])) if n else np.nan,d_GT=float(dg[indices].mean()) if n else np.nan,median_d_GT=float(np.median(dg[indices])) if n else np.nan,q_policy=float(q[indices].mean()) if n else np.nan,median_q_policy=float(np.median(q[indices])) if n else np.nan,core_fraction=float((q[indices]<50).mean()) if n else np.nan,boundary_fraction=float(((q[indices]>=50)&(q[indices]<95)).mean()) if n else np.nan,ood_fraction=float((q[indices]>=95).mean()) if n else np.nan,pool_diversity=pair_mean(a[indices]),filler_fraction=float(fill[indices].mean()) if n else np.nan,unique_fraction=len({fingerprints[i] for i in indices})/n if n else np.nan,boundary_quality_mass=float(((q[indices]>=50)&(q[indices]<95)&(reward[indices]>raw_median)).mean()) if n else np.nan)
                pools.append(dict(base,**vals))
                pools[-1].update(coverage_fraction=float(extra[indices].mean()) if n else np.nan,original_pc_failure=bool(extra.any()))
            held=out/'evaluator/heldout'/method/f'{token}.npz'
            if not held.exists():continue
            hs=np.load(held)['trajectories'];local_reward=hs[...,6]*100
            if len(a):
                local=dict(original_pdms=reward,local_mean=local_reward.mean(1),local_median=np.median(local_reward,axis=1),local_p10=np.quantile(local_reward,.1,axis=1),local_cvar20=cvar20(local_reward),local_feasible_rate=feasible(hs).mean(1),local_hard_failure_rate=hard_failure(hs).mean(1),local_drop=reward-local_reward.mean(1))
                per_amp=local_reward.reshape(len(a),len(cfg['evaluation_perturbation_amplitudes']),cfg['evaluation_directions_per_amplitude']).mean(-1)
                amps=np.asarray(cfg['evaluation_perturbation_amplitudes']);centered=amps-amps.mean();local['robustness_slope']=-(per_amp@centered)/(centered@centered)
                local.update({f'amp_{amp:g}':per_amp[:,i] for i,amp in enumerate(amps)})
                for i in range(len(a)):localrecords.append(dict(token=token,method=method,candidate_id=str(f['candidate_id'][i]),is_fill=bool(fill[i]),is_coverage_fallback=bool(extra[i]),**{k:float(v[i]) for k,v in local.items()}))
            else:local={k:np.array([]) for k in ['original_pdms','local_mean','local_median','local_p10','local_cvar20','local_feasible_rate','local_hard_failure_rate','local_drop','robustness_slope']+[f'amp_{amp:g}' for amp in cfg['evaluation_perturbation_amplitudes']]}
            for subset,indices in subsets:localrows.append(dict(token=token,log=r['log'],method=method,subset=subset,count=len(indices),original_pc_failure=bool(extra.any()),**{k:float(v[indices].mean()) if len(indices) else np.nan for k,v in local.items()}))
        for i,a in enumerate(args.methods):
            for b in args.methods[:i]:
                union=method_sets[a]|method_sets[b];overlaps.append(dict(token=token,a=a,b=b,jaccard=len(method_sets[a]&method_sets[b])/len(union) if union else np.nan))
    dest=out/'metrics'/('conditional_pc' if args.matched_pc_only else '');dest.mkdir(parents=True,exist_ok=True)
    if args.matched_pc_only:save(dest/'cohort.json',dict(identity=identity(cfg,sc),scope='conditional_on_PC_MTS_accepted_parent_not_full_cohort',full_scene_count=len(sc['scenes']),pending_count=len(pending),paired_scene_count=len(sc['scenes'])-len(pending),tokens=[r['token'] for r in sc['scenes'] if r['token'] not in pending]))
    for name,rows in [('candidate_records',records),('candidate_pool_scene',pools),('pool_overlap',overlaps),('local_robustness_scene',localrows),('local_robustness_candidates',localrecords)]:
        if rows:
            df=pd.DataFrame(rows);df.to_csv(dest/(name+'.csv'),index=False);df.to_parquet(dest/(name+'.parquet'),index=False)
    correlation=[]
    for method in args.methods:
        rows=[r for r in records if r['method']==method]
        x=np.array([r['d_GT'] for r in rows]);y=np.array([r['q_policy'] for r in rows])
        corr=float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None
        lr=[r for r in localrows if r['method']==method and r['subset']=='all16' and np.isfinite(r['original_pdms'])]
        xx=np.array([r['original_pdms'] for r in lr]);yy=np.array([r['local_mean'] for r in lr])
        pc=float(pearsonr(xx,yy).statistic) if len(xx)>2 and np.ptp(xx)>0 and np.ptp(yy)>0 else None
        correlation.append(dict(method=method,gt_distance_policy_percentile_spearman_descriptive=corr,original_local_scene_pearson=pc))
    save(dest/'candidate_correlations.json',correlation)
    print('Candidate and robustness statistics saved',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--matched-pc-only',action='store_true');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);main(p.parse_args())
