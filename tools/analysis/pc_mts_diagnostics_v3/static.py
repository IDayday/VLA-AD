"""A/B: outcome-independent gate order and raw-bank quality frontier."""
import argparse, concurrent.futures
from common_v3 import *

def dominators(scores):
    x=np.asarray(scores)[:,[2,3,5]]
    good=np.isfinite(scores).all(1)
    return np.all(x[None,:,:]>=x[:,None,:],2)&np.any(x[None,:,:]>x[:,None,:],2)&good[:,None]&good[None,:]

def ranks(scores, eligible=None):
    dom=dominators(scores);remaining=np.isfinite(scores).all(1)
    if eligible is not None:remaining &= eligible
    out=np.full(len(scores),-1,dtype=int);level=1
    while remaining.any():
        front=remaining & ~np.any(dom & remaining[None,:],axis=1)
        assert front.any();out[front]=level;remaining[front]=False;level+=1
    return out

def choose(indices, s, traj, q, rank, k=16, quality_only=False):
    remaining=list(map(int,indices));chosen=[];dist=distance(traj,traj)
    while remaining and len(chosen)<k:
        boundary=[i for i in remaining if q[i]>=50]
        available=boundary or remaining
        if boundary:
            if not quality_only:
                front=min(rank[i] for i in available);available=[i for i in available if rank[i]==front]
        else:
            bestq=max(q[i] for i in available);available=[i for i in available if q[i]==bestq]
            if not quality_only:
                front=min(rank[i] for i in available);available=[i for i in available if rank[i]==front]
        best=max(s[i,6] for i in available);ties=[i for i in available if s[i,6]>=best-.0001]
        # Suppression is a tie-break only, so it cannot silently change the quality/rank gates.
        if chosen:
            different=[i for i in ties if dist[i,chosen].min()>=.02]
            if different:ties=different
        win=max(ties,key=lambda i:(dist[i,chosen].min() if chosen else 0,s[i,6],-i))
        chosen.append(win);remaining.remove(win)
    return chosen

def conditional(s,traj,q,local,ref,unique,qmax=95,margin=0,lf=.75,quality_only=False):
    hard=np.isfinite(s).all(1)&~hard_failure(s)&unique
    strict=hard&(q<qmax)&(s[:,6]*100>=ref+margin-1e-8)&(local>=lf)
    cr=ranks(s,strict)
    primary=choose(np.flatnonzero(strict),s,traj,q,cr,quality_only=quality_only)
    fallback=hard&(q<qmax)&(s[:,6]*100>=ref-1-1e-8)&(local>=lf)&~strict
    sr=ranks(s,hard&(q<qmax)&(s[:,6]*100>=ref-1-1e-8)&(local>=lf))
    secondary=choose(np.flatnonzero(fallback),s,traj,q,sr,k=16-len(primary),quality_only=quality_only)
    return primary+secondary,[0]*len(primary)+[1]*len(secondary),strict,cr

def exact_sources(raw):
    out=[]
    for i,(s,cid,anchor) in enumerate(zip(raw['source'],raw['candidate_id'],raw['anchor_id'])):
        if 64<=i<128:out.append(['mts_8692','mts_8751','grpo_9041','apr_9145'][(i-64)//16])
        elif i>=128:out.append(str(s)+'_'+str(anchor).rsplit('_',1)[0])
        else:out.append(str(s))
    return out

def load_scene(scene):
    token=scene['token'];path=V2/'raw_candidates'/f'{token}.npz';raw=np.load(path);traj=raw['trajectories'];s=np.load(V2/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];pos=np.load(V2/'positions'/f'{token}.npz');local=feasible(np.load(V2/'evaluator/selection_local'/f'{token}.npz')['trajectories']).mean(1)
    unique=np.zeros(192,dtype=bool);unique[unique_indices(traj)]=True
    ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6]*100)
    return raw,traj,s,pos,local,unique,ref

def scene_a(scene):
    token=scene['token'];raw,traj,s,pos,local,unique,ref=load_scene(scene);q=pos['q_holdout'];global_rank=ranks(s);valid_s=np.isfinite(s).all(1)&unique;hard=valid_s&~hard_failure(s);compatible=hard&(q<95);quality=compatible&(s[:,6]*100>=ref-1e-8)
    idx,levels,strict,cr=conditional(s,traj,q,local,ref,unique)
    rescued=strict&(global_rank>2)&(cr<=2)&(cr>0);dom=dominators(s);rescue=[];domrows=[]
    exact=exact_sources(raw)
    for i in np.flatnonzero(rescued):
        ds=np.flatnonzero(dom[i]);far=ds[q[ds]>=95]
        rescue.append(dict(token=token,raw_index=int(i),candidate_id=str(raw['candidate_id'][i]),PDMS=s[i,6]*100,q=q[i],r_knn=pos['r_knn'][i],source=exact[i],global_rank=global_rank[i],conditional_rank=cr[i],dominator_count=len(ds),far_dominator_count=len(far),far_dominator_poisoning=bool(len(far))))
        for j in ds:domrows.append(dict(token=token,rescued_index=i,dominator_index=j,dominator_q=q[j],dominator_PDMS=s[j,6]*100,dominator_source=exact[j],far=bool(q[j]>=95)))
    choices={'conditional_pc':(idx,levels)}
    ai,al,_,_=conditional(s,traj,q,local,ref,unique,quality_only=True);choices['quality_only']=(ai,al)
    frozen_info={}
    for m,v2m in [('score','score'),('pareto','pareto'),('gt_distance','gt_distance'),('old_pc','pc_mts')]:
        p=np.load(V2/'candidate_pools'/v2m/f'{token}.npz');ri=p['raw_index'];ids=[];ls=[];seen=set()
        for i,level in zip(ri,p['fallback_level']):
            if i>=0:
                b=traj[i].astype(np.float32).tobytes()
                if b not in seen:seen.add(b);ids.append(int(i));ls.append(int(level))
        choices[m]=(ids,ls);frozen_info[m]=dict(frozen_slots=len(ri),nonraw_fillers=int((ri<0).sum()),frozen_PDMS=float(p['scores'][:,6].mean()*100))
    rows=[];poolmeta={}
    for m,(ids,ls) in choices.items():
        ids=np.asarray(ids,dtype=int);strict_selected=int(strict[ids].sum());n=len(ids)
        original_strict=bool(m=='old_pc')
        rows.append(dict(token=token,method=m,strict_candidate_count=strict_selected,strict_available_in_raw=int(strict.sum()),strict_selected_count=strict_selected,unique_candidate_count=n,zero_strict_scene=strict_selected==0,scene_has_4=strict_selected>=4,scene_has_8=strict_selected>=8,scene_has_16=strict_selected>=16,fallback_rate=float((~strict[ids]).mean()) if n else 1.,primary_count=sum(l==0 for l in ls) if m in ['conditional_pc','quality_only'] else strict_selected,secondary_count=sum(l==1 for l in ls) if m in ['conditional_pc','quality_only'] else 0,PDMS=float(s[ids,6].mean()*100) if n else np.nan,strict_PDMS=float(s[ids[strict[ids]],6].mean()*100) if strict_selected else np.nan,core_fraction=float((q[ids]<50).mean()) if n else np.nan,boundary_fraction=float(((q[ids]>=50)&(q[ids]<95)).mean()) if n else np.nan,far_fraction=float((q[ids]>=95).mean()) if n else np.nan,r_knn=float(pos['r_knn'][ids].mean()) if n else np.nan,reference_PDMS=ref,**frozen_info.get(m,{})))
        poolmeta[m]=dict(indices=ids.tolist(),levels=ls,strict=[bool(strict[i]) for i in ids],candidate_ids=[str(raw['candidate_id'][i]) for i in ids])
    save(OUT/'cache/pools'/f'{token}.json',dict(identity=identity(),token=token,raw_sha256=sha(V2/'raw_candidates'/f'{token}.npz'),methods=poolmeta))
    rawdf=pd.DataFrame(dict(token=token,raw_index=np.arange(192),candidate_id=raw['candidate_id'],source=raw['source'],exact_source=exact,PDMS=s[:,6]*100,NC=s[:,0],DAC=s[:,1],EP=s[:,2],TTC=s[:,3],comfort=s[:,4],DDC=s[:,5],q_holdout=q,policy_distance_knn=pos['policy_distance_knn'],r_knn=pos['r_knn'],maha_ratio=pos['maha_ratio'],d_GT=distance(traj,np.asarray(scene['gt'])[None])[:,0],local_feasible=local,unique=unique,hard_safe=hard,conservative_feasible=feasible(s)&valid_s,strict_eligible=strict,global_rank=global_rank,conditional_rank=cr,rescued=rescued,reference_PDMS=ref))
    stages={'raw':unique,'hard_safety':hard,'compatible':compatible,'quality':quality,'local':strict,'global_front2':strict&(global_rank<=2)&(global_rank>0),'conditional_front2':strict&(cr<=2)&(cr>0)}
    funnel=[dict(token=token,stage=k,count=int(v.sum())) for k,v in stages.items()]
    # Same V2 eligibility conditions; change only where Pareto ranks are computed.
    oldbase=valid_s&feasible(s)&(q<95)&(s[:,6]*100>=ref-1-1e-8)&(local>=.75);old_cond=ranks(s,oldbase)
    order=dict(token=token,eligible=int(oldbase.sum()),global_front2=int((oldbase&(global_rank<=2)&(global_rank>0)).sum()),conditional_front2=int((oldbase&(old_cond<=2)&(old_cond>0)).sum()))
    sensitivity=[]
    for qm in CFG['supplementary']['q_max']:
        for margin in CFG['supplementary']['quality_margin_points']:
            for lf in CFG['supplementary']['local_feasibility']:
                ids,_,e,_=conditional(s,traj,q,local,ref,unique,qm,margin,lf)
                sensitivity.append(dict(token=token,q_max=qm,quality_margin=margin,local_threshold=lf,strict_count=int(e.sum()),strict_selected_count=min(16,int(e.sum())),pool_count=len(ids)))
    return rows,rawdf,funnel,rescue,domrows,order,sensitivity

def exp_a():
    identity();assert read(OUT/'manifests/protocol_frozen.json')['no_v3_outcomes_exist']
    start=time.time()
    with concurrent.futures.ProcessPoolExecutor(96) as ex:result=list(ex.map(scene_a,scenes(),chunksize=1))
    rows=pd.DataFrame(sum([r[0] for r in result],[]));raw=pd.concat([r[1] for r in result],ignore_index=True)
    csv(rows,'A_pool_scene.csv');parquet(raw,'A_raw_candidates.parquet');csv(pd.DataFrame(sum([r[2] for r in result],[])),'A_eligibility_funnel.csv');csv(pd.DataFrame(sum([r[3] for r in result],[])),'A_rescued_candidates.csv');parquet(pd.DataFrame(sum([r[4] for r in result],[])),'A_rescued_dominators.parquet');csv(pd.DataFrame([r[5] for r in result]),'A_pure_gate_order.csv');parquet(pd.DataFrame(sum([r[6] for r in result],[])),'A_threshold_sensitivity.parquet')
    summary=rows.groupby('method').mean(numeric_only=True).reset_index();csv(summary,'A_pool_summary.csv')
    discreteness=[]
    for gate,mask in [('raw',raw.unique),('compatible',raw.unique&raw.hard_safe&(raw.q_holdout<95)),('strict',raw.strict_eligible)]:
        for metric in ['NC','DAC','EP','TTC','DDC']:
            vals=raw.loc[mask,metric];discreteness.append(dict(gate=gate,metric=metric,n=len(vals),n_unique=vals.nunique(),std=vals.std(),fraction_one=(vals==1).mean(),values=','.join(map(str,sorted(vals.unique()))) if vals.nunique()<20 else 'continuous'))
    csv(pd.DataFrame(discreteness),'A_metric_discreteness.csv')
    pairs=[]
    for m in METHODS[:-1]:
        for metric in ['strict_selected_count','unique_candidate_count','PDMS','scene_has_4','scene_has_8','scene_has_16']:
            tab=rows.pivot(index='token',columns='method',values=metric).astype(float);pairs.append(dict(method=m,contrast='conditional_pc_minus_baseline',metric=metric,**bootstrap(tab.conditional_pc-tab[m],f'A_{m}_{metric}')))
    csv(pd.DataFrame(pairs),'A_scene_paired_bootstrap.csv')
    assert len(raw)==192000 and raw.token.nunique()==1000
    assert all(len(set(v['indices']))==len(v['indices']) for token in tokens('all') for v in read(OUT/'cache/pools'/f'{token}.json')['methods'].values())
    stage_audit('A',scene_count=1000,raw_candidate_count=len(raw),unique_raw_count=int(raw.unique.sum()),raw_table_sha256=sha(OUT/'metrics/A_raw_candidates.parquet'),pool_summary_sha256=sha(OUT/'metrics/A_pool_summary.csv'),rescue_candidates=int(raw.rescued.sum()),rescue_scenes=int(raw[raw.rescued].token.nunique()),source_composition=raw.exact_source.value_counts().to_dict(),selection_uses_denoising_or_gradients=False,duplicate_candidates_in_primary_pools=0,outcome_conditioned_scene_filtering=False,seconds=time.time()-start)
    print(summary[['method','strict_selected_count','unique_candidate_count','zero_strict_scene','PDMS']].to_string(index=False),flush=True)

def exp_b():
    assert (OUT/'manifests/audit_A.json').exists();raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');rows=[];head=[]
    for token,g in raw.groupby('token',sort=False):
        for definition,mask in [('conservative',g.conservative_feasible),('structural',g.hard_safe&g.unique)]:
            valid=g[mask];oracle=valid.PDMS.max()
            for q in CFG['frontier']['q_thresholds']:
                sub=valid[valid.q_holdout<q] if q<100 else valid;best=sub.PDMS.max();k=CFG['frontier']['top_k']
                rows.append(dict(token=token,feasibility=definition,q_threshold=q,count=len(sub),coverage=len(sub)>0,best_PDMS=best,top4_PDMS=sub.PDMS.nlargest(k).mean() if len(sub)>=k else np.nan,global_oracle=oracle,compatibility_tax=oracle-best))
            core=valid[valid.q_holdout<50].PDMS.max();bound=valid[(valid.q_holdout>=50)&(valid.q_holdout<95)].PDMS.max()
            head.append(dict(token=token,feasibility=definition,best_core=core,best_boundary=bound,boundary_headroom=bound-core,boundary_gain_over_reference=bound-g.reference_PDMS.iloc[0]))
    df=pd.DataFrame(rows);hd=pd.DataFrame(head);csv(df,'B_frontier_scene.csv');csv(hd,'B_boundary_headroom.csv');summary=[]
    for (feas,q),g in df.groupby(['feasibility','q_threshold']):
        for metric in ['best_PDMS','top4_PDMS','compatibility_tax','coverage']:
            x=g[metric].astype(float);summary.append(dict(feasibility=feas,q_threshold=q,metric=metric,**bootstrap(x,f'B_{feas}_{q}_{metric}'),**{f'P{p}':x.quantile(p/100) for p in [25,50,75,90]}))
    csv(pd.DataFrame(summary),'B_frontier_bootstrap.csv');special=[]
    for feas in ['conservative','structural']:
        g=df[(df.feasibility==feas)&(df.q_threshold==95)]
        for gap in [.25,.5,1.]:special.append(dict(feasibility=feas,metric=f'fraction_within_oracle_{gap}',**bootstrap((g.compatibility_tax<=gap+1e-8).astype(float),f'B_gap_{feas}_{gap}')))
        g=hd[hd.feasibility==feas]
        for key in ['boundary_headroom','boundary_gain_over_reference']:
            special.append(dict(feasibility=feas,metric=key,**bootstrap(g[key],f'B_{feas}_{key}')))
            special.append(dict(feasibility=feas,metric=key+'_positive_full1000',**bootstrap((g[key]>1e-8).astype(float),f'B_positive_{feas}_{key}')))
    csv(pd.DataFrame(special),'B_key_results.csv')
    stage_audit('B',scene_count=1000,raw_candidate_count=len(raw),raw_table_sha256=sha(OUT/'metrics/A_raw_candidates.parquet'),q_thresholds=CFG['frontier']['q_thresholds'],all_raw_before_method_selection=True,outcome_conditioned_scene_filtering=False)
    print(pd.DataFrame(special)[['feasibility','metric','mean','ci_low','ci_high']].to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['A','B']);a=p.parse_args();exp_a() if a.phase=='A' else exp_b()
