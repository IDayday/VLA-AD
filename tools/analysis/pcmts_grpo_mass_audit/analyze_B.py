"""Independent B validation, gated on the hash of all A-frozen selections."""
from common_mass import *
import pandas as pd
import concurrent.futures, math, time

def mean(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float('nan')
def median(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else float('nan')

def wilson(count,n):
    z=1.959963984540054;p=count/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)

def candidate_statistics(d,scores,candidate_scores,epsilon=.5,group_size=8):
    near=d<=epsilon;n=d.shape[1];hits=near.sum(1);safe=feasible(scores);pdms=scores[:,6]*100;cp=candidate_scores[:,6]*100
    quality=safe[None,:]&(pdms[None,:]>=cp[:,None]-1-1e-10)
    local_mean=np.divide(near@pdms,hits,out=np.full(len(hits),np.nan),where=hits>0)
    local_med=np.asarray([np.median(pdms[row]) if row.any() else np.nan for row in near])
    safe_local=np.divide(near@safe.astype(float),hits,out=np.full(len(hits),np.nan),where=hits>0)
    group_p=near.reshape(len(near),-1,group_size).mean(2);p=hits/n;se=group_p.std(1,ddof=1)/np.sqrt(group_p.shape[1])
    upper=np.minimum(1,p+1.96*se);upper[hits==0]=1-.05**(1/group_p.shape[1]);lower=np.maximum(0,p-1.96*se);lower[hits==n]=.05**(1/group_p.shape[1])
    return dict(hit_count_B=hits,p_B=p,p_B_safe=(near&safe).mean(1),p_B_quality=(near&quality).mean(1),local_mean_PDMS=local_mean,local_median_PDMS=local_med,local_safe_fraction=safe_local,candidate_local_gain=cp-local_mean,local_gain_primary=np.where(hits>=10,cp-local_mean,np.nan),p_B_group_cluster_ci_low=lower,p_B_group_cluster_ci_high=upper,zero_hit_iid_only_upper=np.where(hits==0,1-.05**(1/n),np.nan)),near,quality

def union_metrics(near,quality,group_size=8):
    n=near.shape[1];covered=near.any(0);good=(near&quality).any(0);marginal=[];previous=np.zeros(n,bool)
    for row in near:
        marginal.append(float((row&~previous).mean()));previous|=row
    groups=covered.reshape(-1,group_size).any(1);low,high=wilson(int(groups.sum()),len(groups))
    return dict(PoolMass=float(covered.mean()),GroupPoolHit=float(groups.mean()),GroupPoolHit_ci_low=low,GroupPoolHit_ci_high=high,QualityMatchedPoolMass=float(good.mean()),marginal_coverage=marginal)

def representatives(traj,threshold):
    if not len(traj):return 0
    d=distance(traj,traj);chosen=[]
    for i in sorted(range(len(traj)),key=lambda i:content_hash(traj[i])):
        if not chosen or np.min(d[i,chosen])>threshold:chosen.append(i)
    return len(chosen)

def source_fractions(records,indices):
    sources=['GT','GT_structured','ddv2','drivor','IL_native_C'];counts={s:0. for s in sources}
    for i in indices:
        tags=records[i]['all_source_tags']
        for s in tags:counts[s]+=1/len(tags)
    n=len(indices)
    return {f'source_{s}_fraction':c/n if n else float('nan') for s,c in counts.items()}

def pool_statistics(scene,variant,method,pool_type,pool,indices,traj,records,cs,stats,near,quality,ref):
    ids=np.asarray(indices,dtype=int);n=len(ids);s=cs[ids];p=s[:,6]*100;gain=stats['local_gain_primary'][ids];items=pool['items'] if pool_type=='operational' else [x for x in pool['items'] if x['strict_qualified']]
    base=dict(token=scene['token'],log=scene['log'],variant=variant,method=method,pool_type=pool_type,valid_unique_count=pool['valid_unique_count'],unique_parent_count=n,strict_count=pool['strict_count'],strict_16_scene_fraction=int(pool['strict_count']==16),empty_scene=int(n==0),insufficient_scene=int(n<16),candidate_PDMS=mean(p),median_PDMS=median(p),min_PDMS=float(p.min()) if n else np.nan,PDMS_gain_over_b_s=mean(p-ref),conservative_feasible_fraction=mean(feasible(s)),hard_safe_fraction=mean(~hard_failure(s)),mean_candidate_p_B=mean(stats['p_B'][ids]),median_candidate_p_B=median(stats['p_B'][ids]),P10_candidate_p_B=float(np.quantile(stats['p_B'][ids],.1)) if n else np.nan,fraction_candidate_p_B_ge_3percent=mean(stats['p_B'][ids]>=.03),mean_candidate_p_B_safe=mean(stats['p_B_safe'][ids]),mean_candidate_p_B_quality=mean(stats['p_B_quality'][ids]),candidate_local_gain=mean(gain),candidate_local_gain_median=median(gain),fraction_local_gain_ge_0_5point=mean(gain[np.isfinite(gain)]>=.5),local_gain_defined_candidate_count=int(np.isfinite(gain).sum()),local_gain_defined_scene_count=int(np.isfinite(gain).any()),local_gain_all_hits_descriptive=mean(stats['candidate_local_gain'][ids]),zero_hit_fraction=mean(stats['hit_count_B'][ids]==0),external_source_fraction=mean([bool({'ddv2','drivor'}&set(records[i]['all_source_tags'])) for i in ids]),C_overlap_fraction=mean([len(records[i]['all_source_tags'])>1 and 'IL_native_C' in records[i]['all_source_tags'] for i in ids]),near_duplicate_representatives_0_02m=representatives(traj[ids],.02),near_duplicate_representatives_0_10m=representatives(traj[ids],.1),fallback_fraction=mean([x['selection_level']>0 for x in items]),selection_optimism=mean(stats['p_A'][ids]-stats['p_B'][ids]),sum_individual_mass= float(stats['p_B'][ids].sum()),pairwise_ADE=mean(distance(traj[ids],traj[ids])[np.triu_indices(n,1)]) if n>1 else np.nan)
    base.update(union_metrics(near[ids],quality[ids]));base.update(source_fractions(records,ids))
    base['marginal_coverage']=json.dumps(base['marginal_coverage'])
    for level in range(4):base[f'level_{level}_fraction']=mean([x['selection_level']==level for x in items])
    for j,k in enumerate(FIELDS[:-1]):base[k]=mean(s[:,j])
    return base

def scene_task(scene):
    t=scene['token'];selectionpath=OUT/'cache/selection'/f'{t}.json';selection=read(selectionpath)
    frozen=read(OUT/'manifests/selection_frozen.json');assert sha(selectionpath)==frozen['selection_hashes'][t]
    # B first becomes visible after this gate. Raw/A selection is never recomputed here.
    bpath=OUT/'cache/rollouts/B'/f'{t}.npz';bscore=OUT/'cache/scores/rollouts/B'/f'{t}.npz'
    b=np.load(bpath)['trajectories'];bs=np.load(bscore)['scores'];assert len(b)==1024 and np.isfinite(bs).all()
    rawpath=OUT/'cache/raw'/f'{t}.npz';raw=np.load(rawpath)['trajectories'];records=read(rawpath.with_suffix('.json'))['candidates'];cs=np.load(OUT/'cache/scores/raw'/f'{t}.npz')['scores'];d=distance(raw,b)
    candidates=[];poolrows=[];selectedrows=[];matchrows=[];allstats={};allnear={};allquality={}
    for epsilon in [.25,.5,1.]:
        stats,near,quality=candidate_statistics(d,bs,cs,epsilon);stats['p_A']=np.asarray(selection['hits_A'][str(epsilon)])/selection['A_denominator'];allstats[epsilon]=stats;allnear[epsilon]=near;allquality[epsilon]=quality
    stats=allstats[.5]
    nearest=d.argmin(1);gt=np.asarray(scene['gt']);fde_gt=np.linalg.norm(raw[:,-1,:2]-gt[-1,:2],axis=1)
    heading_gt=np.abs(np.arctan2(np.sin(raw[:,:,2]-gt[:,2]),np.cos(raw[:,:,2]-gt[:,2]))).mean(1)
    fde_nearest=np.linalg.norm(raw[:,-1,:2]-b[nearest,-1,:2],axis=1)
    heading_nearest=np.abs(np.arctan2(np.sin(raw[:,:,2]-b[nearest,:,2]),np.cos(raw[:,:,2]-b[nearest,:,2]))).mean(1)
    for i,r in enumerate(records):
        row={k:r[k] for k in ['token','candidate_id','unique_parent_id','content_hash','source','raw_index','trajectory_path']}
        row.update(perturbation_family=json.dumps(r['perturbation_family']),checkpoint_hash=json.dumps(r['checkpoint_hash']))
        row.update(FDE_GT=float(fde_gt[i]),heading_error_GT_rad=float(heading_gt[i]),nearest_ADE_B=float(d[i,nearest[i]]),FDE_to_ADE_nearest_B=float(fde_nearest[i]),heading_error_to_ADE_nearest_B_rad=float(heading_nearest[i]))
        row.update(all_source_tags=json.dumps(r['all_source_tags']),C_overlap='IL_native_C' in r['all_source_tags'] and len(r['all_source_tags'])>1,**{k:float(v[i]) for k,v in stats.items()},d_GT=selection['d_GT'][i],hit_count_A=selection['hits_A']['0.5'][i],reference_PDMS=selection['reference_PDMS'],score_status='OK',local_count_status='OK' if stats['hit_count_B'][i]>=10 else 'LOW_COUNT',conservative_feasible=bool(feasible(cs[i:i+1])[0]),hard_safe=bool((cs[i,[0,1]]>=1-1e-8).all()),quality_floor_pass=bool(cs[i,6]*100>=selection['reference_PDMS']-1e-10))
        row.update({k:float(cs[i,j]*(100 if k=='PDMS' else 1)) for j,k in enumerate(FIELDS)});row['selection_optimism']=row['p_A']-row['p_B']
        eligible=row['hard_safe'] and row['quality_floor_pass']
        row['descriptive_type']='quality_or_safety_unqualified' if not eligible else 'low_coverage_high_quality' if row['p_B']<.03 else 'covered_low_count' if row['hit_count_B']<10 else 'covered_expert_gain' if row['candidate_local_gain']>=.5 else 'covered_nearby_quality'
        candidates.append(row)
    for v in frozen['variants']:
        for method,pool in selection['variants'][v['name']].items():
            for pt,key in [('operational','operational_indices'),('strict','strict_indices')]:
                poolrows.append(pool_statistics(scene,v['name'],method,pt,pool,pool[key],raw,records,cs,allstats[v['epsilon']],allnear[v['epsilon']],allquality[v['epsilon']],selection['reference_PDMS']))
            if v['name']=='primary':
                for slot in range(16):
                    if slot>=len(pool['items']):selectedrows.append(dict(token=t,method=method,slot=slot,valid_mask=False));continue
                    item=pool['items'][slot];row=dict(candidates[item['raw_index']],method=method,slot=slot,**item);selectedrows.append(row)
    for pt,comparisons in selection['matched_pairs'].items():
        for other,match in comparisons.items():
            for pair in match['pairs']:
                i,j=pair['pc_index'],pair['other_index'];row=dict(token=t,log=scene['log'],pool_type=pt,comparator=other,pc_candidate_id=records[i]['candidate_id'],other_candidate_id=records[j]['candidate_id'],**pair)
                for k in ['p_B','p_B_safe','p_B_quality','candidate_local_gain','hit_count_B']:
                    row['pc_'+k]=float(stats[k][i]);row['other_'+k]=float(stats[k][j]);row['delta_'+k]=float(stats[k][i]-stats[k][j])
                row['local_gain_pair_valid']=bool(stats['hit_count_B'][i]>=10 and stats['hit_count_B'][j]>=10);matchrows.append(row)
    safe=feasible(bs);pdms=bs[:,6]*100;raw_safe=feasible(cs);oracle=float(cs[raw_safe,6].max()*100) if raw_safe.any() else np.nan;hq=safe&(pdms>=oracle-1) if np.isfinite(oracle) else np.zeros(len(b),bool)
    gs=safe.reshape(-1,8);gp=pdms.reshape(-1,8);best=np.max(np.where(gs,gp,-np.inf),1);best[~gs.any(1)]=np.nan
    c=np.load(OUT/'cache/scores/rollouts/C'/f'{t}.npz')['scores'];gt_id=next(i for i,r in enumerate(records) if 'GT' in r['all_source_tags'])
    glob=dict(token=t,log=scene['log'],B_samples=len(b),real_groups=len(gs),GT_PDMS=cs[gt_id,6]*100,C_mean_PDMS=c[:,6].mean()*100,C_median_PDMS=np.median(c[:,6])*100,C_best_PDMS=c[:,6].max()*100,raw_best_known_safe_PDMS=oracle,HQ_threshold=oracle-1,Global_HQMass=float(hq.mean()) if np.isfinite(oracle) else np.nan,Global_HQGroupHit=float(hq.reshape(-1,8).any(1).mean()) if np.isfinite(oracle) else np.nan,Best_safe_at_G=mean(best),No_safe_member_group_rate=float((~gs.any(1)).mean()),near_ceiling=bool(oracle-selection['reference_PDMS']<=1) if np.isfinite(oracle) else False,B_mean_PDMS=mean(pdms),B_feasible_rate=mean(safe),B_best_known_gap=oracle-float(pdms[safe].max()) if safe.any() else np.nan)
    low,high=wilson(int(hq.reshape(-1,8).any(1).sum()),len(gs));glob.update(Global_HQGroupHit_ci_low=low if np.isfinite(oracle) else np.nan,Global_HQGroupHit_ci_high=high if np.isfinite(oracle) else np.nan,Best_safe_at_G_defined_groups=int(gs.any(1).sum()))
    dest=OUT/'cache/analysis';dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [('candidates',candidates),('pools',poolrows),('selected',selectedrows),('matches',matchrows)]:
        frame=pd.DataFrame(rows)
        if name=='matches' and not rows:
            frame=pd.DataFrame(columns=['token','log','pool_type','comparator','exact_source','PDMS_gap_points','local_gain_pair_valid']+[prefix+k for prefix in ['pc_','other_','delta_'] for k in ['p_B','p_B_safe','p_B_quality','candidate_local_gain','hit_count_B']])
        final=dest/f'{t}_{name}.parquet';temp=final.with_suffix(f'.{os.getpid()}.tmp.parquet');frame.to_parquet(temp,index=False);temp.replace(final)
    save(dest/f'{t}_global.json',{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in glob.items()})
    return t

def main():
    gate=read(OUT/'manifests/selection_frozen.json');assert gate['scene_count']==1000 and gate['protocol_hash']==protocol()
    start=time.time()
    with concurrent.futures.ProcessPoolExecutor(24) as pool:
        for j,t in enumerate(pool.map(scene_task,scenes())):
            if j%100==0:print('B validation',j+1,1000,time.time()-start,flush=True)
    dest=OUT/'metrics';dest.mkdir(parents=True,exist_ok=True)
    for name,filename in [('candidates','candidate_metrics.parquet'),('selected','selected_pools_4x1000x16.parquet'),('matches','source_quality_matched.parquet'),('pools','all_variant_pool_scene_metrics.parquet')]:
        frames=[pd.read_parquet(OUT/'cache/analysis'/f'{s["token"]}_{name}.parquet') for s in scenes()];df=pd.concat(frames,ignore_index=True);df.to_parquet(dest/filename,index=False)
        if name=='pools':df[df.variant=='primary'].to_csv(dest/'pool_scene_metrics.csv',index=False)
    pd.DataFrame([read(OUT/'cache/analysis'/f'{s["token"]}_global.json') for s in scenes()]).to_csv(dest/'global_policy_sampling_baseline.csv',index=False)
    save(OUT/'audits/B_validation.json',dict(scene_count=1000,selection_frozen_hash=sha(OUT/'manifests/selection_frozen.json'),seconds=time.time()-start,B_loaded_only_after_global_selection_freeze=True,completed_at=utc()))
if __name__=='__main__':main()
