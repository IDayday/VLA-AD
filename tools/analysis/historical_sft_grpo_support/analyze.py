"""Scene-paired analysis of actual G16 rollouts and historical supervision."""
from common_support import *
import argparse
METRIC_NAMES=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']
PARENTS={'original_grpo_11970':'official_il','a5_grpo_300':'a5_sft','a5_grpo_4842':'a5_sft','v6_grpo_300':'v6_sft','v6_grpo_3300':'v6_sft'}

def safe(x):return (np.asarray(x)[...,[0,1,3,5]]>=1.-1e-8).all(-1)
def train_reward(x):
    # Actual NAVSIM-v1 scorer: NC and DAC multiply; diagnostic DDC has
    # driving_direction_weight=0 and is NOT a multiplicative score factor.
    x=np.asarray(x);return x[...,0]*x[...,1]*(10*x[...,2]+5*x[...,3]+2*x[...,4])/17
def representatives(t,threshold=.1):
    d=distance(t,t);selected=[]
    for i in range(len(t)):
        if not selected or d[i,selected].min()>threshold:selected.append(i)
    return len(selected)
def mean_or_nan(a):return float(np.mean(a)) if len(a) else np.nan

def group_stats(t,s,hq_threshold):
    assert t.shape==(16,8,3) and s.shape==(16,7)
    pdms=s[:,6]*100;feas=safe(s);hard=(s[:,[0,1]]<1.-1e-8).any(1)
    tr=train_reward(s);z=(tr-tr.mean())/(tr.std(ddof=1)+1e-8);positive=z>0
    hq=feas & (pdms>=hq_threshold)
    spread=np.sqrt(t[:,:,:2].var(0,ddof=1).sum(-1))
    out=dict(mean_PDMS=pdms.mean(),max_PDMS=pdms.max(),min_PDMS=pdms.min(),std_PDMS=pdms.std(ddof=1),range_PDMS=np.ptp(pdms),P10_PDMS=np.quantile(pdms,.1),CVaR25_PDMS=np.sort(pdms)[:4].mean(),feasible_rate=feas.mean(),safe_count=feas.sum(),hard_failure_rate=hard.mean(),zero_score_rate=(pdms==0).mean(),all_safe_group=feas.all(),no_safe_group=not feas.any(),best_safe_PDMS=float(pdms[feas].max()) if feas.any() else np.nan,pairwise_ADE=pair_mean(t),Spread_AUC=spread.mean(),endpoint_spread=spread[-1],nonredundant_0p1m=representatives(t),HQ_mass=hq.mean(),Hit16=hq.any(),D_HQ=pair_mean(t[hq]) if hq.sum()>1 else np.nan,zero_variance_group=np.ptp(tr)<1e-8,train_reward_mean=tr.mean()*100,train_reward_min=tr.min()*100,train_reward_max=tr.max()*100,vanilla_proxy_positive_infeasible_count=int((positive&~feas).sum()),infeasible_count=int((~feas).sum()),vanilla_proxy_positive_count=positive.sum(),best_is_unsafe=not feas[np.argmax(pdms)])
    out.update({k:s[:,i].mean()*100 for i,k in enumerate(METRIC_NAMES[:-1])})
    return out

def bootstrap_difference(diff,logs,seed_suffix):
    d=np.asarray(diff,dtype=float);logs=np.asarray(logs);mask=np.isfinite(d);d=d[mask];logs=logs[mask]
    if not len(d):return dict(n=0)
    rng=np.random.default_rng(seed('bootstrap',0,seed_suffix));b=CFG['bootstrap_replicates'];samples=[]
    for i in range(0,b,100):samples.extend(d[rng.integers(len(d),size=(min(100,b-i),len(d)))].mean(1))
    unique,inv=np.unique(logs,return_inverse=True);sumlog=np.bincount(inv,weights=d);count=np.bincount(inv)
    cl=[]
    for i in range(0,b,100):
        indices=rng.integers(len(unique),size=(min(100,b-i),len(unique)));cl.extend(sumlog[indices].sum(1)/count[indices].sum(1))
    return dict(n=len(d),mean_difference=d.mean(),median_difference=np.median(d),ci_low=np.quantile(samples,.025),ci_high=np.quantile(samples,.975),log_cluster_ci_low=np.quantile(cl,.025),log_cluster_ci_high=np.quantile(cl,.975),win_fraction=(d>1e-10).mean(),tie_fraction=(abs(d)<=1e-10).mean(),log_count=len(unique))

def analyze():
    identity();membership=pd.read_csv(OUT/'metrics/training_membership.csv').set_index('token')
    groups=[];scene_rows=[];teach_rows=[];factorial=[];cross=[]
    for si,scene in enumerate(scenes()):
        token=scene['token'];gt=np.asarray(scene['gt']);common=bool(membership.loc[token,'common_train'])
        b=np.load(OUT/'cache/teachers'/f'{token}.npz');teachers=b['trajectories'];tm=json.loads(str(b['metadata']))['rows']
        ts=np.load(OUT/'cache/scores/teachers'/f'{token}.npz')['trajectories'];assert len(ts)==len(tm)
        banks={};scores={}
        for m in models():
            bp=OUT/'cache/rollouts'/m/f'{token}.npz';sp=OUT/'cache/scores/rollouts'/m/f'{token}.npz'
            banks[m]={k:v for k,v in np.load(bp).items() if k!='metadata'};scores[m]={k:v for k,v in np.load(sp).items() if k!='metadata'}
        # A shared scene-specific standard PDMS reference; impossible >100
        # thresholds remain in the denominator. Never select the best model.
        hq_threshold=scores['official_il']['eval'][...,6].mean()*100+CFG['quality_improvement_points']
        for m in models():
            for protocol,t in banks[m].items():
                s=scores[m][protocol];rs=[]
                for g in range(4):
                    r=dict(token=token,log=scene['log'],command=scene['command'],common_train=common,model=m,protocol=protocol,group=g,**group_stats(t[g],s[g],hq_threshold));groups.append(r);rs.append(r)
                keys=[k for k in rs[0] if k not in ['token','log','command','common_train','model','protocol','group']]
                row=dict(token=token,log=scene['log'],command=scene['command'],common_train=common,model=m,protocol=protocol,**{k:mean_or_nan([r[k] for r in rs if np.isfinite(r[k])]) for k in keys})
                flat=t.reshape(64,8,3);iloc=banks['official_il']['eval'].reshape(64,8,3)
                row['center_shift_IL_eval']=distance(flat.mean(0)[None],iloc.mean(0)[None])[0,0]
                own_eval=banks[m]['eval'].reshape(64,8,3)
                row['center_shift_own_eval']=distance(flat.mean(0)[None],own_eval.mean(0)[None])[0,0]
                row['CRN_shift_own_eval']=np.linalg.norm(flat[:,:,:2]-own_eval[:,:,:2],axis=-1).mean()
                row['GT_center_distance']=distance(flat.mean(0)[None],gt[None])[0,0]
                row['HQ_threshold']=hq_threshold;row['HQ_threshold_above_100']=hq_threshold>100
                if m in PARENTS:
                    parent=banks[PARENTS[m]][protocol].reshape(64,8,3)
                    row['center_shift_SFT_parent']=distance(flat.mean(0)[None],parent.mean(0)[None])[0,0]
                    row['CRN_shift_SFT_parent']=np.linalg.norm(flat[:,:,:2]-parent[:,:,:2],axis=-1).mean()
                scene_rows.append(row)
                if m in CFG['primary_models'] and protocol in CFG['protocols']:
                    d=distance(teachers,flat);fitting=np.load(OUT/'cache/fitting'/m/f'{token}.npz')['epsilon_mse'].mean(0)
                    for j,meta in enumerate(tm):
                        r=dict(**meta,policy=m,protocol=protocol,teacher_PDMS=ts[j,6]*100,teacher_feasible=bool(safe(ts[j])),teacher_GT_ADE=float(distance(teachers[j:j+1],gt[None])[0,0]),nearest_rollout_ADE=float(d[j].min()),mean_rollout_ADE=float(d[j].mean()),epsilon_mse=float(fitting[j]))
                        for threshold in CFG['teacher_ADE_m']:
                            suffix=str(threshold).replace('.','p');hit=(d[j]<=threshold).reshape(4,16)
                            r[f'hit16_{suffix}']=hit.any(1).mean();r[f'hit64_{suffix}']=hit.any();r[f'mass64_{suffix}']=hit.mean()
                        teach_rows.append(r)
            if m in CFG['primary_models'] and 'floor_only' in banks[m]:
                native_t=banks[m]['native_grpo'];both=banks[m]['floor_and_clip']
                factorial.append(dict(token=token,model=m,native_vs_floor_and_clip_CRN_ADE=np.linalg.norm(native_t[...,:2]-both[...,:2],axis=-1).mean(),native_vs_floor_and_clip_max_abs=float(abs(native_t-both).max())))
        if si%100==0:print('ANALYZE',si+1,flush=True)
    gf=pd.DataFrame(groups);sf=pd.DataFrame(scene_rows);tf=pd.DataFrame(teach_rows)
    gf.to_parquet(OUT/'metrics/group16_metrics.parquet',index=False);sf.to_csv(OUT/'metrics/scene_metrics.csv',index=False)
    tf.to_parquet(OUT/'metrics/teacher_coverage.parquet',index=False);csv('factorial_parity.csv',factorial)
    summaries=[]
    for scope,subset in [('FULL1000',sf),('COMMON_TRAIN835',sf[sf.common_train])]:
        for (m,p),g in subset.groupby(['model','protocol']):
            r=dict(scope=scope,model=m,protocol=p,scenes=len(g),groups=len(g)*4)
            for k in g.select_dtypes(include=np.number):r[k]=g[k].mean()
            r['vanilla_proxy_PIA_rate']=g.vanilla_proxy_positive_infeasible_count.sum()/g.infeasible_count.sum() if g.infeasible_count.sum() else np.nan
            summaries.append(r)
    csv('group16_summary.csv',summaries)
    teacher_scene=[]
    for (token,policy,protocol,origin),g in tf.groupby(['token','policy','protocol','origin']):
        for kind,h in [('ALL',g[g.weight>0]),('NON_GT',g[(~g.is_gt)&(g.weight>0)]),('GT',g[g.is_gt & (g.weight>0)])]:
            if not len(h):continue
            weight=h.weight.to_numpy();weight=weight/weight.sum()
            r=dict(token=token,policy=policy,protocol=protocol,origin=origin,kind=kind,common_train=bool(h.common_train.iloc[0]),actual_training_target=bool(h.actual_training_target.iloc[0]),teacher_count=len(h),log=membership.loc[token,'log'])
            for col in ['nearest_rollout_ADE','mean_rollout_ADE','epsilon_mse','teacher_PDMS','teacher_GT_ADE']+[c for c in h if c.startswith(('hit16_','hit64_','mass64_'))]:
                r[col]=float(h[col].mean());r['weighted_'+col]=float(h[col].to_numpy()@weight)
            teacher_scene.append(r)
    tsf=pd.DataFrame(teacher_scene);csv('teacher_scene_metrics.csv',teacher_scene)
    tsummary=[]
    for scope,subset in [('FULL1000',tsf),('COMMON_TRAIN835',tsf[tsf.common_train]),('PSI_HOLDOUT165',tsf[~tsf.common_train])]:
        for (p,pr,o,k),g in subset.groupby(['policy','protocol','origin','kind']):
            r=dict(scope=scope,policy=p,protocol=pr,origin=o,kind=k,scenes=len(g),teachers=int(g.teacher_count.sum()))
            r.update({c:g[c].mean() for c in g.select_dtypes(include=np.number) if c!='teacher_count'});tsummary.append(r)
    csv('teacher_summary.csv',tsummary)
    comparisons=[]
    def compare(frame,a,b,key,metric,label):
        left=frame.query(a).set_index(key);right=frame.query(b).set_index(key)
        ix=left.index.intersection(right.index);assert left.index.is_unique and right.index.is_unique
        diff=left.loc[ix,metric].to_numpy()-right.loc[ix,metric].to_numpy();logs=left.loc[ix,'log'].to_numpy()
        comparisons.append(dict(comparison=label,metric=metric,**bootstrap_difference(diff,logs,label+metric)))
    main_metrics=['mean_PDMS','max_PDMS','min_PDMS','range_PDMS','feasible_rate','pairwise_ADE','Hit16','safe_count','hard_failure_rate','center_shift_IL_eval']
    for m in models():
        for metric in main_metrics:compare(sf,f"model == '{m}' and protocol == 'native_grpo'",f"model == '{m}' and protocol == 'eval'",'token',metric,f'{m}:native-minus-eval')
    factorial_tokens=set(read(OUT/'manifests/scenes.json')['factorial_tokens'])
    fac=sf[sf.token.isin(factorial_tokens)]
    for m in CFG['primary_models']:
        for pr in CFG['factorial_protocols']:
            for metric in ['mean_PDMS','pairwise_ADE','feasible_rate']:
                compare(fac,f"model == '{m}' and protocol == '{pr}'",f"model == '{m}' and protocol == 'eval'",'token',metric,f'factorial:{m}:{pr}-minus-eval')
    for m in [x for x in models() if x!='official_il']:
        parent=PARENTS.get(m,'official_il')
        for pr in CFG['protocols']:
            for metric in main_metrics:compare(sf,f"model == '{m}' and protocol == '{pr}'",f"model == '{parent}' and protocol == '{pr}'",'token',metric,f'{m}-minus-{parent}:{pr}')
    for m in CFG['primary_models']:
        for origin in CFG['primary_models']:
            for kind in ['ALL','NON_GT','GT']:
                subset=tsf[(tsf.origin==origin)&(tsf.kind==kind)&tsf.common_train]
                for metric in ['weighted_hit16_0p5','weighted_hit64_0p5','weighted_mass64_0p5','epsilon_mse','weighted_epsilon_mse']:
                    compare(subset,f"policy == '{m}' and protocol == 'native_grpo'",f"policy == '{m}' and protocol == 'eval'",'token',metric,f'teacher:{m}:{origin}:{kind}:native-minus-eval')
                    if m!='official_il':compare(subset,f"policy == '{m}' and protocol == 'eval'","policy == 'official_il' and protocol == 'eval'",'token',metric,f'teacher:{m}-minus-official:{origin}:{kind}:eval')
                    if m in ['a5_sft','v6_sft']:compare(subset,f"policy == '{m}' and protocol == 'eval'","policy == 'psi_sft' and protocol == 'eval'",'token',metric,f'teacher:{m}-minus-psi:{origin}:{kind}:eval')
    csv('paired_comparisons.csv',comparisons)
    save(OUT/'audits/analysis_complete.json',dict(protocol_hash=identity(),scene_count=len(scenes()),group_rows=len(gf),teacher_policy_rows=len(tf),models=list(models()),bootstrap_replicates=3000,primary_teacher_scope='common835 train',no_new_training=True))
    print(pd.DataFrame(summaries).query("scope == 'FULL1000' and protocol in ['eval','native_grpo']")[[ 'model','protocol','mean_PDMS','max_PDMS','min_PDMS','feasible_rate','pairwise_ADE','Hit16']].to_string(index=False),flush=True)

if __name__=='__main__':analyze()
