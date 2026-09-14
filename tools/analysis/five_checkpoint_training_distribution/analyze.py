"""Paired 1000-scene distribution, actual-supervision, and geometric analyses."""
from __future__ import annotations
from common_fd import *
from analyze_policy_distribution import geometry

def paired_stats(values, logs, seed_key):
    values=np.asarray(values,dtype=float);logs=np.asarray(logs);good=np.isfinite(values)
    values=values[good];logs=logs[good];n=len(values)
    if not n:return dict(n=0,mean=np.nan,median=np.nan,ci_low=np.nan,ci_high=np.nan,win_fraction=np.nan,log_ci_low=np.nan,log_ci_high=np.nan)
    rng=np.random.default_rng(int(digest([CFG['seed'],seed_key])[:16],16));rep=CFG['bootstrap_replicates']
    draws=rng.integers(n,size=(rep,n));boot=values[draws].mean(1)
    groups=np.unique(logs);sums=np.array([values[logs==g].sum() for g in groups]);counts=np.array([np.sum(logs==g) for g in groups])
    draws=rng.integers(len(groups),size=(rep,len(groups)));lb=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(n=n,mean=values.mean(),median=np.median(values),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),
        win_fraction=np.mean(values>1e-10),tie_fraction=np.mean(np.abs(values)<=1e-10),log_count=len(groups),log_ci_low=np.quantile(lb,.025),log_ci_high=np.quantile(lb,.975))

def chain_analysis():
    root=ROOT/'outputs/pc_mts_diagnostics_v2'
    manifest=read(root/'manifests/historical_chains.json');chain=next(c for c in manifest['chains'] if c['name']=='official_il_grpo')
    original=next(Path(chain['run']).glob('**/epoch=8-step=11970.ckpt'))
    original_hash=sha(original);assert original_hash==MODEL_INFO['grpo_9041']['sha256']
    save(OUT/'audits/grpo_9041_lineage.json',dict(original_training_checkpoint=str(original),sha256=original_hash,
        requested_checkpoint=MODEL_INFO['grpo_9041']['checkpoint_path'],same_file_content=True,
        initial_policy=chain['initial_checkpoint'],reference_policy=chain['reference_checkpoint'],
        config_path=chain['config_path'],config_sha256=sha(chain['config_path'])))
    tokens=set(read(root/'manifests/chain_scenes.json')['tokens']);rows=[];identity=[]
    for scene in SCENES:
        token=scene['token']
        if token not in tokens:continue
        il,ils,meta=load_model_scene('official_il',token);threshold=ils[:,6].mean()*100+1
        for snap in chain['snapshots']+[dict(step=11970,name='grpo_9041',reuse_V1='grpo_9041')]:
            if snap.get('reuse_V1'):
                t,s,_=load_model_scene(snap['reuse_V1'],token);t=t[:32];s=s[:32]
                source=v1.OUT/'rollouts'/snap['reuse_V1']/f'{token}.npz'
            else:
                source=root/'chains/rollouts'/snap['name']/f'{token}.npz';scorepath=root/'chains/scores'/snap['name']/f'{token}.npz'
                with np.load(source) as z:t=z['trajectories'].astype(float)
                with np.load(scorepath) as z:
                    s=z['trajectories'];sm=json.loads(str(z['metadata']));assert sm['input_sha256']==sha(source)
            assert t.shape==(32,8,3)
            g,_=geometry(t)
            rows.append(dict(token=token,log=scene['log'],step=snap['step'],**score_metrics(s,threshold),
                pairwise_ADE=g['pairwise_ade'],Spread_AUC=g['spread_auc'],mean_center_shift=distance(center(t)[None],center(il[:32])[None])[0,0]))
            identity.append(dict(token=token,step=snap['step'],path=str(source),sha256=sha(source)))
    csv('historical_grpo_scene.csv',rows)
    f=pd.DataFrame(rows);csv('historical_grpo_summary.csv',f.groupby('step').mean(numeric_only=True).reset_index())
    base=f[f.step==0].set_index('token');comparisons=[]
    for step in sorted(set(f.step)-{0}):
        now=f[f.step==step].set_index('token').loc[base.index]
        for key in ['PDMS','feasible','EP','TTC','pairwise_ADE','Spread_AUC','Hit8']:
            comparisons.append(dict(step=step,metric=key,**paired_stats(now[key]-base[key],base['log'],['chain',step,key])))
    csv('historical_grpo_paired.csv',comparisons)
    save(OUT/'manifests/historical_grpo_inputs.json',dict(chain=chain,inputs=identity,CRN='V2 historical_chains.py uses exact first32 V1 streams; inserted11970 uses first32 of V1 GRPO64',
        caution='Historical training trajectory observational over optimizer time; 11970 is requested90.41,13300 is retained later epoch9, not selected as best'))

def main():
    config_identity();teacher=pd.read_csv(OUT/'metrics/teacher_candidates.csv');ts=pd.read_csv(OUT/'metrics/teacher_scene.csv')
    teacher_groups={(m,t):g for (m,t),g in teacher.groupby(['model','token'])}
    rows=[];target_rows=[];teacher_scores=[];time_rows=[];hybrid=[];reliability=[];audit=[];examples={}
    exampleset=set(read(OUT/'manifests/protocol_frozen.json')['examples'])
    for si,scene in enumerate(SCENES):
        token=scene['token'];bank={};scores={};meta={}
        for m in MODELS:bank[m],scores[m],meta[m]=load_model_scene(m,token)
        seeds=[meta[m]['initial_noise_seeds'] for m in MODELS]
        assert all(s==seeds[0] for s in seeds)
        il=bank['official_il'];ils=scores['official_il'];ilg,ilmed=geometry(il)
        threshold=ils[:,6].mean()*100+1
        oldref=float(np.load(v1.OUT/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])*100
        gt=np.asarray(scene['gt']);gts,gt_identity=exact_gt_score(scene);heading_total=float(np.abs(np.diff(np.unwrap(np.r_[0.,gt[:,2]]))).sum())
        direction=np.stack([np.cos(gt[:,2]),np.sin(gt[:,2])],axis=-1)
        normal=np.stack([-direction[:,1],direction[:,0]],axis=-1)
        for m in MODELS:
            t,s=bank[m],scores[m];g,med=geometry(t);res=residual(t)[...,:2]
            hq=v1.feasible(s)&(s[:,6]*100>=threshold);positive=v1.feasible(s)&(s[:,6]*100>oldref)
            shift=center(t)[:,:2]-center(il)[:,:2]
            base=dict(token=token,log=scene['log'],command=scene['command'],model=m,GT_heading_change=heading_total)
            row=dict(base,**score_metrics(s,threshold),pairwise_ADE=g['pairwise_ade'],Spread_AUC=g['spread_auc'],
                spread_ratio=g['spread_auc']/ilg['spread_auc'],pairwise_ratio=g['pairwise_ade']/ilg['pairwise_ade'],
                R90=g['r90'],effective_rank=g['effective_rank'],endpoint_spread=g['endpoint_spread'],
                mean_center_shift=distance(center(t)[None],center(il)[None])[0,0],medoid_center_shift=distance(med[None],ilmed[None])[0,0],
                mean_GT_distance=distance(t,gt[None]).mean(),D_high_quality=pair_mean(t[hq]),HQ_count=int(hq.sum()),
                D_positive_legacy=pair_mean(t[positive]),positive_count_legacy=int(positive.sum()),
                Hit8_legacy=(v1.feasible(s)&(s[:,6]*100>=oldref+1)).reshape(8,8).any(1).mean(),
                mean_progress_shift=(shift*direction).sum(-1).mean(),mean_lateral_shift=(shift*normal).sum(-1).mean(),
                longitudinal_spread=np.std((res*direction).sum(-1),axis=0,ddof=1).mean(),
                lateral_spread=np.std((res*normal).sum(-1),axis=0,ddof=1).mean(),
                threshold_PDMS=threshold,threshold_above_100=threshold>100,
                CRN_score_win=(s[:,6]>ils[:,6]+1e-10).mean(),
                CRN_safe_repair=(~v1.feasible(ils)&v1.feasible(s)).mean(),
                CRN_safe_loss=(v1.feasible(ils)&~v1.feasible(s)).mean())
            rows.append(row)
            for ti in range(8):time_rows.append(dict(base,time_s=(ti+1)*.5,spread=g[f'spread_t{ti+1}'],
                longitudinal_std=np.std((res[:,ti]*direction[ti]).sum(-1),ddof=1),
                lateral_std=np.std((res[:,ti]*normal[ti]).sum(-1),ddof=1),
                center_ADE_at_time=np.linalg.norm(shift[ti])))
            for k in [16,32,64]:
                gs,_=geometry(t[:k]);reliability.append(dict(token=token,model=m,K=k,pairwise_ADE=gs['pairwise_ade'],Spread_AUC=gs['spread_auc']))
        with np.load(OUT/'cache/scored'/f'{token}.npz') as z:
            zm=json.loads(str(z['metadata']));assert zm['protocol_sha256']==config_identity()
            for a,b in CFG['counterfactual']['contrasts']:
                key=a+'__'+b
                q=[score_metrics(scores[a],threshold),score_metrics(z[key+'__new_center_old_residual'],threshold),
                   score_metrics(z[key+'__old_center_new_residual'],threshold),score_metrics(scores[b],threshold)]
                for metric in ['PDMS','feasible','hard_failure','EP','NC','DAC','TTC','DDC','HQ_mass','Hit8']:
                    q00,q10,q01,q11=[x[metric] for x in q];ce,re=shapley(q00,q10,q01,q11)
                    assert abs(ce+re-(q11-q00))<1e-8
                    hybrid.append(dict(token=token,log=scene['log'],contrast=key,metric=metric,old=q00,new_center_old_residual=q10,
                        old_center_new_residual=q01,new=q11,total_gain=q11-q00,center_contribution=ce,residual_shape_contribution=re))
            s=z['official_il__grpo_width_only'];ms=score_metrics(s,threshold)
            hybrid.append(dict(token=token,log=scene['log'],contrast='official_il__width_only',metric='PDMS',old=ils[:,6].mean()*100,
                new=ms['PDMS'],total_gain=ms['PDMS']-ils[:,6].mean()*100))
            for m in ARCHIVES:
                p,r,idx,t=teacher_record(m,token);g=teacher_groups[m,token];w=g['expected_weight_pre_budget'].values
                assert np.array_equal(idx,g.raw_index.values)
                sc=z['teacher__'+m];ds=distance(t,gt[None])[:,0];non_gt=g.source_code.values!=1
                for j,i in enumerate(idx):teacher_scores.append(dict(model=m,token=token,raw_index=int(i),PDMS=sc[j,6]*100,feasible=bool(v1.feasible(sc[j])),GT_distance=ds[j]))
                mean_t=np.einsum('n,ntd->td',w,t[:,:,:2]);aim=mean_t-center(il)[:,:2];movement=center(bank[m])[:,:2]-center(il)[:,:2]
                norm=np.linalg.norm(aim)*np.linalg.norm(movement)
                target=dict(model=m,token=token,log=scene['log'],teacher_PDMS_weighted=np.dot(w,sc[:,6]*100),
                    teacher_feasible_weighted=np.dot(w,v1.feasible(sc)),teacher_best_PDMS=sc[:,6].max()*100,
                    teacher_GT_PDMS=gts[6]*100,GT_in_selected_support=bool((g.source_code.values==1).any()),
                    teacher_best_gain_over_GT=(sc[:,6].max()-gts[6])*100,
                    teacher_nonGT_PDMS=np.dot(w[non_gt],sc[non_gt,6]*100)/w[non_gt].sum() if w[non_gt].sum() else np.nan,
                    teacher_weighted_GT_distance=np.dot(w,ds),shift_teacher_cosine=float(np.sum(aim*movement)/norm) if norm>1e-10 else np.nan,
                    output_to_weighted_teacher_center=np.linalg.norm(center(bank[m])[:,:2]-mean_t,axis=-1).mean(),
                    il_to_weighted_teacher_center=np.linalg.norm(center(il)[:,:2]-mean_t,axis=-1).mean())
                for model in ['official_il',m]:
                    d=distance(t,bank[model]);mind=d.min(1)
                    target[model+'_teacher_nearest_ADE_weighted']=np.dot(w,mind)
                    for eps in CFG['teacher_coverage_ADE_m']:
                        covered=mind<=eps
                        target[f'{model}_teacher_mass_covered_{eps}']=np.dot(w,covered)
                        target[f'{model}_nonGT_mass_covered_{eps}']=np.dot(w[non_gt],covered[non_gt])/w[non_gt].sum() if w[non_gt].sum() else np.nan
                    # Finite64 sample geometric coverage, not model likelihood or semantic mode coverage.
                target_rows.append(target)
        audit.append(dict(token=token,scene_cache_exists=Path(scene['metric_cache_path']).exists(),
            metric_cache_sha256=zm['metric_cache_sha256'],rollout_hashes=zm['input_hashes'],exact_gt_score_identity=gt_identity,seeds_identical_across_models=True))
        if token in exampleset:
            examples[token]=dict(command=scene['command'],gt=scene['gt'],trajectories={m:bank[m].tolist() for m in MODELS},
                PDMS={m:scores[m][:,6].mean()*100 for m in MODELS},
                teachers={m:dict(trajectory=teacher_record(m,token)[3].tolist(),weights=teacher_groups[m,token]['expected_weight_pre_budget'].tolist()) for m in ARCHIVES})
        if si%100==0:print('scene analysis',si+1,flush=True)
    f=csv('policy_scene.csv',rows);csv('policy_time_scene.csv',time_rows);csv('counterfactual_scene.csv',hybrid)
    td=csv('teacher_output_scene.csv',pd.DataFrame(target_rows).merge(ts,on=['model','token','log'],validate='one_to_one'))
    csv('teacher_candidates_scored.csv',teacher.merge(pd.DataFrame(teacher_scores),on=['model','token','raw_index'],validate='one_to_one'))
    csv('sample_count_reliability.csv',reliability)
    summary=f.groupby('model').mean(numeric_only=True).reindex(MODELS)
    summary['median_spread_ratio']=f.groupby('model').spread_ratio.median();summary['compressed_scene_fraction']=f.groupby('model').spread_ratio.apply(lambda x:(x<1).mean())
    summary['HQ_diversity_defined_scenes']=f.groupby('model').D_high_quality.count()
    summary['D_positive_legacy_defined_scenes']=f.groupby('model').D_positive_legacy.count()
    summary['ratio_of_mean_pairwise_ADE']=summary.pairwise_ADE/summary.loc['official_il','pairwise_ADE']
    csv('policy_summary.csv',summary.reset_index());csv('teacher_output_summary.csv',td.groupby('model').mean(numeric_only=True).reset_index())
    base=f[f.model=='official_il'].set_index('token');comparisons=[]
    metrics=['PDMS','feasible','hard_failure','EP','NC','DAC','TTC','DDC','Comfort','pairwise_ADE','Spread_AUC','HQ_mass','Hit8','D_high_quality','D_positive_legacy','oracle_gap','CVaR20','mean_GT_distance','lateral_spread','longitudinal_spread']
    for a,b in CFG['counterfactual']['contrasts']:
        left=f[f.model==a].set_index('token');right=f[f.model==b].set_index('token').loc[left.index]
        for key in metrics:comparisons.append(dict(contrast=a+'__'+b,metric=key,**paired_stats(right[key]-left[key],left.log,[a,b,key])))
    csv('paired_comparisons.csv',comparisons)
    strata=base[['log','command','Spread_AUC','GT_heading_change']].copy()
    strata['IL_spread_quartile']=pd.qcut(strata.Spread_AUC.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
    strata['heading_tertile']=pd.qcut(strata.GT_heading_change.rank(method='first'),3,labels=['low','middle','high'])
    f=f.merge(strata[['IL_spread_quartile','heading_tertile']],left_on='token',right_index=True,validate='many_to_one')
    parts=[]
    for key in ['IL_spread_quartile','command','heading_tertile']:
        for (value,m),g in f.groupby([key,'model'],observed=True):
            row=dict(stratification=key,stratum=str(value),model=m,n=len(g),**g.select_dtypes(include=[np.number]).mean().to_dict())
            row['median_spread_ratio']=g.spread_ratio.median();row['compressed_scene_fraction']=(g.spread_ratio<1).mean();parts.append(row)
    csv('policy_stratification.csv',parts)
    teacher_strata=[]
    for m in ARCHIVES:
        g=td[td.model==m].merge(f[f.model==m][['token','spread_ratio','Hit8','PDMS']],on='token')
        g['teacher_count_group']=np.select([g.non_gt_count==0,g.non_gt_count<=2,g.non_gt_count<=5],['GT_only','1-2_nonGT','3-5_nonGT'],default='6plus_nonGT')
        for k,h in g.groupby('teacher_count_group'):
            teacher_strata.append(dict(model=m,group=k,n=len(h),median_spread_ratio=h.spread_ratio.median(),**h.select_dtypes(include=[np.number]).mean().to_dict()))
    csv('teacher_availability_stratification.csv',teacher_strata)
    effects=[];h=pd.DataFrame(hybrid)
    for (contrast,metric),g in h.groupby(['contrast','metric']):
        for effect in ['total_gain','center_contribution','residual_shape_contribution']:
            if g[effect].notna().any():effects.append(dict(contrast=contrast,metric=metric,effect=effect,**paired_stats(g[effect],g.log,[contrast,metric,effect])))
    csv('counterfactual_summary.csv',effects)
    te=[]
    for m in ARCHIVES:
        g=td[td.model==m]
        for eps in CFG['teacher_coverage_ADE_m']:
            for kind in ['teacher_mass','nonGT_mass']:
                d=g[f'{m}_{kind}_covered_{eps}']-g[f'official_il_{kind}_covered_{eps}']
                te.append(dict(model=m,metric=kind+'_coverage_gain',epsilon=eps,**paired_stats(d,g.log,[m,kind,eps])))
    csv('teacher_coverage_paired.csv',te)
    save(OUT/'manifests/analysis_inputs.json',audit);save(OUT/'cache/scene_examples.json',examples)
    chain_analysis()
    save(OUT/'audits/analysis_completion.json',dict(scenes=len(SCENES),models=len(MODELS),rollouts=320000,policy_rows=len(rows),
        teacher_records=len(target_rows),new_scored_scenes=len(audit),missing=0,optimizer_updates=0,
        engineering_fixes=[dict(issue='Initial parser used log_name but frozen scene manifest uses log',action='Corrected key and reran teacher extraction; no data or protocol change'),
            dict(issue='Some actual A5 selected supports do not contain GT; selected-GT averaging produced NA',action='Reference GT score now reuses verified exact-GT NAVSIM cache for every scene; no GT added to actual support; missing-GT flag retained')]))
    print(summary[['PDMS','feasible','pairwise_ADE','Hit8','HQ_mass','mean_center_shift','median_spread_ratio']].to_string(),flush=True)

if __name__=='__main__':main()
