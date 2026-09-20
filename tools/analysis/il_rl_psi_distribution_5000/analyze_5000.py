"""Scene-equal statistics. Independent new4000 replication is always reported."""
from common_5000 import *
import analyze as historical
import concurrent.futures

PRIMARY=['pairwise_ADE','pairwise_ADE64','centroid_displacement','feasible_rate','mean_PDMS','min_PDMS','max_PDMS','CVaR25_PDMS','Spread_AUC','hard_failure_rate','zero_score_rate','std_PDMS','no_safe_group','best_safe_PDMS','Hit16','GT_center_distance']

def one(scene):
    token=scene['token'];bank={};scores={}
    for model in CFG['primary_models']:
        with np.load(bank_path(model,token)) as z:bank[model]={p:z[p] for p in CFG['protocols']}
        with np.load(bank_path(model,token,True)) as z:scores[model]={p:z[p] for p in CFG['protocols']}
    threshold=scores['official_il']['eval'][...,6].mean()*100+1
    group_rows=[];scene_rows=[]
    for model in CFG['primary_models']:
        for protocol in CFG['protocols']:
            t=bank[model][protocol];s=scores[model][protocol]
            assert t.shape==(4,16,8,3) and s.shape==(4,16,7)
            assert np.isfinite(t).all() and np.isfinite(s).all() and ((s>=-1e-8)&(s<=1+1e-8)).all()
            key=dict(token=token,log=scene['log'],command=scene['command'],cohort=scene['cohort'],model=model,protocol=protocol)
            rows=[dict(**key,group=g,**historical.group_stats(t[g],s[g],threshold)) for g in range(4)]
            group_rows+=rows
            metric_keys=[k for k in rows[0] if k not in key and k!='group']
            row=dict(**key,**{k:historical.mean_or_nan([r[k] for r in rows if np.isfinite(r[k])]) for k in metric_keys})
            flat=t.reshape(64,8,3);ref=bank['official_il'][protocol].reshape(64,8,3)
            row.update(centroid_displacement=float(distance(flat.mean(0)[None],ref.mean(0)[None])[0,0]),pairwise_ADE64=pair_mean(flat),CRN_policy_shift=float(np.linalg.norm(flat[...,:2]-ref[...,:2],axis=-1).mean()),GT_center_distance=float(distance(flat.mean(0)[None],np.asarray(scene['gt'])[None])[0,0]),HQ_threshold=threshold)
            for g in [0,1]:
                # Two 32-sample halves: diagnostic sampling stability, never selection.
                sl=slice(g*32,(g+1)*32)
                row[f'half{g}_centroid_displacement']=float(distance(flat[sl].mean(0)[None],ref[sl].mean(0)[None])[0,0])
                row[f'half{g}_pairwise_ADE']=pair_mean(flat[sl])
            scene_rows.append(row)
    return scene_rows,group_rows

def main():
    identity();assert (OUT/'audits/gpu_sampling_complete.json').exists()
    rows=scenes();assert len(rows)==5000
    missing=[]
    for row in rows:
        for m in CFG['primary_models']:
            for score in [False,True]:
                p=bank_path(m,row['token'],score)
                if not p.exists():missing.append(dict(token=row['token'],model=m,score=score,path=str(p)))
    save(OUT/'audits/missing.json',dict(missing=missing));assert not missing,missing[:3]
    sf=[];gf=[]
    with concurrent.futures.ProcessPoolExecutor(16) as pool:
        for i,(s,g) in enumerate(pool.map(one,rows,chunksize=8)):
            sf.extend(s);gf.extend(g)
            if i%200==0:print('ANALYZE',i+1,flush=True)
    sf=pd.DataFrame(sf);gf=pd.DataFrame(gf);(OUT/'metrics').mkdir(exist_ok=True)
    sf.to_parquet(OUT/'metrics/scene_metrics.parquet',index=False)
    gf.to_parquet(OUT/'metrics/group16_metrics.parquet',index=False)
    summarize(sf)

def summarize(sf):
    summaries=[];comparisons=[];strata=[];sampling=[]
    for scope,subset in [('FULL5000',sf),('NEW4000',sf[sf.cohort=='NEW4000']),('OLD1000',sf[sf.cohort=='OLD1000'])]:
        for (model,protocol),g in subset.groupby(['model','protocol']):
            r=dict(scope=scope,model=model,protocol=protocol,scenes=len(g),groups=len(g)*4,rollouts=len(g)*64)
            for metric in PRIMARY:
                r[metric]=g[metric].mean();r[metric+'_median']=g[metric].median();r[metric+'_defined_scenes']=int(g[metric].notna().sum())
            summaries.append(r)
        for protocol in CFG['protocols']:
            baseline=subset[(subset.model=='official_il')&(subset.protocol==protocol)].set_index('token')
            for model in CFG['primary_models'][1:]:
                current=subset[(subset.model==model)&(subset.protocol==protocol)].set_index('token').loc[baseline.index]
                for metric in PRIMARY:
                    comparisons.append(dict(scope=scope,protocol=protocol,model=model,reference='official_il',metric=metric,**historical.bootstrap_difference(current[metric]-baseline[metric],baseline.log,f'{scope}/{protocol}/{model}/{metric}')))
            psi=subset[(subset.model=='psi_sft')&(subset.protocol==protocol)].set_index('token').loc[baseline.index]
            rl=subset[(subset.model=='original_grpo_11970')&(subset.protocol==protocol)].set_index('token').loc[baseline.index]
            for metric in PRIMARY:
                comparisons.append(dict(scope=scope,protocol=protocol,model='psi_sft',reference='original_grpo_11970',metric=metric,**historical.bootstrap_difference(psi[metric]-rl[metric],baseline.log,f'{scope}/{protocol}/PSIvsRL/{metric}')))
        for model in CFG['primary_models']:
            ev=subset[(subset.model==model)&(subset.protocol=='eval')].set_index('token')
            ng=subset[(subset.model==model)&(subset.protocol=='native_grpo')].set_index('token').loc[ev.index]
            for metric in PRIMARY:
                sampling.append(dict(scope=scope,model=model,metric=metric,**historical.bootstrap_difference(ng[metric]-ev[metric],ev.log,f'{scope}/{model}/native-minus-eval/{metric}')))
        for (command,model,protocol),g in subset.groupby(['command','model','protocol']):
            strata.append(dict(scope=scope,command=command,model=model,protocol=protocol,scenes=len(g),**{k:g[k].mean() for k in PRIMARY}))
    csv('summary.csv',summaries);csv('paired_comparisons.csv',comparisons);csv('command_strata.csv',strata);csv('sampling_protocol_differences.csv',sampling)
    print(pd.DataFrame(summaries).query('scope=="FULL5000"')[['model','protocol','pairwise_ADE','centroid_displacement','feasible_rate','mean_PDMS','min_PDMS','max_PDMS']].to_string(index=False),flush=True)

if __name__=='__main__':main()
