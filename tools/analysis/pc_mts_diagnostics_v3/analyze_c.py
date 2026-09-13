"""Paired inference and scene/source fixed effects, not pooled correlations."""
from common_v3 import *

def cluster_bootstrap(values,groups,key):
    d=pd.DataFrame({'v':values,'g':groups}).dropna();agg=d.groupby('g').v.agg(['sum','count']);rng=np.random.default_rng(seed(key,'v3_cluster_bootstrap'));means=[]
    for b in range(0,3000,100):
        ix=rng.integers(len(agg),size=(100,len(agg)));means.extend(agg['sum'].to_numpy()[ix].sum(1)/agg['count'].to_numpy()[ix].sum(1))
    return dict(cluster_scenes=len(agg),cluster_ci_low=float(np.quantile(means,.025)),cluster_ci_high=float(np.quantile(means,.975)),scene_win_fraction=float((agg['sum']>1e-10).mean()))

def absorb(matrix,codes):
    z=matrix.astype(float).copy()
    for iteration in range(1000):
        old=z.copy()
        for c in codes:
            counts=np.bincount(c);sums=np.zeros((len(counts),z.shape[1]));np.add.at(sums,c,z);z-=sums[c]/counts[c,None]
        if np.max(abs(z-old))<1e-10:break
    return z,iteration+1

def fixed_effects(d,outcome):
    keys=['r_knn','PDMS','d_GT'];d=d.dropna(subset=keys+[outcome]);scene=pd.factorize(d.token)[0];source=pd.factorize(d.exact_source)[0]
    z,niter=absorb(d[keys+[outcome]].to_numpy(),[scene,source]);x=z[:,:3];y=z[:,3];bread=np.linalg.pinv(x.T@x);coef=bread@x.T@y;resid=y-x@coef
    scores=np.zeros((scene.max()+1,3));np.add.at(scores,scene,x*resid[:,None]);influence=scores@bread.T
    rng=np.random.default_rng(seed(outcome,'v3_FE_wild_scene_bootstrap'));draws=[]
    for b in range(0,3000,100):draws.extend(rng.choice([-1.,1.],size=(100,len(scores)))@influence)
    draws=np.asarray(draws);rows=[]
    for j,key in enumerate(keys):rows.append(dict(outcome=outcome,predictor=key,coefficient=coef[j],ci_low=coef[j]+np.quantile(draws[:,j],.025),ci_high=coef[j]+np.quantile(draws[:,j],.975),n=len(d),scenes=d.token.nunique(),sources=d.exact_source.nunique(),scene_FE=True,source_FE=True,controls=','.join(keys),inference='3000 Rademacher wild scene-cluster bootstrap, fixed design',absorption_iterations=niter,standardized_effect=coef[j]*d[key].std()))
    return rows

def main():
    q=pd.read_parquet(OUT/'metrics/C_query_manifest.parquet');frames=[]
    for token in tokens('all'):
        a=np.load(OUT/'cache/learnability'/f'{token}.npz');g=pd.DataFrame(dict(query_id=a['query_id'],diffusion_loss=a['uniform_loss'].mean(1),E_rec=a['E_rec'],return_rate=a['return_rate']))
        for j,t in enumerate([20,50,80]):g[f'loss_t{t}']=a['fixed_loss'][:,j*2:j*2+2].mean(1)
        frames.append(g)
    d=q.merge(pd.concat(frames,ignore_index=True),on='query_id',validate='one_to_one');raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');d=d.merge(raw,on=['token','raw_index'],validate='many_to_one');assert len(d)==len(q);parquet(d,'C_candidate_measurements.parquet')
    pair=pd.read_csv(OUT/'metrics/C_pair_manifest.csv');joined=[];statistics=[]
    for protocol in ['primary','sensitivity']:
        x=d[d.protocol==protocol];a=x[x.side=='near'].set_index('pair_id');b=x[x.side=='far'].set_index('pair_id');assert set(a.index)==set(b.index)
        b=b.reindex(a.index);j=pair[pair.protocol==protocol].set_index('pair_id').reindex(a.index).copy()
        for metric in ['diffusion_loss','loss_t20','loss_t50','loss_t80','E_rec','return_rate']:
            j['near_'+metric]=a[metric];j['far_'+metric]=b[metric];j['delta_'+metric]=b[metric]-a[metric]
        joined.append(j.reset_index())
        for region in ['all','boundary','core']:
            sub=j if region=='all' else j[j.near_region==region]
            if not len(sub):continue
            for metric in ['diffusion_loss','loss_t20','loss_t50','loss_t80','E_rec','return_rate']:
                delta=sub['delta_'+metric];statistics.append(dict(protocol=protocol,near_region=region,metric=metric,contrast='far_minus_compatible',near_mean=sub['near_'+metric].mean(),far_mean=sub['far_'+metric].mean(),**bootstrap(delta,f'C_pair_{protocol}_{region}_{metric}'),**cluster_bootstrap(delta.to_numpy(),sub.token.to_numpy(),f'C_{protocol}_{region}_{metric}')))
    csv(pd.concat(joined,ignore_index=True),'C_pair_results.csv');csv(pd.DataFrame(statistics),'C_paired_bootstrap.csv')
    regression=d[d.protocol=='regression'];reg=fixed_effects(regression,'diffusion_loss')+fixed_effects(regression,'E_rec');csv(pd.DataFrame(reg),'C_fixed_effects.csv')
    stage_audit('C',scene_count=d.token.nunique(),candidate_queries=len(d),regression_candidates=len(regression),regression_scene_count=regression.token.nunique(),pair_count=len(pair),measurement_sha256=sha(OUT/'metrics/C_candidate_measurements.parquet'),checkpoint_sha256=v1.models()[0]['sha256'],common_random_numbers='same noise_key generates native uniform timestep and epsilon for both members',source_composition=d.exact_source.value_counts().to_dict(),duplicate_supervision_modes_created=False,leakage='Static diagnostic uses Navtrain offline targets/scores; frozen forward conditioning contains observations only',outcome_conditioned_filtering=False)
    print(pd.DataFrame(statistics).query("protocol=='primary' and near_region=='all'")[['metric','near_mean','far_mean','mean','cluster_ci_low','cluster_ci_high']].to_string(index=False));print(pd.DataFrame(reg).query("predictor=='r_knn'").to_string(index=False),flush=True)
if __name__=='__main__':main()
