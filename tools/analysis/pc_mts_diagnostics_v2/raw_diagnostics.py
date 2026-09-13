"""Reservoir-only audit, frozen subset; must run before candidate selection."""
import argparse,concurrent.futures
from v2_common import *
import pandas as pd
from scipy.stats import spearmanr

POS=['policy_distance_knn','r_knn','q_holdout','mahalanobis','maha_ratio','maha_percentile']
def frame(scene):
    token=scene['token'];a=np.load(OUT/'raw_candidates'/f'{token}.npz');p=np.load(OUT/'positions'/f'{token}.npz');s=np.load(OUT/'evaluator/raw_candidates'/f'{token}.npz')['trajectories'];local=np.load(OUT/'evaluator/selection_local'/f'{token}.npz')['trajectories']
    d=pd.DataFrame({k:a[k] for k in ['source','candidate_id','bridge_alpha','anchor_id','reference_index']})
    for k in POS:d[k]=p[k]
    d['token']=token;d['command']=scene['command'];d['raw_index']=np.arange(192);d['pdms']=s[:,6]*100;d['feasible']=feasible(s);d['d_GT']=distance(a['trajectories'],np.asarray(scene['gt'])[None])[:,0];d['selection_local_feasible']=feasible(local).mean(1);d['region']=region(d.q_holdout.to_numpy());d['high_quality']=d.pdms>=d.pdms.quantile(.75)
    for source in d.source.unique():d.loc[d.source==source,'source_pairwise_ADE']=pair_mean(a['trajectories'][d.source==source])
    il=np.load(V1/'rollouts/official_il'/f'{token}.npz')['trajectories'];selfdist=distance(il,il);np.fill_diagonal(selfdist,np.inf);selfdist=np.sort(selfdist,axis=1)[:,:5].mean(1)
    old=np.load(V1/'raw_candidates'/f'{token}.npz')['trajectories'];old_dist=np.sort(distance(old,il),axis=1)[:,:5].mean(1)
    calibration=[]
    for label,vals in [('V1_IL_self_KNN',selfdist),('V1_candidate_KNN',old_dist),('V2_IL_Q_to_R_KNN',p['query_reference_distances']),('V2_candidate_KNN',p['policy_distance_knn']),('V2_Q_mahalanobis',p['query_mahalanobis'])]:
        calibration.append(dict(token=token,distribution=label,**{f'p{q}':np.percentile(vals,q) for q in [0,5,25,50,75,95,99,100]},count=len(vals),exceeds_V1_self_max=float((vals>selfdist.max()).mean()) if label=='V1_candidate_KNN' else np.nan))
    return d,calibration

def main(a):
    rows=scenes()['scenes'];tag='smoke' if a.smoke else 'full'
    if a.smoke:rows=[r for r in rows if r['token'] in set(subset('smoke'))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(96,len(rows))) as ex:result=list(ex.map(frame,rows,chunksize=1))
    d=pd.concat([r[0] for r in result],ignore_index=True);d.to_parquet(OUT/'metrics'/f'raw_candidates_{tag}.parquet',index=False)
    c=pd.DataFrame(sum([r[1] for r in result],[]));c.to_csv(OUT/'metrics'/f'calibration_scene_quantiles_{tag}.csv',index=False)
    c.groupby('distribution').mean(numeric_only=True).to_csv(OUT/'metrics'/f'calibration_quantiles_{tag}.csv')
    summary=d.groupby('source').agg(count=('token','size'),scenes=('token','nunique'),PDMS=('pdms','mean'),d_GT=('d_GT','mean'),d_CR=('policy_distance_knn','mean'),r_knn=('r_knn','mean'),q_holdout=('q_holdout','mean'),maha=('mahalanobis','mean'),maha_ratio=('maha_ratio','mean'),maha_percentile=('maha_percentile','mean'),local_feasible=('selection_local_feasible','mean'),source_diversity=('source_pairwise_ADE','mean'))
    summary.to_csv(OUT/'metrics'/f'raw_source_summary_{tag}.csv')
    subset_rows=[]
    for token,g in d.groupby('token',sort=False):
        counts=g.region.value_counts();core=int(counts.get('core',0));far=int(counts.get('far',0));boundary=int(counts.get('boundary',0));near_good=bool(((g.region=='core')&g.high_quality).any());far_good=bool(((g.region=='far')&g.high_quality).any())
        subset_rows.append(dict(token=token,core=core,far=far,boundary=boundary,near_good=near_good,far_good=far_good,eligible=core>=16 and far>=16 and boundary>=8 and near_good and far_good))
    sd=pd.DataFrame(subset_rows);sd.to_csv(OUT/'metrics'/f'contrastive_eligibility_{tag}.csv',index=False)
    bridge=d[d.bridge_alpha.notna()];p=bridge.pivot(index=['token','anchor_id'],columns='bridge_alpha',values='r_knn');increments=np.diff(p.to_numpy(),axis=1)
    stats=dict(anchor_pairs=len(p),strict_monotonic_fraction=float((increments>0).all(1).mean()),nondecreasing_fraction=float((increments>=-1e-8).all(1).mean()),alpha08_greater02_fraction=float((p[.8]>p[.2]).mean()),median_ratio_08_02=float((p[.8]/p[.2]).median()),mean_r_knn_by_alpha=bridge.groupby('bridge_alpha').r_knn.mean().to_dict(),median_r_knn_by_alpha=bridge.groupby('bridge_alpha').r_knn.median().to_dict(),definition='Same-anchor bridge, alpha fixed before outcomes; KNN can change nearest neighbours.')
    save(OUT/'manifests'/f'raw_audit_{tag}.json',dict(identity=ident(),scene_count=len(rows),candidate_count=len(d),sources=d.source.value_counts().to_dict(),bridge=stats,contrastive_scene_count=int(sd.eligible.sum()),selection_has_not_run=not any((OUT/'candidate_pools').glob('*/*.npz'))))
    if not a.smoke:
        save(OUT/'manifests/contrastive_scenes.json',dict(identity=ident(),definition='Core>=16, Far>=16, Boundary>=8; both Core and Far include PDMS>=raw scene p75; no selected-pool outcomes',tokens=sd[sd.eligible].token.tolist(),count=int(sd.eligible.sum())))
    print(summary.to_string());print(json.dumps(stats));print('Contrastive',int(sd.eligible.sum()),'/',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args())
