from common_v3 import *
from analyze_c import cluster_bootstrap
def main():
    records=[read(OUT/'cache/gradient'/f'{t}.json') for t in tokens('gradient')];methods=pd.DataFrame(sum([r['methods'] for r in records],[]));candidates=pd.DataFrame(sum([r['candidates'] for r in records],[]));pairs=pd.DataFrame(sum([r['pairs'] for r in records],[]));csv(methods,'D_method_scene.csv');csv(candidates,'D_candidate_gradients.csv');csv(pairs,'D_matched_gradients.csv');stats=[]
    for m,g in methods.groupby('method'):
        for metric in ['loss','gradient_norm','cosine','dot','predicted_delta_L_ref']:stats.append(dict(method=m,metric=metric,**bootstrap(g[metric],f'D_{m}_{metric}')))
    csv(pd.DataFrame(stats),'D_method_bootstrap.csv');paired=[]
    for m in METHODS[:-1]:
        for metric in ['loss','gradient_norm','cosine','dot','predicted_delta_L_ref']:
            tab=methods.pivot(index='token',columns='method',values=metric);paired.append(dict(method=m,metric=metric,contrast='conditional_minus_baseline',**bootstrap(tab.conditional_pc-tab[m],f'D_method_pair_{m}_{metric}')))
    csv(pd.DataFrame(paired),'D_scene_paired_comparisons.csv');a=pairs[pairs.side=='near'].set_index('pair_id');b=pairs[pairs.side=='far'].set_index('pair_id').reindex(a.index);rows=[]
    for metric in ['loss','gradient_norm','cosine','dot','predicted_delta_L_ref']:
        delta=b[metric]-a[metric];rows.append(dict(metric=metric,contrast='far_minus_compatible',near_mean=a[metric].mean(),far_mean=b[metric].mean(),**bootstrap(delta,f'D_pair_{metric}'),**cluster_bootstrap(delta.to_numpy(),a.token.to_numpy(),f'D_pair_{metric}')))
    csv(pd.DataFrame(rows),'D_matched_bootstrap.csv');csv(candidates.groupby(['method','region']).agg(n=('token','size'),scenes=('token','nunique'),loss=('loss','mean'),norm=('gradient_norm','mean'),alignment=('cosine','mean')).reset_index(),'D_region_summary.csv')
    stage_audit('D',scene_count=len(records),method_scene_rows=len(methods),missing_method_scenes=methods[methods['count']==0].groupby('method').size().to_dict(),exact_parameter_count=read(OUT/'manifests/gradient_parameters.json')['total'],matched_candidate_counts=True,matched_pair_count=len(a),no_gradient_sketch=True,checkpoint_sha256=v1.models()[0]['sha256'],source_composition=candidates.exact_source.value_counts().to_dict(),no_candidate_duplication=True,raw_table_sha256=sha(OUT/'metrics/A_raw_candidates.parquet'),outcome_conditioned_filtering=False)
    print(pd.DataFrame(rows)[['metric','near_mean','far_mean','mean','cluster_ci_low','cluster_ci_high']].to_string(index=False),flush=True)
if __name__=='__main__':main()
