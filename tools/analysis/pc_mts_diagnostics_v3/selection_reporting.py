"""Clarify identical count scopes; retain the original schema and every selected parent."""
from common_v3 import *
def main():
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');scenes_table=pd.read_csv(OUT/'metrics/A_pool_scene.csv');before=sha(OUT/'metrics/A_pool_scene.csv');archive=OUT/'report/A_pool_scene_original_count_schema.csv'
    if not archive.exists():archive.write_bytes((OUT/'metrics/A_pool_scene.csv').read_bytes())
    rawcount=raw.groupby('token').strict_eligible.sum();scenes_table['strict_candidate_count']=scenes_table.strict_selected_count;scenes_table['strict_available_in_raw']=scenes_table.token.map(rawcount);csv(scenes_table,'A_pool_scene.csv');csv(scenes_table.groupby('method').mean(numeric_only=True).reset_index(),'A_pool_summary.csv')
    metrics=[];sources=[]
    for token,g in raw.groupby('token',sort=False):
        g=g.set_index('raw_index');pools=read(OUT/'cache/pools'/f'{token}.json')['methods']
        for method,p in pools.items():
            s=g.loc[p['indices']]
            metrics.append(dict(token=token,method=method,count=len(s),**{k:s[k].mean() for k in ['PDMS','q_holdout','policy_distance_knn','r_knn','maha_ratio','d_GT','local_feasible']}))
            for source,count in s.exact_source.value_counts().items():sources.append(dict(token=token,method=method,source=source,count=int(count)))
    csv(pd.DataFrame(metrics),'A_continuous_pool_metrics.csv');csv(pd.DataFrame(sources),'A_selected_source_counts.csv')
    save(OUT/'manifests/A_count_schema_clarification.json',dict(identity=identity(),original_csv_sha256=before,current_csv_sha256=sha(OUT/'metrics/A_pool_scene.csv'),definition='strict_candidate_count consistently counts selected unique raw parents satisfying the common V3 strict gate; strict_available_in_raw separately records preselection availability',all_selected_parent_ids_unchanged=True,strict_selected_count_and_all_main_comparisons_unchanged=True,rescoring_or_reselection=False,original_count_schema_preserved=str(archive)))
if __name__=='__main__':main()
