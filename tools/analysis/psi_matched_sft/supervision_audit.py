"""Publish reproducible target identities and actual target presentation counts."""
from common_matched import *
def main():
    rows=[];rawrows=[];frozen=read(OUT/'manifests/selection_frozen.json')['selection_hashes']
    for s in scenes():
        t=s['token'];path=OUT/'cache/raw'/f'{t}.npz'
        with np.load(path) as a:tr=a['trajectories'];sc=a['scores'];meta=json.loads(str(a['metadata']))
        selection=read(OUT/'cache/selection'/f'{t}.json')['methods']
        for method,p in selection.items():
            for idx,w in zip(p['indices'],p['weights']):
                # Read-only audit of the official legacy output envelope. Never
                # used to filter or change any frozen supervision candidate.
                norm=2*(tr[idx]+np.array([1.57,19.68,1.67]))/np.array([66.74,42.,3.53])-1
                rows.append(dict(token=t,split=s['split'],method=method,raw_index=idx,weight=w,trajectory=tr[idx].reshape(-1).tolist(),trajectory_sha256=digest(tr[idx].tolist()),all_source_tags=meta['all_source_tags'][idx],archive_path=meta['archive_path'],archive_sha256=meta['archive_sha256'],archive_indices=meta['raw_indices'][idx],PDMS=sc[idx,6]*100,feasible=bool(safe(sc[idx])),normalized_max_abs=float(abs(norm).max()),outside_native_final_clip=bool((abs(norm)>1+1e-6).any()),selection_sha256=frozen[t]))
        for i,tags in enumerate(meta['all_source_tags']):rawrows.append(dict(token=t,split=s['split'],source=tags[0] if len(set(tags))==1 else 'multi_source',PDMS=sc[i,6]*100,feasible=bool(safe(sc[i])),raw_index=i))
    targets=table('selected_targets.parquet',rows);raw=table('raw_candidates_summary.parquet',rawrows)
    table('raw_source_summary.csv',raw.groupby(['split','source']).agg(candidates=('token','size'),scenes=('token','nunique'),PDMS=('PDMS','mean'),feasible=('feasible','mean')).reset_index())
    if not (OUT/'audits/training_complete.json').exists():return
    out=[]
    for method in CFG['methods']:
        for r in CFG['train_seeds']:
            ledger=pd.read_parquet(OUT/'metrics'/f'ledger_{method}_seed{r}.parquet');counts=ledger.groupby(['token','raw_index']).size()
            for row in targets[(targets.split=='train')&(targets.method==method)].to_dict('records'):
                out.append(dict(token=row['token'],method=method,seed=r,raw_index=row['raw_index'],expected_weight=row['weight'],presentation_count=int(counts.get((row['token'],row['raw_index']),0)),is_gt=row['raw_index']==0))
    f=table('target_presentations.parquet',out)
    table('target_presentation_summary.csv',f.groupby(['method','seed']).agg(unique_targets=('raw_index','size'),never_presented=('presentation_count',lambda x:int((x==0).sum())),mean_presentations=('presentation_count','mean')).reset_index())
if __name__=='__main__':main()
