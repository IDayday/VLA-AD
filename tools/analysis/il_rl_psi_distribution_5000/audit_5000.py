"""Validate imported caches, native/scalar scoring, and final completeness."""
from common_5000 import *
import concurrent.futures, argparse, lzma, pickle, inspect
import evaluate_cached_rollouts as evaluator

def check_old(record):
    assert sha(record['path'])==record['sha256'],record['path']
    return record

def inputs_identity(row):
    return dict(token=row['token'],log=row['log'],cohort=row['cohort'],metric_cache_path=row['metric_cache_path'],metric_cache_sha256=sha(row['metric_cache_path']),feature_path=str(feature_path(row['token'])),feature_sha256=sha(feature_path(row['token'])))

def initial():
    identity();records=read(OLDOUT/'manifests/cache_hashes.json')
    names=set(CFG['primary_models']);selected=[r for r in records if Path(r['path']).parent.name in names and '/rollouts/' in r['path']]
    assert len(selected)==6000,len(selected)
    with concurrent.futures.ThreadPoolExecutor(12) as pool:list(pool.map(check_old,selected))
    save(OUT/'manifests/imported_caches.json',selected)
    evaluator.init_worker();checks=[]
    for scene in scenes()[:4]:
        with np.load(bank_path('official_il',scene['token'])) as z:t=z['native_grpo'][0,:4]
        checks.append(evaluator.parity(scene,t))
        with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        fresh=evaluator.score_arrays(cache,t)
        with np.load(bank_path('official_il',scene['token'],True)) as z:old=z['native_grpo'][0,:4]
        err=float(abs(fresh-old).max());assert err<=1e-8
        checks[-1]['fresh_vs_imported_score_max_abs_error']=err
    import navsim.evaluate.pdm_score_batch as batchmodule
    save(OUT/'audits/imports_and_scoring.json',dict(status='PASS',verified_old_files=len(selected),old_files_unchanged=True,checks=checks,evaluator_runtime_path=inspect.getfile(batchmodule),evaluator_runtime_sha256=sha(inspect.getfile(batchmodule))))

def final():
    identity();initial()
    with concurrent.futures.ThreadPoolExecutor(12) as pool:observations=list(pool.map(inputs_identity,scenes()))
    save(OUT/'manifests/observations.json',observations)
    oldobs={r['token']:r for r in read(OLDOUT/'manifests/observations.json')}
    for row in observations:
        if row['token'] in oldobs:
            for k in ['feature_sha256','metric_cache_sha256']:assert row[k]==oldobs[row['token']][k]
    records=[];model_info=models()
    for model in CFG['primary_models']:
        for row in scenes():
            token=row['token'];p=bank_path(model,token);sp=bank_path(model,token,True)
            with np.load(p) as z:
                meta=json.loads(str(z['metadata']))
                assert meta['checkpoint_hash']==model_info[model]['sha256']
                assert meta['group_seeds']==[seed(token,g) for g in range(4)]
                for pr in CFG['protocols']:assert z[pr].shape==(4,16,8,3) and np.isfinite(z[pr]).all()
            with np.load(sp) as z:
                sm=json.loads(str(z['metadata']));assert sm['input_sha256']==sha(p)
                for pr in CFG['protocols']:assert z[pr].shape==(4,16,7) and np.isfinite(z[pr]).all()
            if row['cohort']=='NEW4000':
                assert meta['protocol_hash']==identity() and sm['protocol_hash']==identity()
            records.extend([dict(path=str(p),sha256=sha(p)),dict(path=str(sp),sha256=sha(sp))])
        for rank in range(8):
            r=read(OUT/'audits'/f'run_{model}_{rank}.json');assert r['state_before']==r['state_after']
        for name in ['sampling','batch','import_parity']:
            assert read(OUT/'audits'/f'{name}_{model}.json')['status']=='PASS'
    sf=pd.read_parquet(OUT/'metrics/scene_metrics.parquet');assert len(sf)==30000
    assert sf.groupby(['model','protocol']).size().eq(5000).all()
    assert (sf[sf.model=='official_il'].centroid_displacement==0).all()
    save(OUT/'manifests/cache_hashes.json',records)
    save(OUT/'audits/final.json',dict(status='PASS',scene_count=5000,old_scenes=1000,new_scenes=4000,scene_model_protocol_cells=30000,groups=120000,rollouts=1920000,new_rollouts=1536000,missing_scenes=0,errors=0,optimizer_updates=0,weight_state_unchanged=True,old_cache_files_unchanged=True,protocol_hash=identity()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
    final() if a.final else initial()
