"""Full-1000 identity, leakage, source and selection acceptance gates."""
from common_mass import *
import pandas as pd
import concurrent.futures, pickle, lzma

def scene_identity(scene):
    import torch
    t=scene['token'];f=torch.load(V1/'features'/f'{t}.pt',map_location='cpu',weights_only=False)
    assert f['_metadata']['observation_only'] and f['_metadata']['token']==t
    source=read(OUT/'audits/scene_cache_identity.json');expected=next(r for r in source['checks'] if r['token']==t)
    assert sha(V1/'features'/f'{t}.pt')==expected['feature_sha256'] and sha(scene['metric_cache_path'])==expected['metric_cache_sha256']
    return dict(token=t,feature_hash=expected['feature_sha256'],metric_cache_hash=expected['metric_cache_sha256'],status='OK')

def inventory():
    inv={};counts=[]
    for source in ['ddv2','drivor']:
        identity=read(OUT/'audits'/f'{source}_identity.json');assert sha(identity['checkpoint_path'])==identity['checkpoint_sha256']
        for path,h in identity['runtime_files'].items():assert sha(path)==h
        records=[]
        for s in scenes():
            p=OUT/'cache/external'/source/f'{s["token"]}.npz'
            if not p.exists():records.append(dict(token=s['token'],status='MISSING',count=0));continue
            z=np.load(p);meta=json.loads(str(z['metadata']));assert meta['token']==s['token'] and meta['checkpoint_sha256']==identity['checkpoint_sha256']
            records.append(dict(token=s['token'],status='OK',count=len(z['trajectories']),unique_count=len({content_hash(t) for t in z['trajectories']}),native_count=meta['native_proposal_count'],cache_sha256=sha(p)))
        inv[source]=dict(identity,coverage=sum(r['status']=='OK' for r in records),scene_denominator=1000,records=records,status='SUPPORTED' if all(r['status']=='OK' for r in records) else 'MISSING')
        counts.append(dict(source=source,total=sum(r['count'] for r in records),scenes=sum(r['status']=='OK' for r in records)))
    inv['ddv2']['native_postprocessing']='20 learned anchors x 10 independent noise groups x (original + 3 native multiplicative augmentations) = 800 before coarse/fine scorers; public bezier_xyyaw preserves XY and computes heading from an origin-anchored derivative. Fixed-index hash subset of 64; native augmentations remain tagged DDV2, not GT perturbation.'
    inv['drivor']['native_postprocessing']='64 final proposal heads, captured as proposals before scorer selection; native inference 4 s / 8 points. Released long-target training is a target deformation, not an instruction to relabel deployment timestamps.'
    inv['GT_structured']=dict(count_per_scene=128,definition=cfg()['perturbations'],implementation_sha256=sha(ROOT/'tools/analysis/pc_mts_diagnostics_v2/perturbations.py'))
    save(OUT/'manifests/source_inventory.json',inv);print('source inventory',counts,flush=True)

def full():
    assert sha(CONFIG)==read(OUT/'manifests/protocol_frozen.json')['sha256'];inventory()
    assert read(OUT/'audits/coordinate_and_token_identity.json')['scene_count']==1000
    with concurrent.futures.ThreadPoolExecutor(16) as pool:feature_checks=list(pool.map(scene_identity,scenes()))
    save(OUT/'audits/feature_observation_identity.json',dict(checks=feature_checks,status='PASS'))
    for path,h in read(OUT/'audits/protected_artifact_hashes.json').items():assert sha(ROOT/path)==h,path
    sampler=read(OUT/'manifests/sampler_identity.json');assert sha(sampler['checkpoint_path'])==sampler['checkpoint_hash'];assert sha(sampler['runtime_code_path'])==sampler['runtime_code_hash']
    source=read(OUT/'audits/scene_cache_identity.json');bytoken={r['token']:r for r in source['checks']};rows=[];seeds=set();sample_count=0;cache_hashes={}
    for scene in scenes():
        t=scene['token'];assert sha(V1/'features'/f'{t}.pt')==bytoken[t]['feature_sha256'];assert sha(scene['metric_cache_path'])==bytoken[t]['metric_cache_sha256']
        for stream,n in [('A',1024),('B',1024),('C',64)]:
            p=OUT/'cache/rollouts'/stream/f'{t}.npz';z=np.load(p);a=z['trajectories'];assert a.shape==(n,8,3) and np.isfinite(a).all()
            assert np.array_equal(z['group_ids'],np.repeat(np.arange(n//8),8)) and np.array_equal(z['member_ids'],np.tile(np.arange(8),n//8))
            assert all(int(x)==random_seed(t,stream,i) for i,x in enumerate(z['group_seeds']))
            assert not seeds&set(z['group_seeds'].tolist());seeds.update(z['group_seeds'].tolist());sample_count+=n
            score=OUT/'cache/scores/rollouts'/stream/f'{t}.npz';zs=np.load(score);assert np.isfinite(zs['scores']).all() and (zs['score_status']=='OK').all()
            assert json.loads(str(zs['metadata']))['input_sha256']==sha(p);cache_hashes[str(p)]=sha(p);cache_hashes[str(score)]=sha(score)
        rawpath=OUT/'cache/raw'/f'{t}.npz';raw=np.load(rawpath)['trajectories'];records=read(rawpath.with_suffix('.json'))['candidates'];ids=[r['unique_parent_id'] for r in records]
        assert len(ids)==len(set(ids))==len(raw) and all(content_hash(x)==r['content_hash'] for x,r in zip(raw,records))
        assert all(not any(str(p.get('parent_candidate_id','')).startswith(('A:','B:')) for p in r['all_parents']) for r in records)
        for r in records:assert len(r['all_source_tags'])==len(set(r['all_source_tags']))
        cache_hashes[str(rawpath)]=sha(rawpath);rows.append(dict(token=t,status='OK',raw_unique=len(raw),A=1024,B=1024,C=64))
    assert sample_count==2112000
    save(OUT/'audits/cache_hash_manifest.json',cache_hashes)
    selected=pd.read_parquet(OUT/'metrics/selected_pools_4x1000x16.parquet');assert len(selected)==64000
    assert len(selected.groupby(['method','token']))==4000 and (selected.groupby(['method','token']).size()==16).all()
    valid=selected[selected.valid_mask.fillna(False)]
    assert not valid.duplicated(['token','method','unique_parent_id']).any()
    assert (valid.strict_qualified==(valid.selection_level==0)).all()
    assert not valid[(valid.method=='grpo_mass_pc')&valid.strict_qualified&~valid.probability_qualified.astype(bool)].shape[0]
    assert not valid[(valid.method=='gt_geometry')&valid.strict_qualified&~valid.geometry_qualified.astype(bool)].shape[0]
    pools=pd.read_csv(OUT/'metrics/pool_scene_metrics.csv');assert len(pools)==8000 and pools.token.nunique()==1000
    assert (pools.PoolMass<=pools.sum_individual_mass+1e-12).all()
    assert (pools.QualityMatchedPoolMass<=pools.PoolMass+1e-12).all()
    assert all(abs(sum(json.loads(r.marginal_coverage))-r.PoolMass)<1e-12 for r in pools.itertuples())
    global_metrics=pd.read_csv(OUT/'metrics/global_policy_sampling_baseline.csv');assert len(global_metrics)==1000 and 'method' not in global_metrics
    flags=read(OUT/'audits/coordinate_and_token_identity.json')['timestamp_flagged_tokens']
    errors=pd.DataFrame([dict(token=t,stage='raw_log_timing',status='TIMESTAMP_IRREGULAR_NATIVE_CONTRACT_RETAINED',error='Raw log future timestamps deviate from nominal .5 s; native frame-indexed GT and all1000 scenes retained') for t in flags],columns=['token','stage','status','error']);errors.to_csv(OUT/'metrics/missing_and_errors.csv',index=False)
    save(OUT/'audits/completion.json',dict(status='PASS',scene_count=1000,missing_scenes=0,error_scenes=0,probability_samples=2048000,C_samples=64000,independent_group_seeds=len(seeds),selected_slot_count=64000,valid_selected_slots=len(valid),old_artifacts_unchanged=True,checkpoints_unchanged=True,config_hash=protocol(),all_scientific_filters_prespecified=True,completed_at=utc(),rows=rows))
    save(OUT/'manifests/run_summary.json',dict(status='DATA_COMPLETE_AWAITING_FINAL_REVIEW',scenes=1000,missing_scenes=0,error_scenes=0,raw_log_timestamp_flagged_scenes=len(flags),raw_candidates=sum(r['raw_unique'] for r in rows),probability_samples=2048000,C_samples=64000,optimizer_updates=0,protocol_hash=protocol(),cache_manifest_hash=sha(OUT/'audits/cache_hash_manifest.json'),completed_at=utc()))
    print('FULL1000 AUDIT PASS; no missing/error scenes; old data unchanged',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--inventory-only',action='store_true');a=p.parse_args();inventory() if a.inventory_only else full()
