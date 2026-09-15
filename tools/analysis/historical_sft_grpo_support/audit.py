"""Cache, scalar evaluator, historical source and no-update final audit."""
from common_support import *
import argparse,subprocess

def scoring_smoke():
    import evaluate_cached_rollouts as e
    from analyze import train_reward
    e.init_worker()
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    import lzma,pickle
    standard=e.STATE['scorer'];training=PDMScorer(e.STATE['sampling'],PDMScorerConfig(progress_weight=10.,ttc_weight=5.,comfortable_weight=2.))
    checks=[]
    for scene in scenes()[:4]:
        for m in CFG['primary_models']:
            tr=np.load(OUT/'cache/rollouts'/m/f"{scene['token']}.npz")['native_grpo'][0,:4]
            parity=e.parity(scene,tr)
            with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
            s=e.score_arrays(cache,tr)
            e.STATE['scorer']=training;actual=e.score_arrays(cache,tr);e.STATE['scorer']=standard
            error=float(abs(train_reward(s)-actual[:,6]).max());assert error<1e-12,(scene['token'],m,error,s.tolist(),actual.tolist())
            checks.append(dict(model=m,**parity,training_reward_formula_error=error))
    save(OUT/'audits/scoring_parity.json',dict(status='PASS',checks=checks,training_weights=[10,5,2],standard_weights=[5,5,2],score_units='underlying fractions; tables PDMS points',batch_code_path=str(v1.CODE/'navsim/evaluate/pdm_score_batch.py'),batch_code_hash=sha(v1.CODE/'navsim/evaluate/pdm_score_batch.py')))

def final():
    rows=scenes();cfg=identity();hashes=[]
    frozen=read(OUT/'manifests/protocol.json')
    assert sha(OUT/'manifests/models.json')==frozen['models_sha256']
    assert sha(OUT/'manifests/scenes.json')==frozen['scene_sha256']
    for ob in read(OUT/'manifests/observations.json'):
        assert sha(ob['feature_path'])==ob['feature_sha256']
        scene=next(s for s in rows if s['token']==ob['token']);assert sha(scene['metric_cache_path'])==ob['metric_cache_sha256']
    for name,m in models().items():
        assert sha(m['checkpoint_path'])==m['sha256'];assert sha(m['runtime_planner_path'])==m['runtime_planner_sha256']
        for s in rows:
            t=s['token'];p=OUT/'cache/rollouts'/name/f'{t}.npz';q=OUT/'cache/scores/rollouts'/name/f'{t}.npz'
            z=np.load(p);a=json.loads(str(z['metadata']));b=np.load(q);score_meta=json.loads(str(b['metadata']))
            assert a['protocol_hash']==cfg and a['checkpoint_hash']==m['sha256']
            assert a['group_seeds']==[seed(t,g) for g in range(4)]
            assert score_meta['input_sha256']==sha(p)
            for protocol in CFG['protocols']:
                assert z[protocol].shape==(4,16,8,3) and np.isfinite(z[protocol]).all()
                assert b[protocol].shape==(4,16,7) and np.isfinite(b[protocol]).all()
            hashes.extend([dict(path=str(p),sha256=sha(p)),dict(path=str(q),sha256=sha(q))])
        audits=[read(p) for p in (OUT/'audits').glob(f'run_{name}_*.json')]
        assert sum(r['count'] for r in audits)==1000
        assert all(r['state_before']==r['state_after'] for r in audits)
    for name in CFG['primary_models']:
        for s in rows:
            p=OUT/'cache/fitting'/name/f"{s['token']}.npz";a=np.load(p);meta=json.loads(str(a['metadata']))
            assert meta['teacher_cache_hash']==sha(OUT/'cache/teachers'/f"{s['token']}.npz")
            assert np.isfinite(a['epsilon_mse']).all();hashes.append(dict(path=str(p),sha256=sha(p)))
    changed=subprocess.check_output(['git','diff',frozen['parent_commit'],'--name-only','--','outputs/pc_mts_diagnostics','outputs/pc_mts_diagnostics_v2','outputs/pc_mts_diagnostics_v3'],text=True)
    assert not changed,changed
    source_hashes=[]
    for family in ['a5','v6','psi']:
        for p in (OUT/'runtime'/family/'navsim').rglob('*.py'):source_hashes.append(dict(path=str(p),sha256=sha(p)))
    save(OUT/'manifests/cache_hashes.json',hashes)
    save(OUT/'manifests/runtime_source_hashes.json',source_hashes)
    save(OUT/'audits/final.json',dict(status='PASS',scene_count=1000,model_count=9,primary_sft_count=4,groups_per_scene=4,group_size=16,primary_rollouts=9*1000*4*16*2,factorial_rollouts=4*256*4*16*3,common_train=835,optimizer_updates=0,missing_PSI_GRPO=True,old_tracked_results_unchanged=True,cache_hash_count=len(hashes),config_sha256=cfg))
    print('FINAL AUDIT PASS',len(hashes),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--scoring-smoke',action='store_true');args=a.parse_args()
    scoring_smoke() if args.scoring_smoke else final()
