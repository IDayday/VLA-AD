"""Completed artifact identities and training/evaluation accounting."""
from common_r import *
import concurrent.futures,subprocess,pandas as pd
CHECKPOINT_HASHES={}
def check_cache(path):
    z=np.load(path);meta=json.loads(str(z['metadata']));assert meta['config_sha256']==sha(CONFIG)
    if path.is_relative_to(OUT/'cache/scores'):
        inp=OUT/'cache/rollouts'/path.relative_to(OUT/'cache/scores');assert meta['input_sha256']==sha(inp)
        assert z['scores'].shape==(64,7) and np.isfinite(z['scores']).all()
    else:
        assert z['trajectories'].shape==(64,8,3) and np.isfinite(z['trajectories']).all()
        assert meta['checkpoint_sha256']==CHECKPOINT_HASHES[(path.parent.parent.name,path.parent.name)]
    return dict(path=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size)
def main():
    identity();original_roots=['tools/analysis/pc_mts_diagnostics','tools/analysis/pc_mts_diagnostics_v2','tools/analysis/pc_mts_diagnostics_v3','configs/pc_mts_diagnostics','configs/pc_mts_diagnostics_v2','configs/pc_mts_diagnostics_v3','outputs/pc_mts_diagnostics','outputs/pc_mts_diagnostics_v2','outputs/pc_mts_diagnostics_v3','reports/PC_MTS_POLICY_DIAGNOSTICS_V3_20260913.md']
    changed=subprocess.check_output(['git','diff','--name-only',CFG['base_commit'],'--',*original_roots],cwd=ROOT,text=True);assert not changed,changed
    for p,h in read(OUT/'manifests/frozen.json')['source_sha256'].items():assert sha(ROOT/p)==h
    assert len(list((OUT/'manifests').glob('train_*.json')))==14
    assert len(list((OUT/'manifests').glob('eval_*.json')))==224
    roster=set(tokens('holdout'));paths=[]
    for method in CFG['methods']:
        for sd in CFG['seeds']:
            for step in CFG['snapshots'][1:]:
                for sub in ['rollouts','scores']:
                    folder=OUT/'cache'/sub/run_name(method,sd)/f'step{step:04d}'
                    ps=[p for p in folder.glob('*.npz') if p.stem in roster];assert {p.stem for p in ps}==roster
                    paths.extend(ps)
    assert len(paths)==33600
    checkpoint_records=[]
    for method in CFG['methods']:
        for sd in CFG['seeds']:
            for step in CFG['snapshots']:
                p=checkpoint(method,sd,step);h=sha(p);assert h==read(p.with_suffix('.json'))['sha256']
                CHECKPOINT_HASHES[(run_name(method,sd),f'step{step:04d}')]=h
                checkpoint_records.append(dict(path=str(p.relative_to(OUT)),sha256=h,bytes=p.stat().st_size))
    with concurrent.futures.ThreadPoolExecutor(32) as pool:records=list(pool.map(check_cache,paths))
    records.extend(checkpoint_records)
    save(OUT/'manifests/CACHE_INDEX.json',records)
    d=pd.read_csv(OUT/'metrics/training_safety_dynamics.csv');skips=d.groupby(['method','seed']).amp_skipped_update.sum().to_dict()
    # Native Lightning increments global_step on an AMP attempt. Report actual
    # successful updates separately if the unmodified scaler ever skipped.
    actual={f'{m}_seed{sd}':110-int(v) for (m,sd),v in skips.items()}
    save(OUT/'manifests/FINAL_AUDIT.json',dict(identity(),cache_files=33600,checkpoint_files=70,eval_trajectories=1075200,training_rollouts=14*56320,
         all_holdout_scene_sets_equal=True,previous_tracked_results_unchanged=True,primary_configuration_unchanged=True,
         successful_optimizer_updates=actual,identical_successful_update_budget=len(set(actual.values()))==1,
         code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/analysis/pc_mts_v3_stage3_rerun').glob('*.py')}))
    print('FINAL AUDIT PASS',actual,flush=True)
if __name__=='__main__':main()
