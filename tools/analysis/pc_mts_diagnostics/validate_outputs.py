"""Full-size, content-based acceptance evidence; partial work is never certified complete."""
import argparse,concurrent.futures
import pandas as pd
from common import *

def validate_scene(task):
    r,cfg,sc_hash,model_rows=task;out=ROOT/cfg['output_dir'];token=r['token'];count=0
    for m in model_rows:
        path=out/'rollouts'/m['name']/f'{token}.npz';z=np.load(path);meta=json.loads(str(z['metadata']))
        assert meta['scene_manifest_hash']==sc_hash and meta['config_hash']==digest(cfg) and meta['checkpoint_hash']==m['sha256']
        assert z['trajectories'].shape==(cfg['num_rollouts'],8,3) and z['trajectories'].dtype==np.float32 and np.isfinite(z['trajectories']).all()
        scored=np.load(out/'evaluator/rollouts'/m['name']/f'{token}.npz');score_meta=json.loads(str(scored['metadata']))
        assert score_meta['input_sha256']==sha(path);assert scored['trajectories'].shape==(cfg['num_rollouts'],7) and np.isfinite(scored['trajectories']).all()
        if m['name']=='official_il':assert z['reference'].shape==(1,8,3)
        else:assert z['external'].shape==(cfg['raw_model_count'],8,3)
        count+=len(z['trajectories'])
    return count

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];model_rows=models();tasks=[(r,cfg,digest(sc),model_rows) for r in sc['scenes']]
    assert len(sc['scenes'])==cfg['num_scenes'] and len({r['token'] for r in sc['scenes']})==cfg['num_scenes']
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:count=sum(ex.map(validate_scene,tasks,chunksize=4))
    assert count==5*cfg['num_scenes']*cfg['num_rollouts']
    gd=pd.read_csv(out/'metrics/policy_distribution.csv');q=pd.read_csv(out/'metrics/grpo_readiness.csv');groups=pd.read_csv(out/'metrics/grpo_groups.csv')
    assert len(gd)==len(q)==5*cfg['num_scenes'];assert len(groups)==5*cfg['num_scenes']*(cfg['num_rollouts']//8)
    assert not gd.duplicated(['token','checkpoint']).any() and not q.duplicated(['token','checkpoint']).any()
    valid=dict(scene_count=cfg['num_scenes'],rollout_trajectories=count,rollout_scene_model_pairs=len(gd),actual_groups=len(groups),checkpoint_hashes_verified=5,all_rollout_files_finite=True,all_score_files_finite=True,score_input_hashes_match=True,main_metric_rows_complete=True,candidate_completion=False)
    if (out/'manifests/candidate_pool_validation.json').exists():
        p=read(out/'manifests/candidate_pool_validation.json');valid['candidate_validation']=p
        valid['candidate_completion']=set(p['methods'])==set(METHODS) and all(v==cfg['num_scenes'] for v in p['complete_scene_counts'].values())
    if (out/'manifests/pc_partial_heldout_validation.json').exists():
        p=read(out/'manifests/pc_partial_heldout_validation.json');assert p['identity']==identity(cfg,sc)
        valid['pc_partial_validation']={k:v for k,v in p.items() if k!='rows'}
    if (out/'manifests/coverage_validation.json').exists():
        p=read(out/'manifests/coverage_validation.json');assert p['identity']==identity(cfg,sc)
        assert p['candidate_count']==cfg['num_scenes']*4*cfg['pool_size'] and p['heldout_scored_count']==p['candidate_count']*12
        valid['coverage_validation']={k:v for k,v in p.items() if k!='rows'}
        valid['full_experiments_complete']=valid['candidate_completion'] and p['full_completion']
    save(out/'manifests/full_validation.json',valid);print(json.dumps({k:v for k,v in valid.items() if k!='candidate_validation'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
