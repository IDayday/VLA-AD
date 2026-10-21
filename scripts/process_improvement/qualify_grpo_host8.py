"""Real eight-rank host qualification, using the unchanged native GRPO trainer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.process_improvement.common import *


def sources():
    return {name:sha(SOURCE/name) for name in [
        'scripts/process_improvement/train_grpo.py',
        'navsim/agents/recogdrive/recogdrive_diffusion_planner.py',
        'navsim/agents/recogdrive/process_improvement/rng_plan.py',
        'scripts/process_improvement/qualify_grpo_host8.py']}


def main():
    import torch
    import numpy as np
    from scripts.process_improvement.qualify_grpo_partition32 import compare_states
    from scripts.process_improvement.endpoint_state import freeze_endpoint_state
    from scripts.process_improvement.calibrate import immutable
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['source','destination'],required=True);a=p.parse_args()
    own=identity();registration=read(ROOT/'selection/host8_execution_registration.json')
    warm=read(ROOT/'selection/warm_start.json')['chosen']['checkpoints']['42']['path']
    inits={'IL':'IL','RL':'RL','WARM':warm}
    def execute(family,label,extra=()):
        name=f'qual_GRPO_host6_B32_world8_{family}_{label}'
        subprocess.run([PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
            'scripts/process_improvement/train_grpo.py','--qualification','--split','legacy_4k',
            '--steps','4','--global-batch','32','--score-budget','1024','--lr','3e-5',
            '--init',inits[family],'--name',name,*extra],cwd=SOURCE,check=True)
        return ROOT/'runs'/name
    def load(path):return torch.load(path,map_location='cpu',weights_only=False)
    if a.stage=='source':
        assert own['host']==registration['source_host']
        gate=read(ROOT/'tests/native_GRPO_partition32.json');assert gate['status']=='PASS'
        assert gate['train_source_sha256']==sources()['scripts/process_improvement/train_grpo.py']
        comparisons=[]
        for family in inits:
            old=ROOT/'runs'/f'qual_GRPO_B32_{family}_continuous8'/'latest.pt'
            new=execute(family,'continuous8')/'latest.pt'
            error=compare_states(load(old),load(new),True)
            comparisons.append(dict(family=family,old_checkpoint=str(old),old_sha256=sha(old),
                                    new_checkpoint=str(new),new_sha256=sha(new),core_max_error=error))
        paused=execute('IL','cross_resume8',['--stop-after','2'])
        state=freeze_endpoint_state(paused,2)
        immutable(ROOT/'tests/native_GRPO_host6_world8_source.json',dict(
            status='PASS_SOURCE_ONLY_NOT_MIGRATION_ADMISSION',world=8,source_host=own['host'],
            comparisons=comparisons,paused_original_state=state,qualification_source_hashes=sources(),owner=own))
        print('REAL_HOST6_WORLD8_THREE_INITIALIZATIONS_AND_PAUSED_SOURCE_PASS',flush=True);return
    first=read(ROOT/'tests/native_GRPO_host6_world8_source.json')
    assert own['host']==registration['destination_host'] and own['host']!=first['source_host']
    assert first['qualification_source_hashes']==sources()
    run=ROOT/'runs/qual_GRPO_host6_B32_world8_IL_cross_resume8'
    pinned=run/'endpoint_states/2/full.pt'
    assert sha(pinned)==first['paused_original_state']['full_state_sha256']
    assert read(run/'config.json')['owner']['host']==first['source_host']
    execute('IL','cross_resume8',['--resume'])
    reference=ROOT/'runs/qual_GRPO_host6_B32_world8_IL_continuous8/latest.pt'
    result=run/'latest.pt';left,right=load(reference),load(result)
    error=compare_states(left,right,True)
    assert left['config']==right['config'] and left['cursor']==right['cursor']
    assert left['reference_sha256']==right['reference_sha256']
    def exact(x,y):
        if isinstance(x,torch.Tensor):assert torch.equal(x,y)
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y)
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for l,r in zip(x,y):exact(l,r)
        else:assert x==y
    for l,r in zip(left['rank_states'],right['rank_states']):
        for key in ['torch_rng','cuda_rng','numpy_rng','python_rng']:exact(l[key],r[key])
        for key in ['physical_scores','logical_scores','reference_simulations']:
            assert l['counts'][key]==r['counts'][key]
    immutable(ROOT/'tests/native_GRPO_cross_host_resume8.json',dict(
        status='PASS',world=8,global_batch=32,source_host=first['source_host'],destination_host=own['host'],
        source_stage_sha256=sha(ROOT/'tests/native_GRPO_host6_world8_source.json'),
        source_original_step2_sha256=sha(pinned),continuous_step4_sha256=sha(reference),
        cross_host_step4_sha256=sha(result),core_max_error=error,optimizer_exact=True,
        all_rank_RNG_exact=True,cursor_reference_budget_exact=True,
        validated_initializations=['IL','RL','WARM'],qualification_source_hashes=sources(),owner=own,
        scope='Paused world8 placement within this actual qualified host pair only; no world repartition or change of scientific recipe'))
    print('REAL_CROSS_HOST_WORLD8_PARAMETERS_OPTIMIZER_RNG_EXACT_PASS',flush=True)


if __name__=='__main__':main()
