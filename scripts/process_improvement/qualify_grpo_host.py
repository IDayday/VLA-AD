"""Real host-parity and cross-host same-world native resume qualification.

No trainer, sampler, reward, loss, RNG or scientific budget is changed. Source
stage reproduces the existing three four-update qualifications on the newly
available host and pauses a fourth IL lineage at update2. Destination stage
resumes that actual complete state on a different host, to update4.
"""
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
        'scripts/process_improvement/qualify_grpo_host.py']}


def main():
    import torch
    import numpy as np
    from scripts.process_improvement.calibrate import immutable
    from scripts.process_improvement.qualify_grpo_partition32 import compare_states
    from scripts.process_improvement.endpoint_state import freeze_endpoint_state
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['source','destination'],required=True);a=p.parse_args()
    own=identity();registration=read(ROOT/'selection/host4_execution_registration.json')
    warm=read(ROOT/'selection/warm_start.json')['chosen']['checkpoints']['42']['path']
    inits={'IL':'IL','RL':'RL','WARM':warm}
    def execute(family,label,extra=()):
        name=f'qual_GRPO_host4_{family}_{label}'
        command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                 'scripts/process_improvement/train_grpo.py','--qualification','--split','legacy_4k',
                 '--steps','4','--global-batch','32','--score-budget','1024','--lr','3e-5',
                 '--init',inits[family],'--name',name,*extra]
        subprocess.run(command,cwd=SOURCE,check=True)
        return ROOT/'runs'/name
    def load(path):return torch.load(path,map_location='cpu',weights_only=False)
    if a.stage=='source':
        assert own['host']==registration['source_host']
        old_gate=read(ROOT/'tests/native_GRPO_partition32.json')
        assert old_gate['status']=='PASS'
        assert old_gate['train_source_sha256']==sources()['scripts/process_improvement/train_grpo.py']
        comparisons=[]
        for family in inits:
            old=ROOT/'runs'/f'qual_GRPO_B32_{family}_continuous4'/'latest.pt'
            new=execute(family,'continuous4')/'latest.pt'
            error=compare_states(load(old),load(new),True)
            comparisons.append(dict(family=family,old_checkpoint=str(old),old_sha256=sha(old),
                                    new_checkpoint=str(new),new_sha256=sha(new),core_max_error=error))
        paused=execute('IL','cross_resume4',['--stop-after','2'])
        preserved=freeze_endpoint_state(paused,2)
        immutable(ROOT/'tests/native_GRPO_host4_source.json',dict(
            status='PASS_SOURCE_ONLY_NOT_MIGRATION_ADMISSION',source_host=own['host'],world=4,
            comparisons=comparisons,paused_original_state=preserved,
            qualification_source_hashes=sources(),owner=own,
            interpretation='Three actual cross-host four-update replicas agree exactly; actual cross-host resume still pending'))
        print('REAL_NEW_HOST_THREE_INITIALIZATIONS_AND_PAUSED_SOURCE_PASS',flush=True)
        return
    first=read(ROOT/'tests/native_GRPO_host4_source.json')
    assert own['host'] in registration['destination_hosts'] and own['host']!=first['source_host']
    assert first['qualification_source_hashes']==sources()
    run=ROOT/'runs/qual_GRPO_host4_IL_cross_resume4'
    state_before=run/'endpoint_states/2/full.pt'
    assert sha(state_before)==first['paused_original_state']['full_state_sha256']
    before=read(run/'config.json')['owner']
    assert before['host']==first['source_host']
    execute('IL','cross_resume4',['--resume'])
    reference=ROOT/'runs/qual_GRPO_host4_IL_continuous4/latest.pt'
    result=run/'latest.pt';left,right=load(reference),load(result)
    error=compare_states(left,right,True)
    assert left['config']==right['config']
    assert left['cursor']==right['cursor'] and left['reference_sha256']==right['reference_sha256']
    def exact(l,r):
        if isinstance(l,torch.Tensor):assert torch.equal(l,r)
        elif isinstance(l,np.ndarray):assert np.array_equal(l,r)
        elif isinstance(l,(list,tuple)):
            assert len(l)==len(r)
            for x,y in zip(l,r):exact(x,y)
        elif isinstance(l,dict):
            assert l.keys()==r.keys()
            for k in l:exact(l[k],r[k])
        else:assert l==r
    for x,y in zip(left['rank_states'],right['rank_states']):
        for key in ['torch_rng','cuda_rng','numpy_rng','python_rng']:exact(x[key],y[key])
        for key in ['physical_scores','logical_scores','reference_simulations']:
            assert x['counts'][key]==y['counts'][key]
    immutable(ROOT/'tests/native_GRPO_cross_host_resume.json',dict(
        status='PASS',world=4,global_batch=32,source_host=first['source_host'],
        destination_host=own['host'],source_stage_sha256=sha(ROOT/'tests/native_GRPO_host4_source.json'),
        source_original_step2_sha256=sha(state_before),continuous_step4_sha256=sha(reference),
        cross_host_step4_sha256=sha(result),core_max_error=error,
        optimizer_exact=True,all_rank_RNG_exact=True,cursor_reference_budget_exact=True,
        validated_initializations=['IL','RL','WARM'],qualification_source_hashes=sources(),owner=own,
        scope='Execution-only host4 admission for paused world4 lineages; no elastic world change, active-job migration or altered scientific recipe'))
    print('REAL_CROSS_HOST_WORLD4_RESUME_PARAMETERS_OPTIMIZER_RNG_EXACT_PASS',flush=True)


if __name__=='__main__':main()
