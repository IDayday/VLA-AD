"""Real native global-B32 GRPO partition4 parity and exact same-world resume.

This qualifies a new execution partition, not a different RL algorithm. All
scene/noise streams and optimizer budgets remain unchanged. No formal policy
is admitted by a CPU-only test or by a failed floating-point assertion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.process_improvement.common import *


def compare_states(left,value,exact):
    import torch
    assert value['step']==left['step']==4 and value['physical_scores']==left['physical_scores']==1024
    maximum=0.
    for key,tensor in left['full_student'].items():
        if key.startswith('action_aware_aux_head.'):continue
        other=value['full_student'][key]
        if exact:assert torch.equal(tensor,other),key
        else:torch.testing.assert_close(tensor,other,atol=2e-6,rtol=1e-5)
        if key in ['eta.eta_logit','old_policy.eta.eta_logit']:
            # Native EtaFixed(base_eta=1): atanh(1)=+inf, tanh(+inf)=1.
            # Both constants must be exactly identical; no core NaN/Inf exemption.
            assert tensor.shape==other.shape==(1,) and torch.isposinf(tensor).all() and torch.isposinf(other).all(),key
            assert torch.equal(tensor,other) and float(torch.tanh(tensor)[0])==1.,key
            continue
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all() and torch.isfinite(other).all(),key
            maximum=max(maximum,float((tensor-other).abs().max()))
    for key,fields in left['optimizer']['state'].items():
        for field,tensor in fields.items():
            other=value['optimizer']['state'][key][field]
            if isinstance(tensor,torch.Tensor):
                assert torch.isfinite(tensor).all() and torch.isfinite(other).all(),(key,field)
                if exact:assert torch.equal(tensor,other),(key,field)
                else:torch.testing.assert_close(tensor,other,atol=2e-8,rtol=2e-4)
    return maximum


def main():
    import torch
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--reuse-reviewed-completed',action='store_true');args=p.parse_args()
    base = ['scripts/process_improvement/train_grpo.py', '--qualification',
            '--split', 'legacy_4k', '--steps', '4', '--global-batch', '32',
            '--score-budget', '1024', '--lr', '3e-5']
    warm=read(ROOT/'selection/warm_start.json')['chosen']['checkpoints']['42']['path']
    initializations={'IL':'IL','RL':'RL','WARM':warm}
    def command(world, family, label, extra=()):
        return [PYTHON, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}',
                *base, '--init',initializations[family],'--name',f'qual_GRPO_B32_{family}_{label}', *extra]
    def load(family,label):
        return torch.load(ROOT/'runs'/f'qual_GRPO_B32_{family}_{label}'/'latest.pt',map_location='cpu',weights_only=False)
    review=None
    if args.reuse_reviewed_completed:
        review=read(ROOT/'audit/partition32_constant_review.json')
        for source,value in review['unchanged_execution_sources'].items():assert sha(SOURCE/source)==value
        for path,value in review['reused_checkpoints'].items():assert sha(path)==value
    def execute(world,family,label,extra=()):
        path=ROOT/'runs'/f'qual_GRPO_B32_{family}_{label}'/'latest.pt'
        if review and str(path) in review['reused_checkpoints']:
            done=read(path.parent/'complete.json');state=load(family,label)
            assert done['status']=='COMPLETE' and done['steps']==4 and done['physical_scores']==1024
            assert state['config']['world']==world and state['config']['global_batch']==32
            assert state['config']['lr']==3e-5 and state['config']['init_sha256']==review['IL_init_sha256']
            return
        subprocess.run(command(world,family,label,extra),cwd=SOURCE,check=True)
    errors={};checkpoint_ids={}
    for family in initializations:
        for world in [8,4]:execute(world,family,f'continuous{world}')
        left,right=load(family,'continuous8'),load(family,'continuous4')
        errors[family+'_partition4']=compare_states(left,right,False)
        checkpoint_ids[family]=left['config']['init_sha256']
        if family=='IL':
            for extra in [['--stop-after','2'],['--resume']]:
                execute(4,family,'resume4',extra)
            errors['IL_exact_resume4']=compare_states(right,load(family,'resume4'),True)
        del left,right
    write(ROOT/'tests/native_GRPO_partition32.json', dict(
        status='PASS', world_sizes=[4, 8], global_batch=32, actual_optimizer_updates=4,
        physical_scores_per_run=1024, errors=errors, exact_resume_optimizer=True,
        validated_initializations=list(initializations),initial_checkpoint_sha256=checkpoint_ids,
        expected_exact_native_constants=['eta.eta_logit=+inf','old_policy.eta.eta_logit=+inf'],
        reviewed_reuse=review,
        train_source_sha256=sha(SOURCE/'scripts/process_improvement/train_grpo.py'),
        planner_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'),
        rng_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/rng_plan.py'),
        qualification_source_sha256=sha(__file__), owner=identity(),
        scope='Execution-only global32 partition; new local lineages only after short-work drain; no world change on resume'))
    print('REAL_NATIVE_GRPO_B32_PARTITION4_AND_RESUME_PASS', errors, flush=True)


if __name__ == '__main__':
    main()
