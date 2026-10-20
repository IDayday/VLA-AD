"""Real native global-B32 GRPO partition4 parity and exact same-world resume.

This qualifies a new execution partition, not a different RL algorithm. All
scene/noise streams and optimizer budgets remain unchanged. No formal policy
is admitted by a CPU-only test or by a failed floating-point assertion.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.process_improvement.common import *


def main():
    import torch
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
    def compare(left,value,exact):
        assert value['step'] == left['step'] == 4
        assert value['physical_scores'] == left['physical_scores'] == 1024
        maximum = 0.
        for key, tensor in left['full_student'].items():
            if key.startswith('action_aware_aux_head.'):
                continue  # Explicitly inactive, not part of the native planner.
            other = value['full_student'][key]
            if exact:
                assert torch.equal(tensor, other), key
            else:
                torch.testing.assert_close(tensor, other, atol=2e-6, rtol=1e-5)
            if tensor.is_floating_point():
                assert torch.isfinite(tensor).all() and torch.isfinite(other).all()
                maximum = max(maximum, float((tensor-other).abs().max()))
        for key, fields in left['optimizer']['state'].items():
            for field, tensor in fields.items():
                other = value['optimizer']['state'][key][field]
                if isinstance(tensor, torch.Tensor):
                    if exact:
                        assert torch.equal(tensor, other), (key, field)
                    else:
                        torch.testing.assert_close(tensor, other, atol=2e-8, rtol=2e-4)
        return maximum
    errors={};checkpoint_ids={}
    for family in initializations:
        for world in [8,4]:subprocess.run(command(world,family,f'continuous{world}'),cwd=SOURCE,check=True)
        left,right=load(family,'continuous8'),load(family,'continuous4')
        errors[family+'_partition4']=compare(left,right,False)
        checkpoint_ids[family]=left['config']['init_sha256']
        if family=='IL':
            for extra in [['--stop-after','2'],['--resume']]:
                subprocess.run(command(4,family,'resume4',extra),cwd=SOURCE,check=True)
            errors['IL_exact_resume4']=compare(right,load(family,'resume4'),True)
        del left,right
    write(ROOT/'tests/native_GRPO_partition32.json', dict(
        status='PASS', world_sizes=[4, 8], global_batch=32, actual_optimizer_updates=4,
        physical_scores_per_run=1024, errors=errors, exact_resume_optimizer=True,
        validated_initializations=list(initializations),initial_checkpoint_sha256=checkpoint_ids,
        train_source_sha256=sha(SOURCE/'scripts/process_improvement/train_grpo.py'),
        planner_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'),
        rng_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/rng_plan.py'),
        qualification_source_sha256=sha(__file__), owner=identity(),
        scope='Execution-only global32 partition; new local lineages only after short-work drain; no world change on resume'))
    print('REAL_NATIVE_GRPO_B32_PARTITION4_AND_RESUME_PASS', errors, flush=True)


if __name__ == '__main__':
    main()
