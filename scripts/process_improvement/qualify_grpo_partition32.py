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
    base = ['scripts/process_improvement/train_grpo.py', '--qualification', '--init', 'IL',
            '--split', 'legacy_4k', '--steps', '4', '--global-batch', '32',
            '--score-budget', '1024', '--lr', '3e-5']
    def command(world, label, extra=()):
        return [PYTHON, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world}',
                *base, '--name', 'qual_GRPO_B32_' + label, *extra]
    for cmd in [command(8, 'continuous8'), command(4, 'continuous4'),
                command(4, 'resume4', ['--stop-after', '2']), command(4, 'resume4', ['--resume'])]:
        subprocess.run(cmd, cwd=SOURCE, check=True)
    states = {name: torch.load(ROOT/'runs'/('qual_GRPO_B32_' + name)/'latest.pt',
                              map_location='cpu', weights_only=False)
              for name in ['continuous8', 'continuous4', 'resume4']}
    reference = states['continuous8']
    errors = {}
    for label in ['continuous4', 'resume4']:
        value = states[label]
        assert value['step'] == reference['step'] == 4
        assert value['physical_scores'] == reference['physical_scores'] == 1024
        left = reference if label == 'continuous4' else states['continuous4']
        maximum = 0.
        for key, tensor in left['full_student'].items():
            if key.startswith('action_aware_aux_head.'):
                continue  # Explicitly inactive, not part of the native planner.
            other = value['full_student'][key]
            if label == 'resume4':
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
                    if label == 'resume4':
                        assert torch.equal(tensor, other), (key, field)
                    else:
                        torch.testing.assert_close(tensor, other, atol=2e-8, rtol=2e-4)
        errors[label] = maximum
    write(ROOT/'tests/native_GRPO_partition32.json', dict(
        status='PASS', world_sizes=[4, 8], global_batch=32, actual_optimizer_updates=4,
        physical_scores_per_run=1024, errors=errors, exact_resume_optimizer=True,
        train_source_sha256=sha(SOURCE/'scripts/process_improvement/train_grpo.py'),
        planner_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'),
        rng_source_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/rng_plan.py'),
        qualification_source_sha256=sha(__file__), owner=identity(),
        scope='Execution-only global32 partition; new local lineages only after short-work drain; no world change on resume'))
    print('REAL_NATIVE_GRPO_B32_PARTITION4_AND_RESUME_PASS', errors, flush=True)


if __name__ == '__main__':
    main()
