"""Preserve native full resume states without changing the qualified trainer.

The trainer replaces latest.pt atomically. A hard link pins the *current inode*,
not the pathname, so later native checkpoints cannot overwrite this archive.
Only paused/completed, exact-step states can be archived. No reconstruction is
performed when a historical full state is already unavailable.
"""
from scripts.process_improvement.common import Path, read, sha


def freeze_endpoint_state(run, step):
    import os
    import torch
    from scripts.process_improvement.calibrate import immutable

    run = Path(run)
    dest = run / 'endpoint_states' / str(step)
    receipt = dest / 'receipt.json'
    target = dest / 'full.pt'
    head = run / f'head_{step}.pt'
    if receipt.exists():
        value = read(receipt)
        if sha(target) != value['full_state_sha256'] or sha(head) != value['head_sha256']:
            raise ValueError('Archived endpoint state or policy changed')
        return value
    dest.mkdir(parents=True, exist_ok=True)
    # link() never overwrites an existing file. An interrupted archive may reuse
    # its pinned file, but must still pass the metadata checks below.
    if not target.exists():
        os.link(run / 'latest.pt', target)
    state = torch.load(target, map_location='meta', weights_only=False)
    policy = torch.load(head, map_location='meta', weights_only=False)
    required = {'full_student', 'student', 'optimizer', 'rank_states', 'config',
                'step', 'cursor', 'physical_scores', 'reference_sha256',
                'scheduler_state', 'amp_state', 'history', 'seconds'}
    if not required.issubset(state):
        raise ValueError('Incomplete native full resume state')
    if state['step'] != step or policy['step'] != step:
        target.unlink()
        raise ValueError('Never relabel a different-step full resume state')
    cfg = state['config']
    if (cfg != policy['config'] or state['cursor'] != policy['cursor'] or
            state['physical_scores'] != policy['physical_scores'] or
            len(state['rank_states']) != cfg['world']):
        raise ValueError('Full resume state and immutable policy identities differ')
    if state['cursor'] != step * cfg['global_batch']:
        raise ValueError('Unexpected native sampler cursor')
    for own in state['rank_states']:
        if not {'torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng', 'counts'}.issubset(own):
            raise ValueError('Missing per-rank native RNG state')
    value = dict(status='PRESERVED_ORIGINAL_FULL_STATE', step=step,
                 full_state_path=str(target), full_state_sha256=sha(target),
                 head_sha256=sha(head), world=cfg['world'],
                 cursor=state['cursor'], physical_scores=state['physical_scores'],
                 reference_sha256=state['reference_sha256'],
                 generator_sha256=sha(__file__),
                 semantics='Original paused/completed latest.pt inode pinned before native atomic replacement; no reconstructed optimizer or RNG')
    immutable(receipt, value)
    return value
