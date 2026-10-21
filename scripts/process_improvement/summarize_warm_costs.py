"""Recover measured selected warm-prefix costs, never later-epoch proxies.

The selected warm runs are single-GPU. Their exact post-update progress entries
retain cumulative elapsed time across resumes even after live ledgers advance.
For a multi-rank run this script refuses to extrapolate rank-zero timing.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from scripts.process_improvement.common import *


def main():
    import torch
    lock=ROOT/'selection/warm_start.json';selected=read(lock)
    if selected['status']!='LOCKED':raise ValueError('Warm recipe must be locked')
    chosen=selected['chosen'];records=[]
    if set(chosen['checkpoints'])!={'42','43','44'}:raise ValueError('Three seed-specific warm policies required')
    for seed in [42,43,44]:
        source=chosen['checkpoints'][str(seed)];checkpoint=Path(source['path'])
        if sha(checkpoint)!=source['sha256']:raise ValueError('Selected warm checkpoint changed')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        step=payload['step'];run=checkpoint.parent;cfg=read(run/'config.json')
        if step!=chosen['step'] or payload['config']['seed']!=seed or cfg['seed']!=seed:
            raise ValueError('Warm policy seed/endpoint identity mismatch')
        if cfg['world']!=1 or payload['config']['world']!=1:
            raise ValueError('Cannot impute multi-rank GPU cost from a rank-zero progress entry')
        progress=run/'progress.json';entries=[p for p in read(progress) if p['step']==step]
        if len(entries)!=1:raise ValueError('No unique actual warm-prefix time measurement')
        entry=entries[0]
        if entry['scene_exposures']!=payload['cursor'] or entry['physical_scores']!=payload['physical_scores']:
            raise ValueError('Observed prefix does not match selected checkpoint')
        native=ROOT/'runs'/f'GRPO_WARM_{seed}'/'config.json'
        if native.exists() and read(native)['init_sha256']!=source['sha256']:
            raise ValueError('Native warm branch uses a different initialization')
        records.append(dict(training_seed=seed,run=run.name,checkpoint=str(checkpoint),
            checkpoint_sha256=source['sha256'],step=step,scene_exposures=entry['scene_exposures'],world=1,
            physical_scores=entry['physical_scores'],logical_scores=entry['logical_scores'],
            observed_cumulative_training_seconds=entry['seconds'],GPU_hours=entry['seconds']/3600,
            source_progress=str(progress),source_progress_sha256=sha(progress),observed_entry=entry,
            source_config_sha256=sha(run/'config.json'),
            peak_GPU_bytes=None,peak_scope='Not recorded at this prefix; never substitute a later peak',
            prefix_teacher_DiT_calls=None,prefix_call_scope='Not separately recorded at this prefix; no later-ledger substitution'))
        del payload
    result=dict(status='MEASURED_SELECTED_SINGLE_GPU_PREFIXES',generator_sha256=sha(__file__),
        selection_sha256=sha(lock),protocol_sha256=sha(ROOT/'protocol.yaml'),records=records,
        mean_inherited_training_GPU_hours=sum(r['GPU_hours'] for r in records)/len(records),
        timing_scope='Actual cumulative post-update elapsed time, world1, including earlier resumed training segments; excludes initial setup and terminal serialization/evaluation. Not a whole-job lease.',
        total_compute_scope='Add the matching seed warm prefix to its native phase. Also report LR/recipe selection, full two-epoch comparisons, shared VLM/cache and evaluations separately; this prefix is not total research cost.',
        interpretation='No timing interpolation, full-two-epoch proxy, other-seed initialization or unrecorded peak/call imputation')
    write(ROOT/'analysis/warm_prefix_costs.json',result)
    write(REPORTS/'tables/warm_prefix_costs.json',result)
    print('ACTUAL_WARM_PREFIX_GPU_HOURS',[(r['training_seed'],r['GPU_hours']) for r in records],flush=True)


if __name__=='__main__':main()
