"""Real fixed-subset local ablations; never reopen selection using their results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import numpy as np
from collections import defaultdict
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.projection_guard import TargetBuilder
    from navsim.agents.recogdrive.process_improvement.candidate_value import evaluate_kernels
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from navsim.agents.recogdrive.process_improvement.statistics import parent_mean_bootstrap,local_noise_bootstrap
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=64);p.add_argument('--split');p.add_argument('--config');p.add_argument('--name')
    a=p.parse_args();configure();plan=diagnostic_plan()
    cfg=config(SOURCE/(a.config or plan['config']))
    cfg.update(candidate_count=16,fit_replicates=8,guard_replicates=8)
    split=a.split or plan['train_split'];rows=membership(split)[:a.limit]
    dest=ROOT/'diagnostics'/(a.name or 'local_parent_ablations')
    if (dest/'results.json').exists():raise FileExistsError('Preserve completed ablation')
    cache_enabled=frozen_suffix_cache_available()
    if cache_enabled:
        from navsim.agents.recogdrive.process_improvement.frozen_suffix import FrozenSuffixAdapter
        NativeAdapter=FrozenSuffixAdapter
    teacher=build_head(cfg['teacher_init'],frozen=True);reference=NativeAdapter(teacher)
    source=build_head(HISTORY/'runs/train_E3_42/latest.pt',frozen=True);roll=NativeAdapter(source)
    scorer=CanonicalScorer(workers=4,metric_paths={r['token']:r['metric_cache_path'] for r in rows})
    write(dest/'configuration.json',dict(scope='Predefined first64 parent-hash D_fit diagnostic scenes, historical OPD rollin, local diagnostic only',split=split,
          scene_tokens=[r['token'] for r in rows],config=cfg,frozen_suffix_cache=cache_enabled,teacher_sha256=sha(cfg['teacher_init']),
          source_checkpoint_sha256=sha(HISTORY/'runs/train_E3_42/latest.pt'),selection_not_changed=True,owner=identity()))
    by_group=defaultdict(list)
    for row in rows:
        tokens,inputs=conditioning(observation(row));key=RNGKey(STUDY,42,0,0,row['token'],0,-1)
        chain=roll.sample_rollin(tokens,inputs,key.plan('rollin',device='cuda'));cache=load_cache(row['metric_cache_path'])
        with torch.no_grad():context=reference.prepare_conditioning(tokens,inputs)
        records=[]
        for index in range(5):
            with torch.no_grad():stats=reference.transition_stats(chain.states[:,index],index,context)
            for bank in range(2):
                for mode in ['antithetic','iid']:
                    own=dict(cfg,candidate_mode=mode);builder=TargetBuilder(reference,scorer,own)
                    for variant in ['PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD']:
                        target=builder.build(chain.states[:,index],context,index,key.at_step(index),cache,bank=bank,
                                             ablation=None if variant=='PROCESS' else variant)
                        test,_=evaluate_kernels(reference,context,stats,[stats.mean,target.target],key.at_step(index),16,
                                                'diagnostic_test',scorer,row['token'],cache,bank)
                        delta=test[1,:,-1]-test[0,:,-1]
                        records.append(dict(index=index,bank=bank,mode=mode,variant=variant,test_delta=delta.tolist(),
                                            accepted=target.accepted,shift_RMS=target.statistics['realized_shift_rms'],
                                            fit_value_range=target.statistics['value_range'],ESS=target.statistics['ess']))
        write(dest/'scenes'/f"{row['token']}.json",dict(token=row['token'],parent=row['driver_record'],records=records))
        for mode in ['antithetic','iid']:
            for variant in ['PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD']:
                data=sorted([r for r in records if r['mode']==mode and r['variant']==variant],key=lambda r:(r['index'],r['bank']))
                noise=np.asarray([r['test_delta'] for r in data]).reshape(5,2,16).mean(0)[...,None]
                by_group[(mode,variant)].append((row['driver_record'],noise,
                    [np.mean([r[k] for r in data]) for k in ['accepted','shift_RMS','fit_value_range','ESS']]))
        print('REAL_LOCAL_ABLATION',row['token'],scorer.counter.state_dict(),flush=True)
    results=[]
    for (mode,variant),data in sorted(by_group.items()):
        result=local_noise_bootstrap([r[1] for r in data],[r[0] for r in data])
        result.update(mode=mode,variant=variant,diagnostics=parent_mean_bootstrap([r[2] for r in data],[r[0] for r in data]))
        results.append(result)
    write(dest/'results.json',dict(status='COMPLETE',results=results,budget=scorer.counter.state_dict(),
          scope='64-scene local stability/negative controls; not a formal student or a substitute for independent confirmation'))
    scorer.close();print('LOCAL_ABLATIONS_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
