"""Measure reference-versus-student suffix mismatch, without fitting to audit data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.projection_guard import TargetBuilder
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from navsim.agents.recogdrive.process_improvement.candidate_value import evaluate_kernels
    from navsim.agents.recogdrive.process_improvement.statistics import parent_mean_bootstrap
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    p.add_argument('--config')
    p.add_argument('--limit',type=int,default=128);a=p.parse_args();configure();plan=diagnostic_plan();cfg=config(a.config or SOURCE/plan['config'])
    dest=ROOT/'diagnostics/suffix_mismatch'/a.name
    if (dest/'results.json').exists():raise FileExistsError('Preserve completed suffix diagnostic')
    student=build_head(a.checkpoint,frozen=True);reference=build_head(cfg['teacher_init'],frozen=True)
    sa=NativeAdapter(student);ra=NativeAdapter(reference);rows=membership(plan['confirm_split'])[:a.limit]
    scorer=CanonicalScorer(workers=4,metric_paths={r['token']:r['metric_cache_path'] for r in rows})
    builder=TargetBuilder(ra,scorer,cfg);records=[];started=time.time()
    for row in rows:
        tokens,inputs=conditioning(observation(row));cache=load_cache(row['metric_cache_path'])
        key=RNGKey(STUDY,42,0,0,row['token'],0,-1);rollin=sa.sample_rollin(tokens,inputs,key.plan('rollin',device='cuda'))
        with torch.no_grad():rc=ra.prepare_conditioning(tokens,inputs);sc=sa.prepare_conditioning(tokens,inputs)
        own=[]
        for index in range(5):
            state=rollin.states[:,index];sk=key.at_step(index)
            target=builder.build(state,rc,index,sk,cache)
            with torch.no_grad():stats=ra.transition_stats(state,index,rc)
            ref,_=evaluate_kernels(ra,rc,stats,[stats.mean,target.target],sk,16,'diagnostic_test',scorer,row['token'],cache)
            actual,_=evaluate_kernels(sa,sc,stats,[stats.mean,target.target],sk,16,'diagnostic_test',scorer,row['token'],cache)
            own.append(dict(index=index,accepted=target.accepted,reference_suffix=ref.tolist(),student_suffix=actual.tolist(),
                            reference_gain=float((ref[1,:,-1]-ref[0,:,-1]).mean()),
                            student_gain=float((actual[1,:,-1]-actual[0,:,-1]).mean()),
                            reference_value_variance=float(ref[0,:,-1].var(ddof=1)),student_value_variance=float(actual[0,:,-1].var(ddof=1))))
        record=dict(token=row['token'],driver_record=row['driver_record'],records=own)
        write(dest/'scenes'/f"{row['token']}.json",record);records.append(record)
    mean=[[np.mean([v[k] for v in r['records']]) for k in ['reference_gain','student_gain','reference_value_variance','student_value_variance']] for r in records]
    result=parent_mean_bootstrap(mean,[r['driver_record'] for r in records])
    result.update(status='COMPLETE',fields=['reference_gain','student_gain','reference_value_variance','student_value_variance'],
                  checkpoint_sha256=sha(a.checkpoint),config_sha256=digest(cfg),budget=scorer.counter.state_dict(),seconds=time.time()-started,
                  caveat='Fixed target and current-step reference noise kernel; only continuation policy changes. Diagnostic, not training or deployment evaluation.')
    scorer.close();write(dest/'results.json',result);print('SUFFIX_MISMATCH',result,flush=True)


if __name__=='__main__':main()
