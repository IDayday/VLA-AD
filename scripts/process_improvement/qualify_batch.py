"""Measure real native planner throughput/memory and batched-gradient semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
from scripts.process_improvement.common import *


def main():
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.contracts import NoisePlan
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.objective import objective
    configure();cfg=config()
    student=build_head(cfg['student_init']);teacher=build_head(cfg['teacher_init'],frozen=True)
    rows=membership('legacy_4k')[:32]
    raw=[conditioning(observation(r)) for r in rows]
    plans=[RNGKey(STUDY,42,0,0,r['token'],j,-1).plan('rollin',device='cuda') for j,r in enumerate(rows)]
    records=[]
    for size in [1,2,4,8,16,32]:
        tokens=torch.cat([r[0] for r in raw[:size]])
        inputs=BatchFeature({k:torch.cat([r[1][k] for r in raw[:size]]) for k in raw[0][1]})
        plan=NoisePlan(torch.cat([p.initial for p in plans[:size]]),torch.cat([p.steps for p in plans[:size]]))
        times=[]
        for rep in range(3):
            student.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.time()
            out=objective(student,teacher,tokens,inputs,plan,0);out['loss'].backward();torch.cuda.synchronize()
            if rep:times.append(time.time()-started)
        gradients={name:p.grad.detach().clone() for name,p in student.named_parameters() if p.grad is not None} if size==4 else None
        if size==4:
            student.zero_grad(set_to_none=True)
            for j in range(size):
                own=BatchFeature({k:v[j:j+1] for k,v in inputs.items()})
                scalar=objective(student,teacher,tokens[j:j+1],own,plans[j],0)
                (scalar['loss']/size).backward()
            maximum=0.
            for name,p in student.named_parameters():
                if p.grad is None:continue
                torch.testing.assert_close(p.grad,gradients[name],atol=3e-7,rtol=2e-3)
                maximum=max(maximum,float((p.grad-gradients[name]).abs().max()))
        else:maximum=None
        record=dict(batch=size,mean_seconds=sum(times)/len(times),seconds_per_scene=sum(times)/len(times)/size,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),gradient_partition_max_error=maximum,
                    loss=float(out['loss'].detach()))
        print('BATCH_PROFILE',record,flush=True);records.append(record)
        del out
    write(ROOT/'tests/batch_profile.json',dict(status='PASS',records=records,owner=identity()))


if __name__=='__main__':main()
