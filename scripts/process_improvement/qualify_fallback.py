"""Real batched exact rejection fallback, including gradients and an update."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.process_improvement.common import *


def main():
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.contracts import NoisePlan
    from navsim.agents.recogdrive.process_improvement.objective import objective_from_rollin
    configure();cfg=config();rows=membership('legacy_4k')[:4]
    moved=[conditioning(observation(r)) for r in rows];tokens=torch.cat([v[0] for v in moved])
    inputs=BatchFeature({k:torch.cat([v[1][k] for v in moved]) for k in moved[0][1]})
    student=build_head(cfg['student_init']);teacher=build_head(cfg['teacher_init'],frozen=True);teacher_hash=state_hash(teacher)
    plans=[RNGKey(STUDY,42,0,0,r['token'],j,-1).plan('rollin',device='cuda') for j,r in enumerate(rows)]
    noise=NoisePlan(torch.cat([v.initial for v in plans]),torch.cat([v.steps for v in plans]));adapter=NativeAdapter(student)
    rollin=adapter.sample_rollin(tokens,inputs,noise);reference=NativeAdapter(teacher);single=[]
    for j in range(4):
        own=BatchFeature({k:v[j:j+1] for k,v in inputs.items()})
        with torch.no_grad():context=reference.prepare_conditioning(tokens[j:j+1],own)
        with torch.no_grad():single.append(reference.transition_stats(rollin.states[j:j+1,j],j,context).mean)
    single=torch.cat(single);index=torch.arange(4,device='cuda')
    ordinary=objective_from_rollin(student,teacher,rollin,tokens,inputs,0);ordinary['loss'].backward()
    gradients={k:p.grad.clone() for k,p in student.named_parameters() if p.grad is not None};student.zero_grad(set_to_none=True)
    old=objective_from_rollin(student,teacher,rollin,tokens,inputs,0,selected_target=single,selected_index=index)
    old['loss'].backward();old_error=max(float((p.grad-gradients[k]).abs().max()) for k,p in student.named_parameters() if p.grad is not None)
    student.zero_grad(set_to_none=True)
    fixed=objective_from_rollin(student,teacher,rollin,tokens,inputs,0,selected_target=single,selected_index=index,
                                selected_mask=torch.zeros(4,device='cuda',dtype=torch.bool))
    fixed['loss'].backward();assert torch.equal(ordinary['targets'],fixed['targets']) and torch.equal(ordinary['loss'],fixed['loss'])
    for k,p in student.named_parameters():
        if p.grad is not None:assert torch.equal(p.grad,gradients[k]),k
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5,weight_decay=cfg['weight_decay'])
    optimizer.step();assert not torch.equal(adapter.sample_rollin(tokens,inputs,noise,1).states,rollin.states)
    assert state_hash(teacher)==teacher_hash
    write(ROOT/'tests/batched_exact_fallback.json',dict(status='PASS',global_batch=4,
          legacy_unmasked_gradient_roundoff=old_error,masked_loss_error=0.,masked_gradient_error=0.,
          actual_student_update=True,teacher_unchanged=True,owner=identity()))
    print('REAL_BATCHED_EXACT_FALLBACK_PASS',old_error,flush=True)


if __name__=='__main__':main()
