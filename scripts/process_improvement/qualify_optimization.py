"""Real FP32 loss/gradient/AdamW parity for scoped observation-linear reuse."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
import time
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.context_cache import context_cache
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.objective import objective
    configure();cfg=config();student=build_head(cfg['student_init']);copy_student=copy.deepcopy(student)
    teacher=build_head(cfg['teacher_init'],frozen=True)
    rows=membership('legacy_4k')[:3];records=[]
    for j,row in enumerate(rows):
        tokens,inputs=conditioning(observation(row));key=RNGKey(STUDY,42,j,0,row['token'],j,-1)
        plan=key.plan('rollin',device='cuda')
        for h in [student,copy_student]:h.zero_grad(set_to_none=True)
        start=time.time();ordinary=objective(student,teacher,tokens,inputs,plan,j);ordinary['loss'].backward();torch.cuda.synchronize();base_time=time.time()-start
        start=time.time()
        with context_cache(copy_student) as student_stats,context_cache(teacher) as teacher_stats:
            optimized=objective(copy_student,teacher,tokens,inputs,plan,j);optimized['loss'].backward()
        torch.cuda.synchronize();cached_time=time.time()-start
        assert torch.equal(ordinary['rollin'].states,optimized['rollin'].states)
        assert torch.equal(ordinary['predictions'],optimized['predictions'])
        assert torch.equal(ordinary['loss'],optimized['loss'])
        maximum=0.
        for (name,p),(name2,q) in zip(student.named_parameters(),copy_student.named_parameters()):
            assert name==name2
            if p.grad is None:assert q.grad is None;continue
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-4)
            maximum=max(maximum,float((p.grad-q.grad).abs().max()))
        opts=[torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=3e-5,weight_decay=1e-4) for h in [student,copy_student]]
        for opt in opts:opt.step()
        update_error=max(float((p-q).abs().max()) for p,q in zip(student.parameters(),copy_student.parameters()))
        # Small roundoff in accumulated gradients may be amplified by Adam near
        # zero; this is explicitly measured rather than assumed bit-identical.
        assert update_error <= 2e-6,update_error
        records.append(dict(token=row['token'],forward_error=0.,loss_error=0.,gradient_max_error=maximum,
                            parameter_update_max_error=update_error,ordinary_seconds=base_time,cached_seconds=cached_time,
                            student_cache=student_stats,teacher_cache=teacher_stats))
        # Reset to the exact same current native head for the next independent case.
        copy_student.load_state_dict(student.state_dict(),strict=True)
    write(ROOT/'tests/context_cache.json',dict(status='PASS',records=records,scope='FP32 native head only; one process/thread; no cross-update retained graph',owner=identity()))
    print('CONTEXT_CACHE_PASS',records,flush=True)


if __name__=='__main__':main()
