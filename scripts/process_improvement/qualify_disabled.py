"""Real old-E3 vs disabled-process loss/gradient/AdamW update parity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import copy
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.trainer import objective as old_objective
    from navsim.agents.recogdrive.process_improvement.objective import objective
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    configure();cfg=config();old=build_head(cfg['student_init']);new=copy.deepcopy(old)
    teacher=build_head(cfg['teacher_init'],frozen=True);records=[]
    optimizers=[torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=3e-5,weight_decay=1e-4) for h in [old,new]]
    for index,row in enumerate(membership('legacy_4k')[:8]):
        tokens,inputs=conditioning(observation(row));plan=RNGKey(STUDY,42,index,0,row['token'],index,-1).plan('rollin',device='cuda')
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        original=old_objective(old,teacher,None,tokens,inputs,experiment='E3',rollout_noise=(plan.initial,plan.steps))
        disabled=objective(new,teacher,tokens,inputs,plan,index)
        assert torch.equal(original['chain'].states,disabled['rollin'].states)
        torch.testing.assert_close(original['opd'],disabled['loss'],atol=1e-11,rtol=1e-6)
        original['opd'].backward();disabled['loss'].backward();gradient_error=0.
        for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
            assert name==other
            if p.grad is None:assert q.grad is None;continue
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-4)
            gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
        for head,optimizer in zip([old,new],optimizers):
            torch.nn.utils.clip_grad_norm_([p for p in head.parameters() if p.requires_grad],1.)
            optimizer.step()
        update_error=0.
        for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
            if name.startswith(('eta.','action_aware_aux_head.')):continue
            update_error=max(update_error,float((p-q).abs().max()))
        assert update_error<=2e-6,update_error
        records.append(dict(token=row['token'],loss_error=float((original['opd']-disabled['loss']).abs()),
                            gradient_error=gradient_error,parameter_error=update_error))
    write(ROOT/'tests/disabled_method.json',dict(status='PASS',records=records,owner=identity()))
    print('DISABLED_METHOD_PARITY_PASS',records,flush=True)


if __name__=='__main__':main()
