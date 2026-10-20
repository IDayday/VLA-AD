"""Real accepted-target gradient integration, explicitly signal-conditioned.

The first lexicographic D_fit case accepted by the locked diagnostic recipe is
used only to exercise the nonzero process target branch. This is not an
unbiased effect estimate, hyperparameter search, or formal trained student.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.projection_guard import TargetBuilder
    from navsim.agents.recogdrive.process_improvement.objective import objective_from_rollin
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from scripts.opd.scoring import load_cache
    configure();cfg=config();case=None
    for row in sorted(membership('diagnostic_train'),key=lambda r:r['token']):
        result=read(ROOT/'diagnostics/local_train_grid/scenes'/f"{row['token']}.json")
        possible=[r for r in result['records'] if r['source']=='IL' and r['bank']==0 and r['temperature_base']==.02 and r['rho']==.5 and r['accepted']]
        if possible:case=(row,sorted(possible,key=lambda r:r['index'])[0]);break
    if case is None:raise ValueError('No accepted real D_fit target to exercise; do not fake an acceptance')
    row,original=case;cfg.update(candidate_count=16,fit_replicates=8,guard_replicates=8,temperature=.02,rho=.5)
    student=build_head(cfg['student_init']);teacher=build_head(cfg['teacher_init'],frozen=True)
    before=state_hash(teacher);tokens,inputs=conditioning(observation(row));index=original['index']
    key=RNGKey(STUDY,42,0,0,row['token'],0,-1);noise=key.plan('rollin',device='cuda')
    adapter=NativeAdapter(student);rollin=adapter.sample_rollin(tokens,inputs,noise)
    reference=NativeAdapter(teacher)
    with torch.no_grad():context=reference.prepare_conditioning(tokens,inputs)
    scorer=CanonicalScorer();target=TargetBuilder(reference,scorer,cfg).build(rollin.states[:,index],context,index,key.at_step(index),load_cache(row['metric_cache_path']),trace=True)
    assert target.accepted and not torch.equal(target.target,target.reference_mean)
    ordinary=objective_from_rollin(student,teacher,rollin,tokens,inputs,0)
    ordinary['loss'].backward();old_grad={k:p.grad.clone() for k,p in student.named_parameters() if p.grad is not None}
    student.zero_grad(set_to_none=True)
    improved=objective_from_rollin(student,teacher,rollin,tokens,inputs,0,selected_target=target.target,selected_index=torch.tensor([index],device='cuda'))
    improved['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    difference=float(sum((p.grad-old_grad[k]).square().sum() for k,p in student.named_parameters() if p.grad is not None).sqrt())
    assert difference>0
    assert any(p.grad is not None and p.grad.norm()>0 for k,p in student.named_parameters() if k.startswith('feature_encoder.'))
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=3e-5,weight_decay=cfg['weight_decay'])
    norm=float(torch.nn.utils.clip_grad_norm_(student.parameters(),1.));optimizer.step()
    updated=adapter.sample_rollin(tokens,inputs,noise,1)
    assert not torch.equal(rollin.states,updated.states)
    assert state_hash(teacher)==before and all(p.grad is None for p in teacher.parameters())
    record=dict(status='PASS',token=row['token'],index=index,guard_gain=target.statistics['guard_gain'],
                selected_target_shift=float((target.target-target.reference_mean).norm()),
                ordinary_vs_process_gradient_L2=difference,gradient_norm=norm,
                ordinary_loss=float(ordinary['loss']),process_loss=float(improved['loss']),
                student_chain_refreshed=True,teacher_unchanged=True,budget=scorer.counter.state_dict(),owner=identity(),
                scope='Signal-conditioned engineering integration only; selection on D_fit is not a performance result')
    save(ROOT/'tests/accepted_target_trace.pt',dict(target=target,record=record))
    write(ROOT/'tests/accepted_target_gradient.json',record);print('REAL_ACCEPTED_TARGET_GRADIENT_PASS',record,flush=True)


if __name__=='__main__':main()
