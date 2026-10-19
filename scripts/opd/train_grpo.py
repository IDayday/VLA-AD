"""Native DiffGRPO with optional frozen-teacher generation-level OPD.

One scene/group at a time with four accumulated scenes per optimizer update.
No offline action is ever substituted for the student's own group rollout.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,signal
from scripts.opd.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.opd.trainer import configure_native_grpo,grpo_with_opd
    p=argparse.ArgumentParser();p.add_argument('--exp',default='E6');p.add_argument('--stage',choices=['smoke','train'],default='smoke')
    p.add_argument('--variant',choices=['IL_GRPO','E5_GRPO','IL_JOINT'],default='IL_JOINT');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--name');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('E6 uses one GPU per independent training seed; the E1-E5 runner provides the validated DDP path')
    configure();cfg=config();steps=a.steps or (4 if a.stage=='smoke' else read(ROOT/'selection/E6_budget.json')['steps'])
    if a.stage=='train' and not read(ROOT/'selection/E6_admission.json')['admitted']:raise ValueError('E6 scientific admission has not passed')
    if a.stage=='train' and (steps!=read(ROOT/'selection/E6_budget.json')['steps'] or a.seed not in cfg['train_seeds']):raise ValueError('E6 formal budget or seed changed')
    name=a.name or f'E6_{a.variant}_{a.seed}_{a.stage}';dest=ROOT/'runs'/name
    if (dest/'complete.json').exists():raise ValueError('Completed experiment cannot be overwritten')
    if (dest/'config.json').exists() and not a.resume:raise ValueError('Partial run needs --resume')
    initial=cfg['student_init'] if a.variant!='E5_GRPO' else str(ROOT/'runs'/f'train_E5_{a.seed}'/'latest.pt')
    actual=dict(experiment='E6',variant=a.variant,stage=a.stage,seed=a.seed,steps=steps,lr=a.lr,global_batch=4,group_size=8,
        initial_checkpoint=initial,initial_checkpoint_sha256=sha(initial),native_BC=.1,external_anchor=0.,
        OPD_initial_weight=.1 if a.variant=='IL_JOINT' else 0.,annealing='linear to0 across all updates',config_sha256=digest(cfg))
    seeds(a.seed);student=build_head(initial);teacher=build_head(cfg['teacher_init'],frozen=True)
    data=rows('train');student=configure_native_grpo(student,data,cfg['anchor_init']);student.eval()
    teacher_hash=state_hash(teacher);reference_hash=state_hash(student.old_policy)
    params=[p for p in student.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=cfg['weight_decay'])
    generator=np.random.default_rng(a.seed);schedule=[]
    while len(schedule)<steps*4:schedule.extend(generator.permutation(len(data)).tolist())
    start=0;history=[];elapsed_before=0.
    if a.resume:
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        if state['config']!=actual:raise ValueError('E6 resume configuration changed')
        student.load_state_dict(state['full_student'],strict=True);optimizer.load_state_dict(state['optimizer']);start=state['step'];history=state['history']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        elapsed_before=state['seconds']
    write(dest/'config.json',dict(actual,owner=identity()));write(dest/'training_schedule.json',[data[i]['token'] for i in schedule[:steps*4]])
    stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True));began=time.time()
    def checkpoint(step):
        full={k:v.detach().cpu() for k,v in student.state_dict().items()}
        state=dict(student={k:v for k,v in full.items() if not k.startswith('old_policy.')},full_student=full,optimizer=optimizer.state_dict(),
            config=actual,step=step,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),seconds=elapsed_before+time.time()-began)
        save(dest/'latest.pt',state)
        if step in (0,100,250,500) or step==steps:save(dest/f'checkpoint_{step}.pt',state)
    if not a.resume:checkpoint(0)
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True);weight=actual['OPD_initial_weight']*max(0.,1-step/max(steps-1,1));entries=[]
        for micro in range(4):
            row=data[schedule[step*4+micro]];tokens,inputs=conditioning(observation(row));seeds(seed_for(row['token'],a.seed,f'GRPO_{step}_{micro}'))
            rewards={};native=student.reward_fn
            def capture(*args,**kwargs):
                out=native(*args,**kwargs);rewards['values']=out.detach().clone();return out
            student.reward_fn=capture
            try:out,aux=grpo_with_opd(student,teacher,tokens,inputs,[row['token']],weight=weight,group_size=8)
            finally:student.reward_fn=native
            if not torch.isfinite(out.loss):raise FloatingPointError('Nonfinite native GRPO loss')
            (out.loss/4).backward();values=rewards['values']
            entries.append(dict(token=row['token'],reward=float(out.reward),reward_std=float(values.std()),valid_policy_signal=bool(values.std()>1e-8),
                policy_loss=float(out.policy_loss),BC_loss=float(out.bc_loss),OPD_loss=float(aux['opd']) if aux else 0.,loss=float(out.loss)))
            if aux is not None:save(dest/'rollouts'/f'{step+1}_{micro}.pt',dict(token=row['token'],states=aux['states'].cpu(),
                sigmas=aux['sigmas'].cpu(),timesteps=aux['timesteps'].cpu(),indices=aux['indices'].cpu(),
                noise_clip=student.randn_clip_value,sampler='native DiffGRPO sample_chain',per_step=aux['per_step'].detach().cpu(),weight=weight))
        norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip_norm']))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite E6 gradient')
        optimizer.step();history.append(dict(step=step+1,OPD_weight=weight,grad_norm=norm,gradient_clipped=norm>cfg['grad_clip_norm'],
            examples=entries,seconds=elapsed_before+time.time()-began))
        write(dest/'progress.json',history);print('E6',name,step+1,weight,norm,flush=True)
        if (step+1)%25==0 or step+1==steps or stop[0]:checkpoint(step+1)
        if stop[0] and step+1<steps:write(dest/'paused.json',dict(step=step+1));return
    assert state_hash(teacher)==teacher_hash and state_hash(student.old_policy)==reference_hash
    assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in student.old_policy.parameters())
    write(dest/'complete.json',dict(status='COMPLETE',config=actual,optimizer_updates=steps,teacher_unchanged=True,reference_unchanged=True,
        seconds=elapsed_before+time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(dest/'latest.pt'),owner=identity(),
        scope='Integration qualification only' if a.stage=='smoke' else 'Registered formal E6 training'))


if __name__=='__main__':main()
