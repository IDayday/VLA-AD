import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,signal
from scripts.opd.common import *


def main():
    import torch
    import torch.distributed as dist
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash,restore
    from navsim.agents.recogdrive.opd.ddim_adapter import Chain
    from navsim.agents.recogdrive.opd.trainer import objective
    p=argparse.ArgumentParser();p.add_argument('--exp',required=True);p.add_argument('--stage',choices=['smoke','tiny','tune','train'],default='train')
    p.add_argument('--config');p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float);p.add_argument('--steps',type=int)
    p.add_argument('--name');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--anchor-weight',type=float);p.add_argument('--il-weight',type=float);a=p.parse_args()
    configure();cfg=config(a.config);world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world>1:torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    if cfg['global_batch']%world:raise ValueError('World must divide global batch')
    gates=read(ROOT/'audit/gates.json');assert gates['G0'] and gates['G1']
    if not read(ROOT/'tests/cpu.json')['passed']:raise ValueError('Unit test gate is incomplete')
    name=a.name or f'{a.stage}_{a.exp}_{a.seed}';dest=ROOT/'runs'/name
    if (dest/'complete.json').exists():raise ValueError('Completed run cannot be overwritten')
    steps=a.steps or {'smoke':cfg['smoke_steps'],'tiny':cfg['tiny_steps'],'tune':cfg['tuning_steps'],'train':cfg['train_steps']}[a.stage]
    lr=a.lr or cfg['lr'];anchor_weight=a.anchor_weight if a.anchor_weight is not None else cfg['lambda_anchor']
    il_weight=a.il_weight if a.il_weight is not None else cfg['lambda_il']
    actual=dict(experiment=a.exp,stage=a.stage,steps=steps,seed=a.seed,lr=lr,global_batch=cfg['global_batch'],world=world,
        anchor_weight=anchor_weight,il_weight=il_weight,config_sha256=digest(cfg),data_sha256=sha(ROOT/'manifests/train.json'))
    seeds(a.seed);student=build_head(cfg['student_init']);teacher=build_head(cfg['teacher_init'],frozen=True)
    anchor=build_head(cfg['anchor_init'],frozen=True) if a.exp in ('E4-A','E4-C','E5') else None
    teacher_before=state_hash(teacher);anchor_before=state_hash(anchor) if anchor else None
    params=[p for p in student.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=lr,weight_decay=cfg['weight_decay'])
    schedule=[];data=rows('train');rng=np.random.default_rng(a.seed)
    if a.stage in ('smoke','tiny'):data=sorted(data,key=lambda r:digest([r['token'],'tiny']))[:cfg['tiny_scenes']]
    while len(schedule)<steps*cfg['global_batch']:schedule.extend(rng.permutation(len(data)).tolist())
    gate=read(ROOT/'gates/static.json')['scenes'] if a.exp=='E5' else {}
    start_step=0;history=[];counts={};start=time.perf_counter();stop=[False]
    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    if a.resume:
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        if state['config']!=actual:raise ValueError('Resume configuration changed')
        restore(student,dest/'latest.pt');opt.load_state_dict(state['optimizer']);start_step=state['step'];history=state['history'];counts=state['forward_counts']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
    elif (dest/'config.json').exists():raise ValueError('Existing partial run requires explicit --resume')
    if rank==0:write(dest/'config.json',dict(actual,owner=identity(),parameter_count=sum(p.numel() for p in params)))
    def checkpoint(step):
        state=dict(student={k:v.detach().cpu() for k,v in student.state_dict().items()},optimizer=opt.state_dict(),step=step,
            config=actual,history=history,forward_counts=counts,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),cursor=step*cfg['global_batch'])
        if world>1:
            states=[None]*world;dist.all_gather_object(states,dict(torch_rng=state['torch_rng'],cuda_rng=state['cuda_rng']));state['rank_rng']=states
        if rank==0:
            save(dest/'latest.pt',state)
            if step in cfg['endpoints'] or step==steps:save(dest/f'checkpoint_{step}.pt',state)
    if not a.resume:checkpoint(0)
    for step in range(start_step,steps):
        opt.zero_grad(set_to_none=True);batch=[data[i] for i in schedule[step*cfg['global_batch']:(step+1)*cfg['global_batch']]]
        effective=[float(gate[r['token']]['gate'] and gate[r['token']]['category']=='A') if a.exp=='E5' else 1. for r in batch]
        protected=[float(gate[r['token']]['category'] in ('B','C')) if a.exp=='E5' else 1. for r in batch]
        valid_count=max(sum(effective),1);protected_count=max(sum(protected),1);metrics=np.zeros(8);per_step=np.zeros(5);micro_records=[]
        for j in range(rank,len(batch),world):
            row=batch[j];token=row['token'];tokens,inputs=conditioning(observation(row));seeds(seed_for(token,a.seed,f'optimizer_step_{step}'))
            bank=torch.load(ROOT/'teacher/chains'/f'{token}.pt',map_location='cpu',weights_only=False)
            z=int(digest([token,a.seed,step])[:8],16)%5
            old=bank['chains'][z]
            teacher_chain=Chain(old['states'].cuda(),old['means'].cuda(),old['sigmas'].cuda(),teacher.ddim_t,
                torch.arange(5,device='cuda'),old['trajectory'].cuda(),tuple(p._version for p in teacher.parameters()),id(teacher),
                cfg['rollout_deterministic'],cfg['rollout_kernel']) if a.exp=='E2' else None
            out=objective(student,teacher,anchor,tokens,inputs,experiment=a.exp,target=gt(row).cuda()[None],
                teacher_trajectory=old['trajectory'].cuda(),teacher_chain=teacher_chain,
                rollout_seed=seed_for(token,3101+z,'distill_chain'),mask=effective[j],protected=protected[j],
                weights=torch.tensor(cfg['step_weights'],device='cuda'),deterministic=cfg['rollout_deterministic'],
                loss_type=cfg['objective'],kernel=cfg['rollout_kernel'])
            if a.exp=='E1':loss=out['il']/len(batch)
            else:loss=cfg['lambda_opd']*out['opd']*effective[j]/valid_count+anchor_weight*out['anchor']*protected[j]/protected_count+il_weight*out['il']/len(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            (loss*world).backward()
            metrics[:4]+=[float(out['opd'].detach())*effective[j],float(out['anchor'].detach())*protected[j],float(out['il'].detach()),float(loss.detach())]
            metrics[4:]+=[effective[j],protected[j],1,0]
            per_step+=out['per_step'].detach().mean(0).cpu().numpy()
            for k,v in out['forward_counts'].items():counts[k]=counts.get(k,0)+v
            if (step+1)%25==0 or step==0:micro_records.append(dict(token=token,seed=3101+z,gate=effective[j],category=gate[token]['category'] if gate else None,
                chain_first=float(out['chain'].states[:,1].float().mean()) if out['chain'] is not None else None))
        if world>1:
            active=[p for p in params if p.grad is not None];flat=torch.cat([p.grad.reshape(-1) for p in active]);dist.all_reduce(flat);flat/=world
            offset=0
            for parameter in active:
                n=parameter.numel();parameter.grad.copy_(flat[offset:offset+n].view_as(parameter));offset+=n
            all_metrics=torch.tensor(np.r_[metrics,per_step],device='cuda',dtype=torch.float64);dist.all_reduce(all_metrics);metrics=all_metrics[:8].cpu().numpy();per_step=all_metrics[8:].cpu().numpy()
        norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip_norm']))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        if step==0 or (step+1)%25==0 or step+1==steps:
            entry=dict(step=step+1,opd=float(metrics[0]/valid_count),anchor=float(metrics[1]/protected_count),il=float(metrics[2]/len(batch)),
                loss=float(metrics[3]),grad_norm=norm,gradient_clipped=norm>cfg['grad_clip_norm'],per_step=(per_step/len(batch)).tolist(),
                gate_fraction=sum(effective)/len(batch),seconds=time.perf_counter()-start,examples=micro_records)
            history.append(entry)
            if rank==0:write(dest/'progress.json',history);print('TRAIN',name,entry,flush=True)
        halt=stop[0] or (a.stop_after is not None and step+1>=a.stop_after)
        if world>1:
            flag=torch.tensor(int(halt),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX);halt=bool(flag)
        if (step+1)%100==0 or step+1==steps or halt:checkpoint(step+1)
        if halt and step+1<steps:
            if rank==0:write(dest/'paused.json',dict(step=step+1,owner=identity()))
            if world>1:dist.barrier();dist.destroy_process_group()
            return
    assert state_hash(teacher)==teacher_before and (anchor is None or state_hash(anchor)==anchor_before)
    assert all(p.grad is None for p in teacher.parameters()) and (anchor is None or all(p.grad is None for p in anchor.parameters()))
    if rank==0:write(dest/'complete.json',dict(status='COMPLETE',config=actual,optimizer_updates=steps,
        teacher_unchanged=True,anchor_unchanged=True,seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        teacher_forward_counts=counts,checkpoint_sha256=sha(dest/'latest.pt'),owner=identity()))
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
