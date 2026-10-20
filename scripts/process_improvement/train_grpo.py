"""Independent native DiffGRPO controls: IL, matched warm start, and RL.

The native sampler, advantage, clipped approximate Gaussian logprob and single
IL BC constraint are unchanged. Only the scorer progress weight is selectable:
10 reproduces the historical reward; 5 is the fair canonical PDMS comparison.
Native scalar scoring is retained and every executed call is charged.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import random
import signal
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    import torch.distributed as dist
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash,restore
    from navsim.agents.recogdrive.opd.trainer import configure_native_grpo
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.compute_ledger import ComputeLedger
    from scripts.opd.scoring import Scoring
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--init',default='IL');p.add_argument('--split',default='train_full')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--steps',type=int);p.add_argument('--score-budget',type=int,default=4000000)
    p.add_argument('--progress-weight',type=float,choices=[5.,10.],default=5.)
    p.add_argument('--global-batch',type=int,default=32);p.add_argument('--resume',action='store_true')
    p.add_argument('--qualification',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();configure();cfg=config();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    if world>1:torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    if a.global_batch%world:raise ValueError('World size must partition global scene exposure')
    if read(ROOT/'tests/parity.json')['status']!='PASS':raise ValueError('Base numerical gate incomplete')
    if not a.qualification:
        assert sha(ROOT/'protocol.yaml')==read(ROOT/'protocol.lock.json')['sha256']
    if a.split not in ('train_full','train_25k','train_50k','legacy_4k'):raise ValueError('Illegal RL training split')
    data=membership(a.split)
    assert all(row['split']=='navtrain' for row in data)
    init=cfg['student_init'] if a.init=='IL' else cfg['teacher_init'] if a.init=='RL' else a.init
    steps=a.steps or int(np.ceil(a.score_budget/(8*a.global_batch)))
    dest=ROOT/'runs'/a.name
    if (dest/'complete.json').exists():raise FileExistsError('Complete GRPO lineage preserved')
    if (dest/'config.json').exists() and not a.resume:raise FileExistsError('Partial GRPO requires --resume')
    actual=dict(method='native_GRPO',init=init,init_sha256=sha(init),seed=a.seed,lr=a.lr,steps=steps,world=world,
                global_batch=a.global_batch,group_size=8,progress_weight=a.progress_weight,BC_coeff=.1,
                split=a.split,data_sha256=sha(ROOT/'manifests'/f'{a.split}.json'),score_budget=a.score_budget,
                sampling_std_floor=.04,noise_clip=5.,logprob_std_floor=.1,logprob_clip=[-5.,2.],gamma_denoising=.6)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    student=build_head(init);student=configure_native_grpo(student,data,cfg['anchor_init']);student.eval()
    scoring=Scoring(progress_weight=a.progress_weight)
    student.train_scorer=scoring.scorer;student.simulator=scoring.simulator
    original_reward=student.reward_fn
    counts=dict(physical_scores=0,logical_scores=0,reference_simulations=0,cpu_scoring_seconds=0.)
    last_reward={}
    def reward(traj,tokens,caches):
        before=time.process_time()
        result=original_reward(traj,tokens,caches)
        counts['cpu_scoring_seconds']+=time.process_time()-before
        counts['physical_scores']+=len(traj);counts['logical_scores']+=len(traj);counts['reference_simulations']+=len(traj)
        last_reward['values']=result.detach().clone()
        return result
    student.reward_fn=reward
    compute=ComputeLedger(student=student,IL_reference=student.old_policy)
    reference_hash=state_hash(student.old_policy)
    params=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=cfg['weight_decay'])
    schedule=np.concatenate([np.random.default_rng(a.seed*100000+epoch).permutation(len(data))
                             for epoch in range(int(np.ceil(steps*a.global_batch/len(data))))])[:steps*a.global_batch]
    start=0;history=[];elapsed=0.;previous=0
    if a.resume:
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        if state['config']!=actual:raise ValueError('Native GRPO resume identity changed')
        student.load_state_dict(state['full_student'],strict=True);optimizer.load_state_dict(state['optimizer'])
        start=state['step'];history=state['history'];elapsed=state['seconds']
        own=state['rank_states'][rank];counts.update(own['counts'])
        if 'compute' in own:compute.load_state_dict(own['compute'])
        torch.set_rng_state(own['torch_rng']);torch.cuda.set_rng_state(own['cuda_rng']);np.random.set_state(own['numpy_rng']);random.setstate(own['python_rng'])
        previous=state['physical_scores']
    if rank==0:write(dest/'config.json',dict(actual,owner=identity(),parameter_count=sum(p.numel() for p in params),
            gradient_path='Original forward_grpo unmodified, including original sample_chain graph behavior',
            probability_limitation='Original Gaussian logprob surrogate for clipped-normal sampler; not exact native kernel density'))
    began=time.time();halt=[False];signal.signal(signal.SIGUSR1,lambda *_:halt.__setitem__(0,True))
    def checkpoint(step,physical,endpoint=False):
        own=dict(counts=counts.copy(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),compute=compute.state_dict())
        write(dest/f'cost_rank_{rank}.json',dict(step=step,compute=compute.state_dict(),budget=counts.copy(),allocated_GPU_seconds=elapsed+time.time()-began))
        states=[own]
        if world>1:states=[None]*world;dist.all_gather_object(states,own)
        if rank==0:
            full={k:v.detach().cpu() for k,v in student.state_dict().items()}
            active={k:v for k,v in full.items() if not k.startswith('old_policy.')}
            record=dict(student=active,full_student=full,optimizer=optimizer.state_dict(),step=step,cursor=step*a.global_batch,
                        config=actual,rank_states=states,physical_scores=physical,history=history,seconds=elapsed+time.time()-began,
                        reference_sha256=reference_hash,scheduler_state=dict(warmup=cfg['warmup_updates'],last_step=step),amp_state=None)
            save(dest/'latest.pt',record)
            if endpoint:
                path=dest/f'head_{step}.pt'
                if path.exists():raise FileExistsError('Do not overwrite an immutable GRPO endpoint')
                save(path,dict(student=active,config=actual,step=step,cursor=step*a.global_batch,physical_scores=physical))
    if not a.resume:checkpoint(0,0,True)
    for step in range(start,steps):
        tick=time.time();optimizer.zero_grad(set_to_none=True);totals=np.zeros(6)
        for group in optimizer.param_groups:group['lr']=a.lr*min(1.,(step+1)/max(1,cfg['warmup_updates']))
        for offset in range(rank,a.global_batch,world):
            row=data[int(schedule[step*a.global_batch+offset])];tokens,inputs=conditioning(observation(row))
            key=RNGKey(STUDY,a.seed,step,0,row['token'],step*a.global_batch+offset,-1)
            # Serial process-local native RNG scope, no threads or head method
            # patching. The semantic seed contains neither rank nor worker id.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                seed=key.seed('rollin');torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                out=student.forward_grpo(tokens,inputs,[row['token']],sample_time=8)
            if not torch.isfinite(out.loss):raise FloatingPointError('Nonfinite native GRPO objective')
            (out.loss/a.global_batch).backward()
            values=last_reward['values']
            totals+=[float(out.loss.detach()),float(out.reward.detach()),float(out.policy_loss.detach()),
                     float(out.bc_loss.detach()),float(values.std()>1e-8),1.]
        active=[p for p in params if p.grad is not None]
        if world>1:
            flat=torch.cat([p.grad.reshape(-1) for p in active]);dist.all_reduce(flat)
            cursor=0
            for parameter in active:
                n=parameter.numel();parameter.grad.copy_(flat[cursor:cursor+n].view_as(parameter));cursor+=n
            aggregate=torch.tensor(totals,device='cuda',dtype=torch.float64);dist.all_reduce(aggregate);totals=aggregate.cpu().numpy()
        norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip_norm']))
        if not np.isfinite(norm):raise FloatingPointError('Nonfinite native GRPO gradient')
        optimizer.step()
        score_count=torch.tensor(counts['physical_scores'],device='cuda',dtype=torch.long)
        if world>1:dist.all_reduce(score_count)
        physical=int(score_count)
        crossed=any(previous<s<=physical for s in cfg['score_endpoints']);previous=physical
        entry=dict(step=step+1,scene_exposures=(step+1)*a.global_batch,physical_scores=physical,
                   loss=totals[0]/a.global_batch,reward=totals[1]/a.global_batch,policy_loss=totals[2]/a.global_batch,
                   BC_loss=totals[3]/a.global_batch,signal_fraction=totals[4]/a.global_batch,grad_norm=norm,
                   gradient_clipped=norm>cfg['grad_clip_norm'],step_seconds=time.time()-tick,seconds=elapsed+time.time()-began)
        if step==start or (step+1)%10==0 or crossed or step+1==steps:
            history.append(entry)
            if rank==0:write(dest/'progress.json',history);print('NATIVE_GRPO',a.name,entry,flush=True)
        stop=halt[0] or (a.stop_after is not None and step+1>=a.stop_after)
        if world>1:
            flag=torch.tensor(int(stop),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX);stop=bool(flag)
        if (step+1)%100==0 or crossed or step+1==steps or stop:checkpoint(step+1,physical,crossed or step+1==steps or stop)
        if stop and step+1<steps:
            assert state_hash(student.old_policy)==reference_hash
            assert all(p.grad is None for p in student.old_policy.parameters())
            if rank==0:write(dest/'paused.json',dict(step=step+1,physical_scores=physical))
            if world>1:dist.barrier();dist.destroy_process_group()
            return
    assert state_hash(student.old_policy)==reference_hash
    assert all(p.grad is None for p in student.old_policy.parameters())
    if rank==0:write(dest/'complete.json',dict(status='COMPLETE',steps=steps,scene_exposures=steps*a.global_batch,
          physical_scores=previous,seconds=elapsed+time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
          IL_BC_reference_unchanged=True,checkpoint_sha256=sha(dest/'latest.pt'),owner=identity()))
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
