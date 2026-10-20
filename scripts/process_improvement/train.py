"""Fresh-chain OPD/process training over reconstructed native populations.

No legacy E5 gates, no final-trajectory labels, no frozen projected features.
Every optimizer update consumes a global scene batch; rank/microbatch partition
changes neither its weights nor its semantic rollout/proposal random streams.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import math
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scripts.process_improvement.common import *


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['OPD','ANCHOR','PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD'],default='OPD')
    p.add_argument('--split',default='train_full')
    p.add_argument('--epochs',type=float,default=2.)
    p.add_argument('--exposures',type=int)
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--global-batch',type=int)
    p.add_argument('--micro-batch',type=int)
    p.add_argument('--init')
    p.add_argument('--name',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--qualification',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--score-budget',type=int)
    p.add_argument('--config')
    return p.parse_args()


def main():
    import torch
    import torch.distributed as dist
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,restore,state_hash
    from navsim.agents.recogdrive.process_improvement.contracts import NoisePlan
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.objective import objective_from_rollin
    from navsim.agents.recogdrive.process_improvement.projection_guard import TargetBuilder
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer,BudgetCounter
    from scripts.opd.scoring import load_cache
    a=parse();configure();cfg=config(a.config)
    if not read(ROOT/'tests/parity.json')['status']=='PASS':
        raise ValueError('Numerical admission has not passed')
    if not a.qualification:
        lock=read(ROOT/'protocol.lock.json')
        assert sha(ROOT/'protocol.yaml')==lock['sha256'], 'Locked protocol changed'
    if a.split.startswith(('cal','dev','navtest','diagnostic_cal')):
        raise ValueError('Optimization outside registered training population prohibited')
    data=membership(a.split)
    if any(r['split']!='navtrain' for r in data):raise ValueError('Training split leakage')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    a.micro_batch=a.micro_batch or min(cfg['micro_batch'],(a.global_batch or cfg['global_batch'])//world)
    if world>1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    batch=a.global_batch or cfg['global_batch']
    if batch%world or (batch//world)%a.micro_batch:raise ValueError('World/microbatch must exactly partition global scene batch')
    planned_exposure=a.exposures or math.ceil(len(data)*a.epochs)
    steps=a.steps or math.ceil(planned_exposure/batch)
    dest=ROOT/'runs'/a.name
    if (dest/'complete.json').exists():raise FileExistsError('Completed lineage is immutable')
    if (dest/'config.json').exists() and not a.resume:raise FileExistsError('Existing lineage requires --resume')
    init=a.init or cfg['student_init']
    actual=dict(method=a.method,split=a.split,seed=a.seed,lr=a.lr,steps=steps,epochs=a.epochs,
                global_batch=batch,micro_batch=a.micro_batch,world=world,init=init,init_sha256=sha(init),
                data_sha256=sha(ROOT/'manifests'/f'{a.split}.json'),config_sha256=digest(cfg),
                score_budget=a.score_budget,qualification=a.qualification)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    student=build_head(init);reference=build_head(cfg['teacher_init'],frozen=True)
    anchor=build_head(cfg['anchor_init'],frozen=True) if a.method=='ANCHOR' else None
    ref_hash=state_hash(reference);anchor_hash=state_hash(anchor) if anchor else None
    initial_hash=state_hash(student)
    parameters=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.lr,weight_decay=cfg['weight_decay'])
    adapter=NativeAdapter(student);reference_adapter=NativeAdapter(reference)
    counter=BudgetCounter();scorer=CanonicalScorer(counter)
    target_builder=TargetBuilder(reference_adapter,scorer,cfg)
    reference_version=0;start=0;elapsed_before=0.;history=[];previous_scores=0
    # Deterministic epoch permutations, concatenated before batching: no partial
    # minibatch padding duplicates and no silently dropped end-of-epoch scenes.
    schedule=np.concatenate([np.random.default_rng(a.seed*100000+epoch).permutation(len(data))
                             for epoch in range(math.ceil(steps*batch/len(data)))])[:steps*batch]
    if a.resume:
        checkpoint=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['config']!=actual:raise ValueError('Resume configuration/world size changed')
        restore(student,dest/'latest.pt');optimizer.load_state_dict(checkpoint['optimizer'])
        start=checkpoint['step'];elapsed_before=checkpoint['seconds'];history=checkpoint['history']
        own=checkpoint['rank_states'][rank]
        torch.set_rng_state(own['torch_rng']);torch.cuda.set_rng_state(own['cuda_rng'])
        np.random.set_state(own['numpy_rng']);random.setstate(own['python_rng'])
        counter.__dict__.update(own['budget']);previous_scores=checkpoint['physical_scores_global']
        reference_version=checkpoint['reference_version']
    if rank==0:
        write(dest/'config.json',dict(actual,parameter_count=sum(p.numel() for p in parameters),owner=identity(),schedule_sha256=digest(schedule.tolist())))
    started=time.time();halt=[False]
    signal.signal(signal.SIGUSR1,lambda *_:halt.__setitem__(0,True))
    endpoint_steps={math.ceil(len(data)*e/batch) for e in [1,2,4,8] if math.ceil(len(data)*e/batch)<=steps}
    endpoint_steps|={math.ceil(64000/batch),steps}
    score_endpoints=cfg['score_endpoints']
    def checkpoint(step,physical,endpoint=False):
        own=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),
                 python_rng=random.getstate(),budget=counter.state_dict())
        rank_states=[own]
        if world>1:
            rank_states=[None]*world;dist.all_gather_object(rank_states,own)
        if rank==0:
            state={k:v.detach().cpu() for k,v in student.state_dict().items()}
            record=dict(student=state,optimizer=optimizer.state_dict(),config=actual,step=step,cursor=step*batch,
                        epoch=step*batch/len(data),student_version=step,reference_version=reference_version,
                        reference_sha256=ref_hash,rank_states=rank_states,history=history,
                        physical_scores_global=physical,seconds=elapsed_before+time.time()-started,
                        scheduler_state=dict(kind='short_linear_warmup_then_constant',warmup=cfg['warmup_updates'],last_step=step),
                        amp_state=None,exact_resume_requires_same_world_size=True)
            save(dest/'latest.pt',record)
            if endpoint:
                path=dest/f'head_{step}.pt'
                if path.exists():raise FileExistsError('Do not overwrite a saved endpoint')
                save(path,dict(student=state,step=step,cursor=step*batch,epoch=step*batch/len(data),config=actual,
                               reference_version=reference_version,physical_scores=physical))
    if not a.resume:checkpoint(0,0,True)
    process_method=a.method in ('PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD')
    with ThreadPoolExecutor(max_workers=2) as loader:
        for step in range(start,steps):
            tick=time.time();optimizer.zero_grad(set_to_none=True)
            lr=a.lr*min(1.,(step+1)/max(1,cfg['warmup_updates']))
            for group in optimizer.param_groups:group['lr']=lr
            batch_rows=[data[int(i)] for i in schedule[step*batch:(step+1)*batch]]
            offsets=list(range(rank,batch,world))
            payloads=list(loader.map(observation,[batch_rows[i] for i in offsets]))
            totals=np.zeros(9,dtype=np.float64)
            diagnostics=[]
            for begin in range(0,len(offsets),a.micro_batch):
                ids=offsets[begin:begin+a.micro_batch];rows=[batch_rows[i] for i in ids]
                moved=[conditioning(payloads[begin+j]) for j in range(len(ids))]
                tokens=torch.cat([v[0] for v in moved]);inputs=BatchFeature({k:torch.cat([v[1][k] for v in moved]) for k in moved[0][1]})
                keys=[RNGKey(STUDY,a.seed,step,reference_version,row['token'],step*batch+i,-1) for row,i in zip(rows,ids)]
                plans=[key.plan('rollin',device=tokens.device) for key in keys]
                noise=NoisePlan(torch.cat([p.initial for p in plans]),torch.cat([p.steps for p in plans]))
                rollin=adapter.sample_rollin(tokens,inputs,noise,step)
                targets=None;selected=None
                if process_method:
                    selected=[(step*batch+i)%5 for i in ids];target_list=[]
                    for j,(row,key,index) in enumerate(zip(rows,keys,selected)):
                        own_inputs=BatchFeature({k:v[j:j+1] for k,v in inputs.items()})
                        with torch.no_grad():context=reference_adapter.prepare_conditioning(tokens[j:j+1],own_inputs)
                        detail=int(digest([row['token'],'trace'])[:8],16)%256==0
                        record=target_builder.build(rollin.states[j:j+1,index],context,index,key.at_step(index),
                                                    load_cache(row['metric_cache_path']),trace=detail,
                                                    ablation=None if a.method=='PROCESS' else a.method)
                        target_list.append(record.target)
                        if detail:save(dest/'target_traces'/f'{step}_{step*batch+ids[j]}.pt',record)
                        diagnostics.append(dict(token=row['token'],index=index,accepted=record.accepted,reason=record.reason,**record.statistics))
                    targets=torch.cat(target_list);selected=torch.tensor(selected,device=tokens.device)
                out=objective_from_rollin(student,reference,rollin,tokens,inputs,step,selected_target=targets,selected_index=selected,
                                          anchor=anchor,anchor_weight=cfg['lambda_anchor'] if anchor else 0.)
                # Sum local scene losses / true global batch. The final allreduce
                # is SUM (not DDP average), including zero-acceptance ranks.
                loss=out['loss']*len(ids)/batch
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
                loss.backward()
                totals[:3]+=[float(out['loss'].detach())*len(ids),float(out['opd'].detach())*len(ids),float(out['anchor'].detach())*len(ids)]
                totals[3:8]+=out['per_step'].sum(0).cpu().numpy();totals[8]+=len(ids)
            active=[p for p in parameters if p.grad is not None]
            if world>1:
                flat=torch.cat([p.grad.reshape(-1) for p in active]);dist.all_reduce(flat,op=dist.ReduceOp.SUM)
                cursor=0
                for parameter in active:
                    n=parameter.numel();parameter.grad.copy_(flat[cursor:cursor+n].view_as(parameter));cursor+=n
                joined=torch.tensor(totals,device='cuda',dtype=torch.float64);dist.all_reduce(joined);totals=joined.cpu().numpy()
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['grad_clip_norm']))
            if not np.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            budget=torch.tensor([counter.logical_scores,counter.physical_scores,counter.score_cache_hits],device='cuda',dtype=torch.long)
            if world>1:dist.all_reduce(budget)
            logical,physical,hits=map(int,budget.cpu().tolist())
            crossed=any(previous_scores<endpoint<=physical for endpoint in score_endpoints)
            previous_scores=physical
            entry=dict(step=step+1,epoch=(step+1)*batch/len(data),scene_exposures=(step+1)*batch,loss=float(totals[0]/batch),
                       opd=float(totals[1]/batch),anchor=float(totals[2]/batch),per_step=(totals[3:8]/batch).tolist(),
                       gradient_norm=norm,gradient_clipped=norm>cfg['grad_clip_norm'],lr=lr,step_seconds=time.time()-tick,
                       seconds=elapsed_before+time.time()-started,logical_scores=logical,physical_scores=physical,score_cache_hits=hits,
                       accepted=sum(d['accepted'] for d in diagnostics),local_proposals=len(diagnostics))
            if step==start or (step+1)%10==0 or step+1 in endpoint_steps or crossed:
                history.append(entry)
                if rank==0:write(dest/'progress.json',history);print('TRAIN',a.name,entry,flush=True)
            if diagnostics:
                path=dest/'target_statistics'/f'rank_{rank}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a') as stream:stream.write(json.dumps(dict(step=step+1,records=diagnostics))+'\n')
            stop=halt[0] or (a.stop_after is not None and step+1>=a.stop_after) or (a.score_budget is not None and physical>=a.score_budget)
            if world>1:
                flag=torch.tensor(int(stop),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX);stop=bool(flag)
            if (step+1)%100==0 or step+1 in endpoint_steps or crossed or stop:
                checkpoint(step+1,physical,step+1 in endpoint_steps or crossed or stop)
            if stop and step+1<steps:
                if rank==0:write(dest/'paused.json',dict(step=step+1,physical_scores=physical,score_budget_reached=a.score_budget is not None and physical>=a.score_budget))
                if world>1:dist.barrier();dist.destroy_process_group()
                return
    assert state_hash(reference)==ref_hash and (anchor is None or state_hash(anchor)==anchor_hash)
    assert all(p.grad is None for p in reference.parameters())
    if rank==0:
        write(dest/'complete.json',dict(status='COMPLETE',steps=steps,scene_exposures=steps*batch,
              seconds=elapsed_before+time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
              initial_student_hash=initial_hash,final_student_hash=state_hash(student),reference_unchanged=True,
              checkpoint_sha256=sha(dest/'latest.pt'),physical_scores=previous_scores,owner=identity()))
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
