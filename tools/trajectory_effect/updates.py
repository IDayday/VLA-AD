"""Independent, real native GRPO updates and paired held-out evaluations."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random
import time
import traceback
import numpy as np
from .common import *
from .bank import load_observation,observation_path,baseline_path,baseline_score_path,score_record


def families():
    values=read(ROOT/'manifests/update_families.json')
    extra=ROOT/'manifests/confirmation_families.json'
    return values+(read(extra) if extra.exists() else [])


def family(name):return next(f for f in families() if f['id']==name)
def directory(name):return ROOT/'updates'/name
def trajectory_path(name,step,token):return directory(name)/f'step_{step}/trajectories/{token}.npz'
def score_path(name,step,token):return directory(name)/f'step_{step}/scores/{token}.npz'
def targets(f):
    cfg,_=protocol();rows=scene_rows('D_'+f['split'])
    return sorted(rows,key=lambda r:digest([r['token'],'target_subset']))[:cfg['updates']['evaluation_targets'][f['split']]]


def wait_file(path,timeout=1800):
    start=time.time()
    while not Path(path).exists():
        if time.time()-start>timeout:raise TimeoutError(f'Required input absent: {path}')
        time.sleep(2)


def policy_state(head):
    return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()
            if not k.startswith(('old_policy.','action_aware_aux_head.'))}


def tensor_digest(state):
    import torch
    h=hashlib.sha256()
    for k,v in state.items():h.update(k.encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore(head,state):
    result=head.load_state_dict(state,strict=False)
    if result.unexpected_keys or any(not k.startswith(('old_policy.','action_aware_aux_head.')) for k in result.missing_keys):
        raise ValueError(result)
    assert tensor_digest(policy_state(head))==tensor_digest(state)


def capture_native(head):
    """Only logs detached copies; calls the existing sampler/reward unmodified."""
    record={};original_sample=head.sample_chain;original_reward=head.reward_fn
    def sample(*a,**kw):
        chains,trajs=original_sample(*a,**kw)
        record['chains']=chains.detach().cpu().numpy();record['trajectories']=trajs.detach().cpu().numpy()
        return chains,trajs
    def reward(*a,**kw):
        rewards=original_reward(*a,**kw);record['rewards']=rewards.detach().cpu().numpy();return rewards
    head.sample_chain=sample;head.reward_fn=reward
    return record


def set_seed(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def train(args):
    import torch
    from .runtime import load_agent,configure_grpo,action_input,predict,trainable_head
    cfg,identity=protocol();f=family(args.family);dest=directory(f['id']);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'train_complete.json').exists():raise ValueError('Completed family must not be overwritten')
    start=time.perf_counter();owner=runtime_identity();atomic_json(dest/'train_owner.json',owner)
    slot=int(f['id'].split('_')[-1]);rows=scene_rows('D_fit')
    pool=sorted([r for r in rows if int(digest([r['token'],'source_pool']),16)%24==slot],
                key=lambda r:digest([r['token'],f['train_seed'],'source_order']))
    assert len(pool)>=8
    # Eligibility can use only the predeclared D_fit baseline and current policy.
    if f['source_kind']=='low_score_reward_diverse':
        for r in pool:wait_file(baseline_score_path(r['token']))
        pool=[r for r in pool if np.load(baseline_score_path(r['token']))['scores'][:5,-1].mean()<cfg['metrics']['source_low_threshold']]
        pool=pool[:cfg['updates']['source_low_qualification_max_scenes_per_family']]
    else:pool=pool[:32]
    if not pool:raise ValueError('No low-score source scene in registered pool; family cannot be silently replaced')
    for r in pool:wait_file(observation_path(r['token']));wait_file(baseline_path(r['token']))
    set_seed(f['train_seed']);agent=load_agent(online=False);head=agent.action_head
    sample=load_observation(pool[0]['token']);inputs=action_input(sample['features']);tokens=sample['tokens'].cuda()
    zero_before=predict(head,tokens,inputs,pool[0]['token'],1101)
    head=configure_grpo(head,rows);head.eval()
    zero_after=predict(head,tokens,inputs,pool[0]['token'],1101)
    bank=np.load(baseline_path(pool[0]['token']))['trajectories'][0]
    zero_error=float(max(abs(zero_before-zero_after).max(),abs(bank-zero_after).max()))
    if zero_error:raise ValueError(f'Zero update/configuration inference changed: {zero_error}')
    initial=policy_state(head);initial_hash=tensor_digest(initial)
    records=capture_native(head);qualification=[];chosen=[]
    if f['source_kind']=='low_score_reward_diverse':
        for r in pool:
            token=r['token'];data=load_observation(token);inp=action_input(data['features'])
            set_seed(seed_for(token,f['train_seed'],'source_qualification'))
            with torch.no_grad():head.forward_grpo(data['tokens'].cuda(),inp,[token],sample_time=8)
            std=float(np.std(records['rewards']));qualification.append(dict(token=token,rewards=records['rewards'].tolist(),std=std))
            atomic_npz(dest/'qualification'/f'{token}.npz',dict(token=token,protocol_sha256=identity),**records)
            if std>1e-12:chosen.append(r)
            if len(chosen)>=8:break
        # Keep no-signal families; do not fabricate an advantage or import actions.
        if not chosen:chosen=pool[:8]
    else:chosen=pool
    assert tensor_digest(policy_state(head))==initial_hash,'Source qualification mutated policy'
    atomic_json(dest/'source_selection.json',dict(family=f,protocol_sha256=identity,
        source_pool_slot=slot,pool_tokens=[r['token'] for r in pool],chosen_tokens=[r['token'] for r in chosen],
        qualification=qualification,zero_update_max_error=zero_error,
        selection_has_reward_diverse_source=any(q['std']>1e-12 for q in qualification) if qualification else None))
    params=trainable_head(head);optcfg=cfg['updates']['optimizer']
    optimizer=torch.optim.AdamW(params,lr=optcfg['lr'],weight_decay=optcfg['weight_decay'],
        betas=tuple(optcfg['betas']),eps=optcfg['eps'])
    reference_hash=sha(IL)
    set_seed(f['train_seed']);schedule=[chosen[j%len(chosen)] for j in range(32)]
    atomic_json(dest/'training_schedule.json',[dict(step=j//8+1,microbatch=j%8,token=r['token']) for j,r in enumerate(schedule)])
    def save(step):
        state=policy_state(head)
        value=dict(policy=state,optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            scheduler=None,step=step,cursor=step*8,family=f,protocol_sha256=identity,
            initial_policy_sha256=initial_hash,policy_state_sha256=tensor_digest(state),
            fixed_reference=str(IL),fixed_reference_sha256=reference_hash,new_training=True)
        path=dest/f'checkpoint_{step}.pt';atomic_torch(path,value)
        atomic_json(dest/f'checkpoint_{step}.json',dict(step=step,file=str(path),sha256=sha(path),
            policy_state_sha256=value['policy_state_sha256'],initial_policy_sha256=initial_hash,protocol_sha256=identity))
    save(0);step_records=[]
    # Native clipping quantiles are0/1 (no clipping), so per-scene gradient
    # accumulation equals the mean of eight native group losses.
    assert head.clip_advantage_lower_quantile==0 and head.clip_advantage_upper_quantile==1
    for step in range(1,5):
        optimizer.zero_grad(set_to_none=True);micro=[];tick=time.perf_counter()
        for j,r in enumerate(schedule[(step-1)*8:step*8]):
            token=r['token'];data=load_observation(token);inp=action_input(data['features'])
            set_seed(seed_for(token,f['train_seed'],f'policy_step_{step}_micro_{j}'))
            output=head.forward_grpo(data['tokens'].cuda(),inp,[token],sample_time=8)
            (output.loss/8).backward()
            atomic_npz(dest/'on_policy'/f'{step}_{j}_{token}.npz',dict(token=token,step=step,microbatch=j,
                family=f['id'],protocol_sha256=identity,reward_weights=[10,5,2],own_current_policy=True),**records)
            micro.append(dict(token=token,reward_mean=float(np.mean(records['rewards'])),reward_std=float(np.std(records['rewards'])),
                policy_loss=float(output.policy_loss),bc_loss=float(output.bc_loss)))
        norm=float(torch.nn.utils.clip_grad_norm_(params,optcfg['clip_grad_norm']))
        if not np.isfinite(norm):raise ValueError('Nonfinite native GRPO gradient')
        optimizer.step();torch.cuda.synchronize()
        step_records.append(dict(step=step,gradient_norm=norm,microbatches=micro,seconds=time.perf_counter()-tick,
            zero_reward_groups=sum(x['reward_std']<=1e-12 for x in micro)))
        atomic_json(dest/'train_progress.json',step_records)
        print('UPDATE',f['id'],step,norm,step_records[-1]['seconds'],flush=True)
        if step in cfg['updates']['save_steps']:save(step)
    final_hash=tensor_digest(policy_state(head))
    atomic_json(dest/'train_complete.json',dict(status='COMPLETE',family=f,protocol_sha256=identity,
        steps=step_records,initial_policy_sha256=initial_hash,final_policy_sha256=final_hash,
        changed=final_hash!=initial_hash,zero_update_max_error=zero_error,
        seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=owner))


def evaluate(args):
    import torch
    from .runtime import load_agent,action_input,predict
    cfg,identity=protocol();f=family(args.family);dest=directory(f['id']);start=time.perf_counter()
    agent=load_agent(online=False);head=agent.action_head;rows=targets(f)[args.rank::args.shards]
    for step in (1,4):
        path=dest/f'checkpoint_{step}.pt';wait_file(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);restore(head,checkpoint['policy'])
        for i,r in enumerate(rows):
            token=r['token'];output=trajectory_path(f['id'],step,token)
            if output.exists():continue
            wait_file(observation_path(token));data=load_observation(token);inputs=action_input(data['features'])
            trajectories=np.stack([predict(head,data['tokens'].cuda(),inputs,token,s) for s in cfg['seeds']['inference']])
            atomic_npz(output,dict(token=token,log=r['log_name'],family=f,step=step,seeds=cfg['seeds']['inference'],
                protocol_sha256=identity,checkpoint_sha256=read(dest/f'checkpoint_{step}.json')['sha256'],
                old_trajectory=str(baseline_path(token))),trajectories=trajectories)
            if (i+1)%32==0:print('EVALUATE',f['id'],step,args.rank,i+1,len(rows),flush=True)
    atomic_json(dest/f'evaluate_{args.rank}_complete.json',dict(status='COMPLETE',rank=args.rank,shards=args.shards,
        scenes=len(rows),seconds=time.perf_counter()-start,protocol_sha256=identity,owner=runtime_identity()))


def score(args):
    cfg,identity=protocol();f=family(args.family);start=time.perf_counter()
    tasks=[(r,str(trajectory_path(f['id'],step,r['token'])),str(score_path(f['id'],step,r['token'])),identity)
           for step in (1,4) for r in targets(f)]
    remaining=tasks[:];pending={};done=0
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        while remaining or pending:
            for task in remaining[:]:
                if len(pending)>=args.workers*3:break
                if Path(task[1]).exists():
                    pending[pool.submit(score_record,task)]=task[1];remaining.remove(task)
            for future in list(pending):
                if future.done():future.result();del pending[future];done+=1
            atomic_json(directory(f['id'])/'score_progress.json',dict(done=done,total=len(tasks),pending=len(pending)))
            if remaining or pending:time.sleep(1)
    atomic_json(directory(f['id'])/'score_complete.json',dict(status='COMPLETE',scenes=len(tasks),
        seconds=time.perf_counter()-start,protocol_sha256=identity,owner=runtime_identity()))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['train','evaluate','score'])
    p.add_argument('--family',required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    try:globals()[args.action](args)
    except Exception:
        atomic_json(directory(args.family)/f'{args.action}_{args.rank}_failure.json',dict(status='FAILED',traceback=traceback.format_exc(),owner=runtime_identity()))
        raise


if __name__=='__main__':main()
