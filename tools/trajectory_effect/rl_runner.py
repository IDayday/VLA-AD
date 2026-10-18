"""Native on-policy PDMS GRPO with matched ordinary-Q/effect auxiliary losses."""
from __future__ import annotations
import argparse
import traceback
import numpy as np
from .common import *
from .bank import load_observation,baseline_path,baseline_score_path,score_record
from .updates import set_seed,policy_state,tensor_digest,restore,capture_native
from .probe_runner import schedule,load_library,load_adapter,run_dir as probe_dir


def run_dir(arm,seed):return ROOT/'rl'/f'{arm}_{seed}'


def check_gate():
    cfg,identity=protocol();gate=read(ROOT/'analysis/stage_admission.json')
    if gate['protocol_sha256']!=identity or gate['stage_E']!='PASS':
        raise RuntimeError('Stage E not admitted by locked B/D evidence')
    return gate


def train(args):
    import torch
    import torch.distributed as dist
    from .runtime import configure,load_agent,configure_grpo,action_input,encode,predict,trainable_head
    from .probes import TrajectoryScoreHead,adapt_vlm,vlm_parameters,vlm_state,joint_training_mode,effect_loss
    from .distributed import initialize,average_gradients,sum_values,finish
    check_gate();rank,world=initialize();configure();cfg,identity=protocol()
    if world not in (1,2,4):raise ValueError('World size must divide both global batches8 and4')
    joint=args.arm!='RL-F';auxiliary=args.arm in ('RL-Q','RL-E');dest=run_dir(args.arm,args.seed)
    selection=read(ROOT/'probes/selection.json');lam=selection['common_lambda'] if args.arm=='RL-E' else 0.
    actual=dict(arm=args.arm,seed=args.seed,steps=500,world_size=world,policy_global_batch=8,aux_global_batch=4,
        protocol_sha256=identity,policy_lr=1e-5,vlm_lr=1e-5,beta=1.,lambda_diff=lam,
        own_current_policy=True,reference_policy=str(IL),fresh_optimizer=True)
    if rank==0:
        if (dest/'complete.json').exists():raise ValueError('Do not overwrite completed RL experiment')
        atomic_json(dest/'config.json',actual);atomic_json(dest/'owner.json',runtime_identity())
    set_seed(args.seed);agent=load_agent(online=joint);policy=configure_grpo(agent.action_head,scene_rows('D_fit'));policy.eval()
    if joint:
        set_seed(seed_for('shared_vlm_adapter',args.seed));adapt_vlm(agent)
    head=None;groups=[dict(params=trainable_head(policy),lr=1e-5)]
    if joint:groups.append(dict(params=vlm_parameters(agent),lr=1e-5))
    warmup=None
    if auxiliary:
        set_seed(seed_for('shared_score_head',41));head=TrajectoryScoreHead().cuda()
        warmup=selection['selected']['FQ']['name'] if 'name' in selection['selected']['FQ'] else None
        # Reuse the selected300-step seed41 frozen-VLM FQ warmup identically.
        lr=selection['selected']['FQ']['head_lr']
        slot=cfg['probes']['head_lr_candidates'].index(lr);warmup=f'tuning_FQ_{slot}'
        warm_state=torch.load(probe_dir(warmup)/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert warm_state['config']['arm']=='FQ' and warm_state['step']==300
        head.load_state_dict(warm_state['head']);groups.append(dict(params=list(head.parameters()),lr=lr))
    params=[p for group in groups for p in group['params']]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    policy_schedule=schedule(args.seed,500,batch=8);aux_schedule=schedule(args.seed+100000,500,batch=4)
    if rank==0:atomic_json(dest/'scene_schedules.json',dict(policy=policy_schedule,auxiliary=aux_schedule if auxiliary else None))
    records=capture_native(policy);start=time.perf_counter();history=[];initial=policy_state(policy)
    initial_vlm=vlm_state(agent) if joint else {};initial_vlm_hash=tensor_digest(initial_vlm) if joint else None
    check_token=policy_schedule[0];payload=load_observation(check_token);inp=action_input(payload['features'])
    context=encode(agent,payload['features']) if joint else payload['tokens'].cuda()
    initial_pred=predict(policy,context,inp,check_token,1101)
    if not np.array_equal(initial_pred,np.load(baseline_path(check_token))['trajectories'][0]):
        raise ValueError('RL starting policy differs from common paired baseline')
    def save(step):
        rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state())
        rngs=[rng]
        if world>1:rngs=[None]*world;dist.all_gather_object(rngs,rng)
        if rank==0:
            state=dict(policy=policy_state(policy),vlm=vlm_state(agent) if joint else {},
                score_head={n:p.detach().cpu() for n,p in head.state_dict().items()} if head else {},
                optimizer=optimizer.state_dict(),scheduler=None,step=step,policy_cursor=step*8,aux_cursor=step*4,
                rank_rng=rngs,config=actual,reference_sha256=sha(IL),warmup=warmup,
                warmup_sha256=sha(probe_dir(warmup)/'checkpoint.pt') if warmup else None)
            path=dest/f'checkpoint_{step}.pt';atomic_torch(path,state)
            atomic_json(dest/f'checkpoint_{step}.json',dict(path=str(path),sha256=sha(path),step=step,protocol_sha256=identity,
                policy_state_sha256=tensor_digest(state['policy']),vlm_sha256=tensor_digest(state['vlm']) if joint else None))
        if world>1:dist.barrier()
    save(0)
    for step in range(500):
        optimizer.zero_grad(set_to_none=True);tick=time.perf_counter();losses=[];zero_groups=0
        for j in range(rank,8,world):
            token=policy_schedule[step*8+j];data=load_observation(token);inputs=action_input(data['features'])
            set_seed(seed_for(token,args.seed,f'rl_step_{step}_micro_{j}'))
            if joint:joint_training_mode(agent);context=encode(agent,data['features'],grad=True)
            else:context=data['tokens'].cuda()
            output=policy.forward_grpo(context,inputs,[token],sample_time=8)
            (output.loss*world/8).backward()
            atomic_npz(dest/'on_policy'/f'{step+1}_{j}.npz',dict(token=token,step=step+1,micro=j,arm=args.arm,
                seed=args.seed,protocol_sha256=identity,own_current_policy=True),**records)
            std=float(np.std(records['rewards']));zero_groups+=int(std<=1e-12)
            losses.append([float(output.policy_loss),float(output.bc_loss),float(output.reward)])
        absolute_value=0.;difference_value=0.
        if auxiliary:
            local=[(t,load_library(t)) for t in aux_schedule[step*4:(step+1)*4][rank::world]]
            counts=sum_values([sum(len(v['labels']) for _,v in local),sum(len(v['pairs']) for _,v in local)],world)
            for token,value in local:
                payload=load_observation(token);joint_training_mode(agent);context=encode(agent,payload['features'],grad=True)
                inputs=action_input(payload['features']);pred=head(context,torch.from_numpy(value['trajectories']).cuda()[None],inputs.state)
                _,absolute,difference=effect_loss(pred,torch.from_numpy(value['labels']).cuda()[None],torch.from_numpy(value['pairs']).cuda(),lam)
                (world*(absolute*len(value['labels'])/counts[0]+lam*difference*len(value['pairs'])/counts[1])).backward()
                absolute_value+=float(absolute)*len(value['labels'])/float(counts[0])
                difference_value+=float(difference)*len(value['pairs'])/float(counts[1])
        average_gradients(params,world)
        vlm_norm=sum(float(p.grad.float().square().sum()) for p in vlm_parameters(agent) if p.grad is not None)**.5 if joint else 0.
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not np.isfinite(norm):raise ValueError('Nonfinite RL gradient')
        optimizer.step();combined=sum_values([*np.sum(losses,axis=0),zero_groups,absolute_value,difference_value],world).cpu().numpy()
        if rank==0:
            history.append(dict(step=step+1,policy_loss=float(combined[0]/8),bc_loss=float(combined[1]/8),
                reward=float(combined[2]/8),zero_reward_groups=int(combined[3]),L_abs=float(combined[4]),L_diff=float(combined[5]),
                gradient_norm=norm,vlm_gradient_norm=vlm_norm,seconds=time.perf_counter()-tick))
            atomic_json(dest/'progress.json',history)
            if (step+1)%10==0:print('RL',args.arm,args.seed,history[-1],flush=True)
        if step+1 in cfg['rl']['save_steps']:save(step+1)
    if rank==0:
        final_vlm=vlm_state(agent) if joint else {}
        if joint and tensor_digest(final_vlm)==initial_vlm_hash:raise ValueError('Shared VLM did not update in RL')
        atomic_json(dest/'complete.json',dict(status='COMPLETE',config=actual,updates=500,seconds=time.perf_counter()-start,
            initial_policy_sha256=tensor_digest(initial),final_policy_sha256=tensor_digest(policy_state(policy)),
            initial_vlm_sha256=initial_vlm_hash,final_vlm_sha256=tensor_digest(final_vlm) if joint else None,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=runtime_identity()))
    finish(world)


def evaluate(args):
    import torch
    from .runtime import configure,load_agent,action_input,encode,predict
    from .probes import adapt_vlm
    configure();cfg,identity=protocol();dest=run_dir(args.arm,args.seed);joint=args.arm!='RL-F'
    if args.split not in ('D_cal','D_audit'):raise ValueError('Navtest needs separate final-confirmation admission')
    agent=load_agent(online=joint)
    if joint:adapt_vlm(agent)
    checkpoint=torch.load(dest/f'checkpoint_{args.step}.pt',map_location='cpu',weights_only=False)
    restore(agent.action_head,checkpoint['policy'])
    if joint:load_adapter(agent,checkpoint['vlm'])
    agent.eval();rows=scene_rows(args.split)[args.rank::args.shards];start=time.perf_counter()
    for i,row in enumerate(rows):
        token=row['token'];path=dest/f'step_{args.step}/trajectories/{token}.npz'
        if path.exists():continue
        data=load_observation(token);inputs=action_input(data['features'])
        context=encode(agent,data['features']) if joint else data['tokens'].cuda()
        trajectories=np.stack([predict(agent.action_head,context,inputs,token,s) for s in cfg['seeds']['inference']])
        atomic_npz(path,dict(token=token,log=row['log_name'],arm=args.arm,training_seed=args.seed,step=args.step,
            protocol_sha256=identity,checkpoint_sha256=read(dest/f'checkpoint_{args.step}.json')['sha256']),trajectories=trajectories)
        if (i+1)%64==0:print('RL_EVAL',args.arm,args.seed,args.step,args.split,args.rank,i+1,flush=True)
    atomic_json(dest/f'eval_{args.split}_{args.step}_{args.rank}.json',dict(status='COMPLETE',scenes=len(rows),
        seconds=time.perf_counter()-start,rank=args.rank,shards=args.shards,protocol_sha256=identity,owner=runtime_identity()))


def score(args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from .updates import wait_file
    _,identity=protocol();dest=run_dir(args.arm,args.seed);rows=scene_rows(args.split)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for i in range(0,len(rows),args.workers*4):
            batch=rows[i:i+args.workers*4];tasks=[]
            for r in batch:
                source=dest/f'step_{args.step}/trajectories/{r["token"]}.npz';wait_file(source)
                tasks.append((r,str(source),str(dest/f'step_{args.step}/scores/{r["token"]}.npz'),identity))
            list(pool.map(score_record,tasks))
    atomic_json(dest/f'score_{args.split}_{args.step}.json',dict(status='COMPLETE',scenes=len(rows),protocol_sha256=identity))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['train','evaluate','score'])
    p.add_argument('--arm',choices=['RL-F','RL-J','RL-Q','RL-E'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--step',type=int,default=500);p.add_argument('--split',default='D_cal')
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    try:globals()[args.action](args)
    except Exception:
        atomic_json(run_dir(args.arm,args.seed)/f'{args.action}_failure_{os.environ.get("RANK",args.rank)}.json',
            dict(status='FAILED',traceback=traceback.format_exc(),owner=runtime_identity()))
        raise


if __name__=='__main__':main()
