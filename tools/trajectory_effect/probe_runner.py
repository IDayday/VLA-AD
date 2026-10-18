"""Matched-capacity frozen/joint probes on actual held-out update consequences."""
from __future__ import annotations
import argparse
from functools import lru_cache
import math
import traceback
import numpy as np
from .common import *
from .bank import load_observation,baseline_path,baseline_score_path
from .candidates import library_path
from .updates import families,targets,trajectory_path,score_path,set_seed,tensor_digest,wait_file


def run_dir(name):return ROOT/'probes'/name


def load_library(token):
    package='confirmation' if (ROOT/'confirmation/activation.json').exists() else 'initial'
    value=np.load(library_path(token,package))
    return dict(trajectories=value['trajectories'],labels=value['labels'][:,-1],pairs=value['pairs'])


def schedule(seed,steps,batch=4):
    rows=scene_rows('D_fit');rng=np.random.default_rng(seed);indices=[]
    while len(indices)<steps*batch:indices.extend(rng.permutation(len(rows)).tolist())
    return [rows[i]['token'] for i in indices[:steps*batch]]


def load_adapter(agent,state):
    import torch
    current=dict(agent.backbone.named_parameters())
    if set(state)!={n for n,p in current.items() if p.requires_grad}:raise ValueError('VLM adaptation scope changed')
    with torch.no_grad():
        for n,v in state.items():current[n].copy_(v.to(current[n]))


def build(arm,seed):
    import torch
    from .runtime import load_agent
    from .probes import TrajectoryScoreHead,adapt_vlm
    joint=arm in ('JQ','JE')
    agent=load_agent() if joint else None
    # Same head initialization across all arms, regardless of VLM construction.
    set_seed(seed_for('shared_score_head',seed));head=TrajectoryScoreHead().cuda()
    if joint:
        set_seed(seed_for('shared_vlm_adapter',seed));adapt_vlm(agent)
    return agent,head


def train(args):
    import torch
    import torch.distributed as dist
    from .runtime import configure,encode,action_input
    from .probes import effect_loss,vlm_parameters,vlm_state,joint_training_mode
    from .distributed import initialize,sum_values,average_gradients,finish
    package='confirmation' if (ROOT/'confirmation/activation.json').exists() else 'initial'
    manifest=ROOT/'candidates'/('confirmation_D_fit_complete.json' if package=='confirmation' else 'D_fit_complete.json')
    wait_file(manifest)
    rank,world=initialize();configure();cfg,identity=protocol()
    assert args.steps==(300 if args.phase=='tuning' else 2000)
    if world not in (1,2,4):raise ValueError('World size must divide fixed global batch4')
    dest=run_dir(args.name);joint=args.arm in ('JQ','JE');start=time.perf_counter()
    if args.arm in ('FQ','JQ') and args.lambda_diff!=0:raise ValueError('Q control cannot receive difference loss')
    if args.arm in ('T','FE','JE') and args.lambda_diff<=0:raise ValueError('Effect arm requires the registered difference loss')
    actual=dict(arm=args.arm,seed=args.seed,steps=args.steps,head_lr=args.head_lr,vlm_lr=cfg['probes']['vlm_lr'],
        lambda_diff=args.lambda_diff,global_batch=4,microbatch=1,world_size=world,protocol_sha256=identity,
        phase=args.phase,library_package=package,fit_library_manifest_sha256=sha(manifest))
    if rank==0:
        if (dest/'complete.json').exists():raise ValueError('Do not overwrite completed probe')
        atomic_json(dest/'config.json',actual);atomic_json(dest/'owner.json',runtime_identity())
    agent,head=build(args.arm,args.seed);initial_head=tensor_digest({n:p.detach().cpu() for n,p in head.state_dict().items()})
    initial_vlm=vlm_state(agent) if joint else {}
    groups=[dict(params=list(head.parameters()),lr=args.head_lr)]
    if joint:groups.append(dict(params=vlm_parameters(agent),lr=cfg['probes']['vlm_lr']))
    params=[p for group in groups for p in group['params']]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    tokens=schedule(args.seed,args.steps)
    if rank==0:atomic_json(dest/'training_scene_schedule.json',tokens)
    history=[];head.train();last_vlm_norm=0.;max_vlm_norm=0.
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True);chosen=tokens[step*4:(step+1)*4]
        local=[(t,load_library(t)) for t in chosen[rank::world]]
        counts=sum_values([sum(len(v['labels']) for _,v in local),sum(len(v['pairs']) for _,v in local)],world)
        values=np.zeros(3,dtype=np.float64)
        for token,value in local:
            payload=load_observation(token);inputs=action_input(payload['features'])
            if joint:
                joint_training_mode(agent);context=encode(agent,payload['features'],grad=True)
            else:context=payload['tokens'].cuda()
            pred=head(context,torch.from_numpy(value['trajectories']).cuda()[None],inputs.state,remove_visual=args.arm=='T')
            pairs=torch.from_numpy(value['pairs']).cuda();labels=torch.from_numpy(value['labels']).cuda()[None]
            _,absolute,difference=effect_loss(pred,labels,pairs,args.lambda_diff)
            # Two separately normalized sample means, even with variable C/P.
            loss=world*(absolute*len(value['labels'])/counts[0]+args.lambda_diff*difference*len(value['pairs'])/counts[1])
            loss.backward()
            values+=np.array([float(absolute)*len(value['labels']),float(difference)*len(value['pairs']),1.])
        average_gradients(params,world)
        if joint:
            last_vlm_norm=sum(float(p.grad.float().square().sum()) for p in vlm_parameters(agent) if p.grad is not None)**.5
            max_vlm_norm=max(max_vlm_norm,last_vlm_norm)
        head_norm=sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        if not np.isfinite(norm):raise ValueError('Nonfinite supervised gradient')
        optimizer.step()
        if step==0 or (step+1)%25==0 or step+1==args.steps:
            metrics=sum_values(values,world).cpu().numpy()
            if rank==0:
                entry=dict(step=step+1,L_abs=float(metrics[0]/float(counts[0])),L_diff=float(metrics[1]/float(counts[1])),
                    vlm_gradient_norm=last_vlm_norm,head_gradient_norm=head_norm,gradient_norm=norm,seconds=time.perf_counter()-start)
                history.append(entry);atomic_json(dest/'progress.json',history);print('PROBE',args.name,entry,flush=True)
    current_vlm=vlm_state(agent) if joint else {}
    changed=[n for n,p in current_vlm.items() if not torch.equal(p,initial_vlm[n])]
    if joint and (max_vlm_norm<=0 or not changed):raise ValueError('Nominally joint model did not update VLM')
    state=dict(head={n:p.detach().cpu() for n,p in head.state_dict().items()},vlm=current_vlm,
        optimizer=optimizer.state_dict(),step=args.steps,cursor=4*args.steps,config=actual,
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state())
    if world>1:
        states=[None for _ in range(world)];dist.all_gather_object(states,dict(torch_rng=state['rng'],cuda_rng=state['cuda_rng']))
        state['rank_rng']=states
    if rank==0:
        atomic_torch(dest/'checkpoint.pt',state)
        atomic_json(dest/'complete.json',dict(status='COMPLETE',config=actual,checkpoint_sha256=sha(dest/'checkpoint.pt'),
            initial_head_sha256=initial_head,initial_vlm_sha256=tensor_digest(initial_vlm) if joint else None,
            final_vlm_sha256=tensor_digest(current_vlm) if joint else None,changed_vlm_parameters=changed,
            max_vlm_gradient_norm=max_vlm_norm,seconds=time.perf_counter()-start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),observations_recomputed=joint,
            optimizer_updates=args.steps,owner=runtime_identity()))
    finish(world)


def evaluate(args):
    import torch
    from .runtime import configure,encode,action_input,load_agent
    from .distributed import initialize,finish
    rank,world=initialize()
    if world>1:args.rank=rank;args.shards=world
    configure();cfg,identity=protocol();dest=run_dir(args.name);run=read(dest/'config.json')
    if args.split=='D_audit' and run['phase']=='tuning':raise ValueError('Tuning cannot access audit')
    checkpoint=torch.load(dest/'checkpoint.pt',map_location='cpu',weights_only=False)
    agent,head=build(run['arm'],run['seed']);head.load_state_dict(checkpoint['head']);head.eval()
    joint=agent is not None
    if joint:load_adapter(agent,checkpoint['vlm']);agent.eval()
    if args.shuffle and agent is None:agent=load_agent();agent.eval()
    donors=read(ROOT/'manifests'/f'{args.split}_observation_shuffle.json')['donors'] if args.shuffle else None
    selected=[f for f in families() if 'D_'+f['split']==args.split]
    rows=targets(selected[0]);all_results=[];start=time.perf_counter()
    for i,row in enumerate(rows[args.rank::args.shards]):
        token=row['token'];payload=load_observation(token);inputs=action_input(payload['features'])
        if args.shuffle:
            # Swap current image only; keep target ego history and command.
            donor=load_observation(donors[token]);features=dict(payload['features'])
            features['image_path_tensor']=donor['features']['image_path_tensor']
            context=encode(agent,features)
        else:context=encode(agent,payload['features']) if joint else payload['tokens'].cuda()
        old=np.load(baseline_path(token))['trajectories'][:5]
        old_scores=np.load(baseline_score_path(token))['scores'][:5,-1]
        arrays=[old];true=[]
        for f in selected:
            for step in (1,4):
                arrays.append(np.load(trajectory_path(f['id'],step,token))['trajectories'])
                true.append(np.load(score_path(f['id'],step,token))['scores'][:,-1]-old_scores)
        with torch.no_grad():
            pred=head(context,torch.from_numpy(np.concatenate(arrays)).cuda()[None],inputs.state,remove_visual=run['arm']=='T').cpu().numpy()[0].reshape(-1,5)
        delta_pred=pred[1:]-pred[0];true=np.array(true)
        all_results.append(dict(token=token,log=row['log_name'],families=[f['id'] for f in selected],
            prediction=delta_pred.reshape(len(selected),2,5).tolist(),target=true.reshape(len(selected),2,5).tolist(),
            base_score=old_scores.tolist(),predicted_base=pred[0].tolist()))
        if (i+1)%128==0:print('PROBE_EVAL',args.name,args.split,args.rank,i+1,flush=True)
    label=args.split+('_shuffled' if args.shuffle else '')
    atomic_json(dest/f'eval_{label}_{args.rank}.json',dict(status='COMPLETE',protocol_sha256=identity,
        checkpoint_sha256=sha(dest/'checkpoint.pt'),rows=all_results,rank=args.rank,shards=args.shards,
        seconds=time.perf_counter()-start,owner=runtime_identity()))
    finish(world)


def merge_eval(name,split):
    dest=run_dir(name);paths=list(dest.glob(f'eval_{split}_*.json'))
    values=[read(p) for p in paths]
    if not values or len(values)!=values[0]['shards'] or set(v['rank'] for v in values)!=set(range(values[0]['shards'])):
        raise ValueError(f'Incomplete probe evaluation: {name}/{split}')
    rows=sorted([r for v in values for r in v['rows']],key=lambda r:r['token'])
    if len({r['token'] for r in rows})!=len(rows):raise ValueError('Duplicate evaluation scene')
    target=np.array([r['target'] for r in rows]);pred=np.array([r['prediction'] for r in rows])
    return rows,pred,target


def select(args):
    cfg,identity=protocol();dest=ROOT/'probes/selection.json'
    if dest.exists():raise ValueError('Selection is already locked')
    candidates={}
    for arm in cfg['probes']['arms']:
        candidates[arm]=[]
        for slot in range(3):
            name=f'tuning_{arm}_{slot}';rows,pred,target=merge_eval(name,'D_cal')
            rmse=float(np.sqrt(np.mean((pred-target)**2)));zero=float(np.sqrt(np.mean(target**2)))
            candidates[arm].append(dict(name=name,slot=slot,RMSE=rmse,zero_RMSE=zero,relative_RMSE=rmse/zero,config=read(run_dir(name)/'config.json')))
    selected={}
    for arm in ('T','FQ','JQ'):
        selected[arm]=min(candidates[arm],key=lambda r:(r['RMSE'],r['slot']))['config']
    slot=min(range(3),key=lambda s:(np.mean([candidates[a][s]['relative_RMSE'] for a in ('FE','JE')]),s))
    for arm in ('FE','JE'):selected[arm]=candidates[arm][slot]['config']
    atomic_json(dest,dict(status='LOCKED',protocol_sha256=identity,candidates=candidates,selected=selected,
        common_lambda=cfg['probes']['effect_lambda_candidates'][slot],selection_split='D_cal x cal update families'))
    print(json.dumps(dict(selected=selected,common_lambda=cfg['probes']['effect_lambda_candidates'][slot]),indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['train','evaluate','select'])
    p.add_argument('--name');p.add_argument('--arm',choices=['T','FQ','FE','JQ','JE']);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--head-lr',type=float,default=3e-4)
    p.add_argument('--lambda-diff',type=float,default=0.);p.add_argument('--phase',choices=['tuning','formal','confirmation'],default='formal')
    p.add_argument('--split',default='D_cal');p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shuffle',action='store_true')
    args=p.parse_args()
    try:globals()[args.action](args)
    except Exception:
        atomic_json(run_dir(args.name or 'selection')/f'failure_{os.environ.get("RANK",args.rank)}.json',
            dict(status='FAILED',traceback=traceback.format_exc(),owner=runtime_identity()))
        raise


if __name__=='__main__':main()
