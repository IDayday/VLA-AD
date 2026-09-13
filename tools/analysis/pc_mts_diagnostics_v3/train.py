"""E/F controlled action-head updates with immutable schedules and explicit parent weights."""
import argparse,copy,math,concurrent.futures,multiprocessing,types
from native import *
from torch.utils.data import DataLoader

def parent_slots(indices,token,train_seed,step,micro,k=16):
    ids=list(indices);fallback=not ids
    if fallback:ids=[0]  # Explicit GT training fallback; never added to the primary pool.
    rng=np.random.default_rng(seed(token,f'v3_train_slots_{train_seed}_{step}',micro));ids=list(np.asarray(ids)[rng.permutation(len(ids))]);slots=np.resize(np.asarray(ids,dtype=np.int64),k)
    unique,counts=np.unique(slots,return_counts=True);multiplicity=dict(zip(unique,counts));weights=np.asarray([1./multiplicity[i]/len(unique) for i in slots],dtype=np.float32)
    assert abs(weights.sum()-1)<1e-6
    for i in unique:assert abs(weights[slots==i].sum()-1/len(unique))<1e-6
    return slots,weights,fallback

def order_rows(train_seed,steps):
    tr=tokens('train');lookup={r['token']:r for r in scenes()};rng=np.random.default_rng(seed(train_seed,'v3_update_scene_order'));order=[]
    while len(order)<steps*8:order.extend(np.asarray(tr)[rng.permutation(len(tr))].tolist())
    return [lookup[t] for t in order[:steps*8]]

def checkpoint_path(stage,method,train_seed,step):return OUT/'checkpoints'/stage/f'{method}_seed{train_seed}'/f'step{step:04d}.pt'
def head_state(p):return {k:v.detach().cpu() for k,v in p.state_dict().items() if not k.startswith('old_policy.')}
def checkpoint(p,opt,stage,method,train_seed,step,metadata):
    path=guard(checkpoint_path(stage,method,train_seed,step));path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(dict(state_dict=head_state(p),optimizer=opt.state_dict(),step=step,metadata=dict(identity=identity(),stage=stage,method=method,seed=train_seed,**metadata)),tmp);tmp.replace(path)
    save(path.with_suffix('.json'),dict(identity=identity(),path=str(path),sha256=sha(path),step=step,method=method,seed=train_seed,**metadata))

def attach_grpo(p,ref,scene_lookup):
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
    cfg=GRPOConfig()
    names=['denoised_clip_value','eval_randn_clip_value','randn_clip_value','final_action_clip_value','eps_clip_value','eval_min_sampling_denoising_std','min_sampling_denoising_std','min_logprob_denoising_std','clip_advantage_lower_quantile','clip_advantage_upper_quantile','gamma_denoising']
    for obj in [p,ref]:
        for key in names:setattr(obj,key,getattr(cfg,key))
    ref.requires_grad_(False);ref.eval();p.old_policy=ref;p.config.grpo=True
    p.metric_cache_loader=types.SimpleNamespace(metric_cache_paths={t:Path(r['metric_cache_path']) for t,r in scene_lookup.items()})
    return {k:getattr(cfg,k) for k in names}

def sft_loss(p,vl,action,trajs,weights,gt,noise_key,device):
    a=torch.tensor(np.concatenate([trajs,gt[None]]),device=device,dtype=torch.float32);n=len(trajs)
    # Every method gets the same timestep/epsilon slot budget, including repeated parents.
    torch.manual_seed(seed(noise_key,'v3_sft_training_noise'));noise=torch.randn_like(a);t=p.sample_time(len(a),device=device,dtype=a.dtype)
    losses,_=diffusion(p,a,t,noise,encode(p,vl,action));w=torch.tensor(weights,device=device)
    candidate=(losses[:n]*w).sum();retention=losses[-1];return candidate+.25*retention,candidate,retention

def main(a):
    stage=a.stage;method=a.method;train_seed=a.seed;cfg=CFG['training']['micro_sft' if stage=='sft' else 'grpo'];steps=cfg['steps']
    assert (OUT/'manifests/audit_D.json').exists() and (OUT/'manifests/audit_native.json').exists()
    if stage=='grpo':assert (OUT/'manifests/audit_E.json').exists() and (OUT/'manifests/audit_F_recipe.json').exists()
    final=checkpoint_path(stage,method,train_seed,steps)
    complete=OUT/'manifests'/f'train_{stage}_{method}_{train_seed}.json'
    if final.exists() and complete.exists():print('Already complete',complete,flush=True);return
    device=setup();torch.manual_seed(seed(train_seed,'v3_model_initialization'));p=model();initial_hash=v1.models()[0]['sha256'];p.train()
    rows=order_rows(train_seed,steps);lookup={r['token']:r for r in scenes()};pool=None;grpo_recipe=None;last_reward={}
    if stage=='grpo':
        initpath=checkpoint_path('sft',method,train_seed,CFG['training']['micro_sft']['steps']);state=torch.load(initpath,map_location='cpu',weights_only=False);p.load_state_dict(state['state_dict'],strict=True);initial_hash=sha(initpath)
        ref=model();grpo_recipe=attach_grpo(p,ref,lookup)
        from scoring import init,online
        pool=concurrent.futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=init)
        def reward_hook(self,trajs,tokens_list,metric_cache):
            assert len(set(tokens_list))==1 and tokens_list[0] in set(tokens('train'))
            standard,reward=pool.submit(online,lookup[tokens_list[0]],trajs.detach().cpu().numpy()).result()
            r=torch.as_tensor(reward[:,6],device=trajs.device,dtype=trajs.dtype).detach();matrix=r.view(1,8);adv=((matrix-matrix.mean(1,keepdim=True))/(matrix.std(1,keepdim=True)+1e-8)).view(-1);adv=adv.clamp(torch.quantile(adv,self.clip_advantage_lower_quantile),torch.quantile(adv,self.clip_advantage_upper_quantile))
            last_reward.update(standard=standard,reward=reward,advantages=adv.cpu().numpy(),trajectories=trajs.detach().cpu().numpy())
            return r
        p.reward_fn=types.MethodType(reward_hook,p)
    params=[v for n,v in p.named_parameters() if v.requires_grad and not n.startswith('old_policy.')]
    opt=torch.optim.AdamW(params,lr=cfg['lr'],betas=tuple(CFG['training']['betas']),weight_decay=CFG['training']['weight_decay'])
    metadata=dict(initial_checkpoint_sha256=initial_hash,reference_checkpoint_sha256=v1.models()[0]['sha256'],train_tokens_sha256=digest(tokens('train')),holdout_tokens_sha256=digest(tokens('holdout')),effective_scene_batch=8,candidate_slots=16,grpo_recipe=grpo_recipe,trainable_parameter_count=sum(v.numel() for v in params),native_loss='epsilon_MSE' if stage=='sft' else 'original forward_grpo with logging-only reward hook')
    checkpoint(p,opt,stage,method,train_seed,0,metadata)
    loader=DataLoader(CachedObservations(rows,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');iterator=iter(loader);logs=[];ledger=[];rollout_logs=[];start=time.time()
    for step in range(1,steps+1):
        if stage=='sft':
            warm=cfg['warmup_steps'];factor=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/(steps-warm)))
            for group in opt.param_groups:group['lr']=cfg['lr']*factor
        opt.zero_grad(set_to_none=True);vals=[]
        for micro in range(8):
            token,f=next(iterator);assert token in set(tokens('train'));vl,action=inputs(f,device);key=f'{train_seed}:{step}:{micro}:{token}'
            if stage=='sft':
                scene=lookup[token];raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];ids=[0] if method=='gt_only' else read(OUT/'cache/pools'/f'{token}.json')['methods'][method]['indices']
                slots,weights,fallback=parent_slots(ids,token,train_seed,step,micro);loss,fit,ret=sft_loss(p,vl,action,raw[slots],weights,np.asarray(scene['gt']),key,device)
                ledger.append(dict(step=step,micro=micro,token=token,parents=slots.tolist(),weights=weights.tolist(),unique_count=len(set(slots)),explicit_GT_fallback=fallback))
                vals.append([float(loss.detach()),float(fit.detach()),float(ret.detach())])
            else:
                torch.manual_seed(seed(key,'v3_GRPO_training_CRN'));out=p.forward_grpo(vl,action(1),[token],sample_time=8,bc_coeff=.1,use_bc_loss=True);loss=out['loss'];vals.append([float(loss.detach()),float(out['policy_loss'].detach()),float(out['bc_loss'].detach())])
                s=last_reward['standard'];adv=last_reward['advantages'];f=feasible(s)
                for j in range(8):rollout_logs.append(dict(step=step,micro=micro,token=token,group=f'{step}:{micro}',member=j,reward=float(last_reward['reward'][j,6]),advantage=float(adv[j]),feasible=bool(f[j]),hard_failure=bool(hard_failure(s[j:j+1])[0]),**{FIELDS[k]:float(s[j,k]) for k in range(7)}))
            assert torch.isfinite(loss);(loss/8).backward()
        norm=torch.nn.utils.clip_grad_norm_(params,CFG['training']['gradient_clip_norm']);assert torch.isfinite(norm);opt.step()
        logs.append(dict(step=step,loss=float(np.mean([v[0] for v in vals])),candidate_or_policy_loss=float(np.mean([v[1] for v in vals])),GT_or_BC_loss=float(np.mean([v[2] for v in vals])),gradient_norm_before_clip=float(norm),lr=opt.param_groups[0]['lr'],seconds=time.time()-start))
        if step in cfg['snapshots']:checkpoint(p,opt,stage,method,train_seed,step,metadata)
        if step%10==0:print(json.dumps(dict(stage=stage,method=method,seed=train_seed,**logs[-1])),flush=True)
    if pool:pool.shutdown()
    csv(pd.DataFrame(logs),f'{stage}_{method}_{train_seed}_training.csv')
    if ledger:save(OUT/'cache/training_ledgers'/f'{stage}_{method}_{train_seed}.json',dict(identity=identity(),records=ledger))
    if rollout_logs:parquet(pd.DataFrame(rollout_logs),f'F_rollouts_{method}_{train_seed}.parquet')
    save(complete,dict(identity=identity(),stage=stage,method=method,seed=train_seed,steps=steps,scene_updates=steps*8,distinct_train_scenes=len(set(r['token'] for r in rows)),snapshots=cfg['snapshots'],initial_checkpoint_sha256=initial_hash,final_checkpoint_sha256=sha(final),peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.time()-start,explicit_GT_fallback_updates=sum(r['explicit_GT_fallback'] for r in ledger),**{k:v for k,v in metadata.items() if k!='initial_checkpoint_sha256'}))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['sft','grpo']);a.add_argument('method',choices=CFG['training']['methods']);a.add_argument('seed',type=int);main(a.parse_args())
