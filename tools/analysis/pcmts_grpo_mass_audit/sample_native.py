"""Read-only extraction of the actual forward_grpo current-policy sample_chain."""
from common_mass import *
import argparse, time, inspect, types, concurrent.futures, multiprocessing
import torch
import native
from train import attach_grpo

def state_hash(p):
    h=hashlib.sha256()
    for k,v in p.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def prepare(rank):
    device=native.setup(rank);p=native.model();ref=native.model()
    recipe=attach_grpo(p,ref,{s['token']:s for s in scenes()})
    p.train();ref.eval();p.requires_grad_(False);p.set_frozen_modules_to_eval_mode()
    bn=[n for n,m in p.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.training]
    assert not bn,('Native state has mutable BN; do not silently use eval',bn)
    return p,device,recipe

def sample_group(p,vl,action,token,stream,group):
    torch.manual_seed(random_seed(token,stream,group))
    p.set_frozen_modules_to_eval_mode()
    a=action(1);g=cfg()['rollouts']['group_size']
    # These are precisely the three repeat_interleave operations in forward_grpo.
    with torch.no_grad():
        chain,traj=p.sample_chain(vl.repeat_interleave(g,0),a.his_traj.repeat_interleave(g,0),a.status_feature.repeat_interleave(g,0),deterministic=False)
    return traj.cpu().numpy(),chain

def cpu_init():
    from scoring import init
    init()
def cpu_score(scene,trajectories):
    from scoring import online
    return online(scene,trajectories)
def cpu_parity(scene,trajectories):
    from evaluate_cached_rollouts import parity
    return parity(scene,trajectories)

def smoke(rank):
    p,device,recipe=prepare(rank);original_hash=state_hash(p);checks=[];std_rows=[]
    drop=[dict(name=n,training=m.training,p=m.p) for n,m in p.named_modules() if isinstance(m,torch.nn.Dropout)]
    original_pm=p.p_mean_variance
    def pm(self,*args,**kw):
        out=original_pm(*args,**kw)
        std=torch.exp(.5*out[1]).clamp(min=self.min_sampling_denoising_std)
        if len(std_rows)<5:std_rows.append(dict(timestep=int(args[1][0]),sampling_std_min=float(std.min()),sampling_std_max=float(std.max())))
        return out
    p.p_mean_variance=types.MethodType(pm,p)
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    ts=TrajectorySampling(time_horizon=4,interval_length=.1)
    p.simulator=PDMSimulator(ts);p.train_scorer=PDMScorer(ts,GRPOConfig().scorer_config)
    original_reward=p.reward_fn;original_logp=p.get_logprobs
    with concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_init) as pool:
        for scene in sorted(scenes(),key=lambda s:digest(['smoke',s['token']]))[:8]:
            token=scene['token'];vl,action=native.inputs(native.feature(token),device);capture={}
            def reward(self,trajs,toks,caches):
                result=original_reward(trajs,toks,caches);capture['trajs']=trajs.cpu().numpy().copy();capture['reward']=result.clone()
                return result
            def logp(self,*args,**kw):
                result=original_logp(*args,**kw)
                if 'logp' not in capture:capture['logp']=result.detach().clone()
                return result
            p.reward_fn=types.MethodType(reward,p);p.get_logprobs=types.MethodType(logp,p)
            torch.manual_seed(random_seed(token,'smoke',0))
            with torch.no_grad():out=p.forward_grpo(vl,action(1),[token],sample_time=8,use_bc_loss=True)
            extracted,chain=sample_group(p,vl,action,token,'smoke',0)
            err=float(np.max(np.abs(extracted-capture['trajs'])));assert err==0,err
            standard,train=pool.submit(cpu_score,scene,extracted).result()
            reward_error=float(np.max(np.abs(capture['reward'].cpu().numpy()-train[:,6].astype(np.float32))));assert reward_error<=1e-8,reward_error
            r=capture['reward'].view(1,8);adv=((r-r.mean(1,keepdim=True))/(r.std(1,keepdim=True)+1e-8)).view(-1)
            adv=adv.clamp(torch.quantile(adv,p.clip_advantage_lower_quantile),torch.quantile(adv,p.clip_advantage_upper_quantile))
            steps=chain.shape[1]-1;discount=p.gamma_denoising**(steps-torch.arange(steps,device=device)-1)
            expected=-(capture['logp'].clamp(-5,2).mean((1,2))*(adv[:,None]*discount).reshape(-1)).mean()
            adv_error=float(abs(expected-out['policy_loss']));assert adv_error<=1e-8,adv_error
            scoring_check=pool.submit(cpu_parity,scene,extracted).result()
            checks.append(dict(token=token,trajectory_max_abs=err,reward_max_abs=reward_error,advantage_loss_max_abs=adv_error,scalar_batch= scoring_check,unique_native_members=len({content_hash(x) for x in extracted})))
            print('smoke',len(checks),token,err,flush=True)
    after=state_hash(p);assert original_hash==after
    source=Path(inspect.getfile(p.__class__))
    import common as v1
    ident=dict(status='PASS',checkpoint_path=v1.models()[0]['checkpoint_path'],checkpoint_hash=sha(v1.models()[0]['checkpoint_path']),runtime_code_path=str(source),runtime_code_hash=sha(source),class_name=str(p.__class__),sampling_steps=p.ddim_steps,eta=float(p.eta(torch.zeros(1,device=device))[0]),initial_noise_distribution='independent standard normal torch.randn; no initial clipping',per_step_sampling_std=std_rows,sampling_noise_clip=p.randn_clip_value,final_action_clip=p.final_action_clip_value,logprob_std_floor=p.min_logprob_denoising_std,network_train_eval_state={n:m.training for n,m in p.named_modules()},dropout_state=drop,group_size=8,within_group_noise_protocol='distinct elementwise standard-normal draws; clipped per-step at native bound; one independently seeded RNG per group',weight_and_buffer_hash_before=original_hash,weight_and_buffer_hash_after=after,optimizer_updates=0,recipe=recipe,checks=checks,group_dependence_assumption='Use independent-group cluster intervals conservatively; no member-level iid claim required',fp32=True,completed_at=utc())
    save(OUT/'manifests/sampler_identity.json',ident)
    print('NATIVE PARITY PASSED',len(checks),flush=True)

def run(rank,world,token_file=None):
    audit=read(OUT/'manifests/sampler_identity.json');assert audit['status']=='PASS'
    from batch_native import sample_groups
    assert read(OUT/'audits/batch_native_parity.json')['status']=='PASS'
    p,device,_=prepare(rank);before=state_hash(p);assert before==audit['weight_and_buffer_hash_before']
    all_rows=scenes()
    if token_file is not None:
        wanted=set(read(token_file)['tokens']);all_rows=[s for s in all_rows if s['token'] in wanted]
    rows=all_rows[rank::world];start=time.time();done=0
    for scene in rows:
        token=scene['token'];vl,action=native.inputs(native.feature(token),device)
        for stream,count in [('A',1024),('B',1024),('C',64)]:
            path=OUT/'cache/rollouts'/stream/f'{token}.npz';meta=dict(token=token,stream=stream,checkpoint_hash=audit['checkpoint_hash'],runtime_code_hash=audit['runtime_code_hash'],group_size=8,samples=count)
            if valid(path,meta):continue
            trajectories=[];seeds=[]
            for begin in range(0,count//8,8):
                groups=list(range(begin,min(begin+8,count//8)))
                t,_=sample_groups(p,vl,action,token,stream,groups);trajectories.append(t);seeds.extend(random_seed(token,stream,g) for g in groups)
            npz(path,dict(meta,batch_groups=8,batch_adapter_hash=sha(Path(__file__).with_name('batch_native.py'))),trajectories=np.concatenate(trajectories),group_seeds=np.asarray(seeds,dtype=np.int64),group_ids=np.repeat(np.arange(count//8),8),member_ids=np.tile(np.arange(8),count//8))
        done+=1
        if done%2==0:print('rollouts',rank,done,len(rows),round(time.time()-start,1),flush=True)
    after=state_hash(p);assert after==before
    label=f'rollouts_rank{rank}' if token_file is None else f'rollouts_rebalanced_rank{rank}'
    save(OUT/'audits'/f'{label}.json',dict(scenes=len(rows),state_before=before,state_after=after,seconds=time.time()-start,rank=rank,world=world,gpu_peak_bytes=torch.cuda.max_memory_allocated(),host=os.uname().nodename,token_file=token_file))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');a.add_argument('--rank',type=int,default=0);a.add_argument('--world',type=int,default=8);a.add_argument('--token-file');args=a.parse_args()
    smoke(args.rank) if args.smoke else run(args.rank,args.world,args.token_file)
