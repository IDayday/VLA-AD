"""Read-only historical sampling through the actual native GRPO entry points.

Capture occurs immediately after native sample_chain, before scoring/updates.
The evaluation-only factorial changes are explicitly diagnostic interventions.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
from common_support import *
import argparse,inspect,copy,types
import torch
from transformers.feature_extraction_utils import BatchFeature

def inputs(token):
    f=torch.load(ROOT/'outputs/pc_mts_diagnostics/features'/f'{token}.pt',map_location='cpu',weights_only=False)
    assert f['_metadata']['observation_only']
    assert set(f)=={'_metadata','last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
    vl=f['last_hidden_state'].unsqueeze(0).cuda().float();h=f['history_trajectory'].reshape(1,-1).cuda().float()
    s=f['status_feature'].unsqueeze(0).cuda().float();c=f['high_command_one_hot'].unsqueeze(0).cuda().float()
    def action(n):return BatchFeature(data={'his_traj':h.expand(n,-1),'history_trajectory':h.reshape(1,4,3).expand(n,-1,-1),'status_feature':s.expand(n,-1),'high_command_one_hot':c.expand(n,-1),'state':torch.cat([s,h],1).expand(n,-1)})
    return vl,action

def load(name):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    m=models()[name];sys.path.insert(0,str(ROOT/'scripts/evaluation/distribution_audit'))
    from run import load_planner
    cfg=dict(m['config']);cfg['grpo']=False
    p=load_planner(dict(m,path=m['checkpoint_path'],config=cfg));p.requires_grad_(False)
    module=sys.modules[p.__class__.__module__];g=module.GRPOConfig();run=m.get('grpo_config',{})
    for key in vars(g):
        native_key='grpo_'+key
        if native_key in run:setattr(g,key,run[native_key])
        elif key in run:setattr(g,key,run[key])
    for key in ['denoised_clip_value','eval_randn_clip_value','randn_clip_value','final_action_clip_value','eps_clip_value','eval_min_sampling_denoising_std','min_sampling_denoising_std','min_logprob_denoising_std','gamma_denoising']:
        if hasattr(g,key):setattr(p,key,getattr(g,key))
    p.grpo_sample_time=16
    p.use_trajectory_level_objective=True;p.use_gspo_ratio=bool(run.get('grpo_use_gspo_ratio',False));p.behavior_policy_sample=bool(run.get('grpo_behavior_policy_sample',False))
    assert not p.use_gspo_ratio,'Behavior-policy sampling requires a separate audited path'
    if m['family'] in ['a5','v6']:
        p.stage3_algorithm='lfp_grpo'
        # These guards are never dereferenced: capture stops before reward access.
        p.lfp_reference_cache=object();p.lfp_metric_adapter=object()
        if m['family']=='v6':
            refm=models()['v6_sft'];refcfg=dict(refm['config']);refcfg['grpo']=False
            ref=load_planner(dict(refm,path=refm['checkpoint_path'],config=refcfg)).requires_grad_(False).eval()
            p.old_policy=ref
    p.train();p.set_frozen_modules_to_eval_mode()
    assert not [n for n,v in p.named_modules() if isinstance(v,torch.nn.modules.batchnorm._BatchNorm) and v.training]
    assert str(Path(inspect.getfile(type(p))).resolve())==str(Path(m['runtime_planner_path']).resolve())
    assert sha(m['runtime_planner_path'])==m['runtime_planner_sha256']
    return p,m

class SamplingCaptured(Exception):pass

def native(p,vl,action,token,capture_detail=False):
    original=p.sample_chain;captured={}
    def hook(*args,**kw):
        chain,traj=original(*args,**kw)
        captured['chain']=chain;captured['trajectory']=traj
        if capture_detail:captured['args']=args;captured['kwargs']=kw
        raise SamplingCaptured()
    p.train();p.set_frozen_modules_to_eval_mode();p.sample_chain=hook
    try:
        entry=p.forward_lfp_grpo if getattr(p,'stage3_algorithm','')=='lfp_grpo' else p.forward_grpo
        with torch.no_grad():entry(vl,action(1),[token],sample_time=16)
    except SamplingCaptured:pass
    finally:p.sample_chain=original
    assert captured and captured['trajectory'].shape==(16,8,3)
    return captured

def draw(p,m,vl,action,token,group,protocol):
    torch.manual_seed(seed(token,group))
    if protocol=='native_grpo':return native(p,vl,action,token)['trajectory'].cpu().numpy()
    p.eval();floor=p.eval_min_sampling_denoising_std;clip=p.eval_randn_clip_value
    if protocol in ['floor_only','floor_and_clip']:p.eval_min_sampling_denoising_std=p.min_sampling_denoising_std
    if protocol in ['clip_only','floor_and_clip']:p.eval_randn_clip_value=float('inf') if m['family'] in ['a5','v6'] else p.randn_clip_value
    try:
        with torch.no_grad():result=p.get_action(vl.expand(16,-1,-1),action(16),deterministic=False)['pred_traj'].cpu().numpy()
    finally:p.eval_min_sampling_denoising_std=floor;p.eval_randn_clip_value=clip
    return result

def smoke(args,p,m):
    before=state_hash(p);checks=[];stds=[];pm=p.p_mean_variance
    def pm_hook(*a,**kw):
        out=pm(*a,**kw)
        if len(stds)<5:
            st=torch.exp(out[1]*.5).clamp(min=p.min_sampling_denoising_std)
            stds.append(dict(timestep=int(a[1][0]),std_min=float(st.min()),std_max=float(st.max())))
        return out
    p.p_mean_variance=pm_hook
    for s in scenes()[:4]:
        t=s['token'];vl,action=inputs(t);torch.manual_seed(seed(t,0));captured=native(p,vl,action,t,True)
        torch.manual_seed(seed(t,0))
        # Exact sampler call captured from the native preamble, including context.
        with torch.no_grad():chain,traj=p.sample_chain(*captured['args'],**captured['kwargs'])
        err=float((traj-captured['trajectory']).abs().max());chainerr=float((chain-captured['chain']).abs().max())
        assert err==0 and chainerr==0,(err,chainerr)
        a=draw(p,m,vl,action,t,0,'eval');b=draw(p,m,vl,action,t,0,'eval');assert np.array_equal(a,b)
        checks.append(dict(token=t,native_entry_vs_extraction_max_abs=err,chain_max_abs=chainerr,eval_repeat_max_abs=float(abs(a-b).max()),unique_members=len(np.unique(traj.cpu().numpy().reshape(16,-1),axis=0))))
    p.p_mean_variance=pm;p.train();p.set_frozen_modules_to_eval_mode();after=state_hash(p);assert before==after
    info=dict(status='PASS',model=args.model,protocol_hash=identity(),checkpoint_hash=m['sha256'],runtime=m['runtime_planner_path'],runtime_hash=m['runtime_planner_sha256'],checks=checks,state_before=before,state_after=after,optimizer_updates=0,group_size=16,steps=p.ddim_steps,per_step_std=stds,eval_std_floor=p.eval_min_sampling_denoising_std,native_std_floor=p.min_sampling_denoising_std,eval_noise_clip=p.eval_randn_clip_value,native_noise_clip='unbounded Gaussian' if m['family'] in ['a5','v6'] else p.randn_clip_value,logprob_std_floor=p.min_logprob_denoising_std,dropout=[dict(name=n,p=v.p,training=v.training) for n,v in p.named_modules() if isinstance(v,torch.nn.Dropout)],capture='actual forward entry interrupted after current-policy sample_chain; before reward/advantage/optimizer',state='train with native frozen-module eval; no mutable BN')
    save(OUT/'audits'/f'sampling_{args.model}.json',info);print('SMOKE PASS',args.model,flush=True)

def run(args,p,m):
    audit=read(OUT/'audits'/f'sampling_{args.model}.json');assert audit['status']=='PASS' and audit['protocol_hash']==identity()
    before=state_hash(p);assert before==audit['state_before'];start=time.time()
    factorial=set(read(OUT/'manifests/scenes.json')['factorial_tokens'])
    rows=scenes()[args.rank::args.world]
    if args.limit:rows=rows[:args.limit]
    for i,s in enumerate(rows):
        token=s['token'];dest=OUT/'cache/rollouts'/args.model/f'{token}.npz'
        meta=dict(protocol_hash=identity(),model=args.model,token=token,checkpoint_hash=m['sha256'],runtime_hash=m['runtime_planner_sha256'],group_size=16,groups=4)
        if v1.valid_npz(dest,meta):continue
        vl,action=inputs(token);out={}
        protocols=list(CFG['protocols'])
        if args.model in CFG['primary_models'] and token in factorial:protocols+=CFG['factorial_protocols']
        for protocol in protocols:out[protocol]=np.stack([draw(p,m,vl,action,token,g,protocol) for g in range(4)])
        assert all(a.shape==(4,16,8,3) and np.isfinite(a).all() for a in out.values())
        npz(dest,dict(meta,group_seeds=[seed(token,g) for g in range(4)]),**out)
        if i%10==0:print('SAMPLE',args.model,args.rank,i+1,len(rows),round(time.time()-start,2),flush=True)
    p.train();p.set_frozen_modules_to_eval_mode();after=state_hash(p);assert before==after
    save(OUT/'audits'/f'run_{args.model}_{args.rank}.json',dict(protocol_hash=identity(),count=len(rows),state_before=before,state_after=after,seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    p,m=load(args.model)
    smoke(args,p,m) if args.smoke else run(args,p,m)
