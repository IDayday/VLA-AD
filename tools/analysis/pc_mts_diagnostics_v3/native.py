"""Native epsilon model primitives; all conditioning is observed cached input."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/mnt/navsim/maps')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
from common_v3 import *
import torch
from distributed_rollout import CachedObservations,load_planner
from transformers.feature_extraction_utils import BatchFeature

def setup(rank=None):
    rank=int(os.environ.get('LOCAL_RANK',0)) if rank is None else rank
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    # A genuine CUDA allocation allows the preexisting pressure script to follow
    # its own cooperative-yield policy. No process is signalled or terminated here.
    probe=torch.empty(1,device=f'cuda:{rank}');torch.cuda.synchronize()
    time.sleep(3)
    return f'cuda:{rank}'

def model(device=None):
    p=load_planner(v1.models()[0]);assert next(p.parameters()).dtype==torch.float32
    return p

def inputs(f,device):
    assert f['_metadata']['observation_only']
    assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot','_metadata'}
    vl=f['last_hidden_state'].unsqueeze(0).to(device).float();h=f['history_trajectory'].view(1,-1).to(device).float();s=f['status_feature'].unsqueeze(0).to(device).float();c=f['high_command_one_hot'].unsqueeze(0).to(device).float()
    def action(n,target=None):
        data=dict(his_traj=h.expand(n,-1),history_trajectory=h.view(1,4,3).expand(n,-1,-1),status_feature=s.expand(n,-1),high_command_one_hot=c.expand(n,-1),state=torch.cat([s,h],1).expand(n,-1))
        if target is not None:data['action']=target
        return BatchFeature(data=data)
    return vl,action

def encode(p,vl,action):
    a=action(1)
    return p.feature_encoder(vl),p.his_traj_encoder(a.his_traj.unsqueeze(1)).repeat(1,8,1),p.ego_status_encoder(a.status_feature)

def predict_epsilon(p,x,t,embeds):
    v,h,e=embeds;n=len(x);v=v.expand(n,-1,-1);h=h.expand(n,-1,-1);e=e.expand(n,*e.shape[1:])
    a=p.action_encoder(x,t)
    if hasattr(p,'position_embedding'):a=a+p.position_embedding(torch.arange(8,device=x.device))
    fused=p.fusion_projector(torch.cat([h,v.mean(1).unsqueeze(1).expand(-1,8,-1),a],2))
    return p.action_decoder(p.model(fused,v,e,t))

def diffusion(p,traj,t,noise,embeds):
    x0=p.norm_odo(traj);alpha=p.extract(p.ddpm_sqrt_alphas_cumprod,t,x0.shape);sigma=p.extract(p.ddpm_sqrt_one_minus_alphas_cumprod,t,x0.shape)
    pred=predict_epsilon(p,alpha*x0+sigma*noise,t,embeds)
    loss=(pred-noise).square().mean((1,2))
    # Epsilon is native. Algebraic x0 estimates are intentionally not reported
    # as a native x0 prediction head; reachability is measured separately.
    return loss,pred

def noise_for(keys,namespace,draw,device):
    return torch.stack([torch.randn((8,3),device=device,generator=torch.Generator(device=device).manual_seed(seed(k,namespace,draw))) for k in keys])

def timesteps_for(keys,namespace,draw,device):
    return torch.cat([torch.randint(0,100,(1,),device=device,generator=torch.Generator(device=device).manual_seed(seed(k,namespace,draw))) for k in keys])

def reverse(p,x,start,embeds):
    n=len(x);v,h,e=embeds
    for i in range(start,p.ddim_steps):
        t=torch.full((n,),int(p.ddim_t[i]),device=x.device,dtype=torch.long);index=torch.full_like(t,i)
        x=p.p_mean_variance(x,t,index,v.expand(n,-1,-1),h.expand(n,-1,-1),e.expand(n,*e.shape[1:]),True)[0]
    return p.denorm_odo(x.clamp(-p.final_action_clip_value,p.final_action_clip_value))

def recon_errors(p,trajectories,keys,embeds,device,namespace='v3_C_reconstruction'):
    x=torch.as_tensor(trajectories,device=device,dtype=torch.float32);xn=p.norm_odo(x);errs=[]
    for ti,t in enumerate([20,40,80]):
        start=int((p.ddim_t==t).nonzero()[0]);alpha=p.ddpm_sqrt_alphas_cumprod[t];sigma=p.ddpm_sqrt_one_minus_alphas_cumprod[t]
        for repeat in range(2):
            rows=[]
            for b in range(0,len(x),16):
                noise=noise_for(keys[b:b+16],namespace,ti*2+repeat,device)
                out=reverse(p,alpha*xn[b:b+16]+sigma*noise,start,embeds)
                rows.append(torch.linalg.vector_norm(out[...,:2]-x[b:b+16,...,:2],dim=-1).mean(-1).detach().cpu().numpy())
            errs.append(np.concatenate(rows))
    return np.stack(errs,1)

def sample(p,vl,action,token,namespace,count,device):
    outputs=[]
    for b in range(0,count,16):
        n=min(16,count-b);keys=[f'{token}:{j}' for j in range(b,b+n)]
        init=noise_for(keys,namespace+'_initial',0,device)
        torch.manual_seed(seed(token,namespace+'_steps',b))
        with torch.no_grad():out=p.get_action(vl.expand(n,-1,-1),action(n),init_actions=init,deterministic=False)['pred_traj']
        outputs.append(out.detach().cpu().numpy())
    return np.concatenate(outputs)

def feature(token):return torch.load(V1/'features'/f'{token}.pt',map_location='cpu',weights_only=False)

def gradient_subset(p):
    lists=[(name,module) for name,module in p.model.named_modules() if isinstance(module,torch.nn.ModuleList) and len(module)==16]
    assert len(lists)==1,[(n,len(m)) for n,m in lists]
    prefix='model.'+lists[0][0]+'.15.'
    params=[(name,param) for name,param in p.named_parameters() if name.startswith(prefix) or name.startswith('action_decoder.')]
    assert params and any(n.startswith(prefix) for n,_ in params)
    return params

def audit():
    import inspect
    device=setup();p=model();rows=[]
    subset=gradient_subset(p)
    save(OUT/'manifests/gradient_parameters.json',dict(identity=identity(),parameters=[dict(name=n,shape=list(v.shape),count=v.numel()) for n,v in subset],total=sum(v.numel() for _,v in subset),selection='last of 16 native DiT blocks plus trajectory action_decoder'))
    for scene in scenes()[:4]:
        token=scene['token'];vl,action=inputs(feature(token),device);traj=torch.tensor(np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'][[0,32,64,128]],device=device,dtype=torch.float32)
        torch.manual_seed(seed(token,'v3_native_loss_parity'))
        noise=torch.randn_like(traj);t=p.sample_time(4,device=device,dtype=traj.dtype)
        with torch.no_grad():ours=diffusion(p,traj,t,noise,encode(p,vl,action))[0].mean()
        torch.manual_seed(seed(token,'v3_native_loss_parity'))
        with torch.no_grad():native=p.forward(vl.expand(4,-1,-1),action(4,traj))['loss']
        err=float(abs(ours-native));assert err<1e-6,err
        # Compare archived official outputs on their exact seed0 single-trajectory path.
        torch.manual_seed(0)
        with torch.no_grad():pred=p.get_action(vl,action(1),deterministic=False)['pred_traj'].cpu().numpy()
        archived=np.load(V1/'rollouts/official_il'/f'{token}.npz')['reference'];pe=float(abs(pred-archived).max());assert pe<1e-6,pe
        rows.append(dict(token=token,native_loss=float(native),implementation_loss=float(ours),loss_abs_error=err,archived_IL_max_abs=pe))
    src=Path(inspect.getfile(p.__class__))
    stage_audit('native',checks=rows,class_name=p.__class__.__module__+'.'+p.__class__.__name__,class_file=str(src),class_sha256=sha(src),checkpoint_sha256=v1.models()[0]['sha256'],parameters=sum(v.numel() for v in p.parameters()),gradient_subset=sum(v.numel() for _,v in subset),timestep_distribution='torch.randint(0,100)',parameterization='epsilon',x0_prediction_error='UNTESTED: model has no native x0 output head',peak_gpu_bytes=torch.cuda.max_memory_allocated())
    print('Native audit PASS',rows,flush=True)

if __name__=='__main__':audit()
