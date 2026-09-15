"""Native forward epsilon objective on the same union of real teacher targets.

Only RNG hooks and a loss-observation hook; no rewritten normalization or loss.
Common noise/time per scene and draw removes teacher/model noise confounding.
"""
from sample import *
import torch.nn.functional as F

def measure(p,vl,action,targets,token,draw_id):
    p.eval();n=len(targets);a=action(n);a['action']=torch.as_tensor(targets,device='cuda',dtype=torch.float32)
    torch.manual_seed(seed(token,draw_id,'teacher_fitting'))
    epsilon=torch.randn((1,8,3),device='cuda').expand(n,-1,-1).clone()
    t=p.sample_time(1,device=vl.device,dtype=vl.dtype).expand(n).clone()
    oldrand=torch.randn_like;oldtime=p.sample_time;oldmse=F.mse_loss;captured=[];calls=[]
    def normal(x,*args,**kw):
        if tuple(x.shape)==(n,8,3):calls.append(1);return epsilon.clone()
        return oldrand(x,*args,**kw)
    def timestep(count,device,dtype):assert count==n;return t.clone()
    def mse(pred,target,*args,**kw):
        if pred.shape==epsilon.shape and torch.equal(target,epsilon):captured.append(((pred-epsilon).float()**2).mean((1,2)).detach())
        return oldmse(pred,target,*args,**kw)
    torch.randn_like=normal;p.sample_time=timestep;F.mse_loss=mse
    try:
        with torch.no_grad():result=p.forward(vl.expand(n,-1,-1),a)
    finally:torch.randn_like=oldrand;p.sample_time=oldtime;F.mse_loss=oldmse
    assert len(captured)==1 and len(calls)==1,(len(captured),len(calls))
    native=result['diffusion_loss'] if 'diffusion_loss' in result else result['loss']
    # Auxiliary loss may appear in `loss`; prefer explicitly logged diffusion loss.
    err=float(abs(captured[0].mean()-native));assert err<1e-5,err
    return captured[0].cpu().numpy(),int(t[0]),err

def main(args):
    p,m=load(args.model);p.eval();before=state_hash(p);start=time.time();checks=[]
    rows=scenes()[args.rank::args.world]
    if args.limit:rows=rows[:args.limit]
    for i,s in enumerate(rows):
        t=s['token'];source=OUT/'cache/teachers'/f'{t}.npz';dest=OUT/'cache/fitting'/args.model/f'{t}.npz'
        meta=dict(protocol_hash=identity(),model=args.model,checkpoint_hash=m['sha256'],teacher_cache_hash=sha(source),token=t,objective='actual native forward epsilon MSE; eval condition dropout disabled',draws=4)
        if v1.valid_npz(dest,meta):continue
        with np.load(source) as bank:targets=bank['trajectories']
        vl,action=inputs(t);values=[];times=[]
        for j in range(4):
            v,ti,e=measure(p,vl,action,targets,t,j);values.append(v);times.append(ti)
            if i==0:checks.append(dict(draw=j,timestep=ti,native_loss_max_abs=e))
        npz(dest,dict(meta,seeds=[seed(t,j,'teacher_fitting') for j in range(4)]),epsilon_mse=np.stack(values),timesteps=np.asarray(times))
        if i%100==0:print('FITTING',args.model,args.rank,i+1,len(rows),round(time.time()-start,1),flush=True)
    assert state_hash(p)==before
    save(OUT/'audits'/f'fitting_{args.model}_{args.rank}.json',dict(status='PASS',protocol_hash=identity(),scenes=len(rows),native_forward_checks=checks,state_before=before,state_after=state_hash(p),optimizer_updates=0,scope='Teacher fitting uses future supervision only as loss target. All rollouts use observation-only action input. Raw loss units differ across legacy and FS parameterizations.'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--rank',type=int,default=0);a.add_argument('--world',type=int,default=1);a.add_argument('--limit',type=int,default=0);main(a.parse_args())
