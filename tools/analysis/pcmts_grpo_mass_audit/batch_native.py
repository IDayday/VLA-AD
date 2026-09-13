"""Native independent groups batched without changing any sampler operation."""
from common_mass import *
import torch, contextlib, types, argparse, time
import native
from sample_native import prepare,state_hash

@contextlib.contextmanager
def group_random_streams(token,stream,groups,device):
    """Route only the 6 native noise draws through independent per-group RNGs.

    Each generator draws exactly the same (8,8,3) initial and five step noises
    as the unbatched entry; member dimensions are not shared or broadcast.
    """
    generators=[torch.Generator(device=device).manual_seed(random_seed(token,stream,g)) for g in groups]
    randn,randn_like=torch.randn,torch.randn_like;calls=[];n=len(groups)*8
    def draw(shape,dev,dtype):
        assert tuple(shape)==(n,8,3),shape
        calls.append(len(calls))
        return torch.cat([randn((8,8,3),device=dev,dtype=dtype,generator=g) for g in generators])
    def normal(*args,**kw):
        shape=args[0] if len(args)==1 and isinstance(args[0],(tuple,list,torch.Size)) else args
        if tuple(shape)==(n,8,3) and len(calls)<6:return draw(shape,kw['device'],kw.get('dtype',torch.float32))
        return randn(*args,**kw)
    def like(x,*args,**kw):
        if tuple(x.shape)==(n,8,3) and len(calls)<6:return draw(x.shape,x.device,x.dtype)
        return randn_like(x,*args,**kw)
    torch.randn,torch.randn_like=normal,like
    try:yield calls
    finally:torch.randn,torch.randn_like=randn,randn_like

def sample_groups(p,vl,action,token,stream,groups):
    n=len(groups)*8;a=action(1);p.set_frozen_modules_to_eval_mode()
    with torch.no_grad(),group_random_streams(token,stream,groups,vl.device) as calls:
        chain,traj=p.sample_chain(vl.repeat_interleave(n,0),a.his_traj.repeat_interleave(n,0),a.status_feature.repeat_interleave(n,0),deterministic=False)
    assert len(calls)==6
    return traj.cpu().numpy(),chain

def audit(rank):
    from sample_native import sample_group,cpu_init,cpu_score
    import concurrent.futures,multiprocessing
    p,device,_=prepare(rank);before=state_hash(p);checks=[];original=p.reward_fn
    assert not [m for m in p.modules() if isinstance(m,torch.nn.Dropout) and m.training and m.p>0]
    with concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_init) as pool:
        for scene in sorted(scenes(),key=lambda s:digest(['smoke',s['token']]))[:8]:
            t=scene['token'];vl,action=native.inputs(native.feature(t),device);groups=list(range(8));capture={}
            def reward(self,trajs,toks,cache):
                capture['trajs']=trajs.cpu().numpy().copy()
                _,r=pool.submit(cpu_score,scene,capture['trajs']).result()
                return torch.tensor(r[:,6],device=device,dtype=torch.float32)
            p.reward_fn=types.MethodType(reward,p)
            with torch.no_grad(),group_random_streams(t,'batch_smoke',groups,vl.device):
                p.forward_grpo(vl.expand(8,-1,-1),action(8),[t]*8,sample_time=8,use_bc_loss=False)
            st=time.time();batched,_=sample_groups(p,vl,action,t,'batch_smoke',groups);bt=time.time()-st
            native_error=float(np.abs(batched-capture['trajs']).max());assert native_error==0
            st=time.time();single=np.concatenate([sample_group(p,vl,action,t,'batch_smoke',g)[0] for g in groups]);single_time=time.time()-st
            error=float(abs(single-batched).max())
            # FP32 batch GEMM rounding is reported explicitly, not seed mismatch.
            assert error<=1e-4,error
            checks.append(dict(token=t,native_batched_max_abs=native_error,unbatched_fp32_rounding_max_abs=error,batch_seconds=bt,single_seconds=single_time,group_count=8))
            print(checks[-1],flush=True)
    assert state_hash(p)==before
    save(OUT/'audits/batch_native_parity.json',dict(status='PASS',checks=checks,batch_groups=8,weights_unchanged=True,noise_protocol_identical=True,notes='Exact parity with actual native forward_grpo at B=8 G=8. Single-group GEMM roundoff explicitly quantified; no model or noise setting changed.',completed_at=utc()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--rank',type=int,default=0);audit(a.parse_args().rank)
