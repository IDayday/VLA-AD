"""64 genuinely new draws, plus native G=8..128 CRN batching parity."""
from shared import *
import argparse,contextlib,time
import sample
import torch
sample.inputs=previous.inputs
@contextlib.contextmanager
def noise(token,start,n):
    randn,like=torch.randn,torch.randn_like
    groups=range(start//16,(start+n+15)//16)
    gens=[torch.Generator(device='cuda').manual_seed(previous.seed(token,g)) for g in groups]
    calls=[]
    def draw(device,dtype):
        calls.append(1)
        bank=torch.cat([randn((16,8,3),generator=g,device=device,dtype=dtype) for g in gens])
        return bank[start%16:start%16+n]
    def rn(*a,**kw):
        shape=a[0] if len(a)==1 and isinstance(a[0],(tuple,list,torch.Size)) else a
        if tuple(shape)==(n,8,3):return draw(kw['device'],kw.get('dtype',torch.float32))
        return randn(*a,**kw)
    def rl(x,*a,**kw):
        if tuple(x.shape)==(n,8,3):return draw(x.device,x.dtype)
        return like(x,*a,**kw)
    torch.randn,torch.randn_like=rn,rl
    try:yield calls
    finally:torch.randn,torch.randn_like=randn,like
def native(p,vl,action,token,start,n):
    original=p.sample_chain;capture={}
    def hook(*a,**kw):
        capture['trajectory']=original(*a,**kw)[1];raise sample.SamplingCaptured()
    p.train();p.set_frozen_modules_to_eval_mode();p.sample_chain=hook
    try:
        with noise(token,start,n) as calls,torch.no_grad():
            try:p.forward_grpo(vl,action(1),[token],sample_time=n)
            except sample.SamplingCaptured:pass
            assert len(calls)==6
    finally:p.sample_chain=original
    return capture['trajectory'].cpu().numpy()
def smoke(p,m):
    before=previous.state_hash(p);checks=[]
    assert not [n for n,v in p.named_modules() if isinstance(v,torch.nn.Dropout) and v.training and v.p>0]
    for row in scenes()[:4]:
        t=row['token'];vl,action=previous.inputs(t);full=native(p,vl,action,t,0,128)
        for n in CFG['group_sizes']:
            one=native(p,vl,action,t,0,n);err=float(abs(one-full[:n]).max());assert err<1e-4,(n,err)
            checks.append(dict(token=t,G=n,max_abs_error=err))
        with np.load(old_bank(m,t)) as z:old=z['native_grpo'].reshape(64,8,3)
        err=float(abs(old-full[:64]).max());assert err<1e-4
        new=native(p,vl,action,t,64,64);assert abs(new-full[64:]).max()<1e-4
        checks.append(dict(token=t,G='old64_cache',max_abs_error=err))
    assert previous.state_hash(p)==before
    save(OUT/'audits'/f'sampler_parity_{m}.json',dict(status='PASS',checks=checks,state_unchanged=True,precision='fp32',no_dropout=True,group_exchangeability='same observation, independent Gaussian draws; native output tested at all five G',checkpoint_hash=models()[m]['sha256']))
def main():
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--rank',type=int,default=0);a.add_argument('--world',type=int,default=1);a.add_argument('--smoke',action='store_true');args=a.parse_args()
    identity();p,m=sample.load(args.model)
    if args.smoke:smoke(p,args.model);return
    assert read(OUT/'audits'/f'sampler_parity_{args.model}.json')['status']=='PASS'
    before=previous.state_hash(p);start=time.time();rows=scenes()[args.rank::args.world]
    for i,row in enumerate(rows):
        t=row['token'];dest=extra_bank(args.model,t)
        meta=dict(protocol_hash=identity(),token=t,checkpoint_hash=m['sha256'],runtime_hash=m['runtime_planner_sha256'],groups=[4,5,6,7],seeds=[previous.seed(t,g) for g in range(4,8)])
        if previous.v1.valid_npz(dest,meta):continue
        vl,action=previous.inputs(t);result=native(p,vl,action,t,64,64)
        assert result.shape==(64,8,3) and np.isfinite(result).all()
        npz(dest,meta,native_grpo=result.reshape(4,16,8,3))
        if i%50==0:print(args.model,args.rank,i+1,len(rows),round(time.time()-start,1),flush=True)
    after=previous.state_hash(p);assert before==after
    save(OUT/'audits'/f'sampling_{args.model}_{args.rank}.json',dict(status='PASS',count=len(rows),seconds=time.time()-start,state_before=before,state_after=after,optimizer_updates=0,peak_memory=torch.cuda.max_memory_allocated()))
if __name__=='__main__':main()
