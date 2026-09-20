"""Distribution-preserving scene batching with independent native G16 streams."""
from common_matched import *
import torch,contextlib
from transformers.feature_extraction_utils import BatchFeature

@contextlib.contextmanager
def streams(tokens):
    randn,like=torch.randn,torch.randn_like;n=len(tokens)*64;calls=[]
    gens=[torch.Generator(device='cuda').manual_seed(legacy.seed(t,g)) for t in tokens for g in range(4)]
    def draw(device,dtype):
        calls.append(len(calls));assert len(calls)<=6
        return torch.cat([randn((16,8,3),device=device,dtype=dtype,generator=g) for g in gens])
    def rn(*a,**kw):
        shape=a[0] if len(a)==1 and isinstance(a[0],(tuple,list,torch.Size)) else a
        return draw(kw['device'],kw.get('dtype',torch.float32)) if tuple(shape)==(n,8,3) else randn(*a,**kw)
    def rl(x,*a,**kw):return draw(x.device,x.dtype) if tuple(x.shape)==(n,8,3) else like(x,*a,**kw)
    torch.randn,torch.randn_like=rn,rl
    try:yield calls
    finally:torch.randn,torch.randn_like=randn,like

def sample_scenes(p,tokens,protocol):
    from sample import SamplingCaptured
    vl,a=batch_input([feature(t) for t in tokens])
    repeats=4 if protocol=='native_grpo' else 64
    v=vl.repeat_interleave(repeats,0);actions=BatchFeature(data={k:x.repeat_interleave(repeats,0) for k,x in a.items()})
    with streams(tokens) as calls,torch.no_grad():
        if protocol=='native_grpo':
            original=p.sample_chain;captured={}
            def hook(*a,**kw):captured['trajectory']=original(*a,**kw)[1];raise SamplingCaptured()
            p.train();p.set_frozen_modules_to_eval_mode();p.sample_chain=hook
            try:p.forward_grpo(v,actions,[t for t in tokens for _ in range(4)],sample_time=16)
            except SamplingCaptured:pass
            finally:p.sample_chain=original
            out=captured['trajectory']
        else:
            p.eval();out=p.get_action(v,actions,deterministic=False)['pred_traj']
    assert len(calls)==6
    return out.cpu().numpy().reshape(len(tokens),4,16,8,3)

def parity(p,m,rows):
    import batched
    tokens=[s['token'] for s in rows[:4]];records=[]
    for pr in CFG['evaluation']['protocols']:
        multi=sample_scenes(p,tokens,pr)
        for i,t in enumerate(tokens):
            vl,action=single_input(t);single=batched.batched(p,m,vl,action,t,pr)
            err=float(abs(multi[i]-single).max());assert err<1e-4,(pr,t,err)
            records.append(dict(token=t,protocol=pr,max_abs_error=err))
    return records
