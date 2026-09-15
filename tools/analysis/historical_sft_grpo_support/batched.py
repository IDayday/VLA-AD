"""Equivalent batching of four independent G16 groups, with checked FP32 error."""
from sample import *
import contextlib

@contextlib.contextmanager
def group_rng(token):
    randn,like=torch.randn,torch.randn_like
    generators=[torch.Generator(device='cuda').manual_seed(seed(token,g)) for g in range(4)];calls=[]
    def draw(dev,dtype):
        calls.append(len(calls));assert len(calls)<=6
        return torch.cat([randn((16,8,3),generator=g,device=dev,dtype=dtype) for g in generators])
    def rn(*a,**kw):
        shape=a[0] if len(a)==1 and isinstance(a[0],(tuple,list,torch.Size)) else a
        if tuple(shape)==(64,8,3):return draw(kw['device'],kw.get('dtype',torch.float32))
        return randn(*a,**kw)
    def rl(x,*a,**kw):
        if tuple(x.shape)==(64,8,3):return draw(x.device,x.dtype)
        return like(x,*a,**kw)
    torch.randn,torch.randn_like=rn,rl
    try:yield calls
    finally:torch.randn,torch.randn_like=randn,like

def batched(p,m,vl,action,token,protocol):
    with group_rng(token) as calls,torch.no_grad():
        if protocol=='native_grpo':
            original=p.sample_chain;capture={}
            def hook(*a,**kw):
                capture['trajectory']=original(*a,**kw)[1];raise SamplingCaptured()
            p.train();p.set_frozen_modules_to_eval_mode();p.sample_chain=hook
            try:
                entry=p.forward_lfp_grpo if getattr(p,'stage3_algorithm','')=='lfp_grpo' else p.forward_grpo
                entry(vl.expand(4,-1,-1),action(4),[token]*4,sample_time=16)
            except SamplingCaptured:pass
            finally:p.sample_chain=original
            result=capture['trajectory']
        else:
            p.eval();floor=p.eval_min_sampling_denoising_std;clip=p.eval_randn_clip_value
            if protocol in ['floor_only','floor_and_clip']:p.eval_min_sampling_denoising_std=p.min_sampling_denoising_std
            if protocol in ['clip_only','floor_and_clip']:p.eval_randn_clip_value=float('inf') if m['family'] in ['a5','v6'] else p.randn_clip_value
            try:result=p.get_action(vl.expand(64,-1,-1),action(64),deterministic=False)['pred_traj']
            finally:p.eval_min_sampling_denoising_std=floor;p.eval_randn_clip_value=clip
    assert len(calls)==6,(protocol,len(calls))
    return result.cpu().numpy().reshape(4,16,8,3)

def check(args,p,m):
    before=state_hash(p);rows=[]
    for s in scenes()[:2]:
        t=s['token'];vl,action=inputs(t)
        for protocol in CFG['protocols']+CFG['factorial_protocols']:
            one=np.stack([draw(p,m,vl,action,t,g,protocol) for g in range(4)])
            batch=batched(p,m,vl,action,t,protocol)
            err=float(abs(one-batch).max());assert err<1e-4,(args.model,protocol,err)
            rows.append(dict(token=t,protocol=protocol,max_abs_error=err))
    assert state_hash(p)==before
    save(OUT/'audits'/f'batch_{args.model}.json',dict(status='PASS',protocol_hash=identity(),checks=rows,state_unchanged=True,group_size=16,batch_groups=4,noise_draws_per_group=6,precision='fp32',tolerance=1e-4))
    print('BATCH PASS',args.model,max(r['max_abs_error'] for r in rows),flush=True)

def run_batch(args,p,m):
    check(args,p,m);before=state_hash(p);start=time.time()
    factorial=set(read(OUT/'manifests/scenes.json')['factorial_tokens'])
    rows=scenes()[args.rank::args.world]
    for i,s in enumerate(rows):
        t=s['token'];dest=OUT/'cache/rollouts'/args.model/f'{t}.npz'
        meta=dict(protocol_hash=identity(),model=args.model,token=t,checkpoint_hash=m['sha256'],runtime_hash=m['runtime_planner_sha256'],group_size=16,groups=4)
        if v1.valid_npz(dest,meta):continue
        vl,action=inputs(t);protocols=list(CFG['protocols'])
        if args.model in CFG['primary_models'] and t in factorial:protocols+=CFG['factorial_protocols']
        out={pr:batched(p,m,vl,action,t,pr) for pr in protocols}
        npz(dest,dict(meta,group_seeds=[seed(t,g) for g in range(4)],batch_groups=4,batch_adapter_hash=sha(__file__)),**out)
        if i%10==0:print('BATCH SAMPLE',args.model,args.rank,i+1,len(rows),round(time.time()-start,1),flush=True)
    after=state_hash(p);assert before==after
    save(OUT/'audits'/f'run_{args.model}_{args.rank}.json',dict(protocol_hash=identity(),count=len(rows),state_before=before,state_after=after,seconds=time.time()-start,batch_groups=4))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--check',action='store_true');a.add_argument('--rank',type=int,default=0);a.add_argument('--world',type=int,default=1);args=a.parse_args()
    p,m=load(args.model)
    if args.check:check(args,p,m)
    else:
        if not (OUT/'audits'/f'sampling_{args.model}.json').exists():smoke(args,p,m)
        run_batch(args,p,m)
