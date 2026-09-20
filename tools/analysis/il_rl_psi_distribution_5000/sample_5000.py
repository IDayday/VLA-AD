"""Run unchanged audited samplers on new scenes; old banks stay immutable."""
from common_5000 import *
import argparse, time, socket
import sample, batched
sample.inputs=inputs;batched.inputs=inputs

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--check',action='store_true');args=a.parse_args()
    import torch
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.cuda.set_device(rank)
    identity();p,m=sample.load(args.model)
    if rank==0:
        sample.smoke(args,p,m);batched.check(args,p,m)
        errors=[]
        for row in scenes()[:2]:
            token=row['token'];vl,action=inputs(token)
            with np.load(bank_path(args.model,token)) as old:
                for protocol in CFG['protocols']:
                    fresh=batched.batched(p,m,vl,action,token,protocol)
                    err=float(abs(fresh-old[protocol]).max());assert err<1e-4,(token,protocol,err)
                    errors.append(dict(token=token,protocol=protocol,max_abs_error=err))
        save(OUT/'audits'/f'import_parity_{args.model}.json',dict(status='PASS',checks=errors,cache_source=str(OLDOUT)))
    if args.check:return
    before=state_hash(p);start=time.time();rows=[r for r in scenes() if r['cohort']=='NEW4000'][rank::world]
    for i,row in enumerate(rows):
        token=row['token'];dest=OUT/'cache/rollouts'/args.model/f'{token}.npz'
        meta=dict(protocol_hash=identity(),token=token,model=args.model,checkpoint_hash=m['sha256'],runtime_hash=m['runtime_planner_sha256'],group_size=16,groups=4)
        if v1.valid_npz(dest,meta):continue
        vl,action=inputs(token)
        arrays={pr:batched.batched(p,m,vl,action,token,pr) for pr in CFG['protocols']}
        assert all(v.shape==(4,16,8,3) and np.isfinite(v).all() for v in arrays.values())
        npz(dest,dict(meta,group_seeds=[seed(token,g) for g in range(4)],batch_groups=4),**arrays)
        if i%20==0:print('SAMPLE',args.model,rank,i+1,len(rows),round(time.time()-start,1),flush=True)
    after=state_hash(p);assert before==after
    save(OUT/'audits'/f'run_{args.model}_{rank}.json',dict(protocol_hash=identity(),host=socket.gethostname(),count=len(rows),state_before=before,state_after=after,seconds=time.time()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
