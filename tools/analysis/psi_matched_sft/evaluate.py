"""Matched read-only G16 sampling and common-target native loss probes."""
from common_matched import *
import torch,argparse,socket

def loss_values(p,vl,action,tr,t):
    n=len(tr);a=action(n);a['action']=torch.as_tensor(tr,device='cuda',dtype=torch.float32)
    rows=[];original=torch.randn_like;capture={}
    def rn(x,*args,**kw):
        out=original(x,*args,**kw);capture['epsilon']=out;return out
    handle=p.action_decoder.register_forward_hook(lambda m,args,out:capture.update(prediction=out))
    torch.randn_like=rn
    try:
        for draw in range(CFG['evaluation']['teacher_loss_draws']):
            torch.manual_seed(seed(t,'native_fit_probe',draw))
            with torch.no_grad():native=p(vl.expand(n,-1,-1),a)['loss'];loss=(capture['prediction']-capture['epsilon']).square().mean((1,2))
            assert float(abs(native-loss.mean()))<1e-6
            rows.append(loss.cpu().numpy())
    finally:torch.randn_like=original;handle.remove()
    return np.stack(rows)

def main(args):
    setup_legacy();legacy.scenes=lambda:scenes('holdout');legacy.CFG['bootstrap_replicates']=3000
    import sample,batched
    sample.inputs=single_input;batched.inputs=single_input
    identity();p,m=sample.load('official_il');name=args.name
    # Shared baseline CRN banks provide a hardware/runtime check on every host.
    checks=[]
    for s in scenes('holdout')[:2]:
        t=s['token'];vl,act=single_input(t)
        for pr in CFG['evaluation']['protocols']:
            fresh=batched.batched(p,m,vl,act,t,pr);cached=np.load(baseline(t))[pr]
            err=float(abs(fresh-cached).max());assert err<1e-4,(pr,err)
            checks.append(dict(token=t,protocol=pr,max_abs_error=err))
    if name!='official_il':
        path=OUT/'checkpoints'/name/f'step{args.step:04d}.pt';a=torch.load(path,map_location='cpu',weights_only=False)
        assert a['protocol_hash']==identity();p.load_state_dict(a['state_dict'],strict=True);m=dict(m,sha256=sha(path),checkpoint_path=str(path))
    evalrows=scenes('holdout');early=sorted(evalrows,key=lambda s:digest([CFG['split_seed'],'early',s['token']]))[:1000]
    probe=scenes('train')[:512];lossids={s['token'] for s in early+probe}
    if args.step==128:rows=early
    else:rows=evalrows+probe
    rows=rows[args.rank::args.world];p.requires_grad_(False)
    label=f'{name}_step{args.step:04d}' if name!='official_il' else name
    auditargs=argparse.Namespace(model=label)
    if args.rank==0:
        sample.smoke(auditargs,p,m);batched.check(auditargs,p,m)
    from multiscene import sample_scenes,parity
    multiscene_checks=parity(p,m,evalrows)
    before=legacy.state_hash(p);start=time.time();done=0
    for start_index in range(0,len(rows),4):
        chunk=rows[start_index:start_index+4];pending=[]
        for s in chunk:
            t=s['token'];dest=OUT/'cache/rollouts'/label/f'{t}.npz'
            meta=dict(protocol_hash=identity(),token=t,checkpoint_hash=m['sha256'],model=label,group_size=16,groups=4)
            if not legacy.v1.valid_npz(dest,meta) and not (name=='official_il' and s['split']=='holdout'):pending.append((s,dest,meta))
        if pending:
            toks=[s['token'] for s,_,_ in pending]
            arrays={pr:sample_scenes(p,toks,pr) for pr in CFG['evaluation']['protocols']}
            for i,(s,dest,meta) in enumerate(pending):npz(dest,dict(meta,generation_host=socket.gethostname(),scene_batch_size=len(pending),group_seeds=[legacy.seed(s['token'],g) for g in range(4)]),**{pr:v[i] for pr,v in arrays.items()})
        for s in chunk:
            t=s['token'];fit=OUT/'cache/fitting'/label/f'{t}.npz'
            if args.step!=128 and t in lossids and not fit.exists():
                vl,action=single_input(t);pool=read(OUT/'cache/selection'/f'{t}.json')['methods'];ids=sorted({i for v in pool.values() for i in v['indices']})
                tr=np.load(OUT/'cache/raw'/f'{t}.npz')['trajectories'][ids]
                p.train();p.set_frozen_modules_to_eval_mode();values=loss_values(p,vl,action,tr,t)
                npz(fit,dict(protocol_hash=identity(),token=t,checkpoint_hash=m['sha256'],raw_indices=ids,draws=16,scope=s['split']),epsilon_mse=values)
        done+=len(chunk)
        if done%20==0:print('EVAL',label,args.rank,done,len(rows),time.time()-start,flush=True)
    after=legacy.state_hash(p);assert before==after
    save(OUT/'audits'/f'eval_{label}_{args.rank}.json',dict(status='PASS',protocol_hash=identity(),host=socket.gethostname(),torch_version=torch.__version__,model=label,count=len(rows),baseline_cache_parity=checks,multiscene_parity=multiscene_checks,state_before=before,state_after=after,checkpoint_hash=m['sha256'],seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('--step',type=int,default=512);a.add_argument('--rank',type=int,default=0);a.add_argument('--world',type=int,default=1);main(a.parse_args())
