"""Four matched continuations from official IL; native forward epsilon loss."""
from common_matched import *
import torch,math,argparse
from torch.utils.data import Dataset,DataLoader

class TrainingData(Dataset):
    def __init__(self,method,runseed):
        self.rows=scenes('train');self.method=method;self.runseed=runseed
        self.order=[];rng=np.random.default_rng(seed(runseed,'scene_order'))
        needed=CFG['training']['updates']*CFG['training']['effective_batch']
        while len(self.order)<needed:self.order.extend(rng.permutation(len(self.rows)).tolist())
        self.order=self.order[:needed]
        self.targets={};self.selections={};frozen=read(OUT/'manifests/selection_frozen.json')['selection_hashes']
        for s in self.rows:
            t=s['token'];self.targets[t]=np.load(OUT/'cache/raw'/f'{t}.npz')['trajectories']
            path=OUT/'cache/selection'/f'{t}.json';assert sha(path)==frozen[t]
            self.selections[t]=read(path)['methods'][method]
    def __len__(self):return len(self.order)
    def __getitem__(self,i):
        s=self.rows[self.order[i]];t=s['token'];v=self.selections[t]
        rng=np.random.default_rng(seed(self.runseed,'target',i))
        j=int(rng.choice(len(v['indices']),p=np.asarray(v['weights'])/sum(v['weights'])));idx=v['indices'][j]
        return t,feature(t),self.targets[t][idx],idx
def collate(rows):return rows

def loss_audit(p,vl,a):
    original=torch.randn_like;capture={}
    def randn(x,*args,**kw):
        out=original(x,*args,**kw);capture['epsilon']=out.detach();return out
    handle=p.action_decoder.register_forward_hook(lambda m,args,out:capture.update(prediction=out.detach()))
    torch.randn_like=randn
    try:
        torch.manual_seed(93012)
        with torch.no_grad():out=p(vl,a)['loss']
    finally:torch.randn_like=original;handle.remove()
    manual=(capture['prediction']-capture['epsilon']).square().mean();err=float(abs(out-manual));assert err<1e-7,err
    return dict(native_loss=float(out),manual_epsilon_MSE=float(manual),abs_error=err)

def main(args):
    identity();assert (OUT/'manifests/selection_frozen.json').exists()
    method=args.method;runseed=args.seed;name=f'{method}_seed{runseed}';cfg=CFG['training'];dest=OUT/'checkpoints'/name;dest.mkdir(parents=True,exist_ok=True)
    if (OUT/'audits'/f'train_{name}.json').exists():return
    p,m=load_model();p.train();p.set_frozen_modules_to_eval_mode()
    params=[v for v in p.parameters() if v.requires_grad];names=[n for n,v in p.named_parameters() if v.requires_grad]
    assert sum(v.numel() for v in params)==34329219
    opt=torch.optim.AdamW(params,lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'])
    ds=TrainingData(method,runseed);loader=DataLoader(ds,batch_size=cfg['microbatch'],num_workers=4,collate_fn=collate,persistent_workers=True,prefetch_factor=2)
    it=iter(loader);logs=[];ledger=[];start=time.time();initial=legacy.state_hash(p);snapshots=[]
    buffer_before={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in p.named_buffers()}
    for step in range(1,cfg['updates']+1):
        warm=cfg['warmup_updates'];f=step/warm if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/(cfg['updates']-warm)))
        lr=cfg['lr']*f if step<=warm else cfg['min_lr']+(cfg['lr']-cfg['min_lr'])*f
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);losses=[]
        for micro in range(cfg['accumulation']):
            rows=next(it);ts=[r[0] for r in rows];vl,a=batch_input([r[1] for r in rows],[r[2] for r in rows])
            if step==1 and micro==0:audit=loss_audit(p,vl,a)
            torch.manual_seed(seed(runseed,'training_noise',step,micro));loss=p(vl,a)['loss'];assert torch.isfinite(loss)
            (loss/cfg['accumulation']).backward();losses.append(float(loss.detach()))
            ledger.extend(dict(step=step,micro=micro,token=r[0],raw_index=r[3]) for r in rows)
        norm=torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']);assert torch.isfinite(norm);opt.step()
        logs.append(dict(step=step,loss=np.mean(losses),lr=lr,gradient_norm=float(norm),seconds=time.time()-start))
        if step%16==0:print(name,logs[-1],flush=True)
        if step in cfg['snapshots']:
            path=dest/f'step{step:04d}.pt';tmp=path.with_suffix('.tmp')
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in p.state_dict().items()},step=step,method=method,seed=runseed,protocol_hash=identity(),initial_checkpoint_sha256=m['sha256']),tmp);tmp.replace(path)
            info=dict(path=str(path),sha256=sha(path),step=step,method=method,seed=runseed,protocol_hash=identity());save(path.with_suffix('.json'),info);snapshots.append(info)
        if step%64==0:table(f'training_{name}.csv',logs)
    table(f'training_{name}.csv',logs);table(f'ledger_{name}.parquet',ledger)
    assert not {x['token'] for x in ledger}&{s['token'] for s in scenes('holdout')}
    buffer_after={n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in p.named_buffers()};assert buffer_before==buffer_after
    save(OUT/'audits'/f'train_{name}.json',dict(status='PASS',protocol_hash=identity(),method=method,seed=runseed,updates=cfg['updates'],supervised_scene_presentations=len(ledger),unique_train_tokens=len({x['token'] for x in ledger}),initial_state_hash=initial,final_state_hash=legacy.state_hash(p),initial_checkpoint_sha256=m['sha256'],trainable_parameter_count=sum(v.numel() for v in params),trainable_names=names,native_loss_audit=audit,buffers_unchanged=True,buffer_hashes=buffer_after,checkpoints=snapshots,seconds=time.time()-start,peak_memory_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('method',choices=CFG['methods']);a.add_argument('seed',type=int);main(a.parse_args())
