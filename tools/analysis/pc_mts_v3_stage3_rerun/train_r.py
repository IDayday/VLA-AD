"""Formal AgentLightningDiT, optimizer, scheduler and Trainer; only I/O is adapted."""
from common_r import *
import argparse, torch, pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
class Observations(Dataset):
    def __init__(self,part):
        self.rows={r['token']:r for r in scenes()};self.ts=tokens(part)
    def __len__(self):return len(self.ts)
    def __getitem__(self,i):
        t=self.ts[i];f=torch.load(V1/'features'/f'{t}.pt',map_location='cpu',weights_only=False)
        assert f.pop('_metadata')['observation_only']
        return f,{'trajectory':torch.tensor(self.rows[t]['gt'],dtype=torch.float32)},t
class Record(pl.Callback):
    def __init__(self,method,sd,last):
        self.method,self.sd,self.last=method,sd,last;self.rows=[];self.updates=[];self.t0=time.time();self.norm=None
    def snapshot(self,tr,module):
        step=int(tr.global_step)
        if not tr.is_global_zero:return
        dest=checkpoint(self.method,self.sd,step);assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
        state={k:v.detach().cpu() for k,v in module.agent.action_head.state_dict().items() if not k.startswith('old_policy.') and not k.startswith('action_aware_aux_head.')}
        data=dict(state_dict=state,optimizer=tr.optimizers[0].state_dict(),scheduler=tr.lr_scheduler_configs[0].scheduler.state_dict(),
                  step=step,epoch=tr.current_epoch,metadata=dict(identity(),initialization_sha256=sha(initial_path(self.method,self.sd)),reference_rule=CFG['reference_policy'],host=socket.gethostname()))
        tmp=dest.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(dest)
        save(dest.with_suffix('.json'),dict(identity(),sha256=sha(dest),step=step,epoch=tr.current_epoch,host=socket.gethostname(),seconds=time.time()-self.t0))
    def on_train_start(self,tr,module):
        assert len(tr.train_dataloader)==11
        assert tr.lr_scheduler_configs[0].interval=='epoch'
        op=tr.optimizers[0];pg=op.param_groups[0]
        assert pg['betas']==(.9,.95) and pg['weight_decay']==1e-4 and pg['lr']==1e-4
        if tr.is_global_zero:
            save(OUT/'manifests'/f'optimizer_{run_name(self.method,self.sd)}.json',dict(identity(),betas=list(pg['betas']),weight_decay=pg['weight_decay'],lr=pg['lr'],interval='epoch',trainable_parameters=sum(p.numel() for p in pg['params']),batch_per_rank=8,world_size=tr.world_size,accumulation=tr.accumulate_grad_batches,precision=str(tr.precision_plugin),sampler=str(type(tr.train_dataloader.sampler))))
        self.snapshot(tr,module)
    def on_before_optimizer_step(self,tr,module,optimizer):
        norms=[p.grad.detach().float().norm() for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        norm=torch.stack(norms).norm()
        # Native AMP owns overflow/skip behavior; diagnostics must not change it.
        self.norm=float(norm) if torch.isfinite(norm) else None
    def on_train_batch_end(self,tr,module,outputs,batch,batch_idx):
        step=int(tr.global_step);b=self.last;s=b['scores'];r=b['rewards'];adv=b['advantages'];f=v3.feasible(s);h=v3.hard_failure(s)
        for i in range(len(s)):
            self.rows.append(dict(step=step,rank=tr.global_rank,epoch=tr.current_epoch,token=b['tokens'][i//8],group=f'{step}:{tr.global_rank}:{i//8}',member=i%8,reward=float(r[i,6]),advantage=float(adv[i]),feasible=bool(f[i]),hard_failure=bool(h[i]),**{v3.FIELDS[k]:float(s[i,k]) for k in range(7)}))
        self.updates.append(dict(step=step,epoch=tr.current_epoch,rank=tr.global_rank,loss=float(outputs['loss']),lr=tr.optimizers[0].param_groups[0]['lr'],gradient_norm=self.norm,amp_nonfinite_gradient=self.norm is None,amp_scale=float(tr.precision_plugin.scaler.get_scale()),seconds=time.time()-self.t0,tokens=b['tokens'],rng_after_sampling=b['rng_after_sampling']))
        if step in CFG['snapshots']:self.snapshot(tr,module)
        if tr.is_global_zero and step%5==0:print(json.dumps(dict(run=run_name(self.method,self.sd),**self.updates[-1])),flush=True)
    def on_train_end(self,tr,module):
        assert tr.global_step==110
        dest=guard(OUT/'metrics'/run_name(self.method,self.sd));dest.mkdir(parents=True,exist_ok=True)
        import pandas as pd
        pd.DataFrame(self.rows).to_parquet(dest/f'rollouts_rank{tr.global_rank}.parquet',index=False)
        save(dest/f'updates_rank{tr.global_rank}.json',self.updates)
        tr.strategy.barrier()
        if tr.is_global_zero:save(OUT/'manifests'/f'train_{run_name(self.method,self.sd)}.json',dict(identity(),method=self.method,seed=self.sd,updates=110,epochs=10,scene_updates=7040,rollouts=56320,seconds=time.time()-self.t0,host=socket.gethostname(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),initialization_sha256=sha(initial_path(self.method,self.sd)),final_sha256=sha(checkpoint(self.method,self.sd,110))))
def main(a):
    assert (OUT/'manifests/parity_native.json').exists()
    assert (OUT/'manifests'/f'host_parity_{socket.gethostname()}.json').exists()
    identity();rank=int(os.environ['LOCAL_RANK']);setup(rank)
    torch.distributed.init_process_group('nccl');pl.seed_everything(a.seed,workers=True)
    agent=build_agent(a.method,a.seed);sys.path.insert(0,str(ROOT))
    from navsim.planning.script.run_training_recogdrive_rl import custom_collate_fn
    from navsim.planning.training.agent_lightning_module import AgentLightningDiT
    from reward import attach
    pool,last,original=attach(agent.action_head,CFG['acceleration']['cpu_workers_per_rank'])
    module=AgentLightningDiT(agent);callback=Record(a.method,a.seed,last)
    train=DataLoader(Observations('train'),collate_fn=custom_collate_fn,shuffle=True,**CFG['dataloader'])
    val=DataLoader(Observations('holdout'),collate_fn=custom_collate_fn,shuffle=False,**CFG['dataloader'])
    options=dict(CFG['trainer']);options.update(default_root_dir=str(OUT/'lightning'/run_name(a.method,a.seed)),enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
    trainer=pl.Trainer(**options,callbacks=[callback])
    try:trainer.fit(module,train,val)
    finally:pool.shutdown()
    torch.distributed.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method',choices=CFG['methods']);p.add_argument('seed',type=int);main(p.parse_args())
