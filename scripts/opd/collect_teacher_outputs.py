import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.ddim_adapter import DDIMAdapter
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    configure();cfg=config();head=build_head(cfg['teacher_init'],frozen=True);adapter=DDIMAdapter(head)
    binding=read(ROOT/'audit/checkpoints.json')['RL']['sha256'];start=time.time()
    for i,row in enumerate(rows('train')[a.rank::a.shards]):
        token=row['token'];path=ROOT/'teacher/chains'/f'{token}.pt'
        if path.exists():
            meta=torch.load(path,map_location='cpu',weights_only=False)['metadata']
            if meta['teacher_sha256']!=binding or meta['config_sha256']!=digest(cfg):raise ValueError('Stale teacher cache')
            continue
        tokens,inputs=conditioning(observation(row));chains=[]
        for z in range(3101,3106):
            seeds(seed_for(token,z,'distill_chain'));c=adapter.sample(tokens,inputs,deterministic=cfg['rollout_deterministic'],kernel=cfg['rollout_kernel'])
            chains.append({k:getattr(c,k).cpu() for k in ('states','means','sigmas','trajectory')})
        metadata=dict(token=token,split='navtrain',teacher_sha256=binding,vlm_sha256=read(ROOT/'audit/checkpoints.json')['VLM']['sha256'],
            config_sha256=digest(cfg),seeds=list(range(3101,3106)),timesteps=head.ddim_t.cpu().tolist(),indices=list(range(5)),owner=identity())
        save(path,dict(metadata=metadata,chains=chains))
        save(ROOT/'teacher/final_trajectories'/f'{token}.pt',dict(metadata=metadata,trajectories=torch.cat([c['trajectory'] for c in chains])))
        if (i+1)%64==0:print('TEACHER_CACHE',a.rank,i+1,flush=True)
    write(ROOT/'teacher'/f'complete_{a.rank}.json',dict(status='COMPLETE',shards=a.shards,seconds=time.time()-start,owner=identity()))

if __name__=='__main__':main()
