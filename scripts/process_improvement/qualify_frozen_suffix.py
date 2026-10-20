"""Real 128/256/512-branch frozen suffix state/score parity and timing."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import argparse
import numpy as np
from scripts.process_improvement.common import *


def aggregate():
    parts=[read(ROOT/'tests'/name) for name in ['frozen_suffix_cache_initial72.json','frozen_suffix_cache_extra120.json']]
    assert all(p['status']=='PASS' for p in parts)
    hashes=parts[0]['source_hashes'];assert all(p['source_hashes']==hashes for p in parts)
    assert all(sha(SOURCE/name)==value for name,value in hashes.items())
    records=sum([p['records'] for p in parts],[])
    identities={(r['source'],r['token'],r['batch'],r['cut']) for r in records}
    assert len(records)==len(identities)==192
    expected={(label,row['token'],batch,cut) for label in ['IL','RL','OLD_OPD']
              for row in membership('diagnostic_parent_train')[:2] for batch in [16,32,48,64,96,128,256,512] for cut in [0,2,4,5]}
    assert identities==expected
    native=sum(r['native_seconds'] for r in records);cached=sum(r['cached_seconds'] for r in records)
    result=dict(status='PASS',records=records,source_hashes=hashes,states_and_trajectories_exact=True,score_error=0.,
                batch_sizes=[16,32,48,64,96,128,256,512],admitted_for_performance=cached<native,
                native_seconds=native,cached_seconds=cached,qualification_parts=['initial72','extra120'],owner=identity())
    write(ROOT/'tests/frozen_suffix_cache.json',result)
    print('FROZEN_SUFFIX_FULL_COVERAGE',len(records),native,cached,flush=True)


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.frozen_suffix import FrozenSuffixAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from scripts.opd.scoring import Scoring,load_cache
    p=argparse.ArgumentParser();p.add_argument('--batches',nargs='+',type=int,default=[16,32,48,64,96,128,256,512])
    p.add_argument('--output-name',default='frozen_suffix_cache.json');p.add_argument('--aggregate',action='store_true')
    args=p.parse_args()
    if args.aggregate:return aggregate()
    configure();cfg=config();records=[];scorer=Scoring()
    paths=dict(IL=cfg['student_init'],RL=cfg['teacher_init'],OLD_OPD=HISTORY/'runs/train_E3_42/latest.pt')
    for label,path in paths.items():
        head=build_head(path,frozen=True);original=NativeAdapter(head);cached=FrozenSuffixAdapter(head);before=state_hash(head)
        for row in membership('diagnostic_parent_train')[:2]:
            tokens,inputs=conditioning(observation(row));key=RNGKey(STUDY,42,0,0,row['token'],0,-1)
            chain=original.sample_rollin(tokens,inputs,key.plan('rollin',device='cuda'))
            with torch.no_grad():context=original.prepare_conditioning(tokens,inputs)
            for batch in args.batches:
                noise=key.normal((batch,5,8,3),'diagnostic_test',batch,device='cuda')
                for cut in [0,2,4,5]:
                    start=chain.states[:,cut].expand(batch,-1,-1).clone()
                    start+=key.normal(start.shape,'candidate',batch+cut,device='cuda')*.001
                    original.rollout_suffix(context,start,cut,noise,True) # shape-specific warmup
                    torch.cuda.synchronize();t0=time.perf_counter()
                    a,sa=original.rollout_suffix(context,start,cut,noise,True)
                    torch.cuda.synchronize();native_seconds=time.perf_counter()-t0;t0=time.perf_counter()
                    b,sb=cached.rollout_suffix(context,start,cut,noise,True)
                    torch.cuda.synchronize();cached_seconds=time.perf_counter()-t0
                    assert torch.equal(sa,sb),(label,row['token'],batch,cut,'latent states')
                    assert torch.equal(a,b),(label,row['token'],batch,cut,'final output')
                    scores0=scorer.batch(load_cache(row['metric_cache_path']),a[:16].cpu().numpy())['scores']
                    scores1=scorer.batch(load_cache(row['metric_cache_path']),b[:16].cpu().numpy())['scores']
                    assert np.array_equal(scores0,scores1)
                    records.append(dict(source=label,token=row['token'],batch=batch,cut=cut,state_error=0.,score_error=0.,
                                        native_seconds=native_seconds,cached_seconds=cached_seconds,cache=cached.last_cache_counts))
        assert state_hash(head)==before
        del head,original,cached;torch.cuda.empty_cache()
    hashes={name:sha(SOURCE/name) for name in ['navsim/agents/recogdrive/process_improvement/frozen_suffix.py',
               'navsim/agents/recogdrive/opd/inference_cache.py','navsim/agents/recogdrive/process_improvement/suffix_rollout.py']}
    native_time=sum(r['native_seconds'] for r in records);cache_time=sum(r['cached_seconds'] for r in records)
    write(ROOT/'tests'/args.output_name,dict(status='PASS',records=records,source_hashes=hashes,
          states_and_trajectories_exact=True,score_error=0.,batch_sizes=args.batches,
          admitted_for_performance=cache_time<native_time,native_seconds=native_time,cached_seconds=cache_time,owner=identity()))
    print('FROZEN_SUFFIX_EXACT_PARITY_PASS',len(records),sum(r['native_seconds'] for r in records),sum(r['cached_seconds'] for r in records),flush=True)


if __name__=='__main__':main()
