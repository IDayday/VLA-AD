"""Real local P0/P1/projection/guard tests with held-out suffix noise.

Selection weights use only fit_suffix. Guard and diagnostic_test are distinct
stateless streams. Tests are never fed back into the constructed target.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey,candidate_noise
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.candidate_value import evaluate_candidates,evaluate_kernels
    from navsim.agents.recogdrive.process_improvement.improvement_target import proposal
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    from scripts.opd.scoring import load_cache
    p=argparse.ArgumentParser()
    p.add_argument('--split',default='diagnostic_train')
    p.add_argument('--sources',nargs='+',default=['IL','RL','OLD_OPD'])
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--limit',type=int);p.add_argument('--banks',type=int,default=2)
    p.add_argument('--candidates',type=int,default=16);p.add_argument('--fit',type=int,default=8)
    p.add_argument('--guard',type=int,default=8);p.add_argument('--test',type=int,default=16)
    p.add_argument('--grid',action='store_true');p.add_argument('--name',required=True)
    p.add_argument('--scoring-workers',type=int,default=0)
    p.add_argument('--config')
    p.add_argument('--frozen-suffix-cache',action='store_true')
    args=p.parse_args();configure();cfg=config(args.config)
    if read(ROOT/'tests/parity.json')['status']!='PASS':raise ValueError('Numerical qualification incomplete')
    if args.frozen_suffix_cache:
        if not frozen_suffix_cache_available():
            raise ValueError('Frozen suffix cache has no current exact numerical admission')
        from navsim.agents.recogdrive.process_improvement.frozen_suffix import FrozenSuffixAdapter
        NativeAdapter=FrozenSuffixAdapter
    if args.split in ('dev','navtest'):raise ValueError('Local method selection must not inspect audit/test data')
    rows=membership(args.split)
    if args.limit:rows=rows[:args.limit]
    rows=rows[args.rank::args.shards]
    dest=ROOT/'diagnostics'/args.name
    completed=dest/f'complete_{args.rank}.json'
    if completed.exists():raise FileExistsError('Completed diagnostic shard is immutable')
    teacher=build_head(cfg['teacher_init'],frozen=True);adapter=NativeAdapter(teacher)
    paths=dict(IL=cfg['student_init'],RL=cfg['teacher_init'],OLD_OPD=str(HISTORY/'runs/train_E3_42/latest.pt'))
    heads={name:build_head(paths[name],frozen=True) for name in args.sources}
    scorer=CanonicalScorer(workers=args.scoring_workers,metric_paths={r['token']:r['metric_cache_path'] for r in rows});started=time.time()
    grid=[(t,r) for t in [.02,.05,.10] for r in [.25,.5]] if args.grid else [(cfg['temperature'],cfg['rho'])]
    dest.mkdir(parents=True,exist_ok=True)
    write(dest/f'config_{args.rank}.json',dict(arguments=vars(args),source_checkpoint_sha256={k:sha(paths[k]) for k in heads},
          scene_manifest_sha256=sha(ROOT/'manifests'/f'{args.split}.json'),config_sha256=digest(cfg),owner=identity(),grid=grid))
    for number,row in enumerate(rows):
        output=dest/'scenes'/f"{row['token']}.json"
        if output.exists():continue
        tokens,inputs=conditioning(observation(row));cache=load_cache(row['metric_cache_path'])
        records=[]
        for source,head in heads.items():
            key=RNGKey(STUDY,42,0,0,row['token'],0,-1)
            chain=NativeAdapter(head).sample_rollin(tokens,inputs,key.plan('rollin',device='cuda'))
            with torch.no_grad():context=adapter.prepare_conditioning(tokens,inputs)
            for index in range(5):
                state=chain.states[:,index];stepkey=key.at_step(index)
                with torch.no_grad():stats=adapter.transition_stats(state,index,context)
                for bank in range(args.banks):
                    raw=candidate_noise(stepkey,args.candidates,device='cuda',bank=bank)
                    candidates=adapter.sample_next(stats,raw)
                    fit,_=evaluate_candidates(adapter,context,candidates,index,stepkey,args.fit,'fit_suffix',scorer,row['token'],cache,bank)
                    independent,_=evaluate_candidates(adapter,context,candidates,index,stepkey,args.test,'diagnostic_test',scorer,row['token'],cache,bank)
                    values=torch.tensor(fit[:,:,-1].mean(1),device='cuda',dtype=torch.float32)
                    for temperature,rho in grid:
                        proposed=proposal(stats.mean[0],stats.std[0],candidates,values,temperature=temperature,rho=rho,
                                          radius=cfg['radius'],ess_fraction=cfg['ess_fraction'],flat_tolerance=cfg['flat_value_tolerance'])
                        native=proposed.mean.unsqueeze(0).to(stats.mean.dtype)
                        guard,_=evaluate_kernels(adapter,context,stats,[stats.mean,native],stepkey,args.guard,
                                                 'target_guard',scorer,row['token'],cache,bank)
                        gd=guard[1]-guard[0]
                        accepted=gd[:,-1].mean()>cfg['guard_margin'] and (gd[:,[0,1,3]].mean(0)>=-cfg['safety_tolerance']).all()
                        final=native if accepted else stats.mean
                        test,trajectories=evaluate_kernels(adapter,context,stats,[stats.mean,native,final],stepkey,args.test,
                                                          'diagnostic_test',scorer,row['token'],cache,bank)
                        # The finite-bank selection expectation needs no additional
                        # resampling: fixed fit weights times independent candidate Q.
                        weighted=(proposed.weights.cpu().numpy()[:,None,None]*independent).sum(0)
                        record=dict(source=source,index=index,timestep=stats.timestep,bank=bank,temperature_base=temperature,rho=rho,
                                    temperature_actual=proposed.temperature,ess=proposed.ess,radius=proposed.radius_after,
                                    flat=proposed.flat,accepted=bool(accepted),fit_value_range=float(values.max()-values.min()),
                                    fit_selection_mean=float((proposed.weights*values).sum()),
                                    P0=test[0].tolist(),P1=weighted.tolist(),proposed=test[1].tolist(),P2=test[2].tolist(),
                                    selection_gain=float((weighted[:,-1]-test[0,:,-1]).mean()),
                                    projection_gain=float((test[1,:,-1]-test[0,:,-1]).mean()),
                                    guarded_gain=float((test[2,:,-1]-test[0,:,-1]).mean()),
                                    guard_gain=float(gd[:,-1].mean()),budget=scorer.counter.state_dict())
                        records.append(record)
                        if int(digest([row['token'],'case'])[:8],16)%64==0 and bank==0:
                            save(dest/'traces'/f"{row['token']}_{source}_{index}_{temperature}_{rho}.pt",
                                 dict(row=row,state=state.cpu(),raw_noise=raw.cpu(),candidates=candidates.cpu(),fit_scores=fit,
                                      independent_scores=independent,weights=proposed.weights.cpu(),guard_scores=guard,test_scores=test,
                                      reference_mean=stats.mean.cpu(),proposed_mean=native.cpu(),std=stats.std.cpu(),
                                      trajectories=trajectories.cpu(),rng_key=stepkey.__dict__,record=record))
        write(output,dict(token=row['token'],driver_record=row['driver_record'],log_name=row['log_name'],records=records))
        print('DIAG',args.name,args.rank,number+1,len(rows),time.time()-started,scorer.counter.state_dict(),flush=True)
    scorer.close()
    write(completed,dict(status='COMPLETE',scenes=len(rows),seconds=time.time()-started,budget=scorer.counter.state_dict(),owner=identity()))


if __name__=='__main__':main()
