"""Exact parity against the actual completed E0 five-seed deployment outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.inference_cache import native_inference_cache
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    configure();cfg=config();records=[]
    for label,path in [('IL',cfg['student_init']),('RL',cfg['teacher_init'])]:
        head=build_head(path,frozen=True);adapter=NativeAdapter(head)
        for row in membership('dev')[:8]:
            tokens,inputs=conditioning(observation(row));actions=[];started=time.time()
            with native_inference_cache(head) as counts:
                for seed in cfg['evaluation_seeds']:
                    key=RNGKey(STUDY,seed,0,0,row['token'],0,-1)
                    actions.append(adapter.sample_rollin(tokens,inputs,key.plan('final_eval',device='cuda')).trajectory[0].cpu().numpy())
            old=np.load(ROOT/'evaluation/dev'/('B0_'+label)/'scenes'/f"{row['token']}.npz")['trajectories']
            assert np.array_equal(np.stack(actions),old)
            records.append(dict(baseline=label,token=row['token'],trajectory_max_error=0.,five_seed_seconds=time.time()-started,cache=counts))
    write(ROOT/'tests/eval_native_cache.json',dict(status='PASS',records=records,owner=identity()))
    print('NATIVE_EVAL_CACHE_PASS',records,flush=True)


if __name__=='__main__':main()
