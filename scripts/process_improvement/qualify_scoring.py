"""Real scalar/batch/CPU-shard score equivalence, orders and foreign candidates."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import time
import numpy as np
from scripts.process_improvement.common import *


def main():
    from scripts.opd.scoring import Scoring,load_cache
    from navsim.agents.recogdrive.process_improvement.budget_counter import CanonicalScorer
    rows=membership('legacy_4k')[:8]
    paths={r['token']:r['metric_cache_path'] for r in rows}
    parallel=CanonicalScorer(workers=4,metric_paths=paths);serial=Scoring();records=[]
    for row in rows:
        path=HISTORY/'teacher/chains'/f"{row['token']}.pt"
        # Use real stored native teacher trajectories and smooth numerical
        # perturbations solely as scoring-protocol inputs, never training labels.
        import torch
        payload=torch.load(path,map_location='cpu',weights_only=False)
        base=payload['chains'][0]['trajectory'][0].numpy()
        array=np.stack([base+np.array([a,0,0])*np.linspace(.125,1,8)[:,None] for a in np.linspace(-3,3,64)])
        cache=load_cache(row['metric_cache_path']);start=time.time();expected=serial.batch(cache,array)['scores'];serial_time=time.time()-start
        start=time.time();actual=parallel.score(row['token'],cache,array);parallel_time=time.time()-start
        assert np.max(np.abs(expected-actual))<=1e-6
        for index in [0,31,63]:assert np.max(np.abs(expected[index]-serial.scalar(cache,array[index])))<=1e-6
        reverse=serial.batch(cache,array[::-1])['scores'][::-1]
        assert np.max(np.abs(expected-reverse))<=1e-6
        extra=np.concatenate([array,array[:1]*0])
        assert np.max(np.abs(serial.batch(cache,extra)['scores'][:64]-expected))<=1e-6
        records.append(dict(token=row['token'],max_error=float(np.abs(expected-actual).max()),serial_seconds=serial_time,parallel_seconds=parallel_time))
    parallel.close()
    write(ROOT/'tests/parallel_scoring.json',dict(status='PASS',records=records,worker_count=4,owner=identity()))
    print('PARALLEL_SCORING_PASS',records,flush=True)


if __name__=='__main__':main()
