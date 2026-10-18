"""Qualify each host on one fixed real observation before deploying its shards."""
import argparse
import time
import numpy as np
from .common import *


def main():
    import torch,numpy,scipy,shapely,transformers
    from .runtime import load_agent,features_for,encode,action_input,predict
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);args=p.parse_args()
    cfg,identity=protocol();start=time.perf_counter();agent=load_agent()
    row=scene_rows('pilot')[0];token=row['token'];features=features_for(agent,row)
    tokens=encode(agent,features)
    reference=torch.load(ROOT/'pilot/observations'/f'{token}.pt',weights_only=False)['tokens']
    old=np.load(ROOT/'pilot/trajectories'/f'{token}.npz')['trajectories'][0]
    prediction=predict(agent.action_head,tokens,action_input(features),token,1101)
    value=dict(status='PASS' if torch.equal(tokens.cpu(),reference) and np.array_equal(prediction,old) else 'INVALID',
        native_tokens_max_error=float((tokens.cpu()-reference).abs().max()),
        trajectory_max_error=float(abs(prediction-old).max()),protocol_sha256=identity,owner=runtime_identity(),
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,
            scipy=scipy.__version__,shapely=shapely.__version__),seconds=time.perf_counter()-start)
    atomic_json(ROOT/'resources'/f'qualified_{args.name}.json',value);print(value,flush=True)
    if value['status']!='PASS':raise RuntimeError('Host output parity failed')


if __name__=='__main__':main()
