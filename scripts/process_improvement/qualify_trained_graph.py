"""Additional B1 graph parity after genuine optimization of each checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import torch
from scripts.process_improvement.common import *


def main():
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.cuda_inference import FrozenNativeGraph
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True)
    a=p.parse_args();configure();head=build_head(a.checkpoint,frozen=True);adapter=NativeAdapter(head);graph=None;records=[]
    for row in membership('dev')[:8]:
        tokens,inputs=conditioning(observation(row))
        if graph is None:graph=FrozenNativeGraph(head,tokens,inputs)
        graph.prepare(tokens,inputs)
        for seed in config()['evaluation_seeds']:
            plan=RNGKey(STUDY,seed,0,0,row['token'],0,-1).plan('final_eval',device='cuda')
            rollin=adapter.sample_rollin(tokens,inputs,plan);native=rollin.trajectory
            accelerated,states=graph.sample(plan,return_states=True)
            assert torch.equal(native,accelerated),(row['token'],seed,float((native-accelerated).abs().max()))
            assert torch.equal(states,rollin.states),(row['token'],seed,'Intermediate graph state mismatch')
            records.append(dict(token=row['token'],seed=seed,trajectory_max_error=0.))
    write(ROOT/'tests/trained_graph'/f'{a.name}.json',dict(status='PASS',checkpoint_sha256=sha(a.checkpoint),records=records,intermediate_states_exact=True,owner=identity()))
    print('TRAINED_NATIVE_GRAPH_PARITY_PASS',a.name,len(records),flush=True)


if __name__=='__main__':main()
