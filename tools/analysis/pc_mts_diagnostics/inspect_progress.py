"""Read authoritative output coverage and process liveness, without assuming completion."""
import argparse,os
from common import *

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];n=len(sc['scenes']);tokens={r['token'] for r in sc['scenes']}
    def count(directory):return len({p.stem for p in directory.glob('*.npz')}&tokens)
    active=read(out/'manifests/active_process.json') if (out/'manifests/active_process.json').exists() else None
    if active:
        try:os.kill(active['pid'],0);active['live']=True
        except ProcessLookupError:active['live']=False
    status=dict(scene_count=n,features=len(list((out/'features').glob('*.pt'))),rollouts={m:count(out/'rollouts'/m) for m in MODELS},scored_rollouts={m:count(out/'evaluator/rollouts'/m) for m in MODELS},raw_reservoir=count(out/'raw_candidates'),candidate_pools={m:count(out/'candidate_pools'/m) for m in METHODS},heldout_scored={m:count(out/'evaluator/heldout'/m) for m in METHODS},active=active)
    save(out/'report/progress.json',status);print(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
