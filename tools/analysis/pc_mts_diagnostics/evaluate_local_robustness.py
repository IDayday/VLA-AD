"""Held-out coherent perturbations, never reused from selection-stage tests."""
import argparse, concurrent.futures
from common import *
from build_candidate_reservoir import perturb_bank

def build_scene(task):
    r,cfg,out,ident,methods=task
    token=r['token']
    for method in methods:
        parent=out/'candidate_pools'/method/f'{token}.npz';dest=out/'heldout'/method/f'{token}.npz';meta=dict(ident,parent_sha256=sha(parent),stream='heldout_local',amplitudes=cfg['evaluation_perturbation_amplitudes'])
        if valid_npz(dest,meta):continue
        a=np.load(parent)['trajectories'];k=len(cfg['evaluation_perturbation_amplitudes'])*cfg['evaluation_directions_per_amplitude']
        values=np.stack([perturb_bank(t,token,'heldout_local',cfg['evaluation_perturbation_amplitudes'],cfg['evaluation_directions_per_amplitude']) for t in a]) if len(a) else np.empty((0,k,8,3))
        save_npz(dest,meta,trajectories=values)
    return r['token']

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];ident=identity(cfg,sc)
    pending=pending_pc_tokens(out,cfg,sc) if args.allow_pending else set()
    tasks=[(r,cfg,out,ident,[m for m in args.methods if m!='pc_mts' or r['token'] not in pending]) for r in sc['scenes']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(cfg['cpu_workers'],len(tasks))) as ex:
        for i,token in enumerate(ex.map(build_scene,tasks,chunksize=1)):
            if i%100==0:print(f'Held-out perturbations {i+1}/{len(tasks)}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--allow-pending',action='store_true');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);main(p.parse_args())
