"""Reproduce the old B8/1000 recipe in a NEW artifact namespace only.

Historical metadata is copied read-only as evidence; its gates/budget govern
only this reproduction and are never consulted by the new training pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import shutil
import subprocess
from scripts.process_improvement.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--exp',choices=['E3','E4-A'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    dest=ROOT/'historical_reproductions'
    complete=dest/'runs'/('reproduce_'+a.exp+'_42')/'complete.json'
    if complete.exists():
        value=read(complete)
        assert value['optimizer_updates']==1000 and value['config']['global_batch']==8
        assert (dest/f'{a.exp}_parameter_comparison.json').exists()
        print('HISTORICAL_ALREADY_COMPLETED_IN_THIS_STUDY',a.exp,flush=True)
        return
    files=['protocol.yaml','protocol.lock.json','audit/gates.json','audit/checkpoints.json','tests/cpu.json',
           'analysis/G2.json','calibration/gradient_scales.json','selection/tuning.json',
           'manifests/train.json','manifests/token_lengths.json']
    for name in files:
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert sha(path)==sha(HISTORY/name)
        else:shutil.copyfile(HISTORY/name,path)
    env=dict(os.environ,OPD_RUN_ROOT=str(dest))
    command=[PYTHON,'scripts/opd/train.py','--exp',a.exp,'--stage','train','--seed','42','--name','reproduce_'+a.exp+'_42']
    if a.resume:command.append('--resume')
    result=subprocess.run(command,cwd=SOURCE,env=env)
    if result.returncode:raise RuntimeError('Historical recipe failed; new independent experiments remain admitted')
    import torch
    actual=torch.load(dest/'runs'/('reproduce_'+a.exp+'_42')/'latest.pt',map_location='cpu',weights_only=False)['student']
    expected=torch.load(HISTORY/'runs'/('train_'+a.exp+'_42')/'latest.pt',map_location='cpu',weights_only=False)['student']
    differences={}
    for key,value in actual.items():
        if key.startswith('action_aware_aux_head.'):continue
        if value.dtype.is_floating_point and torch.isfinite(value).all():differences[key]=float((value-expected[key]).abs().max())
        else:assert torch.equal(value,expected[key])
    write(dest/f'{a.exp}_parameter_comparison.json',dict(experiment=a.exp,global_batch=8,updates=1000,
          max_error=max(differences.values()),per_parameter=differences,reference_checkpoint=str(HISTORY/'runs'/('train_'+a.exp+'_42')/'latest.pt'),
          scope='Historical recipe reproduction only; not the main scaled warm start'))
    print('HISTORICAL_REPRODUCTION',a.exp,max(differences.values()),flush=True)


if __name__=='__main__':main()
