"""Finalize the registered two-epoch endpoint only after the extension decision."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.process_improvement.common import *


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['OPD','ANCHOR'],required=True);a=p.parse_args()
    choice=read(ROOT/'selection/four_epoch_extension.json')
    assert not choice['decisions'][a.method]['extend_to_four_epochs'],'Registered four-epoch extension is required'
    for seed in config()['train_seeds']:
        dest=ROOT/'runs'/f'full_{a.method}_{seed}'
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        assert state['step']==4681 and state['config']['steps']==9361
        assert read(dest/'paused.json')['step']==4681
        value=dict(status='COMPLETE_REGISTERED_TWO_EPOCH_ENDPOINT',steps=4681,scene_exposures=149792,
                   physical_scores=state['physical_scores_global'],seconds=state['seconds'],
                   checkpoint_sha256=sha(dest/'latest.pt'),requested_maximum_updates=9361,
                   additional_four_epoch_extension=False,extension_decision_sha256=sha(ROOT/'selection/four_epoch_extension.json'))
        if (dest/'complete.json').exists():assert read(dest/'complete.json')==value
        else:write(dest/'complete.json',value)
    print('REGISTERED_TWO_EPOCH_COMPLETION',a.method,flush=True)


if __name__=='__main__':main()
