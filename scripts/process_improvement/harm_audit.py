"""Apply only the locked calibration-confirm harm boundary, preserving endpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--evaluation',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--finalize-stop',action='store_true')
    a=p.parse_args();dest=ROOT/'runs'/a.run
    assert (dest/'paused.json').exists(),'Only a paused fixed endpoint may be audited for early stopping'
    paused=read(dest/'paused.json');cfg=read(dest/'config.json');step=paused['step']
    physical=paused['physical_scores']
    epoch_steps={int(__import__('math').ceil(len(membership(cfg['split']))*epoch/cfg['global_batch'])) for epoch in [1,2,4,8]}
    score_endpoint=any(b<=physical<b+cfg['global_batch']*max(8,config()['candidate_count']*config()['fit_replicates']+2*config()['guard_replicates']) for b in config()['score_endpoints'])
    assert step in epoch_steps or score_endpoint,'Early-stop check is outside registered endpoints'
    result=compare([a.evaluation],a.baseline,'cal_confirm')
    point=result['point'];ci=result['CI95']
    triggers=[name for name in ['G','delta_NC','delta_DAC','delta_TTC'] if point[name]<=-.02 and ci[name][1]<0]
    record=dict(status='HARM_BOUNDARY_REACHED' if triggers else 'CONTINUE',run=a.run,step=step,
                physical_scores=physical,triggers=triggers,comparison=result,rule='Locked protocol formal_harm_boundary; cal-confirm only')
    path=dest/f'harm_audit_{step}.json'
    if path.exists():assert read(path)==record
    else:write(path,record)
    if a.finalize_stop:
        if not triggers:raise ValueError('Cannot stop on this harm rule: boundary not reached')
        if (dest/'complete.json').exists():raise FileExistsError('Preserve completed lineage')
        import torch
        state=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        assert state['step']==step
        write(dest/'complete.json',dict(status='STOPPED_PREREGISTERED_HARM',steps=step,
              scene_exposures=step*cfg['global_batch'],physical_scores=physical,seconds=state['seconds'],
              checkpoint_sha256=sha(dest/'latest.pt'),boundary_record=str(path),
              requested_updates=cfg['steps'],higher_endpoints='NOT_RUN_after_precommitted_harm_boundary',owner=identity()))
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
