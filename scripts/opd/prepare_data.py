import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');a=p.parse_args();cfg=config(a.config)
    source=PROJECT/'outputs/trajectory_effect_validation_20261019/manifests';mapping={'train':'D_fit','cal':'D_cal','dev':'D_audit'};logs={};parents={}
    for split,old in mapping.items():
        data=read(source/f'{old}.json')
        assert all(r['split']=='navtrain' and Path(r['metric_cache_path']).is_file() and (Path(cfg['observation_cache'])/f"{r['token']}.pt").is_file() for r in data)
        logs[split]={r['log_name'] for r in data};parents[split]={r['driver_record'] for r in data}
        write(ROOT/'manifests'/f'{split}.json',data)
    for a in logs:
        for b in logs:
            if a!=b:assert not logs[a]&logs[b] and not parents[a]&parents[b]
    write(ROOT/'manifests/identity.json',dict(splits={s:dict(scenes=len(rows(s)),logs=len(logs[s]),sha256=sha(ROOT/'manifests'/f'{s}.json')) for s in mapping},
        source=str(source),source_checkpoint='frozen VLM encoding; the action projection is not cached',
        caveat='All splits are official navtrain; held out from this new training, potentially seen by original pretraining'))
    print(read(ROOT/'manifests/identity.json'))

if __name__=='__main__':main()
