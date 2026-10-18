"""Finite dependency waits for explicitly requested experiment stages."""
from __future__ import annotations
import argparse
from .common import *
from .updates import families,directory,wait_file


def h1():
    from .statistics import collect_h1
    from .controls import ridge
    for f in families():wait_file(directory(f['id'])/'score_complete.json')
    wait_file(ROOT/'baseline/high_screen/complete.json')
    collect_h1(None);ridge(None)


def selection():
    from .probe_runner import select
    cfg,_=protocol()
    for arm in cfg['probes']['arms']:
        for slot in range(3):
            n=4 if arm in ('JQ','JE') else 1
            for rank in range(n):wait_file(ROOT/'probes'/f'tuning_{arm}_{slot}'/f'eval_D_cal_{rank}.json')
    select(None)


def probes():
    from types import SimpleNamespace
    from .probe_metrics import analyze
    from .probe_diagnostics import main as diagnostics
    cfg,_=protocol()
    for arm in cfg['probes']['arms']:
        for seed in cfg['seeds']['training']:
            dest=ROOT/'probes'/f'formal_{arm}_{seed}'
            wait_file(dest/'config.json');world=read(dest/'config.json')['world_size']
            for split in ['D_fit','D_cal','D_audit']+(['D_audit_shuffled'] if arm=='JE' else []):
                for rank in range(world):wait_file(dest/f'eval_{split}_{rank}.json',timeout=7200)
    analyze(SimpleNamespace(prefix='formal',split='D_audit'));diagnostics()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['h1','selection','probes']);a=p.parse_args();globals()[a.action]()
