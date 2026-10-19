"""Separate E6 policy-initialization and teacher-ceiling comparisons."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.analyze import bank,bootstrap,compare


def analyze(split):
    cfg=config();il=bank(split,'E0','IL');rl=bank(split,'E0','RL');screen=bank(split,'E0','IL_screen');stable=(screen[...,-1]>=.8).all(-1);logs=[r['log_name'] for r in rows(split)]
    variants=['IL_GRPO','E5_GRPO','IL_JOINT'] if split!='navtest' else [read(ROOT/'selection/E6_navtest.json')['variant']]
    methods={};arrays={}
    for v in variants:
        y=np.stack([bank(split,f'E6_{v}_{s}_train') for s in cfg['train_seeds']]);arrays[v]=y
        methods[v]=dict(metrics=bootstrap(y,il,logs,stable),relative_teacher=bootstrap(y,rl,logs,stable),
            costs=[read(ROOT/'runs'/f'E6_{v}_{s}_train'/'complete.json')['seconds'] for s in cfg['train_seeds']],
            initialization_cost_included='E5 1000-step pretraining plus teacher/gate preprocessing must be added' if v=='E5_GRPO' else 'Original IL initialization')
    contrasts={}
    if len(arrays)>1:
        for a in ['E5_GRPO','IL_JOINT']:contrasts[a+'_vs_IL_GRPO']=compare(arrays[a],arrays['IL_GRPO'],il,logs,stable)
    write(ROOT/'analysis'/f'E6_{split}.json',dict(status='COMPLETE',split=split,scenes=len(il),methods=methods,contrasts=contrasts))
    print({v:d['relative_teacher']['G'] for v,d in methods.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');a=p.parse_args();analyze(a.split)
