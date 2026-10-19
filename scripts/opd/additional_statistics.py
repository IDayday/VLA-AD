"""Planned baseline/teacher comparisons, without rerunning any policy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.analyze import bank,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--anchor-only',action='store_true');a=p.parse_args();split=a.split;cfg=config()
    base=bank(split,'E0','IL');teacher=bank(split,'E0','RL');stable=(bank(split,'E0','IL_screen')[...,-1]>=.8).all(-1);logs=[r['log_name'] for r in rows(split)]
    if a.anchor_only:
        if split!='dev':raise ValueError('Unselected E4-B/C have no registered final test')
        left=np.stack([bank(split,f'train_E4-C_{s}') for s in cfg['train_seeds']]);right=np.stack([bank(split,f'train_E4-B_{s}') for s in cfg['train_seeds']])
        configs=[read(ROOT/'runs'/f'train_{e}_{s}/config.json') for e in ['E4-B','E4-C'] for s in cfg['train_seeds']]
        assert len({(v['lr'],v['il_weight'],v['global_batch'],v['steps']) for v in configs})==1
        result=compare(left,right,base,logs,stable,cfg['bootstrap_replicates'])
        write(ROOT/'analysis/secondary_anchor_vs_GT_dev.json',dict(contrast='E4-C minus E4-B',result=result,
            scope='Secondary comparison added after viewing dev means; fixed existing models, no new training or queries, no final test, does not replace primary decisions',
            matched=['IL initialization','seed','scene/noise schedule','learning rate','GT weight','updates','capacity'],owner=identity()))
        print(result);return
    result={};arrays={e:np.stack([bank(split,f'train_{e}_{s}') for s in cfg['train_seeds']]) for e in ['E2','E3','E5']}
    for e,y in arrays.items():result[e+'_vs_RL']=compare(y,np.broadcast_to(teacher,y.shape),base,logs,stable,cfg['bootstrap_replicates'])
    path=ROOT/'analysis'/f'formal_{split}.json'
    if split!='navtest':
        sft=np.stack([bank(split,f'train_E1_{s}') for s in cfg['train_seeds']]);extra=compare(arrays['E2'],sft,base,logs,stable,cfg['bootstrap_replicates'])
        d=read(path);d['contrasts']['E2_vs_E1']=extra;write(path,d)
    write(ROOT/'analysis'/f'teacher_comparisons_{split}.json',dict(scope='Secondary paired teacher gaps; all policies fixed, no checkpoint selection',contrasts=result));print({k:v['PDMS'] for k,v in result.items()})


if __name__=='__main__':main()
