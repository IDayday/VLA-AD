"""Post-hoc coarser-cluster sensitivity; primary results remain immutable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *
from scripts.opd.analyze import bank,bootstrap,compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','navtest'],required=True);a=p.parse_args()
    target=ROOT/'analysis'/f'parent_record_sensitivity_{a.split}.json'
    if target.exists():raise FileExistsError('Sensitivity result already completed')
    primary_path=ROOT/'analysis'/f'formal_{a.split}.json';primary_hash=sha(primary_path);d=read(primary_path)
    protocol=sha(SOURCE/'configs/opd/protocol.yaml');selected=rows(a.split)
    clusters=[r.get('driver_record',r['log_name'].rsplit('_',2)[0]) for r in selected]
    registration=dict(scope='POST_HOC sensitivity, registered after primary dev and navtest estimates were viewed; no new training/inference or selection',
        reason='Multiple official NAVSIM logs share a parent driving record; examine residual cross-log dependence',
        primary_sha256=primary_hash,protocol_sha256=protocol,all_methods=list(d['methods']),all_contrasts=list(d['contrasts']),
        cluster_unit='Complete parent driving record including vehicle ID',parents=len(set(clusters)),logs=len({r['log_name'] for r in selected}),
        scenes=len(selected),replicates=2000,owner=identity())
    write(ROOT/'analysis'/f'parent_record_registration_{a.split}.json',registration)
    base=bank(a.split,'E0','IL');stable=(bank(a.split,'E0','IL_screen')[...,-1]>=.8).all(-1)
    values={e:np.stack([bank(a.split,f'train_{e}_{s}') for s in config()['train_seeds']]) for e in d['methods']}
    methods={e:bootstrap(v,base,clusters,stable) for e,v in values.items()};contrasts={}
    for key in d['contrasts']:
        first,second=key.split('_vs_');contrasts[key]=compare(values[first],values[second],base,clusters,stable)
    for e,v in methods.items():
        assert all(abs(m['estimate']-d['methods'][e]['metrics'][k]['estimate'])<1e-12 for k,m in v.items() if m is not None)
    assert sha(primary_path)==primary_hash and sha(SOURCE/'configs/opd/protocol.yaml')==protocol
    write(target,dict(passed=True,registration=registration,methods=methods,contrasts=contrasts,
        interpretation='Secondary robustness description only; original registered log-cluster decisions and intervals are retained'))
    print('PARENT_RECORD_SENSITIVITY_COMPLETE',a.split,len(set(clusters)))


if __name__=='__main__':main()
