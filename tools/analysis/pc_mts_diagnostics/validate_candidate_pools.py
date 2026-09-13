"""Pool invariants and a pre-robustness check for accidentally identical selectors."""
import argparse
import pandas as pd
from common import *

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];n=cfg['pool_size'];missing=[];counts={m:0 for m in args.methods};fills={m:0 for m in args.methods}
    pending=pending_pc_tokens(out,cfg,sc) if args.allow_pending else set()
    for r in sc['scenes']:
        token=r['token'];raw_path=out/'raw_candidates'/f'{token}.npz';raw=np.load(raw_path);a=raw['trajectories'];raw_scores=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories']
        assert len(a)==cfg['raw_gt_count']+4*cfg['raw_model_count'];assert np.array_equal(a[0],np.asarray(r['gt']))
        assert 'official_il' not in set(raw['source'])
        for m in args.methods:
            if m=='pc_mts' and token in pending:continue
            z=np.load(out/'candidate_pools'/m/f'{token}.npz');meta=json.loads(str(z['metadata']));assert meta['raw_sha256']==sha(raw_path)
            if len(z['trajectories'])==0:
                assert m=='pc_mts' and meta['empty_policy']=='missing' and meta['selection_status']=='no_qualified_parent';missing.append(token);continue
            assert z['trajectories'].shape==(n,8,3) and z['scores'].shape==(n,7)
            assert np.isfinite(z['trajectories']).all() and np.isfinite(z['scores']).all()
            for i in np.flatnonzero(~z['is_fill']):
                assert np.array_equal(z['trajectories'][i],a[z['raw_index'][i]])
                assert np.array_equal(z['scores'][i],raw_scores[z['raw_index'][i]])
            for i in np.flatnonzero(z['is_fill']):assert str(z['fill_parent_id'][i]) and z['fallback_level'][i]>=4
            counts[m]+=1;fills[m]+=int(z['is_fill'].sum())
    overlap=pd.read_csv(out/'metrics'/('conditional_pc' if args.allow_pending else '')/'pool_overlap.csv');average=overlap.groupby(['a','b']).jaccard.mean()
    almost_identical=bool(len(average)>0 and (average>.95).all())
    name='candidate_pool_validation_partial.json' if args.allow_pending else 'candidate_pool_validation.json'
    save(out/'manifests'/name,dict(scene_count=len(sc['scenes']),methods=args.methods,pool_size=n,complete_scene_counts=counts,filler_counts=fills,missing_pc_scene_count=len(missing),missing_pc_tokens=missing,pending_pc_scene_count=len(pending),full_completion=not pending and not missing,mean_jaccard={f'{a} / {b}':float(v) for (a,b),v in average.items()},all_methods_almost_identical=almost_identical,all_non_fill_candidates_trace_to_common_raw=True))
    if almost_identical:raise RuntimeError('All candidate pools have mean Jaccard > .95; inspect source diversity/selection before robustness evaluation')
    print('Candidate pool structure and selector-difference gate passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');p.add_argument('--allow-pending',action='store_true');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);main(p.parse_args())
