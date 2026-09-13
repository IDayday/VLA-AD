"""Meaningful preflight gates: model/schema, sampler identity, evaluator, geometry."""
import argparse, json, sys
import numpy as np
from common import *
from build_candidate_reservoir import smooth_perturbation,perturb_bank
from analyze_policy_distribution import geometry
from evaluate_cached_rollouts import init_worker,score_arrays

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];checks={}
    assert len(sc['scenes'])==20 and cfg['num_rollouts']==8
    for m in MODELS:
        paths=list((out/'rollouts'/m).glob('*.npz'));assert len(paths)==20
        for p in paths:
            z=np.load(p);assert z['trajectories'].shape==(8,8,3) and np.isfinite(z['trajectories']).all();assert json.loads(str(z['metadata']))['checkpoint_hash']==next(r for r in models() if r['name']==m)['sha256']
    checks['five_real_checkpoints_20x8']=True
    checks['scalar_batch_parity']=max(r['max_abs_error'] for r in read(out/'manifests/evaluator_parity.json'))
    checks['gt_representation_parity']=max(r['GT_pose_conversion_error'] for r in read(out/'manifests/gt_representation_parity.json'))
    a=np.zeros((8,8,3));g,_=geometry(a);assert g['pairwise_ade']==0 and g['spread_auc']==0 and g['effective_rank']==0
    a[4:,:,0]=4.;g,_=geometry(a);assert abs(g['pairwise_ade']-16*4/28)<1e-12 and abs(g['effective_rank']-1)<1e-12
    shifted=a.copy();shifted[...,:2]+=[3,7];gg,_=geometry(shifted);assert abs(g['pairwise_ade']-gg['pairwise_ade'])<1e-12
    checks['spread_controls_and_translation_invariance']=True
    token=sc['scenes'][0]['token'];gt=np.asarray(sc['scenes'][0]['gt'])
    selection=perturb_bank(gt,token,'selection_local',[.03,.06],2);heldout=perturb_bank(gt,token,'heldout_local',[.02,.05,.1,.2],3)
    assert not {digest(t.tolist()) for t in selection}&{digest(t.tolist()) for t in heldout}
    for i,amp in enumerate([.02,.05,.1,.2]):assert np.linalg.norm(heldout[i*3:(i+1)*3,:,:2]-gt[:,:2],axis=-1).max()<=amp+1e-8
    checks['heldout_seed_separation_and_amplitude_bounds']=True
    assert len(list((out/'figures').glob('*.png')))>=14 and len(list((out/'figures').glob('*.pdf')))>=14
    checks['policy_and_readiness_figure_pipeline']=True
    # Check archived reward weighting against actual PDM scorer, not only a copied formula.
    import lzma,pickle
    init_worker()
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    from navsim.evaluate.pdm_score_batch import pdm_score_batch_same_cache
    from evaluate_cached_rollouts import STATE
    scorer=PDMScorer(STATE['sampling'],PDMScorerConfig(progress_weight=10,ttc_weight=5,comfortable_weight=2))
    errors=[]
    for r in sc['scenes'][:4]:
        with lzma.open(r['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        a=np.load(out/'rollouts/official_il'/f"{r['token']}.npz")['trajectories']
        s=score_arrays(cache,a);formula=s[:,0]*s[:,1]*(10*s[:,2]+5*s[:,3]+2*s[:,4])/17
        native=pdm_score_batch_same_cache(cache,a,STATE['sampling'],STATE['simulator'],scorer,use_exact_array_conversion=True)
        errors.append(float(np.abs(formula-np.array([r.score for r in native])).max()))
    assert max(errors)<=1e-8;checks['native_grpo_reward_max_error']=max(errors)
    if (out/'manifests/coverage_validation.json').exists():
        p=read(out/'manifests/coverage_validation.json');assert p['scene_count']==20 and p['pool_size']==8 and p['full_completion']
        checks['candidate_construction_status']='all four methods complete under original rules plus user-authorized coverage extension'
    else:checks['candidate_construction_status']='empty-parent interpretation pending user clarification; not a completed M4 gate'
    save(out/'manifests/smoke_validation.json',checks);print(json.dumps(checks,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/pc_mts_diagnostics/smoke.yaml'));p.add_argument('--manifest',default=str(OUT/'manifests/scenes_smoke.json'));main(p.parse_args())
