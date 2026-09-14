"""Final integrity, native replay, evaluator, statistics, and artifact audit."""
import subprocess, dataclasses
from common_fd import *
from evaluate_cached_rollouts import init_worker,parity,score_arrays
from teacher_audit import native_helpers

def main():
    config_identity();checks=[]
    for model in MODELS:
        r=read(OUT/'audits'/f'replay_{model}.json')
        assert r['parameters_and_buffers_unchanged']
        assert r['bitwise_cache_replay'] and r['max_abs_error']==0
        assert all(x['current_runtime_repeat_error']==0 for x in r['checks'])
        assert sha(MODEL_INFO[model]['checkpoint_path'])==MODEL_INFO[model]['sha256']
    init_worker();hybrid_checks=[]
    for s in SCENES[:8]:
        b={m:load_model_scene(m,s['token'])[0] for m in ['official_il','grpo_9041']}
        arrays=combine(center(b['grpo_9041']),residual(b['official_il']))[:2]
        hybrid_checks.append(parity(s,arrays))
    save(OUT/'audits/hybrid_scalar_batch_parity.json',hybrid_checks)
    replay_checks=[]
    for scene in SCENES[:8]:
        with lzma.open(scene['metric_cache_path'],'rb') as f:cache=pickle.load(f)
        for m in MODELS:
            t=np.load(OUT/'cache/replay'/m/f"{scene['token']}.npz")['trajectories']
            _,oldscore,_=load_model_scene(m,scene['token'])
            sc=score_arrays(cache,t);err=float(np.abs(sc-oldscore[:16]).max())
            replay_checks.append(dict(token=scene['token'],model=m,max_component_error=err,PDMS_point_error=float(np.abs(sc[:,6]-oldscore[:16,6]).max()*100),
                conservative_feasible_labels_equal=bool(np.array_equal(v1.feasible(sc),v1.feasible(oldscore[:16])))))
    save(OUT/'audits/replay_score_consistency.json',replay_checks)
    protected=read(OUT/'manifests/protected_files.json')
    assert all(sha(ROOT/p)==h for p,h in protected.items()),'Protected old results changed'
    files=sorted((OUT/'cache/scored').glob('*.npz'));assert len(files)==1000
    assert {p.stem for p in files}=={s['token'] for s in SCENES}
    counts={}
    for p in files:
        with np.load(p) as z:
            assert json.loads(str(z['metadata']))['protocol_sha256']==config_identity()
            for key in z.files:
                if key=='metadata':continue
                a=z[key];assert a.ndim==2 and a.shape[1]==7 and np.isfinite(a).all()
                counts[key]=counts.get(key,0)+len(a)
    policy=pd.read_csv(OUT/'metrics/policy_scene.csv');assert len(policy)==5000
    assert policy.groupby('model').token.nunique().eq(1000).all()
    old=pd.read_csv(v1.OUT/'metrics/policy_distribution.csv')
    merged=policy.merge(old,left_on=['model','token'],right_on=['checkpoint','token'],validate='one_to_one')
    np.testing.assert_allclose(merged.pairwise_ADE,merged.pairwise_ade,atol=1e-12,rtol=0)
    np.testing.assert_allclose(merged.Spread_AUC,merged.spread_auc,atol=1e-12,rtol=0)
    t=pd.read_csv(OUT/'metrics/teacher_candidates_scored.csv');assert len(t)==12800
    assert t.groupby(['model','token']).expected_weight_pre_budget.sum().sub(1).abs().max()<1e-6
    targets=pd.read_csv(OUT/'metrics/teacher_output_scene.csv');assert len(targets)==2000
    assert targets.teacher_GT_PDMS.notna().all()
    assert targets.gt_archive_ADE.dropna().max()<2e-6
    source=[]
    raw=pd.read_csv(OUT/'metrics/teacher_raw_sources.csv')
    raw['source_bucket']=raw.source.apply(native_helpers().SourceHelpers._source_bucket)
    source_keys=set(zip(t.model,t.source_bucket))|set(zip(raw.model,raw.source_bucket))
    for m,b in sorted(source_keys):
        g=t[(t.model==m)&(t.source_bucket==b)]
        rg=raw[(raw.model==m)&(raw.source_bucket==b)]
        source.append(dict(model=m,source_bucket=b,raw_count=int(rg.raw_count.sum()),raw_scene_count=rg.token.nunique(),
            selected_support_count=len(g),selected_scene_count=g.token.nunique(),
            expected_loss_mass_pre_budget_per_scene=g.expected_weight_pre_budget.sum()/1000,
            selected_PDMS_pooled=g.PDMS.mean(),selected_feasible_pooled=g.feasible.mean()))
    csv('teacher_source_summary.csv',source)
    chain=pd.read_csv(OUT/'metrics/historical_grpo_scene.csv');assert len(chain)==1500 and chain.groupby('step').token.nunique().eq(300).all()
    assert read(OUT/'audits/grpo_9041_lineage.json')['same_file_content']
    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('test_mechanisms.py'))],capture_output=True,text=True)
    save(OUT/'audits/unit_tests.json',dict(returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    assert proc.returncode==0
    for name in ['Fig1_width_quality','Fig2_actual_teachers_weights','Fig3_teacher_to_output','Fig4_center_residual_counterfactual','Fig5_historical_GRPO_evolution','Fig6_scene_stratification']:
        for ext in ['png','pdf','svg']:assert (OUT/'figures'/f'{name}.{ext}').stat().st_size>1000
    runtime_files=[NATIVE,Path(MODEL_INFO['official_il']['code_root'])/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py',
        v1.CODE/'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',v1.CODE/'navsim/evaluate/pdm_score_batch.py']
    training_files=[]
    for m in ARCHIVES:
        folder=Path(MODEL_INFO[m]['checkpoint_path']).parent
        for filename in ['train_args.json','commands.log','run_training_recogdrive.log','trainable_parameter_counts.json']:
            f=folder/filename
            if f.exists():training_files.append(dict(model=m,path=str(f),sha256=sha(f)))
    save(OUT/'manifests/historical_training_files.json',training_files)
    save(OUT/'manifests/run_summary.json',dict(status='COMPLETE',scene_count=1000,missing_scenes=0,models=5,existing_primary_rollouts=320000,
        actual_teacher_records=2000,actual_selected_teachers=12800,new_NAVSIM_scores=sum(counts.values()),new_scores_by_type=counts,
        historical_chain_scenes=300,historical_chain_draws_per_checkpoint=32,
        replay_draws=640,optimizer_updates=0,config_sha256=config_identity(),old_files_verified=len(protected),
        old_geometry_unchanged=True,checkpoint_hashes_unchanged=True,scene_paired_bootstrap=3000,log_cluster_bootstrap=3000,
        source_identity='Two real historical training archives; no mapping to reconstructed Score/Pareto/PC pools',
        data_scope='Navtrain internal descriptive mechanism analysis; MTS training-data overlap is intentional, no untouched generalization claim',
        replay_bitwise_status={m:read(OUT/'audits'/f'replay_{m}.json')['bitwise_cache_replay'] for m in MODELS},
        replay_max_abs_error={m:read(OUT/'audits'/f'replay_{m}.json')['max_abs_error'] for m in MODELS},
        replay_max_PDMS_point_difference=max(r['PDMS_point_error'] for r in replay_checks),
        engineering_fixes=['log_name→log parser key; selected-GT-absent scenes use existing exact-GT score as reference without adding training targets',
            'Initial replay set all requires_grad flags false and differed from cache by ≤3.815e-6. Restoring the V1 final float cast alone did not remove it; preserving original module requires_grad flags restored exact bitwise equality for all five models. Inference_mode and absent optimizer prevent updates; weights/buffers verified unchanged. Initial audits retained; no tolerance relaxed or scientific cache replaced.',
            'Report rendering optional tabulate dependency unavailable; used a local Markdown table formatter instead of changing the environment.'],
        code_hashes={str(p):sha(p) for p in runtime_files},new_code_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print('AUDIT PASS',sum(counts.values()),'new scores;',len(protected),'old files preserved',flush=True)

if __name__=='__main__':main()
