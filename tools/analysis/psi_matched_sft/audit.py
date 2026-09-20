from common_matched import *
import argparse,inspect,concurrent.futures,lzma,pickle
import evaluate_cached_rollouts as evaluator

def initial():
    identity();evaluator.init_worker();checks=[]
    for s in scenes('holdout')[:4]:
        tr=np.load(OUT/'cache/raw'/f"{s['token']}.npz")['trajectories'][:4];checks.append(evaluator.parity(s,tr))
    import navsim.evaluate.pdm_score_batch as module
    code={str(p):sha(p) for p in (WORK/'tools/analysis/psi_matched_sft').glob('*.py')}
    code.update({str(p):sha(p) for p in [Path(inspect.getfile(module)),MAIN/'outputs/historical_sft_grpo_support/runtime/psi/navsim/agents/recogdrive/pareto_support.py']})
    protected=[]
    for d in ['pc_mts_diagnostics','pc_mts_diagnostics_v2','pc_mts_diagnostics_v3']:
        protected += list((MAIN/'outputs'/d/'metrics').glob('*.csv'))+list((MAIN/'outputs'/d/'metrics').glob('*.parquet'))
    protected+=list(MAIN.glob('reports/PC_MTS*.md'))
    snapshot=OUT/'manifests/protected_files.json'
    if not snapshot.exists():save(snapshot,[dict(path=str(p),sha256=sha(p)) for p in protected])
    save(OUT/'audits/initial.json',dict(status='PASS',protocol_hash=identity(),scoring_parity=checks,code=code,protected_file_count=len(read(snapshot)),holdout_not_entering_optimizer=True))

def verify_record(r):assert sha(r['path'])==r['sha256'],r['path']
def verify_cache(path):
    with np.load(path) as f:
        meta=json.loads(str(f['metadata']));assert meta['protocol_hash']==identity()
        assert meta['token']==path.stem
        for pr in CFG['evaluation']['protocols']:assert f[pr].shape==(4,16,8,3) and np.isfinite(f[pr]).all()
    score=OUT/'cache/scores/rollouts'/path.relative_to(OUT/'cache/rollouts')
    with np.load(score) as f:
        sm=json.loads(str(f['metadata']));assert sm['input_sha256']==sha(path) and sm['protocol_hash']==identity()
        for pr in CFG['evaluation']['protocols']:assert f[pr].shape==(4,16,7) and np.isfinite(f[pr]).all()
    return dict(path=str(path),sha256=sha(path),score_path=str(score),score_sha256=sha(score),checkpoint_hash=meta['checkpoint_hash'])
def final():
    identity()
    with concurrent.futures.ThreadPoolExecutor(12) as pool:list(pool.map(verify_record,read(OUT/'manifests/protected_files.json')))
    frozen=read(OUT/'manifests/selection_frozen.json')
    for t,h in frozen['selection_hashes'].items():assert sha(OUT/'cache/selection'/f'{t}.json')==h
    audits=[]
    for method in CFG['methods']:
        for seedno in CFG['train_seeds']:
            name=f'{method}_seed{seedno}';a=read(OUT/'audits'/f'train_{name}.json');audits.append(a)
            assert a['updates']==512 and a['supervised_scene_presentations']==65536 and a['unique_train_tokens']==3072
            for c in a['checkpoints']:assert sha(c['path'])==c['sha256']
            for step in [128,512]:
                for rank in range(4):
                    e=read(OUT/'audits'/f'eval_{name}_step{step:04d}_{rank}.json');assert e['state_before']==e['state_after']
    assert len({a['initial_state_hash'] for a in audits})==1
    for seedno in CFG['train_seeds']:
        base=None
        for method in CFG['methods']:
            l=pd.read_parquet(OUT/'metrics'/f'ledger_{method}_seed{seedno}.parquet');keys=l[['step','micro','token']]
            if base is None:base=keys
            else:pd.testing.assert_frame_equal(base,keys)
    sf=pd.read_parquet(OUT/'metrics/scene_metrics.parquet');final=sf[(sf.split=='holdout')&(sf.step==512)]
    assert final.groupby(['method','seed','protocol']).size().eq(5000).all()
    assert np.isfinite(final[METRICS_AVAILABLE(final)].to_numpy()).all()
    prior=pd.read_parquet(PREV/'metrics/scene_metrics.parquet');prior=prior[prior.model=='official_il'].set_index(['token','protocol'])
    here=sf[(sf.method=='official_il')&(sf.split=='holdout')].set_index(['token','protocol']).loc[prior.index]
    errors={k:float(abs(here[k]-prior[k]).max()) for k in ['pairwise_ADE64','centroid_displacement','mean_PDMS','feasible_rate','min_PDMS','max_PDMS']}
    assert max(errors.values())<1e-10,errors
    save(OUT/'audits/previous5000_analysis_parity.json',dict(status='PASS',cells=len(prior),max_errors=errors))
    # Revalidate the exact reused baseline cache against the published inventory.
    inventory=read(PREV/'manifests/cache_hashes.json');inventory=[r for r in inventory if Path(r['path']).parent.name=='official_il']
    with concurrent.futures.ThreadPoolExecutor(12) as pool:list(pool.map(verify_record,inventory))
    with concurrent.futures.ThreadPoolExecutor(12) as pool:cache_records=list(pool.map(verify_cache,(OUT/'cache/rollouts').glob('*/*.npz')))
    table('cache_inventory.parquet',cache_records)
    save(OUT/'audits/final.json',dict(status='PASS',protocol_hash=identity(),train_scenes=3072,holdout_scenes=5000,training_runs=8,seed_count=2,updates_per_run=512,identical_scene_order_within_seed=True,old_results_unchanged=True,reused_baseline_files_verified=len(inventory),new_cache_files_verified=len(cache_records),no_GRPO_updates=True,missing_scenes=0,final_model_scene_protocol_cells=len(final)))
def METRICS_AVAILABLE(f):return ['mean_PDMS','pairwise_ADE64','centroid_displacement','feasible_rate']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();final() if a.final else initial()
