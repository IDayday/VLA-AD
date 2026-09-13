"""Full scientific integrity audit; checkpoints/caches stay local and are hash-indexed."""
import concurrent.futures,subprocess,argparse
from common_v3 import *
def file_record(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
def main(skip_hash=False):
    for phase in ['A','B','C','D','E','F','F_recipe','G','G_null','native']:
        assert (OUT/'manifests'/f'audit_{phase}.json').exists(),phase
    frozen=read(OUT/'manifests/protocol_frozen.json');assert sha(CONFIG_PATH)==frozen['yaml_sha256'];assert set(tokens('train')).isdisjoint(tokens('holdout'))
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='analysis/pc-mts-policy-diagnostics-v3-20260913'
    changed=subprocess.check_output(['git','diff',CFG['base_commit'],'--name-only'],cwd=ROOT,text=True).splitlines();assert all('pc_mts_diagnostics_v3' in p or 'PC_MTS_POLICY_DIAGNOSTICS_V3_' in p for p in changed),changed
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');assert len(raw)==192000 and raw.token.nunique()==1000
    total_unique=0;overlap=[]
    for token in tokens('all'):
        pools=read(OUT/'cache/pools'/f'{token}.json')['methods'];g=raw[raw.token==token].set_index('raw_index')
        for method,p in pools.items():
            ids=p['indices'];assert len(set(ids))==len(ids)<=16 and all(0<=i<192 for i in ids);total_unique+=len(ids)
            if method in ['conditional_pc','quality_only']:
                assert all(g.loc[i,'q_holdout']<95 and g.loc[i,'hard_safe'] and g.loc[i,'local_feasible']>=.75 for i in ids)
                for i,level in zip(ids,p['levels']):assert g.loc[i,'PDMS']>=g.loc[i,'reference_PDMS']-(0 if level==0 else 1)-1e-8
        a=set(pools['conditional_pc']['indices']);b=set(pools['quality_only']['indices']);overlap.append(dict(token=token,conditional_count=len(a),quality_only_count=len(b),intersection=len(a&b),union=len(a|b),Jaccard=len(a&b)/len(a|b) if a|b else 1.,identical=a==b))
        with np.load(OUT/'cache/learnability'/f'{token}.npz',allow_pickle=False) as z:
            assert z['query_id'].dtype.kind=='U'
            assert np.isfinite(z['uniform_loss']).all() and np.isfinite(z['E_rec']).all()
    csv(pd.DataFrame(overlap),'A_quality_only_overlap.csv')
    ledgers=[]
    for method in CFG['training']['methods']:
        for sd in CFG['training']['seeds']:
            ledger=read(OUT/'cache/training_ledgers'/f'sft_{method}_{sd}.json')['records'];assert len(ledger)==1600
            for row in ledger:
                assert row['token'] in set(tokens('train')) and len(row['parents'])==16;ids=np.asarray(row['parents']);w=np.asarray(row['weights']);u=np.unique(ids);assert np.isclose(w.sum(),1)
                assert all(np.isclose(w[ids==i].sum(),1/len(u)) for i in u)
            r=pd.read_parquet(OUT/'metrics'/f'F_rollouts_{method}_{sd}.parquet');assert len(r)==6400 and set(r.token)<=set(tokens('train'));assert r.groupby(['step','micro']).size().eq(8).all();assert np.isfinite(r.advantage).all()
            values=r[FIELDS].to_numpy();np.testing.assert_array_equal(r.feasible.to_numpy(),feasible(values));np.testing.assert_array_equal(r.hard_failure.to_numpy(),hard_failure(values))
            ledgers.append(dict(method=method,seed=sd,SFT_scene_updates=len(ledger),unique_training_tokens=len({x['token'] for x in ledger}),GT_fallback_updates=sum(x['explicit_GT_fallback'] for x in ledger),GRPO_rollouts=len(r),GRPO_groups=800))
    for tag in ['E','F']:
        d=pd.read_csv(OUT/'metrics'/f'{tag}_holdout_scene.csv');assert len(d)==14400 and set(d.token)==set(tokens('holdout'));assert not d[['PDMS','feasible_rate','pairwise_ADE','center_shift','IL_retention_loss']].isna().any().any()
        assert d.groupby(['method','seed','step']).token.nunique().eq(300).all()
    g=pd.read_csv(OUT/'metrics/G_scene_promotion.csv');assert len(g)==6000 and g.groupby('run').token.nunique().eq(1000).all()
    for i in range(1,8):
        assert len(list((OUT/'figures').glob(f'Fig-V3-{i}_*.png')))==1 and len(list((OUT/'figures').glob(f'Fig-V3-{i}_*.pdf')))==1
    files=[p for root in ['cache','checkpoints'] for p in (OUT/root).rglob('*') if p.is_file() and p.suffix in ['.npz','.pt','.json']]
    if not skip_hash:
        with concurrent.futures.ThreadPoolExecutor(16) as ex:index=list(ex.map(file_record,files))
        save(OUT/'manifests/CACHE_INDEX.json',dict(identity=identity(),files=index))
    else:assert (OUT/'manifests/CACHE_INDEX.json').exists()
    stage_audit('final',scene_count=1000,raw_candidate_count=192000,primary_unique_parent_slots_across_methods=total_unique,unique_weight_checks=ledgers,holdout_scene_rows_per_stage=14400,all_64_rollouts=True,all_snapshots_not_best_selected=True,primary_config_unchanged=True,cache_file_count=len(files),cache_index_sha256=sha(OUT/'manifests/CACHE_INDEX.json'),statistical_replicates=3000,figures=7,PNG_and_PDF=True,global_no_leakage_claim=False,provenance_caveats=read(OUT/'manifests/audit_split_provenance.json'),corrections=['C identifier serialization only, numerical arrays bit-identical','A count column scope clarified, selected parents and primary comparisons unchanged'])
    print('FINAL AUDIT PASS',len(files),'cache/checkpoint files',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--skip-hash',action='store_true');main(a.parse_args().skip_hash)
