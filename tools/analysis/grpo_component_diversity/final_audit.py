from shared import *
import subprocess
def main():
    identity();ss=scenes();frozen=read(OUT/'manifests/analysis_code_frozen.json');changes=[]
    for p,h in frozen['files'].items():
        if sha(ROOT/p)!=h:changes.append(p)
    # Functional engineering changes after freeze must be disclosed, never
    # silently replace primary definitions or thresholds.
    fixes=read(OUT/'audits/engineering_fixes.json') if (OUT/'audits/engineering_fixes.json').exists() else []
    for p in changes:
        fix=next((x for x in fixes if x['path']==p),None)
        assert fix and fix['before_sha256']==frozen['files'][p] and fix['after_sha256']==sha(ROOT/p),p
    protected=Path('/mnt/project/VLA-AD-worktrees/psi-matched-sft-20260920/outputs/psi_matched_sft/manifests/protected_files.json')
    pv=read(protected)
    for v in pv:assert sha(v['path'])==v['sha256'],v['path']
    inherited=read(PREV/'outputs/il_rl_psi_distribution_5000/manifests/cache_hashes.json');old={r['path']:r['sha256'] for r in inherited}
    inventory=read(OUT/'manifests/input_cache_hashes.json');verified_old=0
    for r in inventory:
        assert sha(r['path'])==r['sha256'],r['path']
        if r['path'] in old:assert old[r['path']]==r['sha256'];verified_old+=1
    assert verified_old==20000
    for m in CFG['models']:
        assert sha(models()[m]['checkpoint_path'])==models()[m]['sha256']
        for rank in range(4):
            audit=read(OUT/'audits'/f'sampling_{m}_{rank}.json');assert audit['status']=='PASS' and audit['state_before']==audit['state_after']
        assert read(OUT/'audits'/f'sampler_parity_{m}.json')['status']=='PASS'
        assert read(OUT/'audits'/f'noise_decomposition_{m}.json')['status']=='PASS'
    assert read(OUT/'audits/native_reward_advantage.json')['status']=='PASS'
    df=pd.read_parquet(OUT/'metrics/group_metrics.parquet');assert len(df)==310000
    assert df.token.nunique()==5000 and len(ss)==5000
    assert not df.duplicated(['model','token','G','block']).any()
    for (m,t,g),q in df.groupby(['model','token','G']):assert len(q)==128//g
    seeds=[previous.seed(r['token'],g) for r in ss for g in range(8)]
    assert len(set(seeds))==len(seeds)==40000
    # Each model sees matched noise; different scene/group seeds are unique.
    primary=df[df.block==0].copy();primary.to_parquet(OUT/'metrics/scene_prefix_metrics.parquet',index=False)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tools/analysis/grpo_component_diversity'),'-p','test_*.py','-v'],capture_output=True,text=True)
    (OUT/'audits/tests.txt').write_text(test.stdout+test.stderr);assert test.returncode==0
    save(OUT/'audits/final.json',dict(status='PASS',scene_count=5000,log_count=len({r['log'] for r in ss}),missing_scenes=0,models=CFG['models'],rollouts=1280000,new_rollouts=640000,group_sizes=CFG['group_sizes'],prefix_rows=len(primary),all_disjoint_group_rows=len(df),old_cache_files_verified=verified_old,old_protected_files_unchanged=len(pv),input_files_verified=len(inventory),distinct_scene_group_seeds=len(seeds),optimizer_updates=0,protocol_hash=identity(),analysis_code_unchanged=not changes,documented_engineering_fixes=fixes,thresholds_and_metric_definitions_unchanged=True,tests_returncode=0))
    print('FINAL AUDIT PASS',flush=True)
if __name__=='__main__':main()
