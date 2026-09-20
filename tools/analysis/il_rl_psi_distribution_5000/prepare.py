"""Freeze outcome-independent nested cohort before any new model inference."""
from common_5000 import *
import concurrent.futures, pickle, subprocess, datetime
from collections import Counter, defaultdict
from build_scene_manifest import inspect_log, details, LOG_ROOT

def main():
    if (OUT/'manifests/protocol.json').exists():
        identity();print('Frozen protocol exists');return
    old=read(OLDOUT/'manifests/scenes.json')['scenes'];oldids={r['token'] for r in old}
    filt=yaml.safe_load((v1.CODE/'navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text())
    paths=[LOG_ROOT/(log+'.pkl') for log in filt['log_names']]
    allowed=set(filt['tokens'])
    with concurrent.futures.ProcessPoolExecutor(24) as pool:
        population=[r for batch in pool.map(inspect_log,paths,chunksize=2) for r in batch if r['token'] in allowed]
    assert len(population)==len({r['token'] for r in population})==len(filt['tokens'])
    assert oldids<={r['token'] for r in population}
    counts=Counter(r['command'] for r in population);oldcounts=Counter(r['command'] for r in old)
    quota={c:int(5000*n/len(population)) for c,n in counts.items()}
    for c in sorted(counts,key=lambda c:-(5000*counts[c]/len(population)-quota[c]))[:5000-sum(quota.values())]:quota[c]+=1
    extra=[]
    for c,n in quota.items():
        extra+=sorted((r for r in population if r['command']==c and r['token'] not in oldids),key=lambda r:digest([CFG['scene_selection_seed'],r['token']]))[:n-oldcounts[c]]
    bylog=defaultdict(list)
    for r in extra:bylog[r['log']].append(r)
    expanded=[]
    for log,rows in bylog.items():
        frames=pickle.load(open(LOG_ROOT/(log+'.pkl'),'rb'))
        expanded.extend(details(r,frames) for r in rows)
    expanded.sort(key=lambda r:digest([CFG['scene_selection_seed'],r['token']]))
    final=[dict(r,cohort='OLD1000') for r in old]+[dict(r,cohort='NEW4000') for r in expanded]
    assert len(final)==len({r['token'] for r in final})==5000
    missing=[r['token'] for r in final if not Path(r['metric_cache_path']).is_file()]
    manifest=dict(scene_count=5000,population=len(population),population_commands=counts,selected_commands=quota,scenes=final,factorial_tokens=[],missing_metric_cache_tokens=missing,selection_seed=CFG['scene_selection_seed'],outcome_conditioned=False)
    save(OUT/'manifests/scenes.json',manifest)
    allmodels=read(OLDOUT/'manifests/models.json');selected={k:allmodels[k] for k in CFG['primary_models']}
    for k,m in selected.items():
        assert sha(m['checkpoint_path'])==m['sha256'],k
        assert sha(m['runtime_planner_path'])==m['runtime_planner_sha256'],k
    save(OUT/'manifests/models.json',selected)
    codepaths=[MAIN/'tools/analysis/historical_sft_grpo_support'/f for f in ['common_support.py','sample.py','batched.py','score.py','analyze.py']]
    codepaths += [MAIN/'tools/analysis/pc_mts_diagnostics'/f for f in ['common.py','build_scene_manifest.py','distributed_rollout.py','evaluate_cached_rollouts.py']]
    frozen=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),config_sha256=sha(CFG_PATH),scene_manifest_sha256=sha(OUT/'manifests/scenes.json'),models_sha256=sha(OUT/'manifests/models.json'),reused_code={str(p):sha(p) for p in codepaths},old_protocol=read(OLDOUT/'manifests/protocol.json'),old_cache_inventory_sha256=sha(OLDOUT/'manifests/cache_hashes.json'))
    save(OUT/'manifests/protocol.json',frozen)
    print(json.dumps(dict(scene_count=5000,new=4000,commands=quota,missing=len(missing),config_sha256=frozen['config_sha256'])),flush=True)

if __name__=='__main__':main()
