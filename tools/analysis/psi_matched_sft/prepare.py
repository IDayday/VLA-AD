from common_matched import *
import concurrent.futures,pickle
from collections import defaultdict,Counter

def main():
    if (OUT/'manifests/protocol.json').exists():identity();return
    sys.path.insert(0,str(MAIN/'tools/analysis/pc_mts_diagnostics'))
    from build_scene_manifest import inspect_log,details,LOG_ROOT
    ev=read(PREV/'manifests/scenes.json')['scenes'];evlogs={s['log'] for s in ev}
    oldargs=read(MAIN/'outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/train_args.json')
    allowedlogs=set(oldargs['train_logs'])-evlogs
    filt=yaml.safe_load((legacy.v1.CODE/'navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml').read_text());allowed=set(filt['tokens'])
    paths=[LOG_ROOT/(log+'.pkl') for log in sorted(allowedlogs & set(filt['log_names']))]
    with concurrent.futures.ProcessPoolExecutor(24) as pool:
        population=[s for block in pool.map(inspect_log,paths,chunksize=2) for s in block if s['token'] in allowed]
    selected=sorted(population,key=lambda s:digest([CFG['split_seed'],s['token']]))[:CFG['train_scene_count']]
    bylog=defaultdict(list)
    for s in selected:bylog[s['log']].append(s)
    tr=[]
    for log,rows in bylog.items():
        frames=pickle.load(open(LOG_ROOT/(log+'.pkl'),'rb'));tr.extend(details(s,frames) for s in rows)
    tr.sort(key=lambda s:digest([CFG['split_seed'],s['token']]))
    assert len(tr)==CFG['train_scene_count'] and len(ev)==5000
    assert not {s['log'] for s in tr}&evlogs
    rows=[dict(s,split='train') for s in tr]+[dict(s,split='holdout') for s in ev]
    assert all(archive(s['token']).exists() and Path(s['metric_cache_path']).exists() for s in rows)
    save(OUT/'manifests/scenes.json',dict(scenes=rows,train_count=len(tr),holdout_count=5000,population_count=len(population),log_overlap=0,outcome_conditioned=False,commands={k:dict(Counter(s['command'] for s in rows if s['split']==k)) for k in ['train','holdout']}))
    m=read(OLD/'manifests/models.json')['official_il'];save(OUT/'manifests/models.json',{'official_il':m})
    assert sha(m['checkpoint_path'])==m['sha256']
    save(OUT/'manifests/protocol.json',dict(config_sha256=sha(CONFIG),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORK,text=True).strip(),scene_sha256=sha(OUT/'manifests/scenes.json'),initial_checkpoint=m['sha256'],historical_training_args_sha256=sha(MAIN/'outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/train_args.json'),previous5000_protocol_sha256=sha(PREV/'manifests/protocol.json'),raw_bank=str(RAW),note='Rebuilt PSI selection using shared available raw candidates. Original AWAC raw bank missing; historical selected PSI tensors remain untouched.'))
    print('FROZEN',identity(),len(tr),len(ev),flush=True)
if __name__=='__main__':main()
