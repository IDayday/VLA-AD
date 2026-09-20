"""Re-score shared archived raw trajectories; apply frozen historical PSI kernel."""
from common_matched import *
import concurrent.futures,lzma,pickle,ast,types,importlib.util,torch
import evaluate_cached_rollouts as evaluator

def psi_kernel():
    path=MAIN/'outputs/historical_sft_grpo_support/runtime/psi/navsim/agents/recogdrive/pareto_support.py'
    spec=importlib.util.spec_from_file_location('matched_psi_original',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    source=subprocess.check_output(['git','show','870d716:scripts/build_stage2_pareto_support_index.py'],cwd=MAIN,text=True)
    names={'_first_source_index','_pdms_core','_pareto_front_mask','_core_pareto_scores'}
    tree=ast.parse(source);selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    pre=ast.parse('from __future__ import annotations\n').body
    env=dict(torch=torch,REQUIRED_COMPONENT_KEYS=['pdms','ego_progress','time_to_collision_within_bound','history_comfort','no_at_fault_collisions','drivable_area_compliance','driving_direction_compliance'])
    exec(compile(ast.Module(body=pre+selected,type_ignores=[]),'historical_build_stage2_pareto_support_index.py','exec'),env)
    return module,env

def raw_work(s):
    t=s['token'];dest=OUT/'cache/raw'/f'{t}.npz';meta=dict(protocol_hash=identity(),token=t,archive_sha256=sha(archive(t)))
    if legacy.v1.valid_npz(dest,meta):return dict(token=t,status='CACHED')
    a=pickle.load(lzma.open(archive(t),'rb'));assert a['token']==t
    trajectories=[];tags=[];raw_ids=[];seen={}
    for i,(tr,src) in enumerate(zip(a['candidates'],a['sources'])):
        tr=np.asarray(tr,dtype=np.float32);assert tr.shape==(8,3) and np.isfinite(tr).all()
        key=np.round(tr,6).tobytes()
        if key in seen:j=seen[key];tags[j].append(src);raw_ids[j].append(i)
        else:seen[key]=len(trajectories);trajectories.append(tr);tags.append([src]);raw_ids.append([i])
    tr=np.stack(trajectories)
    with lzma.open(s['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    scores=evaluator.score_arrays(cache,tr,batch=128);assert np.isfinite(scores).all()
    assert np.max(abs(tr[0]-np.asarray(s['gt'])))<1e-3
    npz(dest,dict(meta,archive_path=str(archive(t)),all_source_tags=tags,raw_indices=raw_ids,original_count=len(a['candidates']),build_metadata=a['build_metadata']),trajectories=tr,scores=scores)
    return dict(token=t,status='OK',count=len(tr))

def front_ranks(values,mask):
    ranks=np.full(len(values),-1,dtype=int);remaining=np.flatnonzero(mask).tolist();rank=0
    while remaining:
        v=values[remaining];dominates=(v[:,None,:]>=v[None,:,:]).all(-1)&(v[:,None,:]>v[None,:,:]).any(-1)
        front=[remaining[i] for i in np.flatnonzero(~dominates.any(0))]
        for i in front:ranks[i]=rank
        remaining=[i for i in remaining if i not in front];rank+=1
    return ranks

def select_arrays(tr,scores,tags,stats,kernel):
    mod,env=kernel
    sources=[sorted(set(x))[0] if len(set(x))==1 else ('gt' if 'gt' in x else 'multi_source') for x in tags]
    fieldmap=dict(pdms=6,no_at_fault_collisions=0,drivable_area_compliance=1,ego_progress=2,time_to_collision_within_bound=3,history_comfort=4,driving_direction_compliance=5)
    components={k:torch.tensor(scores[:,v],dtype=torch.float32) for k,v in fieldmap.items()}
    filtered=dict(components=components,sources=sources,valid_mask=torch.tensor(np.isfinite(tr).all((1,2))))
    score,valid,epok,core=env['_core_pareto_scores'](filtered);eligible=(valid&epok).numpy()
    gt=sources.index('gt');geo=np.linalg.norm(tr[...,:2]-tr[gt,...,:2],axis=-1).mean(-1)
    selection=mod.select_adaptive_pareto_supports(torch.tensor(tr),score,torch.tensor(eligible),descriptor_mean=torch.tensor(stats['mean']),descriptor_std=torch.tensor(stats['std']),pdms=components['pdms'],core=core,nc_dac=components['no_at_fault_collisions']*components['drivable_area_compliance'],geometry_distance=torch.tensor(geo),sources=sources,gt_candidate_index=gt,quality_band=CFG['psi_quality_band'],min_descriptor_distance=CFG['psi_min_descriptor_distance'])
    rank=front_ranks(scores[:,[2,3,4]],eligible);ids=np.flatnonzero(eligible).tolist()
    scoreids=sorted(ids,key=lambda i:(-scores[i,6],i))[:3] or [gt]
    paretoids=sorted(ids,key=lambda i:(rank[i],-scores[i,6],i))[:3] or [gt]
    result={}
    for name,sel in [('il_sft',[gt]),('score',scoreids),('pareto',paretoids),('psi',selection.indices[selection.mask].tolist())]:
        weights=mod._support_weights(sel,gt,best_weight=.5,gt_weight=.2,other_weight=.3,dtype=torch.float32)[:len(sel)].numpy()
        result[name]=dict(indices=sel,weights=weights.tolist(),fallback=bool(not eligible.any()) if name!='il_sft' else False,sources=[sources[i] for i in sel])
    return result,eligible,rank

def main():
    identity();rows=scenes();start=time.time()
    with concurrent.futures.ProcessPoolExecutor(64,initializer=evaluator.init_worker) as pool:
        for i,result in enumerate(pool.map(raw_work,rows,chunksize=1)):
            if i%100==0:print('RAW_SCORE',i,len(rows),time.time()-start,flush=True)
    kernel=psi_kernel();mod=kernel[0];ds=[]
    for s in scenes('train'):
        with np.load(OUT/'cache/raw'/f"{s['token']}.npz") as f:ds.append(mod.trajectory_descriptor(torch.tensor(f['trajectories'])).double())
    ds=torch.cat(ds);stats=dict(mean=ds.mean(0).float().tolist(),std=ds.std(0,unbiased=False).clamp(min=1e-4).float().tolist(),count=len(ds),calibration='train_only')
    save(OUT/'manifests/descriptor_stats.json',stats);allrows=[];hashes={}
    for s in rows:
        t=s['token'];path=OUT/'cache/raw'/f'{t}.npz'
        with np.load(path) as f:tr=f['trajectories'];sc=f['scores'];meta=json.loads(str(f['metadata']))
        methods,elig,rank=select_arrays(tr,sc,meta['all_source_tags'],stats,kernel)
        save(OUT/'cache/selection'/f'{t}.json',dict(token=t,protocol_hash=identity(),methods=methods,raw_sha256=sha(path),eligible_count=int(elig.sum())))
        hashes[t]=sha(OUT/'cache/selection'/f'{t}.json')
        for method,v in methods.items():
            ids=v['indices'];w=np.asarray(v['weights']);sel=tr[ids];d=legacy.distance(sel,sel)
            allrows.append(dict(token=t,split=s['split'],method=method,count=len(ids),weighted_PDMS=float(sc[ids,6]@w*100),weighted_feasible=float(safe(sc[ids])@w),GT_weight=sum(w[j] for j,i in enumerate(ids) if i==0),target_pairwise_ADE=float(d[np.triu_indices(len(ids),1)].mean()) if len(ids)>1 else 0,source='|'.join(v['sources']),fallback=v['fallback'],score_pareto_identical=methods['score']['indices']==methods['pareto']['indices']))
    table('supervision_composition.csv',allrows)
    save(OUT/'manifests/selection_frozen.json',dict(protocol_hash=identity(),selection_hashes=hashes,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_optimizer_updates=True,raw_count=len(rows),descriptor_stats_sha256=sha(OUT/'manifests/descriptor_stats.json')))
    print('SELECTION FROZEN',len(rows),flush=True)
if __name__=='__main__':main()
