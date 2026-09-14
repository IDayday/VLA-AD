"""Actual support records, unchanged native weight helpers, and historical logs.

Weight reconstruction is at the checkpoint epoch, before residual-budget scaling.
Historical logged post-budget weights are exported separately, never invented.
"""
from __future__ import annotations
import ast, dataclasses, itertools, math, types
import torch
from common_fd import *

def native_helpers():
    name='_fd_native_teacher_helpers'
    if name in sys.modules:return sys.modules[name]
    tree=ast.parse(NATIVE.read_text())
    functions={'_dpsi_source_matches','_compute_dpsi_mode_balance','_is_dpsi_frontier_target',
        '_compute_dpsi_support_profile','build_frontier_curriculum_mask','_build_asmi_weights',
        '_sample_asmi_targets','_apply_dpsi_residual_budget','_compute_dpsi_target_hardness_diagnostics'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and
           (n.name in functions or n.name=='OfflineRLConfig')]
    method_names={'_source_bucket','_source_code','_dpsi_weight_for_tag','_compute_awac_candidate_valid_mask'}
    methods=[m for n in tree.body if isinstance(n,ast.ClassDef) for m in n.body if isinstance(m,ast.FunctionDef) and m.name in method_names]
    cls=ast.ClassDef(name='SourceHelpers',bases=[],keywords=[],body=methods,decorator_list=[])
    pre=ast.parse('from __future__ import annotations\nfrom dataclasses import dataclass, field\nfrom typing import *\nimport torch, math, numpy as np\n_DPSI_FRONTIER_TARGET_DISTRIBUTIONS=frozenset({"frontier_v4","learning_frontier_v5","full_training_v6"})\n').body
    module=types.ModuleType(name);sys.modules[name]=module
    exec(compile(ast.fix_missing_locations(ast.Module(body=pre+nodes+[cls],type_ignores=[])),str(NATIVE),'exec'),module.__dict__)
    return module

def native_config(model):
    n=native_helpers();a=read(MODEL_INFO[model]['config_path'])['agent']
    fields={x.name for x in dataclasses.fields(n.OfflineRLConfig)}
    cfg=n.OfflineRLConfig(**{k[len('offline_rl_'):]:v for k,v in a.items() if k.startswith('offline_rl_') and k[len('offline_rl_'):] in fields})
    assert cfg.use_dpsi and cfg.dpsi_filter_to_support_indices
    return cfg

def exact_legacy_sampling_expectation(weights, sources, valid, rewards, cfg):
    """Exact expectation for the native forced-anchor/best + weighted WOR sampler.

    Returns expected normalized loss mass, not the probability of inclusion.
    Enumerates ordered remaining selections, including their subsequent native
    selected-weight renormalization. All scientific parameters come from cfg.
    """
    w=np.asarray(weights,dtype=float);w=w/w.sum();available=np.flatnonzero(w>0).tolist()
    k=min(cfg.dpsi_target_sample_m_after_warmup,len(w))
    if len(available)<=k:return w
    chosen=[]
    if cfg.dpsi_force_anchor_target:
        eligible=[i for i in available if sources[i] in (1,2)] or available
        chosen.append(max(eligible,key=lambda i:rewards[i]))
    if cfg.dpsi_force_best_target:
        eligible=[i for i in available if valid[i]] or available
        best=max(eligible,key=lambda i:rewards[i])
        if best not in chosen:chosen.append(best)
    slots=k-len(chosen);remaining=[i for i in available if i not in chosen]
    result=np.zeros(len(w));mass=0.
    for order in itertools.permutations(remaining,min(slots,len(remaining))):
        probability=1.;denominator=w[remaining].sum()
        for i in order:
            probability*=w[i]/denominator;denominator-=w[i]
        idx=chosen+list(order)
        result[idx]+=probability*w[idx]/w[idx].sum();mass+=probability
    assert abs(mass-1)<1e-7 and abs(result.sum()-1)<1e-7
    return result

def weights_for(model, record, idx, trajs):
    n=native_helpers();cfg=native_config(model);h=n.SourceHelpers()
    sources=[record['sources'][i] for i in idx];tags=[record.get('support_tags',['']*len(record['sources']))[i] for i in idx]
    code=np.array([h._source_code(s) for s in sources]);r=np.asarray(record['rewards'])[idx]
    real=torch.ones((1,len(idx)),dtype=torch.bool)
    valid=torch.as_tensor(np.asarray(record['valid_mask'])[idx][None],dtype=torch.bool)
    if cfg.recompute_buffer_valid_mask_on_load:
        components={k:torch.tensor(np.asarray(v)[idx][None],dtype=torch.float32) for k,v in record['components'].items()}
        valid=h._compute_awac_candidate_valid_mask(components,sources,cfg)[0]
    ts=torch.tensor(trajs[None],dtype=torch.float32);rs=torch.tensor(r[None],dtype=torch.float32);codes=torch.tensor(code[None])
    gt=torch.tensor([record['gt_reward']],dtype=torch.float32);il=torch.tensor([record['il_reward']],dtype=torch.float32)
    profile,diag=n._compute_dpsi_support_profile(ts,rs,real,codes,gt,il,valid,cfg,EPOCHS[model])
    sw=torch.tensor([[h._dpsi_weight_for_tag(t,s,cfg) for t,s in zip(tags,sources)]],dtype=torch.float32)
    extra={}
    if cfg.dpsi_target_distribution=='full_training_v6':
        for dest,key,dtype in [('selected_mode_id','mode_ids',torch.long),('selected_teacher_eligible','teacher_eligible_mask',torch.bool),('selected_frontier_difficulty','learning_frontier_difficulty',torch.float32),('selected_teacher_confidence','teacher_confidence',torch.float32)]:
            extra[dest]=torch.tensor(np.asarray(record[key])[idx][None],dtype=dtype)
    w,wd=n._build_asmi_weights(rs,real,valid,codes,sw,gt,il,profile,cfg,current_epoch=EPOCHS[model],**extra)
    nominal=w[0].numpy();nominal=nominal/nominal.sum()
    expected=nominal.copy() if cfg.dpsi_target_distribution=='full_training_v6' else exact_legacy_sampling_expectation(nominal,code,valid[0].numpy(),r,cfg)
    return dict(nominal=nominal,expected=expected,codes=code,valid=valid[0].numpy(),sources=sources,tags=tags,
                beta=float(profile['beta_profile'][0]),scene_weight=float(profile['scene_weight'][0]),native_args=(ts,w,real,valid,codes,rs,cfg))

def export_logs():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows=[];inventory=[]
    for m in ARCHIVES:
        folder=Path(MODEL_INFO[m]['checkpoint_path']).parent/'lightning_logs/version_0'
        a=EventAccumulator(str(folder),size_guidance={'scalars':0});a.Reload()
        files=sorted(folder.glob('events.out.tfevents.*'))
        inventory.append(dict(model=m,files=[dict(path=str(p),sha256=sha(p)) for p in files],selected_epoch=EPOCHS[m]))
        for tag in a.Tags()['scalars']:
            if not tag.endswith('_epoch'):continue
            events=a.Scalars(tag)
            assert len(events)==200,(m,tag,len(events))
            for epoch,e in enumerate(events):
                if epoch<=EPOCHS[m]:rows.append(dict(model=m,epoch=epoch,optimizer_step=e.step,metric=tag,value=e.value,wall_time=e.wall_time))
    csv('historical_training_curves.csv',rows)
    last=pd.DataFrame(rows);last=last[last.apply(lambda r:r['epoch']==EPOCHS[r['model']],axis=1)]
    csv('historical_training_at_checkpoint.csv',last)
    save(OUT/'manifests/training_logs.json',inventory)

def main():
    config_identity();torch.set_num_threads(1)
    rows=[];scenes=[];hashes=[];raw_sources=[]
    for si,scene in enumerate(SCENES):
        token=scene['token']
        for m in ARCHIVES:
            path,r,idx,t=teacher_record(m,token);w=weights_for(m,r,idx,t)
            hashes.append(dict(model=m,token=token,path=str(path),sha256=sha(path),selected_indices=idx.tolist()))
            d=distance(t,t);expected=w['expected'];gt=w['codes']==1
            unique=len(np.unique(np.round(t,6).reshape(len(t),-1),axis=0))
            scene_row=dict(token=token,log=scene['log'],model=m,raw_count=len(r['candidates']),support_count=len(t),
                unique_support_count=unique,positive_weight_count=int((expected>0).sum()),non_gt_count=int((~gt & (expected>0)).sum()),
                nominal_GT_mass=w['nominal'][gt].sum(),expected_GT_mass_pre_budget=expected[gt].sum(),
                beta=w['beta'],scene_weight=w['scene_weight'],target_ESS=1/np.sum(expected**2),
                weighted_pair_ADE=float(expected@d@expected),unweighted_pair_ADE=pair_mean(t) if len(t)>1 else 0.,
                gt_archive_ADE=float(distance(t[gt],np.asarray(scene['gt'])[None]).max()) if gt.any() else float('nan'))
            scenes.append(scene_row)
            for rawsource,count in zip(*np.unique(np.asarray(r['sources'],dtype=str),return_counts=True)):
                raw_sources.append(dict(model=m,token=token,source=rawsource,raw_count=int(count)))
            for j,i in enumerate(idx):
                rows.append(dict(model=m,token=token,raw_index=int(i),source=w['sources'][j],source_bucket=native_helpers().SourceHelpers._source_bucket(w['sources'][j]),
                    support_tag=w['tags'][j],source_code=int(w['codes'][j]),archive_valid=bool(w['valid'][j]),
                    historical_reward=float(r['rewards'][i]),nominal_weight=float(w['nominal'][j]),expected_weight_pre_budget=float(expected[j]),
                    trajectory_hash=digest(t[j].tolist()),archive_path=str(path),archive_sha256=hashes[-1]['sha256']))
        if si%100==0:print('teacher audit',si+1,flush=True)
    csv('teacher_candidates.csv',rows);csv('teacher_scene.csv',scenes);csv('teacher_raw_sources.csv',raw_sources)
    save(OUT/'manifests/teacher_records.json',hashes)
    export_logs()
    save(OUT/'audits/teacher_weight_scope.json',dict(native_file=str(NATIVE),sha256=sha(NATIVE),
        helper_execution='AST-extracted unchanged native dataclass/functions/source methods; no planner or optimizer constructed',
        historical_code_limitation='Training run retained Hydra config and TensorBoard; byte-identical historical Python snapshot not found. Weight reconstruction uses audited available implementation, cross-checked with actual logged diagnostics.',
        residual_budget='Per-target errors during historical optimization are not recoverable. Per-scene expected weights are PRE-budget; actual post-budget evidence comes only from historical TensorBoard.',
        scenes=len(SCENES),records=len(hashes),selected_rows=len(rows)))

if __name__=='__main__':main()
