"""One-scene causal isolation: replace only patch convolution with real PPU C+bias.

This is a diagnostic intervention, never a teacher generation method.
"""
from pathlib import Path
import sys

ROOT=Path('/mnt/project/recogdrive-teacher2-a800-20261026')
AUDIT=ROOT/'diagnostics/ppu_audit_20261027_v2'
PACK= AUDIT/'unpacked/ppu-a800-embedding-audit-20261027-v2'
TOKEN='0002267a294e52a7'


def main():
    sys.path.insert(0,str(ROOT/'diagnostics/torch240_packages'))
    import torch
    import numpy as np
    import hashlib
    import importlib.util
    from cache_common import REPO,read,write,sha,stamp
    sys.path.insert(0,str(REPO))
    out=AUDIT/'C_INJECTION'
    if (out/'RESULT.json').exists():raise FileExistsError(out/'RESULT.json')
    from interface_teacher.runtime import bootstrap,Runtime
    bootstrap(ROOT/'paths.local.json',out)
    from transformers import AutoModel
    original_loader=AutoModel.from_pretrained
    def load(*a,**kw):
        kw['use_flash_attn']=False
        return original_loader(*a,**kw)
    AutoModel.from_pretrained=load
    from vlm_core import prepare
    from interface_teacher import Library,DecisionTable
    from paired_noise import PairedReplayNoise
    from noise_replay import replay_bank
    from repair_score import score_group
    spec=importlib.util.spec_from_file_location('ppu_input_payload',PACK/'load_inputs.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    ppu_inputs,aux,ppu_tensors,metadata=loader.load_inputs()
    engine=Runtime('raw');model=engine.backbone.model;language=model.language_model.model
    assert language._attn_implementation==language.config._attn_implementation=='eager'
    assert all(type(layer.self_attn).__name__=='Qwen2Attention' for layer in language.layers)
    assert all(not layer.attn.use_flash_attn for layer in model.vision_model.encoder.layers)
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
    def fp(value):
        value=value.detach().cpu().contiguous()
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
    expected=read(ROOT/'diagnostics/PPU_REPORTED_FINGERPRINTS.json')['cases'][TOKEN]
    features=torch.load(ROOT/'diagnostics/backend/inputs'/(TOKEN+'.pt'),map_location='cpu',weights_only=False)
    inputs,_=prepare(engine.backbone,features)
    comparisons={key:fp(inputs[key])['sha256']==fp(ppu_inputs[key])['sha256'] for key in loader.MODEL_KEYS}
    comparisons.update({key:fp(features[key])['sha256']==fp(aux[key])['sha256'] for key in loader.AUX_KEYS})
    valid=inputs['attention_mask'][0].bool();ids=inputs['input_ids'][0]
    image=valid&(ids==engine.backbone.img_context_token_id);text=valid&~image
    comparisons.update({key:fp(value)['sha256']==fp(ppu_tensors[key])['sha256'] for key,value in [('role_image',image),('role_validtext',text),('role_padding',~valid),('pixels_FP32',features['pixel_values'])]})
    assert all(comparisons.values()),comparisons
    with np.load(AUDIT/'ppu_native_no_bias.npz',allow_pickle=False) as z:
        info=__import__('json').loads(z['__metadata_utf8__'].tobytes());raw=z['native_no_bias'].copy()
    assert hashlib.sha256(raw.tobytes()).hexdigest()=='c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b'
    c=torch.from_numpy(raw).view(torch.bfloat16).reshape(info['tensors']['native_no_bias']['shape']).cuda()
    emb=model.vision_model.embeddings;conv=emb.patch_embedding
    ppu_patch=c+conv.bias[None,:,None,None]
    assert fp(ppu_patch)['sha256']=='0d0a5f0e48d4e03f6635140f9043a2fc79b9f8b800a4e95b541bd28e3e55cd6b'
    parameter_versions={k:p._version for k,p in model.named_parameters()}
    setup=dict(time=stamp(),scene=TOKEN,torch=str(torch.__version__),torch_git=torch.version.git_version,
        cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
        TF32_matmul=torch.backends.cuda.matmul.allow_tf32,TF32_cudnn=torch.backends.cudnn.allow_tf32,
        language_impl=language._attn_implementation,vision_flash_count=0,all_12_inputs_match=comparisons,
        PPU_native_C_file_sha256=sha(AUDIT/'ppu_native_no_bias.npz'),
        diagnostic_only=True,bulk_generation_resumed=False)
    write(out/'SETUP.json',setup)
    stages={};layer_fingerprints={};handles=[]
    def capture(name):
        def hook(module,args,output):
            value=output if torch.is_tensor(output) else output[0]
            stages[name]=value.detach().cpu().clone()
        return hook
    for name,module in [('vision_embedding',emb),('vision_last',model.vision_model.encoder.layers[-1]),('bridge',model.mlp1),('language_first',language.layers[0])]:
        handles.append(module.register_forward_hook(capture(name)))
    def record(name):
        def hook(module,args,output):
            value=output if torch.is_tensor(output) else output[0]
            layer_fingerprints[name]=fp(value)
        return hook
    for i,layer in enumerate(model.vision_model.encoder.layers):handles.append(layer.register_forward_hook(record('vision_'+str(i))))
    for i,layer in enumerate(language.layers):handles.append(layer.register_forward_hook(record('language_'+str(i))))
    library=Library('D_C8_double');table=DecisionTable('D_C8_double','train')
    assert table.choice(TOKEN)==6
    noise=PairedReplayNoise(ROOT/'cache/train')
    old_qual=ROOT/'qualification/raw3_primary/cases'/TOKEN[:2]/TOKEN
    qualification=read(old_qual/'COMPLETE.json')
    with np.load(PACK/'original_ppu_golden_outputs.npz',allow_pickle=False) as z:
        golden={key:z[key].copy() for key in z.files}
    write(out/'GOLDEN_KEYS.json',{key:dict(shape=list(value.shape),dtype=str(value.dtype)) for key,value in golden.items()})
    reference_row=next(item for item in read(REPO/'reference/GOLDEN.json')['real_native_records'] if item['population']=='train' and item['library']=='D_C8_double' and item['token']==TOKEN)
    with np.load(REPO/reference_row['arrays']['path'],allow_pickle=False) as z:
        old_B=z['Base_trajectories_float32'].copy()
        old_T=z['trajectories_float32'][np.flatnonzero(z['positions_int64']==5)[0]].copy()
    results=[];arm_stages={}
    for mode in ['native','ppu_C']:
        stages.clear();layer_fingerprints.clear()
        intervention=None
        if mode=='ppu_C':
            intervention=conv.register_forward_hook(lambda module,args,output:ppu_patch)
        try:
            with torch.no_grad():hidden=model(**inputs).hidden_states[-1]
        finally:
            if intervention is not None:intervention.remove()
        stages['language_final_BF16']=hidden.detach().cpu().clone()
        with torch.no_grad(),torch.autocast('cuda',enabled=False):base=engine.agent.action_head.feature_encoder(hidden.float())
        stages['Base_interface']=base[0].detach().cpu().clone()
        fingerprints={key:fp(value) for key,value in stages.items()}
        matches={key:value['sha256']==expected[key] for key,value in fingerprints.items()}
        if mode=='ppu_C':assert matches['vision_embedding'],'Intervention must first reproduce the full original PPU embedding'
        row=dict(mode=mode,fingerprints=fingerprints,matches_PPU=matches,layer_fingerprints=dict(layer_fingerprints),
            first_divergent_recorded_stage=next((key for key in stages if not matches[key]),None))
        target=library.selected_target(base,image,text,TOKEN,table)
        trajectories={}
        for name,interface,reference in [('Base',base,old_B),('teacher',target,old_T)]:
            group=[]
            for j in range(5):
                with replay_bank(noise(TOKEN,j)):group.append(engine.trajectory(features,hidden.float(),interface,TOKEN,j))
            actual=np.asarray(group);trajectories[name]=actual
            delta=np.abs(actual-reference)
            row[name+'_trajectory']=dict(XY_max_abs_m=float(delta[:,:,:2].max()),heading_max_abs_rad=float(delta[:,:,2].max()),raw_equal=np.array_equal(actual,reference))
        scored=score_group(dict(metric_cache=qualification['original_source']['metric_cache'],metric_sha256=qualification['metric_cache_sha256'],scoring=engine.cfg['scoring'],
            records=[dict(arm=name,seed_index=j,trajectory=traj.tolist()) for name,group in trajectories.items() for j,traj in enumerate(group)]))
        row['scores']=scored
        for name in trajectories:
            row[name+'_PDMS']=float(np.mean([item['score']['score'] for item in scored if item['arm']==name])*100)
        torch.save(stages,out/(mode+'_STAGES.pt'))
        with (out/(mode+'_trajectories.npz')).open('xb') as stream:np.savez_compressed(stream,**trajectories)
        arm_stages[mode]={key:value.clone() for key,value in stages.items()}
        results.append(row);write(out/'LIVE.json',dict(setup=setup,cases=results))
        print(__import__('json').dumps(dict(mode=mode,matches=matches,Base_XY=row['Base_trajectory']['XY_max_abs_m'],teacher_XY=row['teacher_trajectory']['XY_max_abs_m'],Base_PDMS=row['Base_PDMS'],teacher_PDMS=row['teacher_PDMS'])),flush=True)
    sensitivity={}
    for name,a in arm_stages['native'].items():
        b=arm_stages['ppu_C'][name];delta=a.float().double()-b.float().double()
        sensitivity[name]=dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/b.double().norm()),different_elements=int((a!=b).sum()))
    assert parameter_versions=={k:p._version for k,p in model.named_parameters()}
    for h in handles:h.remove()
    write(out/'RESULT.json',dict(status='COMPLETE_ONE_SCENE_REAL_PPU_C_CAUSAL_INJECTION',setup=setup,cases=results,
        native_vs_injection=sensitivity,parameters_unchanged=True,
        full_chain_aligned=all(results[1]['matches_PPU'].values()),
        production_fix=False,bulk_generation_resumed=False))


if __name__=='__main__':main()
