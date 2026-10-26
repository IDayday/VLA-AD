"""Portable, one-block PPU/A800 comparison from the same real PPU embedding.

Does not load a model, change precision flags/parameters, or write by default.
Use an already loaded eval model, the verified scene pixels and real PPU C NPZ.
By default only block 0 executes; optional visual-suffix verification checks
continuity to the historical PPU vision_last. No language/action/teacher forward.
"""
import hashlib
import inspect
import json
from pathlib import Path


def trace_first_vision_block(model,pixels,native_C_path,*,reference_file=None,verify_full_vision=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    def info(value):
        value=value.detach().cpu().contiguous()
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
    def state():
        return dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            autocast=torch.is_autocast_enabled(),cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic)
    flags=state()
    assert not flags['autocast'] and not model.training
    assert info(pixels)['sha256']=='f0d5c4a3814c428ac6cca1a0ec5293c5d624fe2d5669c926cc9d3c594e7ab4ac'
    with np.load(native_C_path,allow_pickle=False) as z:
        metadata=json.loads(z['__metadata_utf8__'].tobytes());raw=z['native_no_bias'].copy()
    assert hashlib.sha256(raw.tobytes()).hexdigest()=='c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b'
    c=torch.from_numpy(raw).view(torch.bfloat16).reshape(metadata['tensors']['native_no_bias']['shape']).to(pixels.device)
    emb=model.vision_model.embeddings;layer=model.vision_model.encoder.layers[0]
    assert not layer.training and not layer.attn.use_flash_attn
    parameters={'embedding.'+name:info(value) for name,value in emb.named_parameters()}
    parameters.update({'block0.'+name:info(value) for name,value in layer.named_parameters()})
    versions={name:value._version for name,value in list(emb.named_parameters())+list(layer.named_parameters())}
    report=dict(schema=1,scene='0002267a294e52a7',torch=str(torch.__version__),torch_git=torch.version.git_version,
        device=torch.cuda.get_device_name(pixels.device),precision=flags,parameters=parameters,events=[],controls={},
        diagnostic_only=True,bulk_generation_resumed=False)
    source=Path(inspect.getsourcefile(type(layer)));report['vision_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    handles=[]
    def capture(name):
        def hook(module,args,output):
            report['events'].append(dict(name=name,tensor=info(output)))
        return hook
    names=['norm1','attn.qkv','attn.q_norm','attn.k_norm','attn.attn_drop','attn.proj','attn','drop_path1',
        'norm2','mlp.fc1','mlp.act','mlp.fc2','mlp','drop_path2']
    modules=dict(layer.named_modules())
    for name in names:
        if name in modules:handles.append(modules[name].register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            patch=c+emb.patch_embedding.bias[None,:,None,None]
            assert info(patch)['sha256']=='0d0a5f0e48d4e03f6635140f9043a2fc79b9f8b800a4e95b541bd28e3e55cd6b'
            position=emb._get_pos_embed(emb.position_embedding[:,1:,:],32,32)
            x=torch.cat([emb.class_embedding.expand(pixels.shape[0],1,-1).to(patch.dtype),patch.flatten(2).transpose(1,2)],dim=1)
            x=x+torch.cat([emb.position_embedding[:,:1,:],position],dim=1).to(x.dtype)
            report['embedding']=info(x)
            assert report['embedding']['sha256']=='ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74'
            y=layer(x)
            report['events'].append(dict(name='block0_output',tensor=info(y)))
    finally:
        for handle in handles:handle.remove()
    with torch.no_grad():
        norm=layer.norm1
        report['controls']['norm1_FP32_then_BF16']=info(F.layer_norm(x.float(),norm.normalized_shape,norm.weight.float(),norm.bias.float(),norm.eps).to(x.dtype))
        d=x.double();mean=d.mean(-1,keepdim=True);variance=((d-mean)**2).mean(-1,keepdim=True)
        report['controls']['norm1_FP64_then_BF16']=info(((d-mean)*torch.rsqrt(variance+norm.eps)*norm.weight.double()+norm.bias.double()).to(x.dtype))
        if verify_full_vision:
            vision_last=model.vision_model.encoder(inputs_embeds=x,return_dict=True).last_hidden_state
            report['vision_last']=info(vision_last)
            report['vision_last_matches_historical_PPU']=report['vision_last']['sha256']=='ba7dbbd0efbb5109fa9c593a28cf3528b8b6d3ed1507fa7a832e36cc7abcfc9d'
    assert flags==state(),'Precision state changed'
    assert versions=={name:value._version for name,value in list(emb.named_parameters())+list(layer.named_parameters())}
    report['flags_and_parameter_versions_preserved']=True
    if reference_file is not None:
        reference=json.loads(Path(reference_file).read_text())
        expected={event['name']:event['tensor'] for event in reference['events']}
        report['parameter_matches_A800']={name:value==reference['parameters'].get(name) for name,value in parameters.items()}
        report['event_matches_A800']={event['name']:event['tensor']==expected.get(event['name']) for event in report['events']}
        report['first_divergent_event']=next((event['name'] for event in report['events'] if not report['event_matches_A800'][event['name']]),None)
        report['control_matches_A800']={name:value==reference['controls'].get(name) for name,value in report['controls'].items()}
        report['reference_precision']=reference['precision']
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report
