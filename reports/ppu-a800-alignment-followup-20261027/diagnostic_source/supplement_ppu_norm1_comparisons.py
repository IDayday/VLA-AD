"""Run in the retained PPU trace process, using its original model/pixels/record.

Only norm1 and the supplied FP32/FP64 arithmetic controls are repeated. No
attention, full encoder, language, action, parameter or precision changes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from audit_embedding import compare_tensors, precision_state, tensor_info

supplement_flags = precision_state(torch)
norm1 = model.vision_model.encoder.layers[0].norm1
norm_versions = {name:value._version for name,value in norm1.named_parameters()}
with torch.no_grad():
    with np.load(FOLDER/'ppu_native_no_bias.npz',allow_pickle=False) as archive:
        c_meta=json.loads(archive['__metadata_utf8__'].tobytes())
        c_raw=archive['native_no_bias'].copy()
    c=torch.from_numpy(c_raw).view(torch.bfloat16).reshape(c_meta['tensors']['native_no_bias']['shape']).to(pixels.device)
    emb=model.vision_model.embeddings
    patch=c+emb.patch_embedding.bias[None,:,None,None]
    position=emb._get_pos_embed(emb.position_embedding[:,1:,:],32,32)
    norm_input=torch.cat([emb.class_embedding.expand(9,1,-1).to(patch.dtype),patch.flatten(2).transpose(1,2)],dim=1)
    norm_input=norm_input+torch.cat([emb.position_embedding[:,:1,:],position],dim=1).to(norm_input.dtype)
    assert tensor_info(norm_input)['sha256']==record['embedding']['sha256']
    native_norm=norm1(norm_input)
    control32=F.layer_norm(norm_input.float(),norm1.normalized_shape,norm1.weight.float(),norm1.bias.float(),norm1.eps).to(norm_input.dtype)
    d=norm_input.double();mean=d.mean(-1,keepdim=True);variance=((d-mean)**2).mean(-1,keepdim=True)
    control64=((d-mean)*torch.rsqrt(variance+norm1.eps)*norm1.weight.double()+norm1.bias.double()).to(norm_input.dtype)
assert tensor_info(native_norm)['sha256']==record['events'][0]['tensor']['sha256']
assert tensor_info(control32)['sha256']==record['controls']['norm1_FP32_then_BF16']['sha256']
assert tensor_info(control64)['sha256']==record['controls']['norm1_FP64_then_BF16']['sha256']
norm1_supplement={'scope':'Current PPU norm1 vs temporary FP32/FP64 controls only; not PPU-vs-A800 exact norm1 errors and not replacement teacher outputs.',
    'norm_class':type(norm1).__name__,'normalized_shape':list(norm1.normalized_shape),'eps':norm1.eps,
    'tensors':{name:tensor_info(value) for name,value in [('input',norm_input),('native_norm1',native_norm),('FP32_then_BF16',control32),('FP64_then_BF16',control64)]},
    'comparisons':{'native_norm1_vs_FP32_then_BF16':compare_tensors(native_norm,control32),
                   'native_norm1_vs_FP64_then_BF16':compare_tensors(native_norm,control64),
                   'FP32_then_BF16_vs_FP64_then_BF16':compare_tensors(control32,control64)},
    'precision':supplement_flags,'parameter_versions_preserved':norm_versions=={name:value._version for name,value in norm1.named_parameters()}}
assert norm1_supplement['parameter_versions_preserved'] and supplement_flags==precision_state(torch)
record['norm1_local_numeric_supplement']=norm1_supplement
record['provenance']['norm1_supplement_script_sha256']=hashlib.sha256((FOLDER/'diagnostic_source/supplement_ppu_norm1_comparisons.py').read_bytes()).hexdigest()
(FOLDER/'PPU_FIRST_BLOCK_TRACE.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
(FOLDER/'PPU_NORM1_CONTROLS.json').write_text(json.dumps(norm1_supplement,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
summary['sha256']=hashlib.sha256((FOLDER/'PPU_FIRST_BLOCK_TRACE.json').read_bytes()).hexdigest()
(FOLDER/'PPU_FIRST_BLOCK_TRACE_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'first_divergent_event':record['first_divergent_event'],
       'historical_vision_last_exact':record['vision_last_matches_historical_PPU'],
       'norm1_comparisons':{key:{k:v for k,v in result.items() if k!='first16'} for key,result in norm1_supplement['comparisons'].items()},
       'trace_sha256':summary['sha256']},ensure_ascii=False,indent=2),flush=True)
