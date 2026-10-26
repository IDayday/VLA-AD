"""Capture actual visual-inference conditions and gate a controlled A800 replay.

Capture/check functions never load models, install packages, or change flags.
configure_fresh_receiver() is an explicit opt-in for a NEW diagnostic process.
No language/action/teacher forward. See PPU_CONFIGURATION_AND_A800_REQUIREMENTS.md.
"""
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys

SCENE = '0002267a294e52a7'
FROZEN = '6bb7119583513094a169b4c2f2dc2a05ac293aa3'
COMMON_PACKAGES = ['torchvision','transformers','tokenizers','timm','numpy','Pillow',
                   'safetensors','accelerate','einops','decord']
ENV_KEYS = ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS',
    'CUDA_MODULE_LOADING','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','TORCH_CUDNN_V8_API_DISABLED',
    'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT','KMP_DUPLICATE_LIB_OK','KMP_INIT_AT_FORK',
    'ATEN_CPU_CAPABILITY','DISABLE_ADDMM_CUDA_LT','CUBLASLT_WORKSPACE_SIZE',
    'CUDNN_CONV_WSCAP_DBG','PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF']
MODULE_SCALARS = ['normalized_shape','eps','elementwise_affine','p','inplace','dropout_p',
    'softmax_scale','num_heads','head_dim','scale','qk_normalization','use_flash_attn',
    'drop_prob','scale_by_keep','approximate','in_features','out_features','in_channels',
    'out_channels','kernel_size','stride','padding','dilation','groups','padding_mode',
    'gradient_checkpointing']
ASSET_NAMES = ['config.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json',
    'vocab.json','merges.txt','generation_config.json','configuration_intern_vit.py',
    'configuration_internvl_chat.py','modeling_intern_vit.py','modeling_internvl_chat.py','conversation.py']
MODEL_KEYS = ['pixel_values','input_ids','attention_mask','position_ids','image_flags']
EXTRA_KEYS = ['pixels_FP32','history_trajectory','high_command_one_hot','status_feature',
              'role_image','role_validtext','role_padding']


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_info(value):
    import torch
    return {'shape':list(value.shape),'dtype':str(value.dtype),'stride':list(value.stride()),
            'layout':str(value.layout),'storage_offset':value.storage_offset(),
            'contiguous':value.is_contiguous(),
            'channels_last':value.is_contiguous(memory_format=torch.channels_last) if value.ndim==4 else None,
            'device_type':value.device.type,
            'sha256':hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}


def numeric_policy():
    import torch
    return {'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
      'tf32_cudnn':torch.backends.cudnn.allow_tf32,'cudnn_enabled':torch.backends.cudnn.enabled,
      'bf16_reduced':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
      'fp16_reduced':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
      'matmul_precision':torch.get_float32_matmul_precision(),
      'autocast_cuda':torch.is_autocast_enabled(),'autocast_cpu':torch.is_autocast_cpu_enabled(),
      'autocast_gpu_dtype':str(torch.get_autocast_gpu_dtype()),
      'autocast_cpu_dtype':str(torch.get_autocast_cpu_dtype()),
      'cudnn_benchmark':torch.backends.cudnn.benchmark,
      'cudnn_deterministic':torch.backends.cudnn.deterministic,
      'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
      'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
      'inference_mode_enabled':torch.is_inference_mode_enabled(),
      'distributed_initialized':torch.distributed.is_available() and torch.distributed.is_initialized(),
      'default_dtype':str(torch.get_default_dtype()),'default_device':str(torch.get_default_device()),
      'num_threads':torch.get_num_threads(),'num_interop_threads':torch.get_num_interop_threads(),
      'operator_execution_context':'torch.no_grad(), eval; no autocast; ordinary tensors, not inference_mode'}


def _clean_config(value):
    if isinstance(value,dict):
        return {str(k):_clean_config(v) for k,v in value.items() if k not in ['_name_or_path','_commit_hash']}
    if isinstance(value,(tuple,list)):
        return [_clean_config(v) for v in value]
    if value.__class__.__module__=='torch' and value.__class__.__name__ in ['dtype','device']:
        return str(value)
    return value


def _source(value):
    try:
        filename=Path(inspect.getfile(value))
        body=inspect.getsource(value)
        return {'file_sha256':sha_file(filename),'object_source_sha256':hashlib.sha256(body.encode()).hexdigest()}
    except (OSError,TypeError):
        return {'builtin_or_unavailable':True}


def driver_observation():
    result={}
    for label,command in [('nvidia_smi_compatibility_query',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
                          ('ppu_smi_version',['ppu-smi','--version'])]:
        if shutil.which(command[0]) is None:
            result[label]={'available':False};continue
        try:
            completed=subprocess.run(command,text=True,capture_output=True,timeout=20)
            result[label]={'exit_code':completed.returncode,'stdout':completed.stdout.strip(),'stderr':completed.stderr.strip()}
        except Exception as exc:
            result[label]={'error':type(exc).__name__+': '+str(exc)}
    return result


def capture_configuration(model, model_inputs, *, bundle_dir, vlm_dir, frozen_repo,
                          auxiliary, role_tensors, hash_weights=True):
    """Read actual objects and original repository data; no model arithmetic/writes."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    start=numeric_policy()
    bundle=Path(bundle_dir);vlm=Path(vlm_dir);repo=Path(frozen_repo)
    if not torch.cuda.is_initialized():
        raise RuntimeError('Capture inside the process already holding the actual device model')
    metadata=json.loads((bundle/'inputs/metadata.json').read_text())
    if metadata['scene']!=SCENE:
        raise ValueError('Wrong repository scene')
    source_data={str(p.relative_to(bundle/'inputs')):{'bytes':p.stat().st_size,'sha256':sha_file(p)}
                 for p in sorted((bundle/'inputs').iterdir()) if p.is_file() and p.name!='READ_ME.txt'}
    with np.load(bundle/'inputs/tensors.npz',allow_pickle=False) as archive:
        stored=json.loads(archive['__metadata_utf8__'].tobytes())
        if stored!=metadata:
            raise ValueError('Embedded input metadata mismatch')
        expected_inputs={}
        for key in MODEL_KEYS+EXTRA_KEYS:
            raw=archive[key]
            if raw.dtype!=np.uint8 or hashlib.sha256(raw.tobytes()).hexdigest()!=metadata['tensors'][key]['sha256']:
                raise ValueError('Corrupt repository tensor: '+key)
            expected_inputs[key]={k:metadata['tensors'][key][k] for k in ['shape','dtype','stride','sha256']}
    actual={key:model_inputs[key] for key in MODEL_KEYS}
    actual.update({key:auxiliary[key] for key in ['history_trajectory','high_command_one_hot','status_feature']})
    actual.update({key:role_tensors[key] for key in ['pixels_FP32','role_image','role_validtext','role_padding']})
    actual_inputs={key:tensor_info(value) for key,value in actual.items()}
    input_checks={key:all(actual_inputs[key][k]==value[k] for k in ['shape','dtype','stride','sha256'])
                  for key,value in expected_inputs.items()}
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    frozen_sources={name:sha_file(repo/name) for name in ['vendor/vlm_core.py',
        'vendor/upstream/model/recogdrive_backbone.py','vendor/upstream/model/recogdrive_features.py',
        'vendor/upstream/model/utils/internvl_preprocess.py','vendor/upstream/model/utils/conversation.py']}
    assets={name:{'bytes':(vlm/name).stat().st_size,'sha256':sha_file(vlm/name)}
            for name in ASSET_NAMES if (vlm/name).is_file()}
    weight_files=sorted(vlm.glob('*.safetensors'))
    weights={p.name:{'bytes':p.stat().st_size,'sha256':sha_file(p) if hash_weights else None}
             for p in weight_files}
    module_specs={}
    for name,module in model.vision_model.named_modules():
        item={'class':type(module).__name__,'training':module.training,'scalars':{}}
        for key in MODULE_SCALARS:
            value=getattr(module,key,None)
            if value is None or isinstance(value,(str,int,float,bool)):
                if hasattr(module,key):item['scalars'][key]=value
            elif isinstance(value,(tuple,list)) and all(isinstance(v,(str,int,float,bool)) for v in value):
                item['scalars'][key]=list(value)
        module_specs[name]=item
    parameters={name:tensor_info(value) for name,value in model.vision_model.named_parameters()}
    buffers={name:tensor_info(value) for name,value in model.vision_model.named_buffers()}
    package_versions={}
    for name in COMMON_PACKAGES:
        try:package_versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:package_versions[name]=None
    sources={'vision_module':_source(type(model.vision_model)),
             'LayerNorm_class':_source(torch.nn.LayerNorm),'LayerNorm_forward':_source(torch.nn.LayerNorm.forward),
             'functional_layer_norm':_source(F.layer_norm),'Linear_forward':_source(torch.nn.Linear.forward),
             'activation_class':_source(type(model.vision_model.encoder.layers[0].mlp.act))}
    # Full file hashes for torch wrappers are retained in the backend section;
    # relevant callable bodies are portable requirements (unrelated vendor edits
    # elsewhere in a large torch source file do not silently pass as identical).
    portable_sources={key:({'file_sha256':value['file_sha256']} if key=='vision_module' else
                           {'object_source_sha256':value.get('object_source_sha256')})
                      for key,value in sources.items()}
    portable={'scene':SCENE,'frozen_teacher_commit':commit,'frozen_entry_sources':frozen_sources,
       'python_version':platform.python_version(),'torch_release':str(torch.__version__).split('+')[0],
       'common_package_versions':package_versions,'numeric_policy':start,
       'relevant_environment':{key:os.environ.get(key) for key in ENV_KEYS},
       'source_data_files':source_data,'repository_tensor_identity':expected_inputs,
       'actual_model_and_auxiliary_inputs':actual_inputs,
       'actual_forward_options':{key:model_inputs.get(key) for key in ['use_cache','output_hidden_states','return_dict']},
       'inference_assets':assets,'weight_files':weights,
       'runtime_vision_config':_clean_config(model.vision_model.config.to_dict()),
       'vision_modules':module_specs,'all_vision_parameters':parameters,'vision_buffers':buffers,
       'relevant_operator_source':portable_sources,'model_training':model.training}
    libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/' in line and any(word in line.lower() for word in
                   ['ppu_sdk','libtorch','libcuda','libcudnn','libcublas','libacblas','libacdnn','libc10'])})
    all_packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    backend={'device':torch.cuda.get_device_name(next(model.parameters()).device),
       'logical_device':str(next(model.parameters()).device),'python_build':sys.version,
       'platform':platform.platform(),'torch_version':str(torch.__version__),'torch_git':torch.version.git_version,
       'torch_cuda_build':torch.version.cuda,'torch_build':torch.__config__.show(),
       'driver_observation':driver_observation(),
       'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').is_file() else None,
       'current_container_image_tag':None,
       'container_image_note':'Not determined by this process capture; never substitute a historical DLC image tag.',
       'cudnn_runtime_version':torch.backends.cudnn.version(),
       'PPU_SDK_release':Path('/usr/local/PPU_SDK/release.yaml').read_text() if Path('/usr/local/PPU_SDK/release.yaml').is_file() else None,
       'loaded_libraries':[{'path':path,'bytes':Path(path).stat().st_size if Path(path).exists() else None} for path in libs],
       'operator_source_file_identity':sources,'all_distribution_versions':dict(sorted(all_packages.items())),
       'model_config':model.config.to_dict(),'parameter_requires_grad':{k:v.requires_grad for k,v in model.vision_model.named_parameters()},
       'global_grad_enabled_outside_diagnostic':torch.is_grad_enabled(),
       'cpu_initial_seed_for_record_only':torch.initial_seed(),
       'rng_policy':'Visual eval/no_grad path has no random sampling. Action/teacher replay is outside this test and must use the previously verified explicit five-noise bank, not seeds alone.'}
    if numeric_policy()!=start:
        raise RuntimeError('Capture unexpectedly changed numerical policy')
    return _clean_config({'schema':1,'scope':'Current PPU-verified pure visual reference; not a fresh full VLM/action validation',
       'portable_conditions':portable,'backend_identity':backend,
       'capture_validation':{'all_12_actual_inputs_match_repository':all(input_checks.values()),
                             'per_tensor':input_checks,'frozen_commit_matches':commit==FROZEN,
                             'weights_hashed_in_this_capture':hash_weights and bool(weights),
                             'model_arithmetic_executed':False,'precision_unchanged':True}})


def differences(expected, actual, prefix=''):
    """Pure JSON comparison, reporting every absent/different field without tolerance."""
    if isinstance(expected,dict) and isinstance(actual,dict):
        out=[]
        for key in sorted(set(expected)|set(actual)):
            name=prefix+'.'+str(key) if prefix else str(key)
            if key not in expected:out.append({'field':name,'reason':'unexpected_field','actual':actual[key]})
            elif key not in actual:out.append({'field':name,'reason':'missing_field','expected':expected[key]})
            else:out.extend(differences(expected[key],actual[key],name))
        return out
    if type(expected) is not type(actual):
        return [{'field':prefix,'reason':'JSON_type_mismatch','expected':expected,'actual':actual}]
    if expected!=actual:
        return [{'field':prefix,'expected':expected,'actual':actual}]
    return []


def compare_configurations(reference, actual):
    diff=differences(reference['portable_conditions'],actual['portable_conditions'])
    validation=actual['capture_validation']
    valid=all(validation[key] for key in ['all_12_actual_inputs_match_repository','frozen_commit_matches',
                                         'weights_hashed_in_this_capture','precision_unchanged'])
    backend=differences(reference['backend_identity'],actual['backend_identity'])
    return {'schema':1,'status':'PORTABLE_MATCH_BACKEND_DIFFERENCES_DECLARED' if not diff and valid else 'BLOCKED_CONFIGURATION_MISMATCH',
       'portable_conditions_match':not diff and valid,'portable_differences':diff,
       'input_identity_validation':validation,'backend_differences':backend,
       'all_external_conditions_identical':not diff and valid and not backend,
       'interpretation':'A passing portable gate allows a backend comparison. Different vendor torch builds, cuDNN/SDK/driver/compiler or runtime libraries remain software/backend variables; never conclude hardware-only from this gate.'}


def configure_fresh_receiver(reference_file):
    """EXPLICIT MUTATION, fresh A800 diagnostic child only; never call on PPU model.

    Sets reference flags/environment/threads BEFORE model construction. Refuses
    an already initialized CUDA process. Installs nothing and loads no model.
    """
    reference=json.loads(Path(reference_file).read_text())['portable_conditions']
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise RuntimeError('Use a fresh diagnostic process; refusing to change an active model process')
    for key,value in reference['relevant_environment'].items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA already initialized')
    flags=reference['numeric_policy']
    torch.set_num_interop_threads(flags['num_interop_threads'])
    torch.set_num_threads(flags['num_threads'])
    if str(torch.get_default_dtype())!=flags['default_dtype']:
        torch.set_default_dtype(getattr(torch,flags['default_dtype'].split('.')[-1]))
    if str(torch.get_default_device())!=flags['default_device']:
        torch.set_default_device(flags['default_device'])
    torch.set_float32_matmul_precision(flags['matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=flags['tf32_matmul']
    torch.backends.cudnn.allow_tf32=flags['tf32_cudnn']
    torch.backends.cudnn.enabled=flags['cudnn_enabled']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flags['bf16_reduced']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=flags['fp16_reduced']
    torch.backends.cudnn.benchmark=flags['cudnn_benchmark']
    torch.backends.cudnn.deterministic=flags['cudnn_deterministic']
    torch.use_deterministic_algorithms(flags['deterministic_algorithms'],warn_only=flags['deterministic_warn_only'])
    torch.set_autocast_gpu_dtype(getattr(torch,flags['autocast_gpu_dtype'].split('.')[-1]))
    torch.set_autocast_cpu_dtype(getattr(torch,flags['autocast_cpu_dtype'].split('.')[-1]))
    if numeric_policy()!=flags:
        raise RuntimeError({'failed_to_match_numeric_policy':differences(flags,numeric_policy())})
    return {'configured_new_child':True,'numeric_policy':numeric_policy()}


def run_controlled_check(model, model_inputs, *, auxiliary, role_tensors, bundle_dir,
                         vlm_dir, frozen_repo, folder, output_dir, verify_full_vision=True):
    """Capture -> strict gate -> only on PASS run native C and supplied vision trace.

    Writes ONLY new result files in an existing independent output directory.
    Config mismatch returns before any convolution/norm/encoder operation.
    """
    folder=Path(folder);out=Path(output_dir)
    if not out.is_dir():raise FileNotFoundError('Create independent output directory first')
    reference=json.loads((folder/'PPU_CONTROLLED_CONFIGURATION.json').read_text())
    actual=capture_configuration(model,model_inputs,bundle_dir=bundle_dir,vlm_dir=vlm_dir,
        frozen_repo=frozen_repo,auxiliary=auxiliary,role_tensors=role_tensors,hash_weights=True)
    gate=compare_configurations(reference,actual)
    for filename,value in [('A800_CONTROLLED_CONFIGURATION.json',actual),('A800_CONTROLLED_GATE.json',gate)]:
        with (out/filename).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    if not gate['portable_conditions_match']:
        print(json.dumps(gate,ensure_ascii=False,indent=2));return gate
    import numpy as np
    import torch
    import torch.nn.functional as F
    flags=numeric_policy()
    versions={key:value._version for key,value in model.vision_model.named_parameters()}
    pixels=model_inputs['pixel_values'];conv=model.vision_model.embeddings.patch_embedding
    with np.load(folder/'ppu_native_no_bias.npz',allow_pickle=False) as archive:
        raw=archive['native_no_bias'].copy();meta=json.loads(archive['__metadata_utf8__'].tobytes())
    if hashlib.sha256(raw.tobytes()).hexdigest()!='c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b':
        raise ValueError('Wrong PPU C payload')
    ppu_c=torch.from_numpy(raw).view(torch.bfloat16).reshape(meta['tensors']['native_no_bias']['shape']).to(pixels.device)
    audit=runpy.run_path(str(Path(bundle_dir)/'audit_embedding.py'))
    emb=model.vision_model.embeddings
    with torch.no_grad():
        common_patch=ppu_c+conv.bias[None,:,None,None]
        position=emb._get_pos_embed(emb.position_embedding[:,1:,:],32,32)
        common_x=torch.cat([emb.class_embedding.expand(pixels.shape[0],1,-1).to(common_patch.dtype),
                            common_patch.flatten(2).transpose(1,2)],dim=1)
        common_x=common_x+torch.cat([emb.position_embedding[:,:1,:],position],dim=1).to(common_x.dtype)
    expected_x=reference['numerical_validation_reference']['norm1_input']
    operand_diff=differences(expected_x,tensor_info(common_x))
    operand_gate={'status':'MATCH' if not operand_diff else 'STOP_BEFORE_NORM1',
                  'differences':operand_diff,'actual':tensor_info(common_x),'expected':expected_x}
    with (out/'A800_CONTROLLED_NORM1_INPUT_GATE.json').open('x') as stream:
        json.dump(operand_gate,stream,ensure_ascii=False,indent=2);stream.write('\n')
    if operand_diff:
        print(json.dumps(operand_gate,ensure_ascii=False,indent=2));return operand_gate
    with torch.no_grad():
        native_c=F.conv2d(pixels,conv.weight,None,stride=conv.stride,padding=conv.padding,dilation=conv.dilation,groups=conv.groups)
        native_patch=conv(pixels)
        ppu_patch=ppu_c+conv.bias[None,:,None,None]
        native_embedding=model.vision_model.embeddings(pixels)
    reference_trace=folder/'PPU_FIRST_BLOCK_TRACE.json'
    trace_function=runpy.run_path(str(folder/'trace_first_vision_block.py'))['trace_first_vision_block']
    trace=trace_function(model,pixels,folder/'ppu_native_no_bias.npz',reference_file=reference_trace,
                         verify_full_vision=verify_full_vision)
    trace['reference_platform']='PPU: PPU_FIRST_BLOCK_TRACE.json'
    trace['parameter_matches_PPU']=trace['parameter_matches_A800']
    trace['event_matches_PPU']=trace['event_matches_A800']
    trace['control_matches_PPU']=trace['control_matches_A800']
    result={'schema':1,'scene':SCENE,'portable_gate_passed':True,
       'native_C':tensor_info(native_c),'native_patch':tensor_info(native_patch),'native_embedding':tensor_info(native_embedding),
       'norm1_operand_gate':operand_gate,
       'PPU_C_minus_actual_C':audit['compare_tensors'](ppu_c,native_c),
       'PPU_patch_minus_actual_patch':audit['compare_tensors'](ppu_patch,native_patch),
       'first_block_from_common_PPU_embedding':trace,
       'note':'Original trace helper names its comparison fields *_A800; in this call its reference is PPU_FIRST_BLOCK_TRACE.json, so those booleans compare against PPU. No teacher outputs were generated.',
       'full_chain_alignment_complete':False,'bulk_generation_resumed':False}
    if flags!=numeric_policy() or versions!={key:value._version for key,value in model.vision_model.named_parameters()}:
        raise RuntimeError('Diagnostic changed numerical flags or parameter versions')
    if any(tensor_info(value)['sha256']!=actual['portable_conditions']['all_vision_parameters'][key]['sha256']
           for key,value in model.vision_model.named_parameters()):
        raise RuntimeError('Visual parameter bytes changed')
    with (out/'A800_CONTROLLED_TRACE.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':'CONTROLLED_VISUAL_CHECK_COMPLETE','first_divergent_event':trace['first_divergent_event'],
         'vision_last_matches_historical_PPU':trace.get('vision_last_matches_historical_PPU'),
         'portable_conditions_match':True,'hardware_only_cause_proven':False},ensure_ascii=False,indent=2))
    return result
