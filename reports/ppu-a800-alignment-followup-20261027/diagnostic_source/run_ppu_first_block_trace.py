"""Execute the incoming first-block trace on PPU, keeping the model in Python -i.

The old local diagnostic process has exited. Load identical frozen original VLM
weights through the original backbone entry, constructing non-Flash attention
to match original PPU and the supplied A800 driver. No precision flag override.
No language/action/teacher forward. Absolute paths are this PPU capture context.
"""
from pathlib import Path
import contextlib
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
import runpy
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
FOLDER = Path(__file__).resolve().parents[1]
BUNDLE = Path('/mnt/workspace/project/ppu-a800-embedding-audit-20261027')
FROZEN_REPO = Path('/mnt/data_and_weight/WM_Group/sjc_workspace/ReCogDrive/1026-interface_teacher_portable/repo')
VLM_PATH = Path('/mnt/data_and_weight/WM_Group/yehr_workspace/ReCogDrive/ReCogDrive/recogdrive/ReCogDrive-VLM-2B')
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(FROZEN_REPO/'vendor/upstream'))

import numpy as np
import torch
from audit_embedding import precision_state, tensor_info
from load_inputs import load_inputs

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

frozen_commit = subprocess.check_output(['git','-C',str(FROZEN_REPO),'rev-parse','HEAD'],text=True).strip()
assert frozen_commit == '6bb7119583513094a169b4c2f2dc2a05ac293aa3'
trace_path = FOLDER/'trace_first_vision_block.py'
reference_path = FOLDER/'evidence/A800_FIRST_BLOCK_REFERENCE.json'
trace_sha = file_sha(trace_path)
input_dict, auxiliary, input_tensors, input_metadata = load_inputs()
initial_flags = precision_state(torch)
print('INPUTS_VERIFIED; FLAGS_PRESERVED', json.dumps(initial_flags), flush=True)

# The same child-only optional wandb import exclusion as prior successful local
# captures. This does not edit a package, environment file, parameter or flag.
sys.modules['wandb'] = None
from transformers import AutoModel
from model.recogdrive_backbone import RecogDriveBackbone
original_load = AutoModel.from_pretrained
load_calls = []
def load_original_non_flash(*args, **kwargs):
    previous = kwargs.get('use_flash_attn')
    kwargs['use_flash_attn'] = False
    load_calls.append({'previous_use_flash_attn_argument': previous,
                       'construction_use_flash_attn': False,
                       'basis': '5cf8440 A800_REAL_C_RESULTS.md original PPU actual non-Flash path and supplied A800 construction driver; trace asserts non-Flash.'})
    return original_load(*args, **kwargs)

AutoModel.from_pretrained = load_original_non_flash
try:
    print('LOADING_ORIGINAL_BACKBONE_WITH_NON_FLASH_CONSTRUCTION', flush=True)
    backbone = RecogDriveBackbone('internvl', str(VLM_PATH), 'cuda')
finally:
    AutoModel.from_pretrained = original_load
model = backbone.model
assert not model.training
assert initial_flags == precision_state(torch), 'Model loading changed global precision flags'
assert all(not layer.attn.use_flash_attn for layer in model.vision_model.encoder.layers)
inputs = {key: value if key == 'image_flags' or not isinstance(value, torch.Tensor)
          else value.to(next(model.parameters()).device) for key,value in input_dict.items()}
pixels = inputs['pixel_values']
vision_versions = {name: value._version for name,value in model.vision_model.named_parameters()}
vision_parameter_metadata = {name: tensor_info(value) for name,value in model.vision_model.named_parameters()}
hook_counts_before = {name: len(module._forward_hooks) for name,module in model.vision_model.named_modules()}
print('MODEL_READY; EXECUTING_UNMODIFIED_FIRST_BLOCK_TRACE_WITH_VISUAL_SUFFIX', flush=True)

trace = runpy.run_path(str(trace_path))['trace_first_vision_block']
with (FOLDER/'PPU_FIRST_BLOCK_TRACE.stdout.txt').open('x') as log, contextlib.redirect_stdout(log):
    record = trace(model, pixels, FOLDER/'ppu_native_no_bias.npz',
                   reference_file=reference_path, verify_full_vision=True)

assert file_sha(trace_path) == trace_sha
assert initial_flags == precision_state(torch)
assert vision_versions == {name:value._version for name,value in model.vision_model.named_parameters()}
assert all(tensor_info(value)['sha256'] == vision_parameter_metadata[name]['sha256']
           for name,value in model.vision_model.named_parameters())
assert hook_counts_before == {name:len(module._forward_hooks) for name,module in model.vision_model.named_modules()}
versions = {}
for name in ['torch','torchvision','transformers','tokenizers','timm','numpy','Pillow','safetensors','flash-attn','apex']:
    try: versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name] = None
record['provenance'] = {
    'time_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'frozen_teacher_commit': frozen_commit, 'incoming_audit_commit': '5cf8440',
    'model_lifetime': 'Previous process exited; same frozen original VLM newly loaded through original backbone. This Python -i process remains alive after trace.',
    'original_entry': "RecogDriveBackbone('internvl', original_VLM_path, 'cuda')",
    'construction_override': load_calls,
    'model_parameters_or_attention_changed_after_construction': False,
    'global_precision_flags_changed': False,
    'precision_initial': initial_flags, 'precision_after': precision_state(torch),
    'PPU_SDK_release': Path('/usr/local/PPU_SDK/release.yaml').read_text(),
    'packages': versions, 'python': sys.version,
    'all_visual_parameter_versions_and_hashes_preserved': True,
    'trace_hooks_removed': True, 'trace_script_modified': False,
    'trace_script_sha256': trace_sha,
    'A800_reference_sha256': file_sha(reference_path),
    'input_payload_sha256': file_sha(BUNDLE/'inputs/tensors.npz'),
    'pixels': tensor_info(pixels),
    'native_C_file_sha256': file_sha(FOLDER/'ppu_native_no_bias.npz'),
    'actual_visual_non_flash_layers': len(model.vision_model.encoder.layers),
    'actual_norm1_class': type(model.vision_model.encoder.layers[0].norm1).__name__,
    'actual_norm1_eps': model.vision_model.encoder.layers[0].norm1.eps,
    'actual_language_attention_classes': sorted({type(layer.self_attn).__name__ for layer in model.language_model.model.layers}),
    'actual_language_config_implementation': getattr(model.language_model.config,'_attn_implementation',None),
    'language_forward_executed': False, 'action_forward_executed': False,
    'teacher_search_or_bulk_generation_executed': False,
    'comparison_scope': 'Same PPU-C reconstructed embedding, isolated block0 and full pure vision encoder only. SDK 2.1 current trace is historical PPU evidence only if vision_last matches archived historical SHA.',
}
write_new(FOLDER/'PPU_FIRST_BLOCK_TRACE.json', record)
write_new(FOLDER/'PPU_VISION_PARAMETER_METADATA.json', vision_parameter_metadata)
summary = {'status': 'TRACE_COMPLETE', 'path': str(FOLDER/'PPU_FIRST_BLOCK_TRACE.json'),
           'sha256': file_sha(FOLDER/'PPU_FIRST_BLOCK_TRACE.json'),
           'embedding_sha256': record['embedding']['sha256'],
           'first_divergent_event': record.get('first_divergent_event'),
           'parameter_matches_A800': all(record.get('parameter_matches_A800',{}).values()),
           'control_matches_A800': record.get('control_matches_A800'),
           'vision_last': record.get('vision_last'),
           'vision_last_matches_historical_PPU': record.get('vision_last_matches_historical_PPU'),
           'precision': record['precision'], 'model_retained_in_process': True,
           'pid': os.getpid()}
write_new(FOLDER/'PPU_FIRST_BLOCK_TRACE_SUMMARY.json', summary)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
print('PPU_MODEL_RETAINED: model, backbone, inputs, pixels, record, trace are available in this Python process.', flush=True)
