"""CPU-only validation that unmatched conditions stop before model arithmetic."""
from pathlib import Path
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode=True
FOLDER=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('controlled_gate_validation',FOLDER/'controlled_conditions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=json.loads((FOLDER/'PPU_CONTROLLED_CONFIGURATION.json').read_text())
assert module.compare_configurations(reference,copy.deepcopy(reference))['portable_conditions_match']
cases=[]
def check(name,mutation):
    actual=copy.deepcopy(reference);mutation(actual['portable_conditions'])
    result=module.compare_configurations(reference,actual)
    assert result['status']=='BLOCKED_CONFIGURATION_MISMATCH' and result['portable_differences'],name
    cases.append(name)
check('TF32 flag',lambda c:c['numeric_policy'].__setitem__('tf32_cudnn',False))
check('CPU threads',lambda c:c['numeric_policy'].__setitem__('num_threads',1))
check('inference mode',lambda c:c['numeric_policy'].__setitem__('inference_mode_enabled',True))
check('norm epsilon',lambda c:c['vision_modules']['encoder.layers.0.norm1']['scalars'].__setitem__('eps',1e-5))
check('normalized shape',lambda c:c['vision_modules']['encoder.layers.0.norm1']['scalars'].__setitem__('normalized_shape',[2048]))
check('input stride',lambda c:c['actual_model_and_auxiliary_inputs']['pixel_values'].__setitem__('stride',[602112,200704,1,448]))
check('raw JPG bytes',lambda c:c['source_data_files']['raw_front.jpg'].__setitem__('sha256','0'*64))
check('actual BF16 input bytes',lambda c:c['actual_model_and_auxiliary_inputs']['pixel_values'].__setitem__('sha256','0'*64))
check('actual norm weight bytes',lambda c:c['all_vision_parameters']['encoder.layers.0.norm1.weight'].__setitem__('sha256','0'*64))
check('checkpoint file bytes',lambda c:c['weight_files']['model.safetensors'].__setitem__('sha256','0'*64))
check('functional norm source',lambda c:c['relevant_operator_source']['functional_layer_norm'].__setitem__('object_source_sha256','0'*64))
check('common package version',lambda c:c['common_package_versions'].__setitem__('transformers','4.37.0'))
check('training state',lambda c:c.__setitem__('model_training',True))
check('missing scalar',lambda c:c['numeric_policy'].pop('default_dtype'))
check('JSON boolean type',lambda c:c['numeric_policy'].__setitem__('tf32_cudnn',1))
invalid=copy.deepcopy(reference);invalid['capture_validation']['all_12_actual_inputs_match_repository']=False
assert not module.compare_configurations(reference,invalid)['portable_conditions_match']
backend=copy.deepcopy(reference);backend['backend_identity']['torch_git']='different_vendor_build';backend['backend_identity']['cudnn_runtime_version']=90700
backend_gate=module.compare_configurations(reference,backend)
assert backend_gate['portable_conditions_match'] and backend_gate['backend_differences']
assert backend_gate['all_external_conditions_identical'] is False

# A passing metadata vector is insufficient if repository identity failed;
# every mismatch case must return without accessing a model or running an op.
class NoArithmetic:
    def __getattr__(self,name):
        raise AssertionError('Model touched despite failed gate: '+name)
original_capture=module.capture_configuration
blocked=copy.deepcopy(reference);blocked['portable_conditions']['numeric_policy']['tf32_cudnn']=False
module.capture_configuration=lambda *args,**kwargs:blocked
try:
    with tempfile.TemporaryDirectory(prefix='a800-control-gate-') as directory,contextlib.redirect_stdout(io.StringIO()):
        result=module.run_controlled_check(NoArithmetic(),{},auxiliary={},role_tensors={},
            bundle_dir='/not-used',vlm_dir='/not-used',frozen_repo='/not-used',
            folder=FOLDER,output_dir=directory)
        assert not result['portable_conditions_match']
        assert {p.name for p in Path(directory).iterdir()}=={'A800_CONTROLLED_CONFIGURATION.json','A800_CONTROLLED_GATE.json'}
finally:
    module.capture_configuration=original_capture

# The explicit settings helper must refuse an existing CUDA process before
# changing even one environment variable; this test uses no actual GPU/model.
previous_torch=sys.modules.get('torch');before=dict(os.environ)
sys.modules['torch']=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True))
try:
    try:module.configure_fresh_receiver(FOLDER/'PPU_CONTROLLED_CONFIGURATION.json')
    except RuntimeError as exc:assert 'fresh' in str(exc)
    else:raise AssertionError('Existing CUDA process accepted')
    assert dict(os.environ)==before
finally:
    if previous_torch is None:sys.modules.pop('torch',None)
    else:sys.modules['torch']=previous_torch

portable=reference['portable_conditions']
old=json.loads((FOLDER/'PPU_VISION_PARAMETER_METADATA.json').read_text())
assert len(portable['all_vision_parameters'])==340
assert len(portable['vision_modules'])==341
for name,info in portable['all_vision_parameters'].items():
    assert all(info[key]==old[name][key] for key in ['shape','dtype','stride','sha256'])
assert reference['capture_source_sha256']==hashlib.sha256((FOLDER/'controlled_conditions.py').read_bytes()).hexdigest()
assert portable['weight_files']['model.safetensors']['sha256']=='79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c'
assert reference['numerical_validation_reference']['norm1_input']['sha256']=='ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74'
for filename in ['controlled_conditions.py','run_a800_controlled_visual.py']:
    compile((FOLDER/filename).read_text(),str(FOLDER/filename),'exec')
result={'status':'PASS','scope':'CPU-only metadata/gate validation; no model or GPU arithmetic',
    'mismatch_cases_blocked':cases,'invalid_input_identity_blocks':True,
    'failed_gate_returns_before_any_model_access':True,
    'backend_build_differences_are_disclosed_without_hardware_only_claim':True,
    'fresh_settings_helper_refuses_active_CUDA_before_mutation':True,
    'all_340_actual_parameter_fingerprints_match_previous_verified_capture':True,
    'source_and_checkpoint_reference_integrity':True}
(FOLDER/'CONTROLLED_GATE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
