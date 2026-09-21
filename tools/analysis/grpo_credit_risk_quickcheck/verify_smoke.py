from common import *
paths=[OUT/'audits'/x for x in ['features_smoke_0.json','sampling_a5_sft_smoke_0.json','sampling_a5_grpo_300_smoke_0.json','scoring_smoke.json']]
for p in paths:
 x=read(p);assert x['status']=='PASS'
 if 'state_before' in x:assert x['state_before']==x['state_after']
 if p.name.startswith('sampling'):assert len(x['checks'])==32 and x['group_size']==16
 if p.name=='scoring_smoke.json':assert len(x['results'])==16 and len(x['errors'])==0
f=read(OUT/'manifests/frozen.json');assert f['config_sha256']==sha(OUT/'resolved_config.yaml');assert f['scene_manifest_sha256']==sha(OUT/'manifests/scenes.json')
save(OUT/'audits/SMOKE_COMPLETE.json',dict(status='PASS',checks={str(p):sha(p) for p in paths},scenes=16,optimizer_steps=0,current_reconstructed_identity='PASS',historical_bit_exact_identity='NOT_PROVABLE: original hidden cache and dirty diff unavailable',historical_precision='BF16 VLM compute -> FP32 feature storage; FP16 autocast action head native GRPO; FP32 deployment',numerical_tolerance=1e-8))
print('SMOKE PASS')
