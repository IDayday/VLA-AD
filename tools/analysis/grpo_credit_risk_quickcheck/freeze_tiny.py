from common import *
from pairwise_credit import require_update_gate
require_update_gate(read(OUT/'audits/SMOKE_COMPLETE.json'),read(OUT/'audits/signal_gate.json'))
p=OUT/'manifests/tiny_update.json';assert not p.exists()
from tiny_support import order
seeds=cfg()['tiny_update']['seeds'];save(p,dict(status='FROZEN_BEFORE_UPDATE',gate_sha256=sha(OUT/'audits/signal_gate.json'),config_sha256=sha(OUT/'resolved_config.yaml'),seeds=seeds,
 scene_order={str(s):{str(step):[r['token'] for r in order(s,step)] for step in range(1,9)} for s in seeds},
 arms=dict(A='archived LFP advantage and original loss',B='EP numerator 10->5, denominator17 retained; selected reference scalar recomputed likewise, nonlinear reference margin and centering/moments/Pareto recomputed; original EP/safety/progress eligibility values retained; official PDMS untouched',C='original plus symmetric nominal-safe pairwise risk, fixed G-1 denominator',D='C plus simulator raw progress <=.5m; action-dependent surrogate'),
 lambda_risk=1.,train_risk_stream='training_v1 from stage2 frozen rule, independent from selection_v1 and evaluation_v1',
 effective_batch=64,microbatch_scenes=1,trajectory_group_size=16,gradient_accumulation_scenes=64,device_allocation='one GPU per arm/seed, eight parallel jobs',
 historical_replay=False,execution_difference='Same effective batch64, G16, loss and global advantage moments; serialize single-scene gradients instead of original 8-GPU DDP microbatch8; floating-point reduction order differs. No batch/learning-rate difference between arms.',
 optimizer=dict(type='AdamW',lr=1e-5,betas=[.9,.95],weight_decay=1e-4,gradient_clip_norm=1.,precision='16-mixed',grad_scaler='PyTorch default65536; nonfinite gradients stop arm'),
 scheduler='Archived epoch-based WarmupCosLR(10 epochs,1 warmup,min1e-6); initial epoch0 lr1e-5 fixed for 8 steps, no epoch scheduler advancement',
 reference='fixed A5 epoch155; exact archived reverse-transition KL .005; BC0; cached VLM frozen',
 saves=[0,1,8],main_endpoint=8,first_step='one shared real rollout/chain cache per seed; all arms read exactly same arrays and scores',later_steps='resample each own current behavior; never reuse previous policy actions',
 evaluation=dict(scenes='frozen256 confirmation',native='two G16 groups',deployment='two pools16 actual get_action sampler',risk='8 evaluation_v1 draws per sampled candidate, no training seeds',nominal='official scalar unchanged',norm_control='single-step original A update displacement rescaled to D actual trainable-parameter displacement norm'),
 stop='nonfinite loss/logprob/gradient, stale policy identity or invalid numeric score stops arm; no adjusted rerun'))
print('TINY CONFIG FROZEN')
