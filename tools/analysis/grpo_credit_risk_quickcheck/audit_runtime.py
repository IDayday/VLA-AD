"""Record resolved archived callable/config identity, without starting training."""
from common import *
import inspect,dataclasses,torch
install_runtime()
from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig,ReCogDriveDiffusionPlanner
from navsim.agents.recogdrive.stage3_lfp_grpo import compute_lfp_advantages
from credit import setup
from scoring import PDMScorerConfig
from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter
m=read(OUT/'manifests/models.json');run=read(OUT/'manifests/archived_training_config.json');a=run['agent'];g=GRPOConfig()
# Exact mapping through archived ReCogDriveAgent constructor is snapshotted too.
for k in vars(g):
 if 'grpo_'+k in a:setattr(g,k,a['grpo_'+k])
 elif k in ['metric_cache_path','reference_policy_checkpoint','bc_anneal','bc_coeff_start','bc_coeff_end','bc_anneal_epochs','reference_kl_coeff','reference_kl_chunk_size'] and k in a:setattr(g,k,a[k])
g.sample_time=a['grpo_sample_time']
l,ref=setup();checks={}
for name,fun in [('forward_grpo',ReCogDriveDiffusionPlanner.forward_grpo),('forward_lfp_grpo',ReCogDriveDiffusionPlanner.forward_lfp_grpo),('sampler_train',ReCogDriveDiffusionPlanner.sample_chain),('sampler_deploy',ReCogDriveDiffusionPlanner.get_action),('advantage',compute_lfp_advantages),('evaluate_rollouts',ReCogDriveDiffusionPlanner._evaluate_lfp_rollouts),('trajectory_logprob_reduce',ReCogDriveDiffusionPlanner._reduce_chain_logprobs),('transition_kl',ReCogDriveDiffusionPlanner._chain_transition_reference_kl)]:
 source=inspect.getsource(fun);p=OUT/'source_snapshots'/f'callable_{name}.py';p.write_text(source);checks[name]=dict(path=inspect.getfile(fun),line=inspect.getsourcelines(fun)[1],sha256=sha(inspect.getfile(fun)),source_snapshot_sha256=sha(p))
assert 'forward_lfp_grpo' in inspect.getsource(ReCogDriveDiffusionPlanner.forward_grpo)
assert g.min_logprob_denoising_std==g.min_sampling_denoising_std==.04
assert g.scorer_config.progress_weight==10 and g.scorer_config.ttc_weight==5 and g.scorer_config.comfortable_weight==2
assert l.ttc_positive_credit_guard and l.progress_gate_enabled and l.pareto_gate_enabled
snapshot=dict(status='PASS',actual_call_chain='forward_grpo -> forward_lfp_grpo -> compute_lfp_advantages',callables=checks,grpo_config=dataclasses.asdict(g),lfp_config=dataclasses.asdict(l),scorer_training=dataclasses.asdict(g.scorer_config),scorer_official=dataclasses.asdict(PDMScorerConfig()),
 loss='-mean(detached clipped LFP advantage * discounted_mean reverse-chain Gaussian log-prob) + 0.005 exact transition reference KL',BC=0,KL=.005,reference_policy_checkpoint=a['reference_policy_checkpoint'],reference_metric_path=l.reference_cache_path,reference_metric_sha256=sha(l.reference_cache_path),
 normalization_and_gate_order=['training scalar','nonlinear reference margin','feasibility/progress/TTC/Pareto eligibility','within-group feasible mean if >=2 else all-candidate mean','global masked moments on centered scores, std floor .05','z=(centered-global_mean)/std','positive credit restricted to eligibility; nonqualified feasible min(z,0); infeasible -1 if any feasible else rescue','clip +/-3','detach'],
 reference_policy_identity=m['a5_sft']['sha256'],scene_sampling='Historical epoch0 uniform curriculum warmup. Probe uses predeclared log/command hash subset and fixed hash minibatches, equal scene/group weights; not historical sampling replay.',effective_batch=64,world_size=8,precision='16-mixed',trainable_scope='Archived action head including PTA/FS-DiT; cached VLM remains frozen. No modules trainable in this read-only probe.',
 identity_limit=read(OUT/'manifests/frozen.json')['runtime_limit'],scorer_fields='NAVSIM v1 NC/DAC product; TTC,Comfort,EP additive; DDC used for actual LFP feasibility guard with filtered scorer field; raw DDC unavailable in archived reference',recomputed_advantages='fresh samples only',initial_moments_batch='fixed 64 scene subsets, 2 groups separately, per model/protocol')
save(OUT/'manifests/resolved_runtime.json',snapshot)
print('RESOLVED RUNTIME PASS')
