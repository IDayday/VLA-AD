import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,copy,importlib.metadata
from scripts.opd.common import *


def main():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.opd.ddim_adapter import DDIMAdapter
    from navsim.agents.recogdrive.opd.losses import transition_loss
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--device',default='cuda');a=p.parse_args()
    configure();cfg=config(a.config);start=time.perf_counter();owner=identity();ROOT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    env=dict(owner=owner,python=sys.version,versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','scipy','pytest','safetensors')},
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True),
        data=read(ROOT/'manifests/identity.json'),initial_HEAD='884ea1d0f67c60821d61033d61efb10916d16929')
    write(ROOT/'audit/environment.json',env)
    student=build_head(cfg['student_init'],a.device);teacher=build_head(cfg['teacher_init'],a.device,True);anchor=build_head(cfg['anchor_init'],a.device,True)
    copied=build_head(cfg['teacher_init'],a.device,True)
    weights={name:dict(path=cfg[k],sha256=sha(cfg[k]),load_report=h._opd_load_report) for name,k,h in [('IL','student_init',student),('RL','teacher_init',teacher),('COPY','teacher_init',copied)]}
    weights['VLM']=dict(path=cfg['vlm'],sha256=sha(Path(cfg['vlm'])/'model.safetensors'),deployment_shared=True,
        limitation='IL/RL action-head checkpoints omit VLM tensors; original training VLM cannot be proved from these files alone')
    write(ROOT/'audit/checkpoints.json',weights)
    for key in ['ddim_t','ddim_alphas','ddim_alphas_prev','ddim_sqrt_one_minus_alphas','ddim_sigmas']:
        assert torch.equal(getattr(student,key),getattr(teacher,key)),key
    architecture=dict(parameters=sum(p.numel() for p in student.parameters() if p.requires_grad),
        flags={k:getattr(student.config,k) for k in ('use_expert_features','use_last_rd','use_bit_drive','use_risk_vla','use_action_aware_aux','sampling_method','num_inference_steps','input_embedding_dim','vlm_feature_dim','action_horizon','action_dim')},
        timesteps=student.ddim_t.tolist(),indices=list(range(student.ddim_steps)),
        saved_sigma_buffer=student.ddim_sigmas.tolist(),eta_state={k:(v.tolist() if torch.isfinite(v).all() else str(v)) for k,v in student.eta.state_dict().items()},
        sampler_note='get_action deterministic: sigma=0; stochastic: actual EtaFixed then minimum1e-4, clipped standard-normal noise[-1,1]; final latent clip[-1,1]. sample_chain used by GRPO has a distinct minimum0.04. Gaussian KL rejected for clipped kernels.')
    write(ROOT/'audit/architecture.json',architecture)
    checks=[];times=[]
    for row in rows('train')[:4]+rows('cal')[:4]:
        payload=observation(row);tokens,inputs=conditioning(payload,a.device)
        for deterministic in (True,False):
            seed=seed_for(row['token'],777);traces=[];original=teacher.p_mean_variance
            def record(x,t,index,*args,**kw):
                result=original(x,t,index,*args,**kw);traces.append((x.detach().clone(),result[0].detach().clone()));return result
            teacher.p_mean_variance=record
            seeds(seed)
            with torch.no_grad():native=teacher.get_action(tokens,inputs,deterministic=deterministic)['pred_traj']
            teacher.p_mean_variance=original
            seeds(seed);adapted=DDIMAdapter(teacher).sample(tokens,inputs,deterministic=deterministic)
            seeds(seed);replica=DDIMAdapter(copied).sample(tokens,inputs,deterministic=deterministic)
            errors=dict(native_final=float((native-adapted.trajectory).abs().max()),
                native_states=float((torch.stack([r[0] for r in traces],1)-adapted.states[:,:-1]).abs().max()),
                native_means=float((torch.stack([r[1] for r in traces],1)-adapted.means).abs().max()),
                copy_states=float((adapted.states-replica.states).abs().max()),copy_final=float((adapted.trajectory-replica.trajectory).abs().max()))
            assert max(errors.values())==0,errors
            assert torch.allclose(teacher.denorm_odo(teacher.norm_odo(native)),native,rtol=1e-6,atol=1e-5)
            checks.append(dict(token=row['token'],deterministic=deterministic,errors=errors,sigmas=adapted.sigmas.flatten(2)[:,:,0].cpu().tolist()))
    # Real gradients and refresh; targets/rollouts detached, teacher/anchor immutable.
    payload=observation(rows('train')[0]);tokens,inputs=conditioning(payload,a.device)
    before_teacher=state_hash(teacher);before_anchor=state_hash(anchor);before_student=state_hash(student)
    seeds(991);chain=DDIMAdapter(student).sample(tokens,inputs)
    with torch.no_grad():target=DDIMAdapter(teacher).evaluate_chain(chain,tokens,inputs)
    pred=DDIMAdapter(student).evaluate_chain(chain,tokens,inputs,require_current=True)
    loss,per_step=transition_loss(pred,target)
    same,_=transition_loss(pred,pred.detach());assert float(same)==0
    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.))
    assert norm>0 and np.isfinite(norm)
    assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in anchor.parameters())
    opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4)
    opt.step();student.zero_grad(set_to_none=True)
    with torch.no_grad():after=float(transition_loss(DDIMAdapter(student).evaluate_chain(chain,tokens,inputs),target)[0])
    seeds(991);refreshed=DDIMAdapter(student).sample(tokens,inputs)
    refresh=float((refreshed.states-chain.states).abs().max());assert refresh>0
    try:DDIMAdapter(student).evaluate_chain(chain,tokens,inputs,require_current=True)
    except ValueError:stale_rejected=True
    else:raise AssertionError('Stale chain admitted')
    assert state_hash(teacher)==before_teacher and state_hash(anchor)==before_anchor and state_hash(student)!=before_student
    elapsed=time.perf_counter()-start
    result=dict(status='PASS',G0=True,G1=True,checks=checks,identical_loss=float(same),student_grad_norm=norm,
        initial_loss=float(loss),fixed_state_loss_after_one_update=after,per_step_loss=per_step.detach().cpu().tolist(),
        refreshed_chain_max_change=refresh,stale_rejected=stale_rejected,teacher_unchanged=True,anchor_unchanged=True,
        seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=owner)
    write(ROOT/'audit/gates.json',result)
    for name,title,data in [('00_environment_audit.md','环境与数据审计',env),('01_architecture_audit.md','原生动作规划器与DDIM审计',architecture),('02_checkpoint_audit.md','严格权重加载审计',weights),('03_algorithm_validation.md','真实权重与场景算法门禁',result)]:
        (REPORTS/name).write_text('# '+title+'\n\n```json\n'+json.dumps(data,ensure_ascii=False,indent=2)+'\n```\n')
    print('AUDIT_PASS',elapsed,result['initial_loss'],after,refresh,flush=True)

if __name__=='__main__':main()
