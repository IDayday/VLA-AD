"""Strict active-policy restoration. RL old_policy is never a teacher source."""
from pathlib import Path
import hashlib
import torch


def read_state(path):
    path=Path(path)
    if path.suffix=='.safetensors':
        from safetensors.torch import load_file
        value=load_file(str(path),device='cpu')
    else:value=torch.load(path,map_location='cpu',weights_only=False)
    for key in ('state_dict','policy','model','student'):
        if isinstance(value,dict) and key in value and isinstance(value[key],dict):value=value[key];break
    if not isinstance(value,dict) or not all(isinstance(v,torch.Tensor) for v in value.values()):
        raise ValueError('Checkpoint must contain a tensor state dictionary')
    return value


def restore(model,path):
    source=read_state(path);mapped={};ignored=[]
    for original,value in source.items():
        key=original
        for prefix in ('module.','agent.','action_head.'):
            if key.startswith(prefix):key=key[len(prefix):]
        if key.startswith(('old_policy.','reference_a0_policy.')):
            ignored.append(dict(key=original,reason='inactive RL/IL reference, never selected as active teacher'));continue
        if key.startswith('action_aware_aux_head.') and not model.config.use_action_aware_aux:
            ignored.append(dict(key=original,reason='disabled auxiliary head'));continue
        if key in mapped:raise ValueError(f'Ambiguous checkpoint key {key}')
        mapped[key]=value
    target=model.state_dict();inactive={k for k in target if k.startswith('action_aware_aux_head.') and not model.config.use_action_aware_aux}
    missing=sorted(set(target)-set(mapped)-inactive)
    unexpected=sorted(set(mapped)-set(target))
    mismatch=[dict(key=k,checkpoint=list(mapped[k].shape),model=list(target[k].shape)) for k in set(mapped)&set(target) if mapped[k].shape!=target[k].shape]
    report=dict(path=str(Path(path).resolve()),missing=missing,missing_all=sorted(set(target)-set(mapped)),
        missing_noncore=sorted((set(target)-set(mapped))&inactive),
        missing_semantics='missing lists required core tensors; missing_all includes explicitly disabled auxiliary targets left unrestored',
        unexpected=unexpected,shape_mismatch=mismatch,ignored=ignored,
        restored_tensors=len(set(mapped)&set(target)),dtypes=sorted({str(v.dtype) for v in mapped.values()}))
    model._opd_load_report=report
    if missing or unexpected or mismatch:raise ValueError(f'Strict checkpoint failure: {report}')
    model.load_state_dict(mapped,strict=False)
    for key,value in mapped.items():
        if not torch.equal(model.state_dict()[key].cpu(),value.to(model.state_dict()[key].dtype)):
            raise ValueError(f'Restored tensor mismatch: {key}')
    return report


def build_head(path,device='cuda',frozen=False):
    from navsim.agents.recogdrive.recogdrive_agent import make_recogdrive_config
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import ReCogDriveDiffusionPlanner
    cfg=make_recogdrive_config('small',action_dim=3,action_horizon=8,input_embedding_dim=384,
        vlm_feature_dim=1536,sampling_method='ddim',num_inference_steps=5,grpo=False,model_dtype='float32')
    for key in ('use_expert_features','use_last_rd','use_bit_drive','use_risk_vla','use_action_aware_aux','allow_random_init','allow_future_targets_in_inference'):
        setattr(cfg,key,False)
    cfg.last_rd_stage='disabled';cfg.policy_kd_mode='none';cfg.policy_kd_loss_weight=0.
    cfg.jepa_alignment_weight=0.;cfg.vggt_alignment_weight=0.;cfg.diffusion_loss_weight=1.
    head=ReCogDriveDiffusionPlanner(cfg).to(device=device,dtype=torch.float32)
    restore(head,path);head.eval()
    for name,p in head.named_parameters():
        p.requires_grad_(not frozen and not name.startswith(('action_aware_aux_head.','eta.')))
    return head


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if k.startswith('action_aware_aux_head.'):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
