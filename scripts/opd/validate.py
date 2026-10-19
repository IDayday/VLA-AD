"""Real checkpoint integration tests, separate from research performance claims."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def batch():
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.opd.ddim_adapter import DDIMAdapter,Chain
    from navsim.agents.recogdrive.opd.losses import transition_loss
    configure();cfg=config();lengths=read(ROOT/'manifests/token_lengths.json');groups={}
    for r in rows('train'):groups.setdefault(lengths[r['token']],[]).append(r)
    chosen=max(groups.values(),key=len)[:8];pairs=[conditioning(observation(r)) for r in chosen]
    tokens=torch.cat([p[0] for p in pairs]);inputs=BatchFeature({k:torch.cat([p[1][k] for p in pairs]) for k in pairs[0][1]})
    s=build_head(cfg['student_init']);t=build_head(cfg['teacher_init'],frozen=True);a=DDIMAdapter(s)
    params=[p for p in s.parameters() if p.requires_grad];initial=[];noises=[]
    for r in chosen:
        g=torch.Generator(device='cuda').manual_seed(seed_for(r['token'],3101,'distill_chain'))
        initial.append(torch.randn((1,8,3),device='cuda',generator=g));noises.append(torch.stack([torch.randn((1,8,3),device='cuda',generator=g) for _ in range(5)],1))
    initial=torch.cat(initial);noises=torch.cat(noises)
    start=time.perf_counter();chain=a.sample(tokens,inputs,init=initial,noises=noises)
    with torch.no_grad():target=DDIMAdapter(t).evaluate_chain(chain,tokens,inputs)
    pred=a.evaluate_chain(chain,tokens,inputs,require_current=True);loss=transition_loss(pred,target)[0]
    grad=torch.autograd.grad(loss,params,allow_unused=True);torch.cuda.synchronize();batched=time.perf_counter()-start
    scalar_grad=[torch.zeros_like(p) for p in params];max_state=0.;max_mean=0.;max_traj=0.;scalar_loss=0.
    start=time.perf_counter()
    for i,pair in enumerate(pairs):
        scalar=a.sample(pair[0],pair[1],init=initial[i:i+1],noises=noises[i:i+1])
        max_state=max(max_state,float((scalar.states-chain.states[i:i+1]).abs().max()));max_traj=max(max_traj,float((scalar.trajectory-chain.trajectory[i:i+1]).abs().max()))
        fixed=Chain(chain.states[i:i+1],chain.means[i:i+1],chain.sigmas[i:i+1],chain.timesteps,chain.indices,
            chain.trajectory[i:i+1],chain.versions,chain.source,False,'native')
        with torch.no_grad():truth=DDIMAdapter(t).evaluate_chain(fixed,*pair)
        value=a.evaluate_chain(fixed,*pair,require_current=True)
        max_mean=max(max_mean,float((value-pred[i:i+1]).abs().max()))
        l=transition_loss(value,truth)[0];scalar_loss+=float(l)/8
        gs=torch.autograd.grad(l/8,params,allow_unused=True)
        for j,g in enumerate(gs):
            if g is not None:scalar_grad[j].add_(g)
    torch.cuda.synchronize();scalar_seconds=time.perf_counter()-start
    numerator=sum(float((x-y).square().sum()) for x,y in zip(grad,scalar_grad) if x is not None)
    denominator=sum(float(y.square().sum()) for y in scalar_grad)
    relative=(numerator/max(denominator,1e-30))**.5
    assert max_state<1e-5 and max_mean<1e-5 and max_traj<1e-4 and relative<1e-4,(max_state,max_mean,max_traj,relative)
    # Inactive future/expert fields must not affect the baseline forward.
    tokens0,inputs0=pairs[0];dirty=BatchFeature(dict(inputs0,jepa_target_tokens=torch.full((1,12,1024),float('nan'),device='cuda'),vggt_target_tokens=torch.full((1,12,2048),float('nan'),device='cuda')))
    seeds(713)
    with torch.no_grad():clean=s.get_action(tokens0,inputs0)['pred_traj']
    seeds(713)
    with torch.no_grad():poison=s.get_action(tokens0,dirty)['pred_traj']
    assert torch.equal(clean,poison)
    assert not hasattr(s,'old_policy')
    write(ROOT/'tests/real_batch.json',dict(passed=True,scenes=[r['token'] for r in chosen],native_future_poison_error=0.,
        max_state_error=max_state,max_mean_error=max_mean,max_trajectory_error=max_traj,relative_gradient_error=relative,
        batched_loss=float(loss),scalar_loss=scalar_loss,batched_seconds=batched,scalar_seconds=scalar_seconds,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),owner=identity(),scope='Batching validation; not PDMS evidence'))
    print('REAL_BATCH_PASS',max_state,max_mean,max_traj,relative,batched,scalar_seconds,flush=True)


def grpo():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,state_hash
    from navsim.agents.recogdrive.opd.trainer import configure_native_grpo,grpo_with_opd
    cfg=config();configure();candidates=[]
    for r in rows('train'):
        p=ROOT/'evaluation/train/E0/scores'/f"{r['token']}.json"
        if p.exists():
            v=read(p)['scores']['IL']['scores'];mean=np.mean(np.array(v)[:,-1])
            if mean<.8:candidates.append((mean,r))
    if not candidates:raise RuntimeError('No real low-score qualification inputs yet')
    r=sorted(candidates,key=lambda x:digest([x[1]['token'],'grpo_validation']))[0][1]
    tokens,inputs=conditioning(observation(r));s=build_head(cfg['student_init']);t=build_head(cfg['teacher_init'],frozen=True)
    s=configure_native_grpo(s,[r],cfg['anchor_init']);s.eval();params=[p for p in s.parameters() if p.requires_grad]
    seeds(7721);raw=s.forward_grpo(tokens,inputs,[r['token']],sample_time=8);g1=torch.autograd.grad(raw.loss,params,allow_unused=True)
    seeds(7721);zero,_=grpo_with_opd(s,t,tokens,inputs,[r['token']],weight=0);g2=torch.autograd.grad(zero.loss,params,allow_unused=True)
    assert torch.equal(raw.loss,zero.loss)
    error=max(float((x-y).abs().max()) for x,y in zip(g1,g2) if x is not None and y is not None);assert error==0
    before_t=state_hash(t);before_s=state_hash(s)
    seeds(7721);joint,aux=grpo_with_opd(s,t,tokens,inputs,[r['token']],weight=.1)
    gn_grpo=sum(float(x.square().sum()) for x in g1 if x is not None)**.5
    gn_opd=torch.autograd.grad(aux['opd'],params,retain_graph=True,allow_unused=True)
    gn_opd=sum(float(x.square().sum()) for x in gn_opd if x is not None)**.5
    joint.loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.));opt=torch.optim.AdamW(params,lr=1e-5);opt.step()
    assert gn_opd>0 and np.isfinite(norm) and state_hash(t)==before_t and state_hash(s)!=before_s
    write(ROOT/'tests/real_grpo.json',dict(passed=True,token=r['token'],optimizer_updates=1,
        zero_weight_loss_error=0.,zero_weight_gradient_error=error,GRPO_gradient_norm=gn_grpo,OPD_gradient_norm=gn_opd,
        combined_gradient_norm=norm,policy_loss=float(raw.policy_loss),BC_loss=float(raw.bc_loss),OPD_loss=float(aux['opd']),
        teacher_unchanged=True,native_BC_weight=s.bc_coeff if hasattr(s,'bc_coeff') else .1,
        formal_E6=False,owner=identity(),scope='Real integration only; not a teacher-ceiling performance experiment'))
    print('REAL_GRPO_PASS',gn_grpo,gn_opd,norm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['batch','grpo'],required=True);a=p.parse_args();globals()[a.mode]()
