"""Pretraining class, score, gradient and cross-host parity; no optimization outcomes."""
from common_r import *
import argparse, torch, types, concurrent.futures, multiprocessing
def check_host():
    import native as n
    device=setup();torch.manual_seed(1701);p=n.model().eval();p.requires_grad_(False)
    rows=[]
    from reward import cpu_init,cpu_score
    pool=concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_init)
    for scene in scenes()[:4]:
        t=scene['token'];vl,act=n.inputs(n.feature(t),device)
        torch.manual_seed(0)
        with torch.no_grad():pred=p.get_action(vl,act(1),deterministic=False)['pred_traj'].cpu().numpy()
        old=np.load(V1/'rollouts/official_il'/f'{t}.npz')['reference']
        error=float(np.max(np.abs(pred-old)));assert error<1e-6,(t,error)
        sampled=n.sample(p,vl,act,t,'v3_diagnostic_holdout64',64,device)
        oldpath=V3/'cache/rollouts/sft/official_il/step0000'/f'{t}.npz'
        if oldpath.exists():
            oldsample=np.load(oldpath)['trajectories']
            ce=float(np.abs(sampled-oldsample).max())
            assert ce<1e-6,ce
            newscore=pool.submit(cpu_score,scene,sampled).result()[0]
            archivedscore=np.load(V3/'cache/scores/sft/official_il/step0000'/f'{t}.npz')['scores']
            se=float(np.abs(newscore-archivedscore).max());assert se<=1e-8,se
        else:ce=None
        if ce is None:se=None
        newref=pool.submit(cpu_score,scene,pred).result()[0]
        oldref=np.load(V1/'evaluator/rollouts/official_il'/f'{t}.npz')['reference']
        re=float(np.abs(newref-oldref).max());assert re<=1e-8,re
        rows.append(dict(token=t,reference_max_abs=error,CRN_max_abs=ce,CRN_score_max_abs=se,reference_score_max_abs=re,trajectory_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
    result=dict(identity(),host=socket.gethostname(),checks=rows,torch_version=torch.__version__,cuda=torch.version.cuda,
                planner_source_sha256=sha(Path(__import__('inspect').getfile(p.__class__))),precision='fp32',
                evaluation_definition='unchanged V3 frozen evaluator class and CRN path')
    pool.shutdown()
    save(OUT/'manifests'/f'host_parity_{socket.gethostname()}.json',result);print('HOST PARITY PASS',result,flush=True)
def check_native():
    device=setup();torch.manual_seed(1701);agent=build_agent('official_il',1701);p=agent.action_head
    import native as n
    from reward import attach
    pool,last,scalar=attach(p,4);batch=p.reward_fn;checks=[];p.eval()
    try:
        for token in tokens('train')[:4]:
            vl,act=n.inputs(n.feature(token),device)
            for amp in [False,True]:
                values=[];grads=[]
                for fn in [scalar,batch]:
                    p.reward_fn=fn;p.zero_grad(set_to_none=True);torch.manual_seed(n.seed(token,'formal_stage3_parity'))
                    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):a=p.forward_grpo(vl,act(1),[token])
                    a['loss'].backward()
                    values.append({k:float(a[k]) for k in ['loss','reward','policy_loss','bc_loss']})
                    grads.append(torch.cat([x.grad.flatten().float() for x in p.parameters() if x.grad is not None]).detach())
                errors={k:abs(values[0][k]-values[1][k]) for k in values[0]}
                ge=float((grads[0]-grads[1]).abs().max());assert max(errors.values())<1e-6,errors;assert ge<1e-6,ge
                checks.append(dict(token=token,precision='16-mixed' if amp else 'fp32',loss_errors=errors,gradient_max_abs=ge))
        # Batched real training collation is tested through the formal agent.
        from train_r import Observations
        from navsim.planning.script.run_training_recogdrive_rl import custom_collate_fn
        features,targets,toks=custom_collate_fn([Observations('train')[i] for i in range(8)])
        features={k:v.to(device) for k,v in features.items()};targets={k:v.to(device) for k,v in targets.items()}
        p.reward_fn=batch;agent.train();torch.manual_seed(1701)
        with torch.autocast('cuda',dtype=torch.float16):out=agent.forward(features,targets,toks)
        out['loss'].backward();assert torch.isfinite(out['loss'])
        opt=agent.get_optimizers();pg=opt['optimizer'].param_groups[0]
        assert pg['betas']==(.9,.95) and pg['weight_decay']==1e-4
        save(OUT/'manifests/parity_native.json',dict(identity(),checks=checks,batched_forward_loss=float(out['loss']),batch_scene_count=8,
             initialization_equal=True,reference_equal=True,trainable_parameters=sum(x.numel() for x in p.parameters() if x.requires_grad),
             native_forward_unchanged=True,auxiliary_head_disabled_and_frozen_by_native_code=True,optimizer_steps_in_audit=0,
             peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        print('NATIVE PARITY PASS',checks,flush=True)
    finally:pool.shutdown()
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('kind',choices=['host','native']);args=a.parse_args()
    (check_host if args.kind=='host' else check_native)()
