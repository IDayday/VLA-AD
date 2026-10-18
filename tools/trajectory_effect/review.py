"""Post-completion numerical review of saved probes; no fitting or new labels.

The original protocol, report, audit labels and decisions are immutable inputs.
This review uses fixed D_fit/D_cal subsets to replay checkpoints, measure
candidate sensitivity and verify the shared native VLM path. It cannot admit RL.
"""
from __future__ import annotations

import argparse
import numpy as np
from .common import *
from .probe_runner import merge_eval, load_library, run_dir

DEST = ROOT/'review_20261019'
ARMS = ('T', 'FQ', 'FE', 'JQ', 'JE')


def register():
    cfg, identity = protocol()
    path = DEST/'registration.json'
    if path.exists():
        raise ValueError('Review registration already exists')
    selected = {}; backward = {}
    for split in ('D_fit', 'D_cal'):
        rows, _, _ = merge_eval('formal_FQ_42', split)
        chosen = sorted(rows, key=lambda r: digest([r['token'], 'numerical_review_20261019']))[:64]
        selected[split] = [r['token'] for r in chosen]
        backward[split] = max(chosen, key=lambda r: (
            float(np.ptp(load_library(r['token'])['labels'])), r['token']))['token']
    protected = [SOURCE/'configs/trajectory_effect_validation_20261019/protocol.yaml',
                 ROOT/'protocol.lock.json', ROOT/'analysis/decisions.json',
                 ROOT/'analysis/probes_formal_D_audit.json', ROOT/'analysis/accounting.json']
    protected += list((REPORT_DIR).rglob('*'))
    protected = [p for p in protected if p.is_file()]
    value = dict(protocol_sha256=identity, status='LOCKED_NUMERICAL_REVIEW',
        post_completion=True, purpose='Implementation and optimization diagnostics; not a new confirmatory comparison',
        subsets=selected, subset_rule='First64 saved evaluation targets by token hash, separately D_fit/D_cal; no labels in membership',
        backward_tokens=backward, backward_rule='Within each fixed subset, greatest true candidate score range; ties by token',
        training_seeds=cfg['seeds']['training'], arms=list(ARMS),
        limits=dict(optimizer_updates=0, new_scoring_labels=0, audit_forward_passes=0,
                    scene_model_pairs=64*2*5*3, backward_scenes_per_model=2),
        tolerances=dict(saved_prediction_absolute=2e-6, zero_adapter_tokens=0.,
                        repeated_candidate_scores=2e-6, loss_formula_absolute=2e-6),
        immutable_inputs={str(p):sha(p) for p in protected}, owner=runtime_identity())
    atomic_json(path,value)
    print(path)


def offline():
    reg=read(DEST/'registration.json'); summary={}; libraries={}
    for split in ('D_fit','D_cal'):
        all_y=[];all_d=[];ranges=[]
        for row in scene_rows(split):
            lib=load_library(row['token']);y=lib['labels'];a,b=lib['pairs'].T
            all_y.append(y);all_d.append(y[a]-y[b]);ranges.append(np.ptp(y))
        y=np.concatenate(all_y);d=np.concatenate(all_d)
        libraries[split]=dict(candidates=len(y),mean_score=float(y.mean()),score_std=float(y.std()),
            material_pair_fraction=float((abs(d)>.02).mean()),pair_RMS=float(np.sqrt(np.mean(d*d))),
            scenes=len(ranges),material_range_fraction=float((np.array(ranges)>.02).mean()))
        summary[split]={}
        for arm in ARMS:
            values=[]
            for seed in reg['training_seeds']:
                rows,pred,target=merge_eval(f'formal_{arm}_{seed}',split)
                q=np.array([r['predicted_base'] for r in rows]);y=np.array([r['base_score'] for r in rows])
                hist=read(run_dir(f'formal_{arm}_{seed}')/'progress.json')
                values.append(dict(seed=seed,scenes=len(rows),absolute_Q_RMSE=float(np.sqrt(np.mean((q-y)**2))),
                    predicted_delta_std=float(pred.std()),target_delta_std=float(target.std()),
                    predicted_delta_max_abs=float(abs(pred).max()),
                    prediction_to_target_std_ratio=float(pred.std()/target.std()),
                    first_logged_L_abs=hist[0]['L_abs'],last_logged_L_abs=hist[-1]['L_abs'],
                    last_logged_L_diff=hist[-1]['L_diff'],
                    max_vlm_gradient_norm=read(run_dir(f'formal_{arm}_{seed}')/'complete.json')['max_vlm_gradient_norm']))
            summary[split][arm]=values
    atomic_json(DEST/'offline.json',dict(protocol_sha256=reg['protocol_sha256'],library_distributions=libraries,
        saved_evaluation_diagnostics=summary,use='Descriptive numerical review; no hyperparameter/checkpoint selection'))
    print(DEST/'offline.json')


def replay(args):
    import torch
    from .bank import load_observation, baseline_path, baseline_score_path
    from .probe_runner import build,load_adapter
    from .probes import effect_loss,joint_training_mode,vlm_parameters
    from .runtime import configure,encode,action_input,predict
    from .updates import trajectory_path,score_path,tensor_digest
    reg=read(DEST/'registration.json');configure()
    if args.arm not in reg['arms'] or args.seed not in reg['training_seeds']:
        raise ValueError('Unregistered replay')
    name=f'formal_{args.arm}_{args.seed}';dest=DEST/name
    if (dest/'complete.json').exists():raise ValueError('Replay already complete')
    start=time.perf_counter();owner=runtime_identity();atomic_json(dest/'owner.json',owner)
    path=run_dir(name)/'checkpoint.pt';before_hash=sha(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert before_hash==read(run_dir(name)/'complete.json')['checkpoint_sha256']
    agent,head=build(args.arm,args.seed);head.load_state_dict(checkpoint['head']);head.eval()
    initial_head=tensor_digest(head.state_dict())
    if agent is not None:
        if not isinstance(agent.action_head.vlm_pre_norm,torch.nn.Identity):
            raise ValueError('Feature hook would capture transformed rather than raw tokens')
        load_adapter(agent,checkpoint['vlm']);agent.eval()
    output=[];gradients=[];shared=[];max_replay=0.;max_formula=0.;max_repeat=0.
    for split,tokens in reg['subsets'].items():
        saved_rows,_,_=merge_eval(name,split);saved={r['token']:r for r in saved_rows}
        for index,token in enumerate(tokens):
            payload=load_observation(token);inputs=action_input(payload['features']);value=load_library(token)
            trajectories=torch.as_tensor(value['trajectories'],device='cuda')[None]
            labels=torch.as_tensor(value['labels'],device='cuda')[None]
            pairs=torch.as_tensor(value['pairs'],device='cuda')
            with torch.no_grad():
                context=encode(agent,payload['features']) if agent is not None else payload['tokens'].cuda()
                q=head(context,trajectories,inputs.state,remove_visual=args.arm=='T')
                _,la,ld=effect_loss(q,labels,pairs,checkpoint['config']['lambda_diff'])
                q_np=q.cpu().numpy()[0]
                row=saved[token];old=np.load(baseline_path(token))['trajectories'][:5]
                old_y=np.load(baseline_score_path(token))['scores'][:5,-1]
                endpoint=[old];true=[]
                for family in row['families']:
                    for step in (1,4):
                        endpoint.append(np.load(trajectory_path(family,step,token))['trajectories'])
                        true.append(np.load(score_path(family,step,token))['scores'][:,-1]-old_y)
                qe=head(context,torch.as_tensor(np.concatenate(endpoint),device='cuda')[None],inputs.state,
                        remove_visual=args.arm=='T').cpu().numpy()[0].reshape(-1,5)
                replay_error=max(float(abs((qe[1:]-qe[0]).reshape(np.array(row['prediction']).shape)-row['prediction']).max()),
                                 float(abs(qe[0]-row['predicted_base']).max()))
                assert np.array_equal(np.array(true).reshape(np.array(row['target']).shape),np.array(row['target']))
                max_replay=max(max_replay,replay_error)
                a,b=value['pairs'].T
                formula=max(abs(float(la)-np.mean((q_np-value['labels'])**2)),
                    abs(float(ld)-np.mean(((q_np[a]-q_np[b])-(value['labels'][a]-value['labels'][b]))**2)))
                max_formula=max(max_formula,float(formula))
                if token==reg['backward_tokens'][split]:
                    same=trajectories[:,:1].expand_as(trajectories)
                    repeated=head(context,same,inputs.state,remove_visual=args.arm=='T')
                    max_repeat=max(max_repeat,float((repeated-repeated[:,:1]).abs().max()))
            output.append(dict(split=split,token=token,prediction=q_np.tolist(),labels=value['labels'].tolist(),
                pairs=value['pairs'].tolist(),absolute_loss=float(la),difference_loss=float(ld),
                candidate_std_pred=float(q_np.std()),candidate_std_true=float(value['labels'].std()),
                saved_prediction_max_error=replay_error))
            if token==reg['backward_tokens'][split]:
                if agent is not None:joint_training_mode(agent)
                head.zero_grad(set_to_none=True)
                context=encode(agent,payload['features'],grad=True) if agent is not None else payload['tokens'].cuda()
                q=head(context,trajectories,inputs.state,remove_visual=args.arm=='T')
                _,la,ld=effect_loss(q,labels,pairs,checkpoint['config']['lambda_diff'])
                groups=[('query',list(head.query.parameters())),('head',list(head.parameters()))]
                if agent is not None:groups.append(('vlm',vlm_parameters(agent)))
                params=list(head.parameters())+(vlm_parameters(agent) if agent is not None else [])
                ga=torch.autograd.grad(la,params,retain_graph=True,allow_unused=True)
                gd=torch.autograd.grad(ld,params,allow_unused=True)
                lookup={id(p):i for i,p in enumerate(params)};stats={}
                for key,ps in groups:
                    indices=[lookup[id(p)] for p in ps]
                    na=sum(float(ga[i].float().square().sum()) for i in indices if ga[i] is not None)**.5
                    nd=sum(float(gd[i].float().square().sum()) for i in indices if gd[i] is not None)**.5
                    dot=sum(float((ga[i].float()*gd[i].float()).sum()) for i in indices if ga[i] is not None and gd[i] is not None)
                    stats[key]=dict(absolute_gradient_norm=na,difference_gradient_norm=nd,
                        cosine=dot/(na*nd) if na*nd else None)
                gradients.append(dict(split=split,token=token,groups=stats))
                del ga,gd,q,la,ld,context
                if agent is not None:
                    agent.eval();baseline_state={n:torch.zeros_like(v) if n.endswith('.b') else v for n,v in checkpoint['vlm'].items()}
                    load_adapter(agent,baseline_state)
                    with torch.no_grad():base_context=encode(agent,payload['features'])
                    parity=float((base_context-payload['tokens'].cuda()).abs().max())
                    load_adapter(agent,checkpoint['vlm'])
                    with torch.no_grad():new_context=encode(agent,payload['features'])
                    old_action=predict(agent.action_head,base_context,inputs,token,1101)
                    new_action=predict(agent.action_head,new_context,inputs,token,1101)
                    shared.append(dict(split=split,token=token,cached_token_max_error=parity,
                        token_change_max=float((new_context-base_context).abs().max()),
                        token_change_RMS=float((new_context-base_context).float().square().mean().sqrt()),
                        action_change_xy_RMS_m=float(np.sqrt(np.mean((new_action[:,:2]-old_action[:,:2])**2))),
                        action_change_heading_RMS_rad=float(np.sqrt(np.mean((new_action[:,2]-old_action[:,2])**2)))))
            if (index+1)%16==0:print('REPLAY',name,split,index+1,flush=True)
    tolerance=reg['tolerances']
    errors=[]
    if max_replay>tolerance['saved_prediction_absolute']:errors.append('Saved prediction replay mismatch')
    if max_formula>tolerance['loss_formula_absolute']:errors.append('Loss formula mismatch')
    if max_repeat>tolerance['repeated_candidate_scores']:errors.append('Identical candidates differ')
    if any(v['cached_token_max_error']>tolerance['zero_adapter_tokens'] for v in shared):errors.append('Zero adapter/native cache mismatch')
    if any(v['token_change_max']==0 or v['action_change_xy_RMS_m']==0 for v in shared):errors.append('Joint VLM did not change shared action features')
    if any(v['groups']['query']['difference_gradient_norm']==0 for v in gradients):errors.append('Trajectory query gradient disconnected')
    assert tensor_digest(head.state_dict())==initial_head
    assert sha(path)==before_hash
    if agent is not None:
        current=dict(agent.backbone.named_parameters())
        assert all(torch.equal(current[n].detach().cpu(),v) for n,v in checkpoint['vlm'].items())
    atomic_json(dest/'results.json',dict(rows=output,gradient_checks=gradients,shared_feature_checks=shared))
    atomic_json(dest/'complete.json',dict(status='PASS' if not errors else 'INVALID',errors=errors,
        protocol_sha256=reg['protocol_sha256'],registration_sha256=sha(DEST/'registration.json'),
        checkpoint_sha256=before_hash,model=name,scene_count=len(output),
        saved_prediction_max_error=max_replay,loss_formula_max_error=max_formula,
        repeated_candidates_max_error=max_repeat,optimizer_updates=0,new_labels=0,
        seconds=time.perf_counter()-start,owner=owner))
    if errors:raise RuntimeError(errors)
    print('REPLAY_COMPLETE',name,flush=True)


def summarize():
    reg=read(DEST/'registration.json');rows=[]
    for arm in ARMS:
        for seed in reg['training_seeds']:
            name=f'formal_{arm}_{seed}';complete=read(DEST/name/'complete.json')
            if complete['status']!='PASS':raise RuntimeError(f'Invalid replay {name}')
            data=read(DEST/name/'results.json')
            for split in reg['subsets']:
                v=[r for r in data['rows'] if r['split']==split];y=np.concatenate([r['labels'] for r in v]);q=np.concatenate([r['prediction'] for r in v])
                d=[];pred=[];centered=[];centered_truth=[]
                for r in v:
                    qq=np.array(r['prediction']);yy=np.array(r['labels']);a,b=np.array(r['pairs']).T
                    pred.extend(qq[a]-qq[b]);d.extend(yy[a]-yy[b]);centered.extend(qq-qq.mean());centered_truth.extend(yy-yy.mean())
                d=np.array(d);pred=np.array(pred)
                base=read(DEST/'offline.json')['library_distributions']['D_fit']['mean_score']
                rows.append(dict(arm=arm,seed=seed,split=split,scenes=len(v),candidates=len(y),
                    absolute_Q_RMSE=float(np.sqrt(np.mean((q-y)**2))),
                    fit_constant_Q_RMSE=float(np.sqrt(np.mean((base-y)**2))),
                    pair_RMSE=float(np.sqrt(np.mean((pred-d)**2))),zero_pair_RMSE=float(np.sqrt(np.mean(d*d))),
                    within_scene_std_ratio=float(np.std(centered)/np.std(centered_truth)),
                    saved_prediction_max_error=complete['saved_prediction_max_error']))
    protected=[p for p,h in reg['immutable_inputs'].items() if not Path(p).exists() or sha(p)!=h]
    if protected:raise RuntimeError(f'Original protected artifacts changed: {protected}')
    atomic_json(DEST/'summary.json',dict(status='PASS',scope='Post-completion numerical validity; descriptive only',
        protocol_sha256=reg['protocol_sha256'],registration_sha256=sha(DEST/'registration.json'),
        original_artifacts_unchanged=True,rows=rows,
        interpretation='Saved outputs and labels replay. The learned Q varies much less across real candidates than their rewards; this is not evidence that the frozen representation is sufficient. No RL gate or hyperparameter is changed.'))
    print(DEST/'summary.json')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['register','offline','replay','summarize'])
    p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
    if args.action=='replay':replay(args)
    else:globals()[args.action]()


if __name__=='__main__':main()
