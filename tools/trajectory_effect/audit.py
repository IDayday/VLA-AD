"""Real-scene qualification and cost measurement, separate from confirmation."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .common import *


def inference(args):
    from .runtime import load_agent, features_for, encode, action_input, predict
    import torch
    start = time.perf_counter()
    agent = load_agent(); load_seconds = time.perf_counter()-start
    rows = scene_rows('pilot')[:args.limit]
    out = ROOT / 'pilot'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'inference_owner.json', runtime_identity())
    timings = []; checks = []
    for i, row in enumerate(rows):
        token = row['token']; features = features_for(agent, row)
        begin = time.perf_counter()
        tokens = encode(agent, features)
        torch.cuda.synchronize(); encoding = time.perf_counter()-begin
        inputs = action_input(features)
        begin = time.perf_counter()
        seeds = list(range(1101, 1106)) + list(range(2101, 2106))
        trajectories = np.stack([predict(agent.action_head, tokens, inputs, token, seed) for seed in seeds])
        again = predict(agent.action_head, tokens, inputs, token, seeds[0])
        assert np.array_equal(trajectories[0], again), 'Same noise/model differs'
        # Compare the cached native-head input against the whole native agent path.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            seed = seed_for(token, seeds[0]); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
            native = agent.forward(features)['pred_traj'].float().cpu().numpy()[0]
        native_error = float(np.max(np.abs(native-trajectories[0])))
        assert native_error == 0, ('Native input-cache parity', token, native_error)
        torch.cuda.synchronize(); prediction_seconds = time.perf_counter()-begin
        # Qualification probes test score batch invariance, not H1 or H2.
        probes = []
        for factor in (.5, 1.5, 2.0):
            p = trajectories[0].copy(); p[:, :2] *= factor; probes.append(p)
        for sign in (-1, 1):
            p = trajectories[0].copy(); p[:, 1] += sign * 4 * (np.arange(1, 9)/8)**2; probes.append(p)
        all_trajectories = np.concatenate([trajectories, again[None], np.stack(probes)])
        metadata = dict(token=token, log=row['log_name'], phase='qualification_only',
                        seeds=seeds, exact_repeat_index=10, native_path_max_error=native_error)
        atomic_npz(out/'trajectories'/f'{token}.npz', metadata, trajectories=all_trajectories)
        payload = dict(features={k:v.cpu() for k,v in features.items()}, tokens=tokens.detach().cpu(),
                       token=token, checkpoint_sha256=sha(RL) if i == 0 else checks[0]['checkpoint_sha256'],
                       input_contract='native_current_front_and_four_ego_history_no_targets')
        path = out/'observations'/f'{token}.pt'; path.parent.mkdir(exist_ok=True)
        torch.save(payload, path)
        timings.append(dict(token=token, encoding_seconds=encoding,
                            sampling_and_native_repeat_seconds=prediction_seconds, bytes=path.stat().st_size,
                            token_shape=list(tokens.shape)))
        checks.append(dict(token=token, native_path_max_error=native_error,
                           repeated_trajectory_max_error=float(abs(again-trajectories[0]).max()),
                           checkpoint_sha256=payload['checkpoint_sha256']))
        atomic_json(out/'inference_progress.json', dict(done=i+1, total=len(rows), latest=timings[-1]))
        print('INFERENCE', i+1, len(rows), 'seconds', round(time.perf_counter()-start, 2), flush=True)
    atomic_json(out/'inference_complete.json', dict(status='PASS', scenes=len(rows),
        checkpoint=dict(path=str(RL), sha256=checks[0]['checkpoint_sha256'], verified_tensors=agent._effect_verified_tensors),
        load_seconds=load_seconds, total_seconds=time.perf_counter()-start, rows=timings, checks=checks,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), observation_targets_present=False,
        inference_seeds=list(range(1101,1106)), independent_comparison_seeds=list(range(2101,2106))))


def score_one(row):
    from .scoring import Scoring, load_cache
    start = time.perf_counter(); scorer = Scoring(); cache = load_cache(row['metric_cache_path'])
    source = ROOT/'pilot/trajectories'/f"{row['token']}.npz"
    tr = np.load(source)['trajectories']
    batch = scorer.batch(cache, tr)
    scalar = np.stack([scorer.scalar(cache, t) for t in tr])
    reverse = scorer.batch(cache, tr[::-1])['scores'][::-1]
    partitioned = np.concatenate([scorer.batch(cache, tr[j:j+3])['scores'] for j in range(0,len(tr),3)])
    repeat = scorer.scalar(cache, tr[0])
    errors = dict(batch_vs_scalar=float(abs(batch['scores']-scalar).max()),
                  order=float(abs(batch['scores']-reverse).max()),
                  partitions=float(abs(batch['scores']-partitioned).max()),
                  repeat=float(abs(scalar[0]-repeat).max()),
                  zero_update=float(abs(scalar[0]-scalar[10]).max()))
    discrete = [0,1,3,4,5]
    discrete_equal = bool(np.array_equal(batch['scores'][:,discrete], scalar[:,discrete]))
    valid = max(errors.values()) <= 1e-6 and discrete_equal
    atomic_npz(ROOT/'pilot/scores'/f"{row['token']}.npz",
        dict(token=row['token'], input_sha256=sha(source), score_fields=FIELDS, status='PASS' if valid else 'INVALID'),
        **batch, scalar_scores=scalar)
    return dict(token=row['token'], log=row['log_name'], status='PASS' if valid else 'INVALID', errors=errors,
        discrete_equal=discrete_equal, seconds=time.perf_counter()-start,
        naive_batch_error=float(abs(batch['naive_batch_scores']-scalar[:,-1]).max()),
        base_scores=scalar[:5,-1], independent_scores=scalar[5:10,-1],
        baseline_components=scalar[:5].mean(0))


def score(args):
    require_host(); rows = scene_rows('pilot')[:args.limit]
    completed = read(ROOT/'pilot/inference_complete.json')
    assert completed['scenes'] >= len(rows)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        values = list(pool.map(score_one, rows))
    status = 'PASS' if all(r['status']=='PASS' for r in values) else 'INVALID'
    means = np.array([np.mean(r['base_scores']) for r in values])
    self_delta = np.array([np.mean(r['independent_scores'])-np.mean(r['base_scores']) for r in values])
    atomic_json(ROOT/'pilot/score_complete.json', dict(status=status, scenes=len(rows), rows=values,
        wall_seconds=time.perf_counter()-start, workers=args.workers,
        max_errors={k:max(r['errors'][k] for r in values) for k in values[0]['errors']},
        naive_batch_max_error=max(r['naive_batch_error'] for r in values),
        independent_seed_self_comparison=dict(mean_delta=float(self_delta.mean()),
            mean_abs_scene_delta=float(abs(self_delta).mean()),
            scene_delta_quantiles=np.quantile(self_delta,[0,.05,.5,.95,1]),
            single_draw_sd=float(np.mean([np.std(r['base_scores'],ddof=1) for r in values]))),
        score_coverage=dict(high=int((means>=.9).sum()), low=int((means<.9).sum()), zero=int((means<=1e-12).sum())),
        events=dict(collision='native earliest at-fault collision index',
                    dac='derived earliest scorer NON_DRIVABLE_AREA index',
                    ttc='native first TTC check failure, not physical time remaining to collision')))
    print(status, 'score seconds', time.perf_counter()-start,
          'max score error', max(r['errors']['batch_vs_scalar'] for r in values), flush=True)
    if status != 'PASS': raise RuntimeError('Scoring gate failed')


def train(args):
    assert read(ROOT/'pilot/score_complete.json')['status'] == 'PASS'
    import torch
    from .runtime import load_agent, action_input, configure_grpo, trainable_head, state_digest
    rows = scene_rows('pilot')[:args.limit]
    agent = load_agent(online=False); head = configure_grpo(agent.action_head, rows)
    params = trainable_head(head); optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
    row = rows[0]; payload = torch.load(ROOT/'pilot/observations'/f"{row['token']}.pt", weights_only=False)
    inputs = action_input(payload['features']); tokens = payload['tokens'].cuda()
    head.eval(); before = state_digest(head)
    torch.manual_seed(7300); torch.cuda.manual_seed(7300)
    original_reward = head.reward_fn; records = {}
    def reward(*a, **kw):
        result = original_reward(*a, **kw); records['rewards'] = result.detach().cpu().tolist(); return result
    head.reward_fn = reward
    begin = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
    output = head.forward_grpo(tokens, inputs, [row['token']], sample_time=8)
    output.loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    optimizer.step(); torch.cuda.synchronize(); seconds=time.perf_counter()-begin
    after = state_digest(head)
    if not np.isfinite(grad_norm): raise ValueError('Nonfinite training gradient')
    path=ROOT/'pilot/training_state.pt'
    torch.save(dict(model=head.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), updates=1, qualification_only=True),path)
    atomic_json(ROOT/'pilot/train_complete.json',dict(status='PASS',new_training=True,updates=1,
        not_historical_resume=True, scene=row['token'], seconds=seconds, gradient_norm=grad_norm,
        pre_state_sha256=before, post_state_sha256=after, parameters_changed=(before!=after),
        policy_loss=float(output.policy_loss), bc_loss=float(output.bc_loss), reward=records,
        reward_std=float(np.std(records['rewards'])), group_size=8,
        optimizer_state_bytes=path.stat().st_size, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        no_formal_data=True, objective='native forward_grpo; native progress weight10/TTC5/comfort2; BC0.1'))
    print('TRAIN',seconds,grad_norm,records,flush=True)


def joint(args):
    assert read(ROOT/'pilot/score_complete.json')['status']=='PASS'
    import torch
    from .runtime import load_agent, encode, action_input, predict
    from .probes import TrajectoryScoreHead, adapt_vlm, joint_training_mode, vlm_parameters, vlm_state, effect_loss
    row=scene_rows('pilot')[0];token=row['token']
    payload=torch.load(ROOT/'pilot/observations'/f'{token}.pt',weights_only=False)
    bank=np.load(ROOT/'pilot/trajectories'/f'{token}.npz')['trajectories']
    labels=np.load(ROOT/'pilot/scores'/f'{token}.npz')['scores'][:,-1]
    agent=load_agent(); initial=encode(agent,payload['features'])
    torch.manual_seed(42); names=adapt_vlm(agent)
    adapted=encode(agent,payload['features'])
    assert torch.equal(initial,adapted), 'Zero LoRA changes native tokens'
    head=TrajectoryScoreHead().cuda();joint_training_mode(agent)
    before=vlm_state(agent); optimizer=torch.optim.AdamW([
        {'params':head.parameters(),'lr':3e-4}, {'params':vlm_parameters(agent),'lr':1e-4}],weight_decay=.01)
    torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
    output=encode(agent,payload['features'],grad=True)
    inputs=action_input(payload['features']); tr=torch.from_numpy(bank).cuda()[None]
    pred=head(output,tr,inputs.state)
    pairs=torch.tensor([[0,1],[1,2],[0,11],[0,12],[0,13],[0,14],[0,15]],device='cuda')
    loss,absolute,difference=effect_loss(pred,torch.tensor(labels,device='cuda')[None],pairs,1.)
    loss.backward()
    vlm_grad=sum(float(p.grad.float().square().sum()) for p in vlm_parameters(agent) if p.grad is not None)**.5
    head_grad=sum(float(p.grad.float().square().sum()) for p in head.parameters() if p.grad is not None)**.5
    torch.nn.utils.clip_grad_norm_([*head.parameters(),*vlm_parameters(agent)],1.); optimizer.step()
    torch.cuda.synchronize();seconds=time.perf_counter()-begin
    changed=[n for n,p in vlm_state(agent).items() if not torch.equal(p,before[n])]
    assert vlm_grad>0 and changed, 'VLM auxiliary gradient/update missing'
    agent.eval();new_tokens=encode(agent,payload['features'])
    token_change=float((new_tokens-initial).abs().max())
    old_tr=predict(agent.action_head,initial,inputs,token,1101)
    new_tr=predict(agent.action_head,new_tokens,inputs,token,1101)
    atomic_json(ROOT/'pilot/joint_complete.json',dict(status='PASS',qualification_only=True,
        updates=1,scene=token,seconds=seconds,vlm_gradient_norm=vlm_grad,head_gradient_norm=head_grad,
        trainable_vlm_parameters=sum(p.numel() for p in vlm_parameters(agent)),
        head_parameters=sum(p.numel() for p in head.parameters()),adapted_projections=names,
        changed_parameter_names=changed,max_token_change=token_change,
        max_action_change_m_or_rad=float(abs(new_tr-old_tr).max()),
        loss_absolute=float(absolute),loss_difference=float(difference),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        input_path='recomputed native online VLM; frozen observation cache not used for trainable encoding',
        precision='native BF16 base; FP32 LoRA master and score head; TF32 off; no dropout'))
    print('JOINT',seconds,vlm_grad,token_change,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',choices=['inference','score','train','joint'],required=True)
    p.add_argument('--limit',type=int,default=64);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    try: globals()[args.phase](args)
    except Exception:
        atomic_json(ROOT/'pilot'/f'{args.phase}_failure.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
