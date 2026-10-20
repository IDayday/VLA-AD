"""Real-checkpoint and real-scene numerical admission; no dummy fallback."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import copy
import time
from unittest.mock import patch
from scripts.process_improvement.common import *


def parity(online=False):
    import torch
    import numpy as np
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head, state_hash
    from navsim.agents.recogdrive.process_improvement.suffix_rollout import NativeAdapter
    from navsim.agents.recogdrive.process_improvement.rng_plan import RNGKey
    from navsim.agents.recogdrive.process_improvement.objective import objective
    from scripts.opd.scoring import Scoring, load_cache
    configure()
    cfg = config()
    started = time.time()
    student = build_head(cfg['student_init'])
    teacher = build_head(cfg['teacher_init'], frozen=True)
    copied = copy.deepcopy(teacher)
    anchor = build_head(cfg['anchor_init'], frozen=True)
    frozen = [state_hash(h) for h in (teacher, anchor)]
    records = []
    data = membership('legacy_4k')[:4]+membership('dev')[:4]
    for number, row in enumerate(data):
        tokens, inputs = conditioning(observation(row))
        key = RNGKey(STUDY,42,0,0,row['token'],number,-1)
        noise = key.plan('rollin',device=tokens.device)
        for name, head in [('IL',student), ('RL',teacher), ('COPY',copied)]:
            adapter = NativeAdapter(head)
            for deterministic in (False,True):
                adapter.deterministic = deterministic
                rollin = adapter.sample_rollin(tokens, inputs, noise)
                with torch.no_grad():
                    context = adapter.prepare_conditioning(tokens, inputs)
                    for cut in range(6):
                        suffix, states = adapter.rollout_suffix(context,rollin.states[:,cut],cut,noise.steps,True)
                        assert torch.equal(suffix,rollin.trajectory), (name,number,cut)
                        assert torch.equal(states,rollin.states[:,cut:]), (name,number,cut,'states')
                # Native path receives exactly the same initial and step noises.
                captured = []
                original = head.p_mean_variance
                def capture(*args,**kwargs):
                    captured.append(args[0].detach().clone())
                    return original(*args,**kwargs)
                draws = iter(noise.steps[:,i] for i in range(5))
                with patch.object(head,'p_mean_variance',side_effect=capture), patch('torch.randn_like',side_effect=lambda x:next(draws).clone()):
                    with torch.no_grad():
                        native = head.get_action(tokens,inputs,deterministic=deterministic,init_actions=noise.initial.clone())['pred_traj']
                assert torch.equal(torch.stack(captured,1),rollin.states[:,:-1]),(name,'native intermediate mismatch')
                assert torch.equal(native,rollin.trajectory),(name,'native final mismatch')
                records.append(dict(token=row['token'],head=name,deterministic=deterministic,native_error=0.,suffix_error=0.))
            if name=='RL':
                reference_rollin = adapter.sample_rollin(tokens,inputs,noise)
            elif name=='COPY':
                value = adapter.sample_rollin(tokens,inputs,noise)
                assert torch.equal(value.states,reference_rollin.states)
        scorer = Scoring()
        cache = load_cache(row['metric_cache_path'])
        t = torch.cat([NativeAdapter(h).sample_rollin(tokens,inputs,noise).trajectory for h in [student,teacher,copied]]).cpu().numpy()
        canonical = np.stack([scorer.scalar(cache,v) for v in t])
        batched = scorer.batch(cache,t)['scores']
        assert np.max(np.abs(canonical-batched)) <= 1e-6
        for order in ([2,0,1],[0],[1,2]):
            result = scorer.batch(cache,t[order])['scores']
            assert np.max(np.abs(canonical[order]-result)) <= 1e-6
    row = data[0]
    tokens, inputs = conditioning(observation(row))
    key = RNGKey(STUDY,42,0,0,row['token'],0,-1)
    noise = key.plan('rollin',device='cuda')
    same = copy.deepcopy(teacher)
    same.requires_grad_(True)
    for name,p in same.named_parameters():
        if name.startswith(('eta.','action_aware_aux_head.')):p.requires_grad_(False)
    identical = objective(same,teacher,tokens,inputs,noise,0)
    assert float(identical['loss']) == 0.
    initial_chain = NativeAdapter(student).sample_rollin(tokens,inputs,noise)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4)
    out = objective(student,teacher,tokens,inputs,noise,0,anchor=anchor,anchor_weight=.1)
    out['loss'].backward()
    gradients = {name:float(p.grad.norm()) for name,p in student.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(p.grad).all() for p in student.parameters() if p.grad is not None)
    assert any(v>0 for k,v in gradients.items() if k.startswith('feature_encoder.'))
    assert sum(gradients.values()) > 0
    optimizer.step()
    updated_chain = NativeAdapter(student).sample_rollin(tokens,inputs,noise,1)
    assert not torch.equal(initial_chain.states,updated_chain.states)
    assert frozen == [state_hash(h) for h in (teacher,anchor)]
    assert all(p.grad is None for h in (teacher,anchor) for p in h.parameters())
    online_error = None
    if online:
        from scripts.opd.inputs import agent,features,encode
        model = agent(official=True)
        errors = []
        for row in data[:2]:
            actual = encode(model,features(model,row))
            expected = observation(row)['tokens']
            assert torch.equal(actual,expected)
            errors.append(float((actual.float()-expected.float()).abs().max()))
        online_error = max(errors)
    result = dict(status='PASS',records=records,heads={name:h._opd_load_report for name,h in [('IL',student),('RL',teacher)]},
                  same_teacher_loss=0.,online_cache_max_error=online_error,core_projection_gradient_nonzero=True,
                  teacher_anchor_unchanged=True,student_chain_refreshed=True,gradients=gradients,
                  timestep=student.ddim_t.cpu().tolist(),seconds=time.time()-started,owner=identity())
    write(ROOT/'tests/parity.json',result)
    print('REAL_PARITY_PASS',len(records),time.time()-started,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--config')
    args=parser.parse_args()
    parity(args.online)
