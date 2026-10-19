"""Read actual rollout archives and measure auxiliary-gradient conflicts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def chain_distributions():
    import torch
    all_seeds=[]
    for seed in config()['train_seeds']:
        result=[]
        for step in [1,100,500,1000]:
            p=ROOT/'runs'/f'train_E2_{seed}'/'rollouts'/f'{step}_0.pt'
            q=ROOT/'runs'/f'train_E3_{seed}'/'rollouts'/f'{step}_0.pt'
            a=torch.load(p,map_location='cpu',weights_only=False);b=torch.load(q,map_location='cpu',weights_only=False)
            assert a['scene_tokens']==b['scene_tokens'] and a['seeds']==b['seeds']
            initial=float((a['states'][:,0]-b['states'][:,0]).abs().max());assert initial==0
            state_rms=(a['states']-b['states']).square().mean((0,2,3)).sqrt()
            response_rms=(a['teacher_means']-b['teacher_means']).square().mean((0,2,3)).sqrt()
            result.append(dict(step=step,initial_noise_max_error=initial,scene_tokens=a['scene_tokens'],
                state_RMS=(a['states']-b['states']).square().mean((0,2,3)).sqrt().tolist(),
                teacher_response_RMS=response_rms.tolist(),
                empirical_teacher_response_to_state_ratio=[float(y/x) if float(x)>1e-12 else None for x,y in zip(state_rms[:-1],response_rms)],
                E2_discrepancy=(a['student_means']-a['teacher_means']).square().mean((0,2,3)).tolist(),
                E3_discrepancy=(b['student_means']-b['teacher_means']).square().mean((0,2,3)).tolist()))
        all_seeds.append(dict(seed=seed,records=result))
    write(ROOT/'diagnostics/chain_distributions.json',dict(training_seeds=all_seeds,
        scope='Same actual training rows and initial noise; different teacher/student state distributions; no behavioral categories inferred'))
    # Already-saved, first-scheduled fit probes: no extra policy sampling or
    # favorable scene selection. One fixed draw per seed is descriptive only.
    from scripts.opd.scoring import Scoring,load_cache
    lookup={r['token']:r for r in rows('train')};probes=[]
    for experiment in ['E2','E3']:
        for seed in config()['train_seeds']:
            path=ROOT/'runs'/f'train_{experiment}_{seed}'/'policy_probes'
            original=torch.load(path/'0.pt',map_location='cpu',weights_only=False)
            row=lookup[original['token']];scorer=Scoring();cache=load_cache(row['metric_cache_path'])
            for step in [0,100,500,1000]:
                source=path/f'{step}.pt';value=torch.load(source,map_location='cpu',weights_only=False)
                assert value['token']==original['token'] and value['seed']==original['seed']
                d=value['trajectory']-original['trajectory'];metrics=scorer.scalar(cache,value['trajectory'].numpy()[0])
                probes.append(dict(experiment=experiment,training_seed=seed,step=step,token=row['token'],
                    normalized_state_RMS=float((value['states']-original['states']).square().mean().sqrt()),
                    xy_RMS_m=float(d[...,:2].square().mean().sqrt()),heading_RMS_rad=float(d[...,2].square().mean().sqrt()),
                    score=metrics.tolist(),PDMS=float(metrics[-1]),source=str(source),source_sha256=sha(source)))
    write(ROOT/'diagnostics/policy_refresh_probes.json',dict(records=probes,
        scope='Already-saved first scheduled fit input per training seed, fixed independent noise6101; one draw, descriptive only, not held-out policy comparison',no_new_policy_queries=True))


def gradients():
    import torch
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head
    from navsim.agents.recogdrive.opd.trainer import objective
    configure();cfg=config();teacher=build_head(cfg['teacher_init'],frozen=True);anchor=build_head(cfg['anchor_init'],frozen=True)
    chosen=sorted(rows('cal'),key=lambda r:digest([r['token'],'gradient_diagnostics']))[:16];records=[]
    for exp in ['E3','E4-A','E4-B','E4-C','E5']:
        student=build_head(ROOT/'runs'/f'train_{exp}_42'/'latest.pt');params=[p for p in student.parameters() if p.requires_grad]
        for row in chosen:
            tokens,inputs=conditioning(observation(row));seeds(seed_for(row['token'],9201,'gradient_diagnostics'))
            # Diagnostic GT never uses calibration future: only measure teacher/IL-anchor transitions here.
            out=objective(student,teacher,anchor,tokens,inputs,experiment='E4-A',rollout_seed=9201)
            vectors=[]
            for loss in [out['opd'],out['anchor']]:
                g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
                vectors.append(torch.cat([(v if v is not None else torch.zeros_like(p)).flatten() for p,v in zip(params,g)]))
            x,y=vectors;nx=float(x.norm());ny=float(y.norm());cos=float((x@y)/(x.norm()*y.norm()).clamp_min(1e-30))
            step_norms=[]
            for k in range(5):
                g=torch.autograd.grad(out['per_step'][:,k].mean()/5,params,retain_graph=True,allow_unused=True)
                step_norms.append(sum(float(v.square().sum()) for v in g if v is not None)**.5)
            records.append(dict(experiment=exp,token=row['token'],OPD_gradient_norm=nx,anchor_gradient_norm=ny,cosine=cos,
                per_step_gradient_norm=step_norms,loss_OPD=float(out['opd']),loss_anchor=float(out['anchor'])))
        del student
    write(ROOT/'diagnostics/gradients.json',dict(scope='Fixed16 calibration inputs, seed42 final models; diagnostic only, no checkpoint/loss selection',records=records,owner=identity()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['chains','gradients']);a=p.parse_args()
    if a.action=='chains':chain_distributions()
    else:gradients()
