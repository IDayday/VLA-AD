"""Quality and actual fixed groups; execute the archived GRPO advantage code."""
import argparse, ast, types
import pandas as pd
from common import *

def advantage_function(out):
    import torch
    path=ARCHIVE/'official_recogdrive/navsim/agents/recogdrive/recogdrive_diffusion_planner.py'
    source=path.read_text();lines=source.splitlines()
    begin=next(i for i,l in enumerate(lines) if 'mean_r = rewards_matrix.mean(dim=1' in l)
    end=next(i for i in range(begin,len(lines)) if 'advantages = advantages.clamp' in lines[i])
    snippet='\n'.join(l[8:] for l in lines[begin:end+1])
    # The historical job has no quantile override; defaults are 0 and 1.
    save(out/'manifests/grpo_advantage_source.json',dict(path=str(path),sha256=sha(path),first_line=begin+1,last_line=end+1,source=snippet,clip_quantiles=[0.,1.],reward_weights=dict(progress=10,ttc=5,comfort=2,driving_direction=0),reward_precision='float32',reward_scale='0 to 1'))
    compiled=compile(snippet,str(path),'exec')
    def execute(rewards):
        env=dict(torch=torch,rewards_matrix=torch.tensor(rewards,dtype=torch.float32),self=types.SimpleNamespace(clip_advantage_lower_quantile=0.,clip_advantage_upper_quantile=1.))
        exec(compiled,env);return env['advantages'].numpy().reshape(rewards.shape)
    return execute

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];advantage=advantage_function(out);rows=[];groups=[]
    for r in sc['scenes']:
        token=r['token'];ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])*100
        for m in MODELS:
            s=np.load(out/'evaluator/rollouts'/m/f'{token}.npz')['trajectories'];reward=s[:,6]*100;f=feasible(s);h=hard_failure(s)
            g=cfg['group_size'];rg=reward.reshape(-1,g);fg=f.reshape(-1,g)
            # Exact archived reward_fn uses score with 10/5/2 weights; standard report PDMS uses 5/5/2.
            training=s[:,0]*s[:,1]*(s[:,2]*10+s[:,3]*5+s[:,4]*2)/17
            advantages=advantage(training.reshape(-1,g));pos=advantages>0;unsafe=~fg&pos;safe=fg&pos
            npos=int(pos.sum());oracles={f'oracle_{k}':float(reward[:min(k,len(reward))].max()) for k in [8,16,32,64] if k<=len(reward)}
            stats=dict(token=token,log=r['log'],command=r['command'],checkpoint=m,mean_pdms=float(reward.mean()),median_pdms=float(np.median(reward)),feasible_rate=float(f.mean()),hard_failure_rate=float(h.mean()),reference_pdms=ref,**oracles,oracle_mean_gap=float(reward.max()-reward.mean()),all_infeasible_group_rate=float((fg.sum(1)==0).mean()),mostly_infeasible_group_rate=float((fg.sum(1)<=2).mean()),unsafe_positive_advantage=float(unsafe.sum()/npos) if npos else np.nan,feasible_positive_advantage=float(safe.sum()/npos) if npos else np.nan,positive_advantage_count=npos,unsafe_positive_count=int(unsafe.sum()),feasible_positive_count=int(safe.sum()))
            stats.update({f'p{q}_pdms':float(np.percentile(reward,q)) for q in [10,25,75,90]})
            for delta in cfg['reward_deltas_points']:
                positive=f&(reward>ref+delta);stats[f'p_plus_{delta}']=float(positive.mean());stats[f'hit8_{delta}']=float(positive.reshape(-1,g).any(1).mean())
            rows.append(stats)
            for i in range(len(rg)):
                groups.append(dict(token=token,checkpoint=m,group=i,reward_std=float(rg[i].std(ddof=1)),best_minus_median=float(rg[i].max()-np.median(rg[i])),best_minus_worst=float(np.ptp(rg[i])),feasible_count=int(fg[i].sum()),positive_advantage_count=int(pos[i].sum()),feasible_positive_count=int(safe[i].sum()),training_reward_std=float(training.reshape(-1,g)[i].std(ddof=1))))
    dest=out/'metrics';dest.mkdir(exist_ok=True)
    for name,values in [('grpo_readiness',rows),('grpo_groups',groups)]:
        df=pd.DataFrame(values);df.to_csv(dest/(name+'.csv'),index=False);df.to_parquet(dest/(name+'.parquet'),index=False)
    print('Readiness scene rows:',len(rows),'groups:',len(groups),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
