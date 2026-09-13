"""Equivalent parallel NAVSIM scoring and observation-only diagnostic logging."""
from common_r import *
import concurrent.futures, multiprocessing, types
def cpu_init():
    from scoring import init
    init()
def cpu_score(scene, trajectories):
    from scoring import online
    return online(scene,trajectories)
def attach(p, workers=4):
    import torch
    pool=concurrent.futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=cpu_init)
    lookup={r['token']:r for r in scenes()};allowed=set(tokens('train'));last={}
    def reward(self,trajs,toks,metric_cache):
        assert set(toks)<=allowed and len(toks)%8==0
        a=trajs.detach().cpu().numpy();groups=[toks[b] for b in range(0,len(toks),8)]
        assert all(len(set(toks[b:b+8]))==1 for b in range(0,len(toks),8))
        futures=[pool.submit(cpu_score,lookup[t],a[j*8:(j+1)*8]) for j,t in enumerate(groups)]
        result=[f.result() for f in futures];s=np.concatenate([x[0] for x in result]);r=np.concatenate([x[1] for x in result])
        out=torch.as_tensor(r[:,6],device=trajs.device,dtype=trajs.dtype).detach()
        matrix=out.view(-1,8);adv=((matrix-matrix.mean(1,keepdim=True))/(matrix.std(1,keepdim=True)+1e-8)).view(-1)
        adv=adv.clamp(torch.quantile(adv,self.clip_advantage_lower_quantile),torch.quantile(adv,self.clip_advantage_upper_quantile))
        last.update(scores=s,rewards=r,advantages=adv.cpu().numpy(),tokens=groups,trajectories=a,
                    rng_after_sampling=digest(torch.cuda.get_rng_state().cpu().tolist()))
        return out
    original=p.reward_fn;p.reward_fn=types.MethodType(reward,p)
    return pool,last,original
