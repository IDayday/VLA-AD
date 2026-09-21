"""Supplementary numerical edge audit; primary CUDA advantages unchanged."""
from shared import *
from metrics import reward
from native_audit import extract_advantage
import torch,types
def main():
    torch.set_num_threads(1);fn,_=extract_advantage(models()[CFG['models'][0]]['runtime_planner_path']);settings=types.SimpleNamespace(clip_advantage_lower_quantile=0.,clip_advantage_upper_quantile=1.)
    rows=[]
    for m in CFG['models']:
        s=np.stack([arrays(m,r['token'],True) for r in scenes()]);r=reward(s.reshape(-1,7)).reshape(5000,128).astype(np.float32)
        with np.load(OUT/'cache/advantages'/f'{m}.npz') as cache:
            for g in CFG['group_sizes']:
                matrix=r.reshape(-1,g);a=cache[f'G{g}'].reshape(-1,g);constant=np.ptp(matrix,axis=1)==0
                cpu=fn(torch.tensor(matrix).reshape(-1),len(matrix),g,settings).reshape(-1,g).numpy()
                discrep=abs(a-cpu).max(1);indices=np.flatnonzero(discrep>1e-3)[:16]
                scalar_error=[]
                for i in indices:
                    single=fn(torch.tensor(matrix[i],device='cuda'),1,g,settings).cpu().numpy()
                    scalar_error.append(float(abs(single-a[i]).max()))
                rows.append(dict(model=m,G=g,groups=len(matrix),constant_reward_groups=int(constant.sum()),cuda_nonzero_constant_groups=int((constant&(abs(a).max(1)>1e-8)).sum()),cpu_nonzero_constant_groups=int((constant&(abs(cpu).max(1)>1e-8)).sum()),cpu_cuda_large_difference_groups=int((discrep>1e-3).sum()),max_cpu_cuda_difference=float(discrep.max()),single_vs_batched_cuda_max_error=max(scalar_error,default=0.),scope='Numerical edge audit only; primary CUDA arrays unchanged.'))
    csv('advantage_numerical_audit.csv',rows)
if __name__=='__main__':main()
