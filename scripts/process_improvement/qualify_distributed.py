"""Real single-vs-multiple GPU update reduction and exact same-world resume."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import subprocess
from scripts.process_improvement.common import *


def main():
    import torch
    base=['scripts/process_improvement/train.py','--qualification','--split','legacy_4k','--steps','4',
          '--global-batch','4','--micro-batch','1']
    runs=[('continuous',[PYTHON,*base,'--name','qual_exact_continuous']),
          ('paused',[PYTHON,*base,'--name','qual_exact_resume','--stop-after','2']),
          ('resumed',[PYTHON,*base,'--name','qual_exact_resume','--resume']),
          ('distributed',[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',*base,'--name','qual_exact_ddp'])]
    for name,command in runs:
        print('RUN_QUALIFICATION',name,flush=True)
        result=subprocess.run(command,cwd=SOURCE)
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    a=torch.load(ROOT/'runs/qual_exact_continuous/latest.pt',map_location='cpu',weights_only=False)
    b=torch.load(ROOT/'runs/qual_exact_resume/latest.pt',map_location='cpu',weights_only=False)
    c=torch.load(ROOT/'runs/qual_exact_ddp/latest.pt',map_location='cpu',weights_only=False)
    assert a['cursor']==b['cursor']==c['cursor']==16
    errors={}
    for label,value in [('resume',b),('ddp',c)]:
        maximum=0.
        for name,tensor in a['student'].items():
            if name.startswith('action_aware_aux_head.'):continue
            other=value['student'][name]
            if label=='resume':assert torch.equal(tensor,other),name
            else:torch.testing.assert_close(tensor,other,atol=2e-6,rtol=1e-5,equal_nan=True)
            if tensor.dtype.is_floating_point and torch.isfinite(tensor).all():maximum=max(maximum,float((tensor-other).abs().max()))
        errors[label]=maximum
    for key,value in a['optimizer']['state'].items():
        for name,tensor in value.items():
            if isinstance(tensor,torch.Tensor):assert torch.equal(tensor,b['optimizer']['state'][key][name])
    write(ROOT/'tests/distributed_resume.json',dict(status='PASS',errors=errors,
          exact_resume_optimizer=True,global_batch=4,world_sizes=[1,2],microbatch=1,owner=identity()))
    print('DISTRIBUTED_RESUME_PASS',errors,flush=True)


if __name__=='__main__':main()
