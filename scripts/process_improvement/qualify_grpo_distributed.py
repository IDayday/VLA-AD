"""Real native GRPO global-loss partition and same-world exact-resume checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.process_improvement.common import *


def main():
    import torch
    base=['scripts/process_improvement/train_grpo.py','--qualification','--init','IL','--split','legacy_4k',
          '--steps','4','--global-batch','8','--score-budget','256','--lr','1e-5']
    parallel=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8']
    for command in [[PYTHON,*base,'--name','qual_GRPO_single'],
                    [*parallel,*base,'--name','qual_GRPO_continuous8'],
                    [*parallel,*base,'--name','qual_GRPO_resume8','--stop-after','2'],
                    [*parallel,*base,'--name','qual_GRPO_resume8','--resume']]:
        subprocess.run(command,cwd=SOURCE,check=True)
    states={name:torch.load(ROOT/'runs'/('qual_GRPO_'+name)/'latest.pt',map_location='cpu',weights_only=False)
            for name in ['single','continuous8','resume8']}
    left=states['continuous8'];errors={}
    for name in ['single','resume8']:
        right=states[name];maximum=0.
        assert left['step']==right['step']==4 and left['physical_scores']==right['physical_scores']==256
        for key,value in left['full_student'].items():
            other=right['full_student'][key]
            if key.startswith('action_aware_aux_head.'):continue
            if name=='resume8':assert torch.equal(value,other),key
            else:torch.testing.assert_close(value,other,atol=2e-6,rtol=1e-5)
            if value.dtype.is_floating_point and torch.isfinite(value).all():maximum=max(maximum,float((value-other).abs().max()))
        errors[name]=maximum
    for key,value in left['optimizer']['state'].items():
        for field,tensor in value.items():
            if isinstance(tensor,torch.Tensor):assert torch.equal(tensor,states['resume8']['optimizer']['state'][key][field])
    write(ROOT/'tests/native_GRPO_distributed_resume.json',dict(status='PASS',errors=errors,exact_resume_optimizer=True,
          world_sizes=[1,8],global_batch=8,actual_optimizer_updates=4,physical_scores_per_run=256,owner=identity()))
    print('NATIVE_GRPO_DISTRIBUTED_RESUME_PASS',errors,flush=True)


if __name__=='__main__':main()
