"""Verify actual tensor identity, not only checkpoint filenames, at every controlled start."""
from common_v3 import *
import torch
def state_hash(path):
    state=torch.load(path,map_location='cpu',weights_only=False)['state_dict'];h=hashlib.sha256()
    for key in sorted(state):
        v=state[key].detach().cpu().numpy();h.update(key.encode());h.update(str(v.shape).encode());h.update(str(v.dtype).encode());h.update(v.tobytes())
        if key!='eta.eta_logit':assert np.isfinite(v).all(),key
    return h.hexdigest()
def main():
    reference=None;rows=[]
    for method in CFG['training']['methods']:
        for sd in CFG['training']['seeds']:
            run=f'{method}_seed{sd}';s0=OUT/'checkpoints/sft'/run/'step0000.pt';sf=OUT/'checkpoints/sft'/run/'step0200.pt';g0=OUT/'checkpoints/grpo'/run/'step0000.pt';a=state_hash(s0);b=state_hash(sf);c=state_hash(g0)
            if reference is None:reference=a
            assert a==reference and b==c
            rows.append(dict(method=method,seed=sd,SFT_step0_tensor_sha256=a,SFT_final_tensor_sha256=b,GRPO_step0_tensor_sha256=c,exact_SFT_to_GRPO_identity=True))
    save(OUT/'manifests/INITIALIZATION_TENSOR_AUDIT.json',dict(identity=identity(),rows=rows,all_SFT_starts_identical=True,all_GRPO_starts_equal_respective_SFT_final=True))
if __name__=='__main__':main()
