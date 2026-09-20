"""Reuse the already parity-audited observation-only FP32 feature path."""
from common_matched import *
import torch,inspect
from torch.utils.data import DataLoader
def main():
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    identity();sys.path.insert(0,str(legacy.v1.CODE))
    from distributed_rollout import RawObservations
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from transformers import AutoModel
    original=AutoModel.from_pretrained
    def fp32(*a,**kw):kw.update(torch_dtype=torch.float32,use_flash_attn=False);return original(*a,**kw)
    AutoModel.from_pretrained=fp32
    rows=[r for r in scenes('train')[rank::world] if not feature_path(r['token']).exists()]
    if not rows:return
    b=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(MAIN/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),device=f'cuda:{rank}',use_expert_features=False)
    start=time.time();records=[]
    for i,(t,obs) in enumerate(DataLoader(RawObservations(rows),batch_size=None,num_workers=4,multiprocessing_context='fork')):
        with torch.inference_mode():f=b.compute_features(obs)
        assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
        f={k:v.cpu() for k,v in f.items()};f['_metadata']=dict(token=t,protocol_hash=identity(),observation_only=True,precision='fp32',feature_code_hash=sha(inspect.getfile(ReCogDriveFeatureBuilder)))
        dest=OUT/'cache/features'/f'{t}.pt';dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');torch.save(f,tmp);tmp.replace(dest)
        records.append(dict(token=t,sha256=sha(dest)))
        if i%20==0:print('FEATURE',rank,i+1,len(rows),time.time()-start,flush=True)
    save(OUT/'audits'/f'features_{rank}.json',dict(status='PASS',records=records,seconds=time.time()-start))
if __name__=='__main__':main()
