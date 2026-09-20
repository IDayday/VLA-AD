"""Identical FP32 observation-only feature path, with old-cache parity smoke."""
from common_5000 import *
import argparse, time, socket

def main():
    import torch
    from torch.utils.data import DataLoader
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a=argparse.ArgumentParser();a.add_argument('--smoke',action='store_true');args=a.parse_args()
    identity();sys.path.insert(0,str(v1.CODE))
    from distributed_rollout import RawObservations
    from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
    from transformers import AutoModel
    original=AutoModel.from_pretrained
    def fp32(*a,**kw):
        kw['torch_dtype']=torch.float32;kw['use_flash_attn']=False
        return original(*a,**kw)
    AutoModel.from_pretrained=fp32
    rows=scenes()[:4] if args.smoke else [r for r in scenes() if r['cohort']=='NEW4000']
    rows=rows[rank::world];pending=rows if args.smoke else [r for r in rows if not feature_path(r['token']).exists()]
    if not pending:return
    start=time.time();b=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(MAIN/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),device=f'cuda:{rank}',use_expert_features=False)
    assert next(b.backbone.model.parameters()).dtype==torch.float32
    checks=[]
    loader=DataLoader(RawObservations(pending),batch_size=None,num_workers=4,multiprocessing_context='fork')
    for i,(token,observation) in enumerate(loader):
        with torch.inference_mode():f=b.compute_features(observation)
        assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
        payload={k:v.cpu() for k,v in f.items()}
        if args.smoke:
            old=torch.load(feature_path(token),map_location='cpu',weights_only=False)
            errors={k:float((payload[k]-old[k]).abs().max()) for k in payload}
            assert max(errors.values())<1e-5,errors
            checks.append(dict(token=token,max_abs_errors=errors))
        else:
            payload['_metadata']=dict(protocol_hash=identity(),token=token,observation_only=True,precision='fp32',feature_code_hash=sha(inspect_file(ReCogDriveFeatureBuilder)))
            dest=OUT/'cache/features'/f'{token}.pt';dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix(f'.{os.getpid()}.tmp');torch.save(payload,tmp);tmp.replace(dest)
        if i%10==0:print('FEATURE',rank,i+1,len(pending),round(time.time()-start,1),flush=True)
    save(OUT/'audits'/f'features_{"smoke" if args.smoke else "run"}_{rank}.json',dict(status='PASS',host=socket.gethostname(),protocol_hash=identity(),count=len(pending),checks=checks,seconds=time.time()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated()))

def inspect_file(cls):
    import inspect
    return inspect.getfile(cls)

if __name__=='__main__':main()
