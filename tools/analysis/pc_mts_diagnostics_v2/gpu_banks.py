"""Only new IL samples; V1 feature/rollout files are never written."""
import argparse,time
from v2_common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from distributed_rollout import CachedObservations,load_planner

def inputs(f,device):
    import torch
    from transformers.feature_extraction_utils import BatchFeature
    vl=f['last_hidden_state'].unsqueeze(0).to(device).float();h=f['history_trajectory'].view(1,-1).to(device).float();s=f['status_feature'].unsqueeze(0).to(device).float();c=f['high_command_one_hot'].unsqueeze(0).to(device).float()
    def action(n):return BatchFeature(data={'his_traj':h.expand(n,-1),'history_trajectory':h.view(1,4,3).expand(n,-1,-1),'status_feature':s.expand(n,-1),'high_command_one_hot':c.expand(n,-1),'state':torch.cat([s,h],1).expand(n,-1)})
    return vl,action

def sample(p,vl,action,token,namespace,count,device):
    import torch
    result=[];seeds=[seed(token,namespace,j) for j in range(count)]
    for begin in range(0,count,CFG['inference_batch']):
        n=min(CFG['inference_batch'],count-begin)
        init=torch.stack([torch.randn((8,3),generator=torch.Generator(device=device).manual_seed(seeds[j]),device=device) for j in range(begin,begin+n)])
        torch.manual_seed(seed(token,namespace+'_steps',begin))
        with torch.inference_mode():v=p.get_action(vl.expand(n,-1,-1),action(n),init_actions=init,deterministic=False)['pred_traj']
        result.append(v.cpu().float().numpy())
    return np.concatenate(result),seeds

def main(a):
    import torch
    from torch.utils.data import DataLoader
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=f'cuda:{rank}'
    torch.cuda.set_device(rank);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=v1.models()[0];p=load_planner(m);assert next(p.parameters()).dtype==torch.float32
    rows=scenes()['scenes']
    if a.smoke:rows=[r for r in rows if r['token'] in set(subset('smoke'))]
    rows=rows[rank::world];base=dict(ident(),checkpoint_hash=m['sha256'],protocol='new_IL_R128_Q128_native32')
    pending=[r for r in rows if not valid(OUT/'il_banks'/f"{r['token']}.npz",base)]
    loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');checks=[];start=time.time()
    for i,(token,f) in enumerate(loader):
        assert f.pop('_metadata')['scene_manifest_hash']==ident()['scene_manifest_hash'];vl,action=inputs(f,device);arrays={};partitions={}
        for key,namespace,n in [('R','v2_il_support_R',128),('Q','v2_il_support_Q',128),('native','v2_candidate_il',32)]:arrays[key],partitions[key]=sample(p,vl,action,token,namespace,n,device)
        assert len(set(sum(partitions.values(),[])))==288
        assert all(x.shape[1:]==(8,3) and np.isfinite(x).all() for x in arrays.values())
        if i==0:
            replay,_=sample(p,vl,action,token,'v2_candidate_il',32,device);err=float(abs(replay-arrays['native']).max());assert err==0;checks.append(dict(token=token,replay_max_error=err))
        npz(OUT/'il_banks'/f'{token}.npz',dict(base,token=token,seeds=partitions,feature_sha256=sha(V1/'features'/f'{token}.pt')),**arrays)
        if i%10==0:print(json.dumps(dict(rank=rank,done=i+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'il_banks_{"smoke" if a.smoke else "full"}_rank{rank}.json',dict(base,rank=rank,scenes=len(rows),new_scenes=len(pending),wall_seconds=time.time()-start,checks=checks,peak_gpu_memory=torch.cuda.max_memory_allocated()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args())
