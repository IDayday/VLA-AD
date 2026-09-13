"""Provenance-based snapshot choice, no checkpoint selection by evaluation score."""
import argparse,re,copy,concurrent.futures,lzma,pickle
from v2_common import *

def agent_config(path):
    text=Path(path).read_text();lines=text.splitlines();start=next(i for i,l in enumerate(lines) if l=='agent:');end=next((i for i in range(start+1,len(lines)) if lines[i] and not lines[i].startswith(' ')),len(lines));return yaml.safe_load('\n'.join(lines[start:end]))['agent']

def discover():
    arch=Path('/mnt/project/container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl');old=Path('/mnt/project/VLA-AD_last_vla_dev/outputs');models={m['name']:m for m in v1.models()};chains=[]
    runs=[('official_il_grpo','official_il',arch/'outputs/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z',None),('mts86_lfp_grpo','mts_8692',old/'stage3_lfp_grpo_v1_v6_epoch165_formal_20260715T145650Z',[300,1800,3300]),('mts87_lfp_grpo','mts_8751',old/'stage3_lfp_grpo_v1_exact_b8g16_epoch155_r4_20260712T083012Z',[300,4200,8100])]
    for name,base,run,steps in runs:
        cfgpath=next(run.glob(('navsim_exp' if base=='official_il' else 'hydra')+'/training_recogdrive_agent/*/code/hydra/config.yaml'));agent=agent_config(cfgpath);m=models[base];assert agent['checkpoint_path']==m['checkpoint_path'] and agent['reference_policy_checkpoint']==m['checkpoint_path'] and agent['grpo']
        if steps is None:
            folder=next(run.glob('**/lightning_logs/version_0/checkpoints'));paths=[folder/'epoch=0-step=1330.ckpt',folder/'epoch=4-step=6650.ckpt',folder/'epoch=9-step=13300.ckpt'];steps=[1330,6650,13300]
        else:
            folder=next(run.glob('hydra/**/step_checkpoints'));paths=[folder/f'step-step={s}.ckpt' for s in steps]
        snapshots=[dict(name=name+'_step0',role='step0',step=0,reuse_V1=base,model=m)]
        for role,step,path in zip(['early','middle','final_available'],steps,paths):
            assert path.exists();entry=copy.deepcopy(m);entry.update(name=name+'_'+str(step),checkpoint_path=str(path),sha256=sha(path),config_path=str(cfgpath),reported_pdms=None)
            snapshots.append(dict(name=entry['name'],role=role,step=step,reuse_V1=None,model=entry))
        chains.append(dict(name=name,baseline=base,run=str(run),config_path=str(cfgpath),config_sha256=sha(cfgpath),algorithm=agent.get('stage3_algorithm','original_GRPO'),initial_checkpoint=agent['checkpoint_path'],reference_checkpoint=agent['reference_policy_checkpoint'],group_size=agent.get('grpo_sample_time'),learning_rate=agent.get('lr'),snapshots=snapshots,final_definition='Last retained step checkpoint, not necessarily completed planned training; official GRPO uses final epoch9',selection='Earliest post-init, near midpoint of retained steps, last retained; fixed before V2 evaluation'))
    provenance=[]
    for base in ['mts_8692','mts_8751']:
        p=Path(models[base]['config_path']);cfg=read(p);agent=cfg['agent'];provenance.append(dict(model=base,path=str(p),sha256=sha(p),agent={k:v for k,v in agent.items() if k in ['checkpoint_path','offline_rl_enabled','offline_rl_mode','offline_rl_support_archive_path','offline_rl_dpsi_target_distribution','offline_rl_dpsi_target_sample_m','offline_rl_dpsi_target_sample_m_after_warmup','offline_rl_dpsi_force_anchor_target','offline_rl_dpsi_force_best_target','use_fs_norm','use_planning_token_adapter']},warning='Top-level stage2_target_source=gt is not sufficient to identify effective target; offline DPSI support archive is configured. Neither archive is mapped to reconstructed M1/M2/M3.'))
    save(OUT/'manifests/historical_chains.json',dict(identity=ident(),chains=chains,MTS_provenance=provenance,requested_recipe_chains={k:'matched causal chain unavailable: no verified recipe-to-training-run mapping' for k in ['Score-MTS → GRPO degradation','Pareto-MTS → GRPO degradation','GT-distance-MTS → GRPO limited gain']},notes='V6/A5 matched LFP-GRPO histories are additional observed chains, not the three requested selection-recipe ablations.'))
    print([(c['name'],[(s['role'],s['step']) for s in c['snapshots']]) for c in chains],flush=True)

def sample_snapshot(name):
    import torch,time
    from torch.utils.data import DataLoader
    from distributed_rollout import CachedObservations,load_planner
    from gpu_banks import inputs
    s=next(s for c in read(OUT/'manifests/historical_chains.json')['chains'] for s in c['snapshots'] if s['name']==name);assert not s['reuse_V1']
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=f'cuda:{rank}';torch.cuda.set_device(rank);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;p=load_planner(s['model']);meta=dict(identity=ident(),snapshot=s,seed_stream='Exact first32 V1 policy streams; common random numbers with step0')
    rows=[r for r in scenes()['scenes'] if r['token'] in set(subset('chain'))][rank::world];pending=[r for r in rows if not valid(OUT/'chains/rollouts'/name/f"{r['token']}.npz",meta)];start=time.time()
    for i,(token,f) in enumerate(DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork')):
        f.pop('_metadata');vl,action=inputs(f,device);results=[]
        for begin in [0,16]:
            init=torch.stack([torch.randn((8,3),device=device,generator=torch.Generator(device=device).manual_seed(v1.seed_for(token,'policy',j))) for j in range(begin,begin+16)]);torch.manual_seed(v1.seed_for(token,'policy_steps',begin))
            with torch.inference_mode():results.append(p.get_action(vl.expand(16,-1,-1),action(16),init_actions=init,deterministic=False)['pred_traj'].cpu().numpy())
        traj=np.concatenate(results);assert traj.shape==(32,8,3) and np.isfinite(traj).all();npz(OUT/'chains/rollouts'/name/f'{token}.npz',meta,trajectories=traj)
        if i%10==0:print(name,rank,i+1,len(pending),time.time()-start,flush=True)
    save(OUT/'manifests'/f'chain_{name}_rank{rank}.json',dict(meta,rank=rank,count=len(rows),new_count=len(pending),seconds=time.time()-start))

def score_scene(task):
    from evaluate_cached_rollouts import score_arrays
    r,s=task;src=OUT/'chains/rollouts'/s['name']/f"{r['token']}.npz";dest=OUT/'chains/scores'/s['name']/f"{r['token']}.npz";meta=dict(identity=ident(),input_sha256=sha(src))
    if valid(dest,meta):return
    with lzma.open(r['metric_cache_path'],'rb') as f:cache=pickle.load(f)
    npz(dest,meta,trajectories=score_arrays(cache,np.load(src)['trajectories']))

def scoring():
    from evaluate_cached_rollouts import init_worker
    rows=[r for r in scenes()['scenes'] if r['token'] in set(subset('chain'))];snaps=[s for c in read(OUT/'manifests/historical_chains.json')['chains'] for s in c['snapshots'] if not s['reuse_V1']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=96,initializer=init_worker) as ex:list(ex.map(score_scene,[(r,s) for r in rows for s in snaps],chunksize=1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--discover',action='store_true');p.add_argument('--snapshot');p.add_argument('--score',action='store_true');a=p.parse_args()
    if a.discover:discover()
    elif a.snapshot:sample_snapshot(a.snapshot)
    elif a.score:scoring()
