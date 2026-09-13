"""D: exact gradients on the prespecified final DiT block and output layer."""
from native import *
from torch.utils.data import DataLoader

def grad(p,trajectories,noise_key,embeds,params,device):
    traj=torch.as_tensor(trajectories,dtype=torch.float32,device=device);losses=[];vectors=[]
    for draw in range(2):
        keys=[noise_key]*len(traj);eps=noise_for(keys,'v3_D_noise',draw,device);t=timesteps_for(keys,'v3_D_time',draw,device)
        loss=diffusion(p,traj,t,eps,embeds)[0].mean();gg=torch.autograd.grad(loss,params,allow_unused=False)
        vectors.append(torch.cat([g.detach().flatten() for g in gg]));losses.append(float(loss.detach()))
    return sum(vectors)/2,float(np.mean(losses))

def measures(g,ref):
    norm=float(g.norm());rn=float(ref.norm());dot=float(torch.dot(g,ref))
    return dict(gradient_norm=norm,reference_norm=rn,cosine=dot/max(norm*rn,1e-20),dot=dot,predicted_delta_L_ref=-CFG['gradient']['eta']*dot)

def main():
    assert (OUT/'manifests/audit_C.json').exists(),'Execution order C before D'
    rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=setup(rank);p=model();p.requires_grad_(False);subset=gradient_subset(p);params=[v for _,v in subset]
    for v in params:v.requires_grad_(True)
    expected=read(OUT/'manifests/gradient_parameters.json')['parameters'];assert [n for n,_ in subset]==[v['name'] for v in expected]
    selected=set(tokens('gradient'));rows=[r for r in scenes() if r['token'] in selected][rank::world];pair=pd.read_csv(OUT/'metrics/C_pair_manifest.csv');pair=pair[pair.protocol=='primary'];rawframe=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet').set_index(['token','raw_index']);start=time.time()
    pending=[r for r in rows if not (OUT/'cache/gradient'/f"{r['token']}.json").exists()]
    loader=DataLoader(CachedObservations(pending,V1),batch_size=None,num_workers=4,multiprocessing_context='fork');scene_lookup={r['token']:r for r in rows}
    for j,(token,f) in enumerate(loader):
        vl,action=inputs(f,device)
        with torch.no_grad():embeds=encode(p,vl,action)
        raw=np.load(V2/'raw_candidates'/f'{token}.npz')['trajectories'];pools=read(OUT/'cache/pools'/f'{token}.json')['methods'];counts=[len(pools[m]['indices']) for m in METHODS];n=min([4]+[k for k in counts if k>0]);il=np.load(V2/'il_banks'/f'{token}.npz')['native'][:n]
        gt=np.asarray(scene_lookup[token]['gt'])[None];g_il,l_il=grad(p,il,token,embeds,params,device);g_gt,l_gt=grad(p,gt,token,embeds,params,device);gref=g_il+.25*g_gt
        cache={};methods=[];individual=[]
        for method in METHODS:
            ids=pools[method]['indices'][:n]
            if not ids:
                methods.append(dict(token=token,method=method,count=0,missing_reason='zero unique raw parents; no invented supervision'));continue
            gs=[];ls=[]
            for i in ids:
                if i not in cache:cache[i]=grad(p,raw[i:i+1],token,embeds,params,device)
                g,l=cache[i];gs.append(g);ls.append(l);r=rawframe.loc[(token,i)]
                individual.append(dict(token=token,method=method,raw_index=i,loss=l,region=str(region([r.q_holdout])[0]),q_holdout=r.q_holdout,PDMS=r.PDMS,exact_source=r.exact_source,**measures(g,gref)))
            methods.append(dict(token=token,method=method,count=n,loss=float(np.mean(ls)),reference_loss=l_il+.25*l_gt,**measures(sum(gs)/n,gref)))
        pairs=[]
        g=pair[pair.token==token];g=g.iloc[sorted(range(len(g)),key=lambda k:seed(g.iloc[k].pair_id,'v3_D_pair_order'))[:2]]
        for _,r in g.iterrows():
            for side,index in [('near',r.near_index),('far',r.far_index)]:
                gv,l=grad(p,raw[int(index):int(index)+1],r.pair_id,embeds,params,device)
                pairs.append(dict(token=token,pair_id=r.pair_id,side=side,raw_index=int(index),loss=l,near_region=r.near_region,exact_source=r.exact_source,PDMS_gap=r.PDMS_gap,**measures(gv,gref)))
        save(OUT/'cache/gradient'/f'{token}.json',dict(identity=identity(),token=token,methods=methods,candidates=individual,pairs=pairs,matched_count=n,counts=counts,feature_sha256=sha(V1/'features'/f'{token}.pt')))
        if j%5==0:print(json.dumps(dict(phase='D',rank=rank,done=j+1,total=len(pending),seconds=time.time()-start)),flush=True)
    save(OUT/'manifests'/f'D_gpu_rank{rank}.json',dict(identity=identity(),scenes=len(rows),seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':main()
