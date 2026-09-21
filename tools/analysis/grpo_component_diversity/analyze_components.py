from shared import *
from metrics import *
import argparse,concurrent.futures,time
ADV={};INDEX={}
def init_worker():
    global ADV,INDEX
    INDEX={r['token']:i for i,r in enumerate(scenes())}
    ADV={m:{g:np.load(OUT/'cache/advantages'/f'{m}.npz')[f'G{g}'] for g in CFG['group_sizes']} for m in CFG['models']}
def worker(row):
    token=row['token'];records=[];patterns=[];components=[];inventory=[]
    for m in CFG['models']:
        t=arrays(m,token);s=arrays(m,token,True)
        assert len(t)==len(s)==128 and abs(reward(s,False)-s[:,6]).max()<1e-8
        for score in [False,True]:
            for p in [old_bank(m,token,score),extra_bank(m,token,score)]:inventory.append(dict(path=str(p),sha256=sha(p)))
        for g in CFG['group_sizes']:
            for start in range(0,128,g):
                records.append(dict(token=token,log=row['log'],command=row['command'],cohort=row['cohort'],model=m,G=g,block=start//g,**group_stats(t[start:start+g],s[start:start+g],ADV[m][g][INDEX[token],start:start+g])))
        for code,count in zip(*np.unique(np.round(s[:,SAFETY],8),axis=0,return_counts=True)):
            patterns.append(dict(token=token,model=m,NC=code[0],DAC=code[1],TTC=code[2],DDC=code[3],count=count,fraction=count/128))
        for i,n in enumerate(NAMES):
            # Raw observed discrete levels, not presupposed binary metrics.
            if n!='EP':
                for v,c in zip(*np.unique(s[:,i],return_counts=True)):components.append(dict(token=token,model=m,component=n,value=v,count=c))
    return records,patterns,components,inventory
def bootstrap_matrix(delta,logs,seed):
    """All metrics paired by scene; resample whole log clusters additionally."""
    rng=np.random.default_rng(seed);x=np.asarray(delta,float);valid=np.isfinite(x);x=np.nan_to_num(x)
    n,k=x.shape;boot=[];cluster=[]
    labels,inv=np.unique(logs,return_inverse=True);sums=np.zeros((len(labels),k));ns=np.zeros_like(sums)
    np.add.at(sums,inv,x);np.add.at(ns,inv,valid)
    for b in range(0,CFG['bootstrap_replicates'],100):
        count=min(100,CFG['bootstrap_replicates']-b)
        # Multinomial counts avoid materializing scenes x metrics x bootstrap.
        w=rng.multinomial(n,np.ones(n)/n,size=count);den=w@valid.astype(float)
        boot.extend(np.divide(w@x,den,out=np.full_like(den,np.nan),where=den>0))
        w=rng.multinomial(len(labels),np.ones(len(labels))/len(labels),size=count);den=w@ns
        cluster.extend(np.divide(w@sums,den,out=np.full_like(den,np.nan),where=den>0))
    return np.asarray(boot),np.asarray(cluster)
def summarize(df):
    numeric=[c for c in df.select_dtypes(include=['number','bool']).columns if c not in ['G','block']]
    tables=[];comparisons=[]
    for mode,data in [('prefix',df[df.block==0]),('all_blocks',df.groupby(['token','log','command','cohort','model','G'],as_index=False)[numeric].mean())]:
        for (m,g),q in data.groupby(['model','G']):
            rr=dict(grouping=mode,model=m,G=g,scenes=len(q),missing_scenes=5000-len(q))
            for c in numeric:rr[c]=q[c].mean();rr[c+'_median']=q[c].median();rr[c+'_n']=q[c].notna().sum()
            tables.append(rr)
        for g,q in data.groupby('G'):
            a=q[q.model==CFG['models'][0]].set_index('token').sort_index();b=q[q.model==CFG['models'][1]].set_index('token').loc[a.index]
            delta=b[numeric].astype(float)-a[numeric].astype(float);boot,cluster=bootstrap_matrix(delta,a.log.to_numpy(),CFG['analysis_seed']+g)
            for j,c in enumerate(numeric):
                z=delta[c].dropna();dist=boot[:,j];cl=cluster[:,j]
                if len(z):
                    comparisons.append(dict(grouping=mode,G=g,metric=c,scenes=len(z),mean_difference=z.mean(),median_difference=z.median(),ci_low=np.nanquantile(dist,.025),ci_high=np.nanquantile(dist,.975),log_ci_low=np.nanquantile(cl,.025),log_ci_high=np.nanquantile(cl,.975),scene_win_fraction=(z>1e-10).mean(),scene_tie_fraction=(abs(z)<=1e-10).mean(),bootstrap_sign_tail=min(1.,2*min((np.count_nonzero(dist<=0)+1)/3001,(np.count_nonzero(dist>=0)+1)/3001))))
    csv('method_summary.csv',tables);cf=pd.DataFrame(comparisons)
    # Prespecified primary family: eight metrics at G16 prefix.
    mask=(cf.grouping=='prefix')&(cf.G==16)&cf.metric.isin(CFG['primary_metrics']);ix=cf[mask].sort_values('bootstrap_sign_tail').index
    adjusted=np.maximum.accumulate([min(1,cf.at[j,'bootstrap_sign_tail']*(len(ix)-k)) for k,j in enumerate(ix)])
    cf.loc[ix,'holm_primary_bootstrap_sign_tail']=adjusted;cf.to_csv(OUT/'metrics/paired_comparisons.csv',index=False)
    # Stratify only by preexisting command / OLD1000 vs NEW4000 identities.
    st=[]
    for col in ['command','cohort']:
        for keys,q in df[df.block==0].groupby([col,'model','G']):
            st.append(dict(stratification=col,stratum=keys[0],model=keys[1],G=keys[2],scenes=len(q),**{c:q[c].mean() for c in numeric}))
    csv('stratified_summary.csv',st)
def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=32);a.add_argument('--summarize-only',action='store_true');args=a.parse_args();identity()
    if not (OUT/'audits/native_reward_advantage.json').exists():
        import subprocess
        subprocess.run([sys.executable,str(Path(__file__).parent/'native_audit.py')],check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
    dest=OUT/'metrics/group_metrics.parquet'
    if not args.summarize_only:
        records=[];patterns=[];components=[];inventory=[];start=time.time()
        with concurrent.futures.ProcessPoolExecutor(args.workers,initializer=init_worker) as pool:
            for i,(r,p,c,v) in enumerate(pool.map(worker,scenes(),chunksize=4)):
                records.extend(r);patterns.extend(p);components.extend(c);inventory.extend(v)
                if i%100==0:print('ANALYZE',i+1,round(time.time()-start),flush=True)
        dest.parent.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(records);df.to_parquet(dest,index=False)
        pd.DataFrame(patterns).to_parquet(OUT/'metrics/scene_outcome_patterns.parquet',index=False)
        csv('component_observed_levels.csv',pd.DataFrame(components).groupby(['model','component','value'],as_index=False)['count'].sum())
        save(OUT/'manifests/input_cache_hashes.json',inventory)
    else:df=pd.read_parquet(dest)
    summarize(df);save(OUT/'audits/analysis_complete.json',dict(status='PASS',scenes=df.token.nunique(),rows=len(df),models=CFG['models'],groups=CFG['group_sizes'],protocol_hash=identity()))
if __name__=='__main__':main()
