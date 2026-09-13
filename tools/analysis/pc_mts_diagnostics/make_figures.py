"""PNG/PDF figures for all four experiments, drawn from actual cached tables."""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from scene_statistics import summary_and_tests

COLORS=['#697782','#27a68b','#9770bb','#3581bc','#d65c47']
M_LABELS=['Score-only','Pareto','Pareto + GT','PC-MTS']

def main(args):
    global M_LABELS
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];figdir=out/'figures';figdir.mkdir(exist_ok=True);metrics=out/'metrics'
    if len(list((out/'candidate_pools/pc_mts').glob('*.npz')))<sc['scene_count']:M_LABELS=M_LABELS[:3]+['PC-MTS (pending)']
    elif (out/'manifests/coverage_completion.json').exists():M_LABELS=M_LABELS[:3]+['PC-MTS + coverage']
    geom=pd.read_csv(metrics/'policy_distribution.csv');quality=pd.read_csv(metrics/'grpo_readiness.csv')
    joined=geom.merge(quality,on=['token','log','command','checkpoint'],validate='one_to_one')
    joined.to_csv(metrics/'table_a_d_scene.csv',index=False)
    summary=summary_and_tests(joined,'checkpoint',metrics,'policy',cfg['bootstrap_replicates'])
    def export(fig,name):
        fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def mean_ci(df,col,order=MODELS,key='checkpoint'):
        result=[];rng=np.random.default_rng(20260913)
        for m in order:
            v=df[df[key]==m][col].dropna().to_numpy()
            if not len(v):result.append([0.,0.]);continue
            boot=v[rng.integers(0,len(v),(cfg['bootstrap_replicates'],len(v)))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);mu=v.mean();result.append([max(mu-lo,0),max(hi-mu,0)])
        return np.array(result).T
    def categorical(df,col,name,ylabel=None,order=MODELS,labels=LABELS,key='checkpoint'):
        fig,ax=plt.subplots(figsize=(8,4));values=[df[df[key]==m][col].dropna().to_numpy() for m in order]
        good=[(i,v) for i,v in enumerate(values) if len(v)>1 and np.ptp(v)>1e-12]
        if good:
            vp=ax.violinplot([v for i,v in good],positions=[i for i,v in good],showextrema=False)
            for b,(i,v) in zip(vp['bodies'],good):b.set_facecolor(COLORS[i]);b.set_alpha(.35)
        ax.boxplot(values,positions=np.arange(len(order)),widths=.2,showfliers=False)
        ax.errorbar(np.arange(len(order))+.17,[v.mean() if len(v) else np.nan for v in values],yerr=mean_ci(df,col,order,key),fmt='D',markersize=4,color='#222222',capsize=3,label='Mean and scene-bootstrap 95% CI')
        ax.legend(fontsize=8)
        ax.set_xticks(range(len(order)),labels,rotation=15);ax.set_ylabel(ylabel or col);ax.grid(axis='y',alpha=.2);export(fig,name)
    for col,tag,label in [('spread_auc','Fig-0a','Spread-AUC (m)'),('pairwise_ade','Fig-0b','Mean pairwise ADE (m)'),('r90','Fig-0c','R90 around trajectory medoid (m)'),('effective_rank','Fig-0f','Flattened covariance effective rank')]:categorical(geom,col,tag,label)
    fig,ax=plt.subplots(figsize=(8,4))
    for m,label,c in zip(MODELS,LABELS,COLORS):
        rows=summary[(summary.group==m)&summary.metric.isin([f'spread_t{i}' for i in range(1,9)])].set_index('metric').loc[[f'spread_t{i}' for i in range(1,9)]]
        x=np.arange(1,9)*.5;ax.plot(x,rows['mean'],label=label,color=c);ax.fill_between(x,rows.ci_low,rows.ci_high,color=c,alpha=.2)
    ax.set_xlabel('Future time (s)');ax.set_ylabel('Spatial spread (m)');ax.legend();export(fig,'Fig-0d')
    fig,ax=plt.subplots(figsize=(7,5))
    for m,label,c in zip(MODELS,LABELS,COLORS):
        d=geom[geom.checkpoint==m];ax.scatter(d.center_shift_from_il,d.spread_ratio_il,s=7,alpha=.2,color=c,label=label)
    ax.axhline(1,color='grey',ls='--');ax.set_xlabel('Center shift from IL (m)');ax.set_ylabel('Spread-AUC / IL Spread-AUC');ax.set_yscale('log');ax.legend();export(fig,'Fig-0e')
    tokenmap={r['token']:r for r in sc['scenes']}
    selected=sc['representatives'] or [dict(token=r['token'],kind='preregistered random smoke') for r in sc['scenes'][:6]]
    for page in range((len(selected)+1)//2):
        scenes=selected[page*2:page*2+2];fig,axes=plt.subplots(len(scenes),5,figsize=(17,4.5*len(scenes)),squeeze=False)
        for row,item in enumerate(scenes):
            token=item['token'];clouds=[np.load(out/'rollouts'/m/f'{token}.npz')['trajectories'] for m in MODELS];gt=np.asarray(tokenmap[token]['gt']);xy=np.concatenate([a.reshape(-1,3) for a in clouds]+[gt,np.zeros((1,3))])
            for col,(cloud,label,c) in enumerate(zip(clouds,LABELS,COLORS)):
                ax=axes[row,col]
                for t in cloud:ax.plot(t[:,1],t[:,0],color=c,alpha=.2,lw=.6)
                ax.plot(gt[:,1],gt[:,0],'k--',lw=1);ax.set_title(label+'\n'+item['kind']+'\n'+token,fontsize=9);ax.set_xlim(xy[:,1].min()-1,xy[:,1].max()+1);ax.set_ylim(xy[:,0].min()-1,xy[:,0].max()+1);ax.set_xlabel('Lateral y (m)');ax.set_ylabel('Forward x (m)');ax.grid(alpha=.2)
        fig.tight_layout();export(fig,f'Fig-0g-page{page+1}')
    means=joined.groupby('checkpoint').mean(numeric_only=True).loc[MODELS]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col,label in zip(axes,['feasible_rate','mean_pdms'],['Feasible rate','Mean PDMS (points)']):ax.bar(LABELS,means[col],color=COLORS,yerr=mean_ci(joined,col),capsize=3);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=25)
    export(fig,'Fig-3a')
    oracle='oracle_64' if 'oracle_64' in means else 'oracle_8'
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(5);ax.bar(x-.15,means.mean_pdms,.3,label='Mean',yerr=mean_ci(joined,'mean_pdms'),capsize=3);ax.bar(x+.15,means[oracle],.3,label=oracle,yerr=mean_ci(joined,oracle),capsize=3);ax.set_xticks(x,LABELS,rotation=15);ax.legend();ax.set_ylabel('PDMS points');export(fig,'Fig-3b')
    fig,ax=plt.subplots(figsize=(8,4))
    for i,delta in enumerate([0,1,2]):ax.bar(x+(i-1)*.23,means[f'hit8_{delta}'],.23,label=f'Delta={delta}',yerr=mean_ci(joined,f'hit8_{delta}'),capsize=2)
    ax.set_xticks(x,LABELS,rotation=15);ax.set_ylabel('Actual Hit@8');ax.legend();export(fig,'Fig-3c')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(x-.16,means.all_infeasible_group_rate,.32,label='All infeasible',yerr=mean_ci(joined,'all_infeasible_group_rate'),capsize=3);ax.bar(x+.16,means.mostly_infeasible_group_rate,.32,label='Feasible count <=2',yerr=mean_ci(joined,'mostly_infeasible_group_rate'),capsize=3);ax.set_xticks(x,LABELS,rotation=15);ax.legend();export(fig,'Fig-3d')
    categorical(quality,'unsafe_positive_advantage','Fig-3e','Unsafe positive advantage fraction')
    fig,ax=plt.subplots(figsize=(7,5))
    for m,label,c in zip(MODELS,LABELS,COLORS):
        d=joined[joined.checkpoint==m];ax.scatter(d.spread_auc,d.hit8_1,s=8,alpha=.2,color=c,label=label)
    ax.set_xlabel('Spread-AUC (m)');ax.set_ylabel('Hit@8 (Delta=1 point)');ax.legend();export(fig,'Fig-3f')
    fig,ax=plt.subplots(figsize=(8,4))
    for i,col in enumerate(['D_all','D_feasible','D_positive']):ax.bar(x+(i-1)*.23,means[col],.23,label=col,yerr=mean_ci(joined,col),capsize=2)
    ax.set_xticks(x,LABELS,rotation=15);ax.set_ylabel('Pairwise ADE (m), valid subsets only');ax.legend();export(fig,'Fig-3g')
    if (metrics/'candidate_pool_scene.csv').exists():
        pools=pd.read_csv(metrics/'candidate_pool_scene.csv');cand=pd.read_parquet(metrics/'candidate_records.parquet');primary=pools[pools.subset=='all16'];summary_and_tests(primary,'method',metrics,'candidate_pools',cfg['bootstrap_replicates'])
        mean=primary.groupby('method').mean(numeric_only=True).reindex(METHODS)
        fig,ax=plt.subplots(figsize=(8,4));bottom=np.zeros(4);cumulative=primary.copy();cumulative['cumulative_fraction']=0.
        for region in ['core_fraction','boundary_fraction','ood_fraction']:
            v=mean[region].fillna(0);ax.bar(M_LABELS,v,bottom=bottom,label=region);bottom+=v
            cumulative['cumulative_fraction']+=cumulative[region]
            if region!='ood_fraction':ax.errorbar(np.arange(4),bottom,yerr=mean_ci(cumulative,'cumulative_fraction',METHODS,'method'),fmt='none',color='black',capsize=3,lw=1)
        ax.legend();ax.set_ylabel('Fraction');ax.set_title('Cumulative boundaries: scene-bootstrap 95% CI',fontsize=10);export(fig,'Fig-1a')
        fig,ax=plt.subplots(figsize=(8,4))
        for m,label in zip(METHODS,M_LABELS):
            v=np.sort(cand[cand.method==m].q_policy.dropna());ax.plot(v,np.arange(1,len(v)+1)/max(len(v),1),label=label)
        ax.set_xlabel('q_policy percentile');ax.set_ylabel('ECDF');ax.legend();export(fig,'Fig-1b')
        for xcol,ycol,name in [('q_policy','pdms','Fig-1c'),('d_GT','q_policy','Fig-1d')]:
            fig,axes=plt.subplots(1,4,figsize=(16,4),sharey=True)
            for ax,m,label in zip(axes,METHODS,M_LABELS):
                d=cand[cand.method==m];ax.hexbin(d[xcol],d[ycol],gridsize=30,mincnt=1,bins='log');ax.set_title(label);ax.set_xlabel(xcol);ax.set_ylabel(ycol)
            export(fig,name)
        categorical(primary,'pool_diversity','Fig-1e','Pool pairwise ADE (m)',METHODS,M_LABELS,'method')
    if (metrics/'local_robustness_scene.csv').exists():
        local=pd.read_csv(metrics/'local_robustness_scene.csv');summary_and_tests(local[local.subset=='all16'],'method',metrics,'local_robustness',cfg['bootstrap_replicates']);lc=local[local.subset=='all16'];mean=lc.groupby('method').mean(numeric_only=True).reindex(METHODS)
        fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4);ax.bar(x-.15,mean.original_pdms,.3,label='Original',yerr=mean_ci(lc,'original_pdms',METHODS,'method'),capsize=3);ax.bar(x+.15,mean.local_mean,.3,label='Held-out local mean',yerr=mean_ci(lc,'local_mean',METHODS,'method'),capsize=3);ax.set_xticks(x,M_LABELS);ax.legend();export(fig,'Fig-2a')
        for col,name in [('local_feasible_rate','Fig-2b'),('local_drop','Fig-2c'),('local_cvar20','Fig-2d')]:categorical(lc,col,name,None,METHODS,M_LABELS,'method')
        fig,ax=plt.subplots(figsize=(8,4))
        for m,label in zip(METHODS,M_LABELS):
            y=np.array([mean.loc[m,f'amp_{amp:g}'] for amp in cfg['evaluation_perturbation_amplitudes']]);err=np.column_stack([mean_ci(lc,f'amp_{amp:g}',[m],'method')[:,0] for amp in cfg['evaluation_perturbation_amplitudes']]);ax.errorbar(cfg['evaluation_perturbation_amplitudes'],y,yerr=err,marker='o',label=label,capsize=3)
        ax.set_xlabel('Smooth perturbation amplitude (m)');ax.set_ylabel('Mean PDMS points');ax.legend();export(fig,'Fig-2e')
        fig,ax=plt.subplots(figsize=(7,5))
        for m,label in zip(METHODS,M_LABELS):
            d=lc[lc.method==m];ax.scatter(d.original_pdms,d.local_mean,s=8,alpha=.3,label=label)
        ax.plot([0,100],[0,100],'k--');ax.set_xlabel('Original PDMS');ax.set_ylabel('Held-out local mean PDMS');ax.legend();export(fig,'Fig-2f')
    print(f'Figures written to {figdir}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
