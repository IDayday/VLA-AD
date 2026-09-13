"""Exportable figures from frozen observational tables, full and contrastive side by side."""
from v2_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'score':'#c44e52','pareto':'#dd9b2f','gt_distance':'#4c72b0','pc_mts':'#55a868'}
TITLES={'score':'Score','pareto':'Pareto','gt_distance':'GT-distance','pc_mts':'PC-MTS V2'}
MODEL_LABELS={'official_il':'Official IL','mts_8692':'MTS-86.92','mts_8751':'MTS-87.51','grpo_9041':'GRPO-90.41','apr_9145':'APR-91.45'}
def finish(fig,n,title):
    fig.suptitle(title)
    if not fig.get_constrained_layout():fig.tight_layout()
    name=f'Fig-V2-{n}'
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
def readcsv(name):return pd.read_csv(OUT/'metrics'/f'{name}.csv')
def regions(ax,tag):
    d=readcsv('pool_summary_'+tag).set_index('method').loc[METHODS];bottom=np.zeros(4)
    for field,color in [('core','#4c72b0'),('boundary','#55a868'),('far','#c44e52')]:
        vals=d[field].to_numpy()*100;ax.bar(range(4),vals,bottom=bottom,color=color,label=field.capitalize())
        for i,v in enumerate(vals):
            if v>=4:ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',fontsize=9,color='white')
        bottom+=vals
    ax.set_xticks(range(4),[TITLES[m] for m in METHODS],rotation=12);ax.set_ylabel('Selected candidates (%)');ax.set_ylim(0,100);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3);ax.set_title(f'{tag}: {int(d.n_scenes.iloc[0])} scenes; q-regions are selection sanity checks')
def main():
    raw=pd.read_parquet(OUT/'metrics/raw_candidates_full.parquet');d=readcsv('candidates');p=readcsv('policy_stratification_scene');srcs=sorted(raw.source.unique());scmap=dict(zip(srcs,plt.get_cmap('tab10').colors));sample=raw.groupby('source',group_keys=False).sample(n=800,random_state=CFG['seed'])
    fig,ax=plt.subplots(figsize=(9,5))
    for source,g in sample.groupby('source'):ax.scatter(g.r_knn,g.pdms,s=7,alpha=.35,label=source,color=scmap[source],rasterized=True)
    ax.set_xscale('log');ax.set_xlabel('Held-out KNN ratio r_knn (log scale)');ax.set_ylabel('Actual NAVSIM PDMS');ax.legend(loc='center left',bbox_to_anchor=(1, .5));finish(fig,1,'Diagnostic reservoir V2: same 192 candidates per scene')
    bridge=raw[raw.bridge_alpha.notna()];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,label in [(axes[0],'r_knn','KNN ratio r_knn'),(axes[1],'policy_distance_knn','KNN distance to R (m)')]:
        g=bridge.groupby('bridge_alpha')[metric];x=np.array([.2,.4,.6,.8]);ax.plot(x,g.median(),marker='o',label='Median');ax.fill_between(x,g.quantile(.25),g.quantile(.75),alpha=.2,label='25–75%');ax.plot(x,g.mean(),linestyle='--',label='Mean');ax.set_xlabel('Fixed bridge alpha');ax.set_ylabel(label);ax.set_xticks(x);ax.legend()
    finish(fig,2,'Bridge distance gradient: 16,000 matched anchors')
    for n,tag in [(3,'full1000'),(4,'contrastive')]:
        fig,ax=plt.subplots(figsize=(8,5));regions(ax,tag);finish(fig,n,'Core / Boundary / Far by selection method')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,label in [(axes[0],'r_knn','KNN ratio'),(axes[1],'maha_ratio','Mahalanobis ratio')]:
        for source,g in sample.groupby('source'):ax.scatter(g.d_GT,g[metric],s=6,alpha=.3,label=source,color=scmap[source],rasterized=True)
        ax.set_yscale('log');ax.set_xscale('symlog',linthresh=.05);ax.set_xlabel('Distance to GT (m; symlog)');ax.set_ylabel(label+' (log)')
    axes[1].legend(loc='center left',bbox_to_anchor=(1,.5),fontsize=8);finish(fig,5,'GT proximity and policy compatibility measure different directions')
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    for ax,m in zip(axes.flat,METHODS):
        g=d[(d.method==m)&d.E_rec.notna()];ax.scatter(g.pdms,g.E_rec,s=5,alpha=.18,color=COLORS[m],rasterized=True);ax.set_title(TITLES[m]+f' (n={g.token.nunique()} scenes)');ax.set_xlabel('Candidate PDMS');ax.set_ylabel('Denoising reconstruction ADE (m)')
    finish(fig,6,'Independent frozen-IL denoising reachability')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));curve=readcsv('robustness_curves_full1000');summary=readcsv('pool_summary_full1000').set_index('method')
    for m in METHODS:
        g=curve[curve.method==m].sort_values('amplitude');x=np.r_[0,g.amplitude];axes[0].plot(x,np.r_[summary.loc[m,'pdms'],g.local_PDMS],marker='o',label=TITLES[m],color=COLORS[m]);axes[1].plot(x,np.r_[0,g.PDMS_drop],marker='o',label=TITLES[m],color=COLORS[m])
    for ax in axes:ax.set_xlabel('Perturbation amplitude (m)');ax.legend(fontsize=8)
    axes[0].set_ylabel('Mean local PDMS');axes[1].set_ylabel('PDMS drop from original (points)');finish(fig,7,'Stronger held-out local perturbations: Full-1000')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,tag in zip(axes,['full1000','contrastive']):
        scene=readcsv('pool_scene_'+tag);vals=[scene[scene.method==m].robust_radius.to_numpy() for m in METHODS];b=ax.boxplot(vals,patch_artist=True,showmeans=True,meanline=True,labels=[TITLES[m] for m in METHODS],showfliers=False)
        for patch,m in zip(b['boxes'],METHODS):patch.set_facecolor(COLORS[m]);patch.set_alpha(.55)
        ax.set_title(tag);ax.tick_params(axis='x',rotation=12);ax.set_ylabel('Scene-mean robust radius (m)');ax.set_ylim(0,.52)
    finish(fig,8,'Largest consistently passing amplitude: feasibility and PDMS-drop gates')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for m in MODELS[1:]:
        values=np.sort(p[p.checkpoint==m].spread_ratio_il);ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=MODEL_LABELS[m])
    ax.axvline(1,color='gray',linestyle='--');ax.set_xscale('log');ax.set_xlabel('Per-scene Spread-AUC / Official-IL Spread-AUC');ax.set_ylabel('ECDF');ax.legend();finish(fig,9,'Observed checkpoint compression and expansion; V1 rollouts reused')
    fig,ax=plt.subplots(figsize=(9,5));globald=readcsv('policy_global');frame=globald
    sc=ax.scatter(frame.spread_auc,frame.hit8_1,c=frame.feasible_rate,s=frame.D_positive.fillna(0)*2200+30,cmap='viridis',vmin=.90,vmax=1,edgecolors='black')
    for _,r in frame.iterrows():ax.annotate(MODEL_LABELS[r.checkpoint],(r.spread_auc,r.hit8_1),fontsize=10,xytext=(8,5),textcoords='offset points')
    ax.set_xlabel('Spread-AUC (m)');ax.set_ylabel('Hit@8 (feasible, PDMS > IL reference + 1)');ax.margins(x=.20,y=.16)
    fig.colorbar(sc,ax=ax,label='Feasible rate');finish(fig,10,'Useful spread: color = feasibility; area = D_positive')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for m,g in p.groupby('checkpoint'):
        g=g.dropna(subset=['D_positive']);axes[0].scatter(g.center_shift_from_il,g.D_positive,s=6,alpha=.15,label=MODEL_LABELS[m],rasterized=True)
    for _,r in globald.iterrows():axes[1].scatter(r.center_shift_from_il,r.D_positive,s=150);axes[1].annotate(MODEL_LABELS[r.checkpoint],(r.center_shift_from_il,r.D_positive),fontsize=9,xytext=(3,4),textcoords='offset points')
    axes[0].set_xscale('symlog',linthresh=.1);axes[0].set_yscale('symlog',linthresh=.1);axes[0].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Center shift from IL (m)');ax.set_ylabel('D_positive: feasible and PDMS > reference (m)')
    finish(fig,11,'Center shift and useful diversity; conditional on two positive trajectories')
    # Additional mechanism figures requested in Parts 8–10.
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,m in zip(axes,['mts_8692','mts_8751']):
        g=p[p.checkpoint==m];ax.scatter(g.spread_ratio_il,g.hit8_1,s=8,alpha=.2,rasterized=True);ax.set_xscale('log');ax.axvline(1,color='gray',ls='--');ax.set_title(MODEL_LABELS[m]);ax.set_xlabel('Spread ratio to IL');ax.set_ylabel('Hit@8 delta=1')
    finish(fig,'S1','Compression and high-quality hits by scene')
    chainpath=OUT/'metrics/chain_evolution_summary.csv'
    if chainpath.exists():
        chain=pd.read_csv(chainpath);fig,axes=plt.subplots(1,3,figsize=(13,4))
        for name,g in chain.groupby('chain'):
            g=g.sort_values('step')
            for ax,metric in zip(axes,['spread_ratio_step0','mean_pdms','hit8_1']):ax.plot(g.step,g[metric],marker='o',label=name);ax.set_xlabel('Training step');ax.set_ylabel(metric)
        axes[0].legend(fontsize=7);finish(fig,'S2','Matched checkpoint histories; final = last retained snapshot')
        gains=readcsv('chain_gain_scene');fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,name in zip(axes,['mts86_lfp_grpo','mts87_lfp_grpo']):
            g=gains[gains.chain==name];ax.scatter(g.step0_SR,g.pdms_gain,s=10,alpha=.3);ax.set_xscale('log');ax.set_xlabel('Step-0 spread / IL spread');ax.set_ylabel('Matched final - step0 PDMS');ax.set_title(name)
        finish(fig,'S3','Initial compression and subsequent matched LFP-GRPO gain')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,tag in zip(axes,['full1000','contrastive']):
        curve=readcsv('robustness_curves_'+tag)
        for m in METHODS:
            g=curve[curve.method==m].sort_values('amplitude');ax.plot(g.amplitude,g.failure_rate*100,marker='o',color=COLORS[m],label=TITLES[m])
        ax.set_xlabel('Perturbation amplitude (m)');ax.set_ylabel('Held-out infeasible rate (%)');ax.set_title(tag);ax.legend(fontsize=8)
    finish(fig,'S4','Seeded held-out failure-rate curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4));stability=readcsv('sample_count_summary')
    for ax,metric in zip(axes,['pairwise_ADE','hit8_1']):
        for m in MODELS:
            g=stability[(stability.checkpoint==m)&(stability.metric==metric)].sort_values('n_rollouts');ax.plot(g.n_rollouts,g['mean'],marker='o',label=MODEL_LABELS[m]);ax.fill_between(g.n_rollouts,g.subsample_p025,g.subsample_p975,alpha=.15)
        ax.set_xlabel('Rollouts per scene');ax.set_xticks([16,32,64]);ax.set_ylabel(metric);ax.legend(fontsize=8)
    finish(fig,'S5','100 cached-rollout subsamples; 64 is the primary model comparison')
    fig,axes=plt.subplots(1,5,figsize=(16,3.8),sharey=True,layout='constrained')
    for ax,m in zip(axes,MODELS):
        g=p[p.checkpoint==m];sc=ax.scatter(g.spread_auc,g.hit8_1,c=g.feasible_rate,s=4+g.D_positive.fillna(0)*50,cmap='viridis',vmin=0,vmax=1,alpha=.45,rasterized=True);ax.set_xscale('log');ax.set_title(MODEL_LABELS[m]);ax.set_xlabel('Spread-AUC (m, log)')
    axes[0].set_ylabel('Hit@8 delta=1');fig.colorbar(sc,ax=axes.tolist(),label='Feasible rate',shrink=.8);finish(fig,'S6','Per-scene useful spread; undefined D_positive receives minimum marker size')

if __name__=='__main__':main()
