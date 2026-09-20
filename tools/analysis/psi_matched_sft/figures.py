"""Plots from measured tables only; retain negative results and both samplers."""
from common_matched import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'official_il':'#555555','il_sft':'#4c78a8','score':'#e45756','pareto':'#f2a541','psi':'#2a9d8f'}
LABELS={'official_il':'Initial IL','il_sft':'GT-only continuation','score':'Score','pareto':'Pareto','psi':'PSI'}
def finish(fig,name,df):
    d=OUT/'figures';d.mkdir(exist_ok=True);df.to_csv(d/f'{name}.csv',index=False)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(d/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    p=d/f'{name}.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n');plt.close(fig)
def main():
    f=pd.read_csv(OUT/'metrics/summary.csv');f=f[f.split=='holdout'];methods=['official_il']+CFG['methods']
    fig,axs=plt.subplots(2,4,figsize=(17,8))
    for row,pr in enumerate(['eval','native_grpo']):
        for col,(metric,title,scale) in enumerate([('mean_PDMS','Mean sampled PDMS',1),('feasible_rate','Joint feasible samples (%)',100),('centroid_displacement','Centroid displacement (m)',1),('pairwise_ADE64','Pairwise ADE64 (m)',1)]):
            ax=axs[row,col];g=f[f.protocol==pr].set_index('method').loc[methods];v=g[metric].to_numpy()*scale
            ax.bar(range(5),v,color=[COLORS[m] for m in methods]);ax.set_xticks(range(5),['IL','GT SFT','Score','Pareto','PSI'],rotation=35);ax.set_title(pr+' / '+title)
            for j,x in enumerate(v):ax.annotate(f'{x:.2f}',(j,x),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
            ax.margins(y=.18);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Matched SFT: 5000 held-out scenes, mean over 2 training seeds',y=1.02)
    finish(fig,'Fig1_matched_distribution_quality',f)
    comp=pd.read_csv(OUT/'metrics/paired_comparisons.csv');comp=comp[(comp.split=='holdout')&(comp.left=='psi')&comp.right.isin(CFG['methods'])]
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,pr in enumerate(['eval','native_grpo']):
        for j,(metric,label,factor) in enumerate([('mean_PDMS','PSI minus baseline PDMS',1),('feasible_rate','PSI minus baseline feasibility (pp)',100),('centroid_displacement','PSI minus baseline centroid shift (m)',1)]):
            ax=axs[i,j];g=comp[(comp.protocol==pr)&(comp.metric==metric)].set_index('right').loc[['il_sft','score','pareto']]
            vals=g.mean_difference.to_numpy()*factor;lo=g.ci_low.to_numpy()*factor;hi=g.ci_high.to_numpy()*factor
            ax.errorbar(vals,range(3),xerr=[vals-lo,hi-vals],fmt='o',color=COLORS['psi'],capsize=4);ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(3),['GT-only','Score','Pareto']);ax.set_title(pr+' / '+label);ax.grid(alpha=.2)
    finish(fig,'Fig2_scene_paired_differences',comp)
    tf=pd.read_parquet(OUT/'metrics/teacher_scene_metrics.parquet');tf=tf[(tf.protocol=='eval')&((tf.step==512)|(tf.method=='official_il'))&(tf.teacher_origin=='psi')&(tf.kind=='NON_GT')]
    # First average seeds within a scene; then average scenes. Equal-scene comparisons.
    t=tf.groupby(['split','method','token'],as_index=False)[['native_epsilon_loss','Hit64_0p5','nearest_teacher_ADE']].mean()
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,split in enumerate(['train','holdout']):
        for j,(metric,label,scale) in enumerate([('native_epsilon_loss','Common PSI teacher epsilon MSE',1),('Hit64_0p5','Common PSI teacher Hit64 @ 0.5m (%)',100),('nearest_teacher_ADE','Nearest sampled trajectory ADE (m)',1)]):
            g=t[t.split==split].groupby('method')[metric].mean().reindex(methods)*scale;ax=axs[i,j]
            ax.bar(range(5),g,color=[COLORS[m] for m in methods]);ax.set_xticks(range(5),['IL','GT SFT','Score','Pareto','PSI'],rotation=30);ax.set_title(split+' / '+label);ax.grid(axis='y',alpha=.2)
    finish(fig,'Fig3_common_supervision_absorption',t)
    ev=pd.read_csv(OUT/'metrics/matched1000_evolution.csv');fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,pr in enumerate(['eval','native_grpo']):
        for j,(metric,label,scale) in enumerate([('mean_PDMS','Mean PDMS',1),('feasible_rate','Feasible rate (%)',100),('centroid_displacement','Centroid displacement (m)',1)]):
            ax=axs[i,j];base=ev[(ev.method=='official_il')&(ev.protocol==pr)][metric].iloc[0]
            for method in CFG['methods']:
                for r in CFG['train_seeds']:
                    g=ev[(ev.method==method)&(ev.seed==r)&(ev.protocol==pr)].sort_values('step');ax.plot([0]+list(g.step),np.r_[base,g[metric]]*scale,color=COLORS[method],alpha=.55,lw=1,ls='-' if r==1701 else '--')
                ax.plot([],[],color=COLORS[method],label=LABELS[method])
            ax.set_title(pr+' / '+label);ax.set_xlabel('SFT optimizer updates');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);finish(fig,'Fig4_fixed1000_training_evolution',ev)
    sf=pd.read_parquet(OUT/'metrics/scene_metrics.parquet');sf=sf[(sf.split=='holdout')&((sf.step==512)|(sf.method=='official_il'))]
    ecdf=sf.groupby(['token','method','protocol'],as_index=False)[['pairwise_ADE64','centroid_displacement']].mean()
    fig,axs=plt.subplots(2,2,figsize=(11,8))
    for i,pr in enumerate(['eval','native_grpo']):
        for j,metric in enumerate(['pairwise_ADE64','centroid_displacement']):
            ax=axs[i,j]
            for m in methods:
                if metric=='centroid_displacement' and m=='official_il':continue
                vals=np.sort(ecdf[(ecdf.method==m)&(ecdf.protocol==pr)][metric].to_numpy());ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=LABELS[m],color=COLORS[m])
            ax.set_title(pr+' / '+metric);ax.set_xlabel('Meters (full observed range)');ax.set_ylabel('Fraction of scenes');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);finish(fig,'Fig5_scene_distribution_ECDF',ecdf)
if __name__=='__main__':main()
