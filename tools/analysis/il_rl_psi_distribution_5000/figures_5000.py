"""Figures from measured FULL5000 and separately reported NEW4000 statistics."""
from common_5000 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'official_il':'Initial GT-IL','original_grpo_11970':'After original GRPO','psi_sft':'After PSI candidate SFT'}
COLORS={'official_il':'#64748b','original_grpo_11970':'#1874b5','psi_sft':'#cc7241'}

def finish(fig,name,source):
    out=OUT/'figures';out.mkdir(exist_ok=True)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    source.to_csv(out/f'{name}.csv',index=False);plt.close(fig)

def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    summary=pd.read_csv(OUT/'metrics/summary.csv');full=summary[summary.scope=='FULL5000']
    comparisons=pd.read_csv(OUT/'metrics/paired_comparisons.csv')
    metrics=[('pairwise_ADE64','Pairwise ADE, 64 samples (m)',1),('centroid_displacement','Centroid displacement from IL (m)',1),('feasible_rate','Conservative feasible samples (%)',100),('mean_PDMS','Mean sampled PDMS (points)',1)]
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for i,protocol in enumerate(CFG['protocols']):
        data=full[full.protocol==protocol].set_index('model').loc[CFG['primary_models']]
        for j,(metric,label,scale) in enumerate(metrics):
            ax=axes[i,j];values=data[metric].to_numpy()*scale
            ax.bar(range(3),values,color=[COLORS[m] for m in data.index],width=.65)
            for x,v in enumerate(values):ax.text(x,v,f'{v:.3f}' if scale==1 and 'ADE' in metric or metric=='centroid_displacement' else f'{v:.2f}',ha='center',va='bottom',fontsize=9)
            ax.set_xticks(range(3),['GT-IL','GRPO','PSI-SFT']);ax.set_title(label);ax.set_ylim(0,max(values)*1.18 if max(values)>0 else 1)
            if j==0:ax.set_ylabel('Evaluation sampler' if protocol=='eval' else 'Native GRPO sampler (G16)')
    fig.suptitle('5,000 frozen scenes; 64 samples per checkpoint and sampler; no weight updates',y=1.02)
    finish(fig,'Fig1_distribution_and_quality',full)

    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,protocol in zip(axes,CFG['protocols']):
        data=full[full.protocol==protocol].set_index('model')
        for i,m in enumerate(CFG['primary_models']):
            vals=[data.loc[m,k] for k in ['min_PDMS','mean_PDMS','max_PDMS']]
            ax.plot(range(3),vals,'o-',color=COLORS[m],label=LABELS[m])
            for x,v in enumerate(vals):ax.annotate(f'{v:.2f}',(x,v),xytext=(3,(-12 if i==2 else 5)),textcoords='offset points',fontsize=8)
        ax.set_xticks(range(3),['Group minimum','Group mean','Group maximum'])
        ax.set_ylabel('PDMS (0–100 points)');ax.set_ylim(0,103);ax.set_title(protocol);ax.legend(fontsize=8)
    fig.suptitle('Quality tails: statistics within each real G16 group, then scene-equal means')
    finish(fig,'Fig2_group16_quality_tails',full[['scope','model','protocol','scenes','groups','min_PDMS','mean_PDMS','max_PDMS','CVaR25_PDMS','std_PDMS']])

    fig,axes=plt.subplots(2,3,figsize=(14,8))
    forest=comparisons[comparisons.reference=='official_il']
    for i,protocol in enumerate(CFG['protocols']):
        for j,(metric,label,scale) in enumerate([('mean_PDMS','PDMS change (points)',1),('feasible_rate','Feasibility change (percentage points)',100),('pairwise_ADE64','Pairwise ADE change (m)',1)]):
            ax=axes[i,j];labels=[];pos=0
            for model in CFG['primary_models'][1:]:
                for scope in ['OLD1000','NEW4000','FULL5000']:
                    r=forest[(forest.scope==scope)&(forest.protocol==protocol)&(forest.model==model)&(forest.metric==metric)].iloc[0]
                    ax.errorbar(r.mean_difference*scale,pos,xerr=np.array([[r.mean_difference-r.ci_low],[r.ci_high-r.mean_difference]])*scale,fmt='o',color=COLORS[model],capsize=3)
                    labels.append(f'{"GRPO" if model.startswith("original") else "PSI"} / {scope}');pos+=1
                pos+=1;labels.append('')
            ax.axvline(0,color='.5',lw=1,ls='--');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlabel(label);ax.set_title(protocol)
    fig.suptitle('Replication without selecting outcomes: paired differences from IL, scene-bootstrap 95% CI',y=1.02)
    finish(fig,'Fig3_independent_replication',forest)

    scenes_df=pd.read_parquet(OUT/'metrics/scene_metrics.parquet')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    cdf_metrics=[('pairwise_ADE64','Per-scene pairwise ADE64 (m)'),('centroid_displacement','Centroid displacement from IL (m)'),('mean_PDMS','Per-scene mean sampled PDMS')]
    for i,protocol in enumerate(CFG['protocols']):
        for j,(metric,label) in enumerate(cdf_metrics):
            ax=axes[i,j]
            for model in CFG['primary_models']:
                vals=np.sort(scenes_df[(scenes_df.model==model)&(scenes_df.protocol==protocol)][metric].dropna().to_numpy())
                ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=LABELS[model],color=COLORS[model])
            ax.set_xlabel(label);ax.set_ylabel('Fraction of 5,000 scenes');ax.set_title(protocol)
            ax.set_ylim(0,1);ax.set_xlim(left=0)
            if metric=='mean_PDMS':ax.set_xlim(0,100)
            ax.legend(fontsize=7)
    fig.suptitle('Full-range scene distributions; no scene omitted or selected by outcome',y=1.01)
    finish(fig,'Fig4_scene_distribution_ECDF',scenes_df[['token','log','cohort','command','model','protocol',*[m[0] for m in cdf_metrics]]])

if __name__=='__main__':main()
