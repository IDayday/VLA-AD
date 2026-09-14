"""Only plots measured June/July historical data; never filters winning cases."""
from analyze import OUT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

F=OUT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
colors=['#177e89','#6a4c93','#e07a36','#b23a48']
labels=['Official IL → original GRPO','IL → candidate SFT → SR-PGRPO','Random multi-target A5 → LFP-GRPO','Random multi-target V6 → LFP-GRPO']

def finish(fig,name,data):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=180,bbox_inches='tight')
    svg=F/(name+'.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    data.to_csv(F/(name+'.csv'),index=False);plt.close(fig)

def main():
    curve=pd.read_csv(OUT/'metrics/historical_curves.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,fam,label,color in zip(axes.flat,['official_original','psi_sr','a5_lfp','v6_lfp'],labels,colors):
        d=curve[curve.family==fam].sort_values('step');ax.plot(d.step,d.PDMS,'o-',color=color,ms=3,label='Historical checkpoints')
        ax.axhline(d.iloc[0].PDMS,color=color,ls=':',lw=1);ax.set(title=label,xlabel='Optimizer updates',ylabel='NAVSIM-v1 PDMS (points)',ylim=(84,92))
        ax.grid(alpha=.2)
        if fam=='a5_lfp':
            w=curve[curve.family=='a5_watcher'];ax.plot(w.step,w.PDMS,'x--',color='gray',label='Watcher: separate, non-CRN evaluation');ax.legend(fontsize=8)
        note='Common-noise seed 0 reevaluation' if fam in ['a5_lfp','v6_lfp'] else 'Historical evaluations: baseline and RL not CRN'
        ax.text(.03,.04,note,transform=ax.transAxes,fontsize=8)
    finish(fig,'Fig-H1_historical_curves',curve)
    pairs=pd.read_csv(OUT/'metrics/paired_comparisons.csv');tail=pd.read_csv(OUT/'metrics/tail_decomposition.csv')
    selected=['official_11970','psi_21000','a5_e2','v6_s3300'];names=['Official: step 11970','PSI: step 21000','A5: step 4842','V6: step 3300']
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    rows=[]
    for i,(key,label,col) in enumerate(zip(selected,names,colors)):
        for j,m in enumerate(['EP','NC','TTC','DAC']):
            r=pairs[(pairs.comparison==key)&(pairs.metric==m)].iloc[0];x=j+(i-1.5)*.18
            axes[0].errorbar(x,r.mean_difference,yerr=[[r.mean_difference-r.ci_low],[r.ci_high-r.mean_difference]],fmt='o',color=col,label=label if j==0 else None)
            rows.append(r.to_dict())
    axes[0].set(xticks=range(4),xticklabels=['EP','NC','TTC','DAC'],ylabel='Change vs own SFT (percentage points)',title='Progress versus safety; scene-cluster 95% CI')
    axes[0].axhline(0,color='gray',lw=.8);axes[0].legend(fontsize=8)
    t=tail.set_index('comparison').loc[selected]
    axes[1].barh(np.arange(4),t.safety_regression_contribution,color='#b23a48',label='Records with NC/DAC/TTC regression')
    axes[1].barh(np.arange(4),t.nonregression_contribution,color='#177e89',label='All remaining records')
    axes[1].scatter(t.total_delta,np.arange(4),marker='D',color='black',label='Total PDMS gain',zorder=4)
    axes[1].set(yticks=range(4),yticklabels=names,xlabel='Contribution to overall PDMS change (points)',title='Exact accounting of the safety tail')
    axes[1].axvline(0,color='gray',lw=.8);axes[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.15))
    finish(fig,'Fig-H2_progress_and_safety',pd.concat([pd.DataFrame(rows),t.reset_index()],ignore_index=True))
    phase=pd.read_csv(OUT/'metrics/training_phase_summary.csv');d=phase[phase.phase=='epoch_observation']
    fig,axes=plt.subplots(1,3,figsize=(13,4));sources=[]
    for run,col,label in [('a5_lfp',colors[2],'A5'),('v6_lfp',colors[3],'V6')]:
        h=d[(d.run==run)&(d.tag=='train/lfp_delta_scalar_mean_epoch')];axes[0].plot(h.step+1,h['mean']*100,'o-',color=col,label=label);sources.append(h)
    axes[0].set(title='Training sampled PDMS − frozen reference',xlabel='Optimizer updates',ylabel='Points; curriculum-weighted train data');axes[0].legend()
    for ax,tag,title,ylabel in [(axes[1],'train/lfp_group_pairwise_ade_m_mean_epoch','V6: actual GRPO group spread','Pairwise ADE (m)'),
                              (axes[2],'train/lfp_group_scalar_span_mean_epoch','V6: within-group quality range','PDMS range (points)')]:
        h=d[(d.run=='v6_lfp')&(d.tag==tag)];scale=100 if 'scalar' in tag else 1
        ax.plot(h.step+1,h['mean']*scale,'o-',color=colors[3]);ax.set(title=title,xlabel='Optimizer updates',ylabel=ylabel);ax.set_ylim(bottom=0);sources.append(h)
    for ax in axes:ax.grid(alpha=.2)
    finish(fig,'Fig-H3_formal_training_diagnostics',pd.concat(sources,ignore_index=True))

if __name__=='__main__':main()
