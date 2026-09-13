"""Fixed endpoints and full holdout, with the old recipe shown explicitly."""
from common_r import *
import pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS=dict(official_il='Official IL (direct)',gt_only='GT-SFT',score='Score-MTS',pareto='Global-Pareto',gt_distance='GT-distance',old_pc='Old-PC-V2',conditional_pc='Conditional-PC')
COLORS=dict(zip(CFG['methods'],plt.get_cmap('tab10').colors[:7]))
def output(fig,name):
    d=OUT/'figures';d.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(d/f'{name}.png',dpi=200,bbox_inches='tight');fig.savefig(d/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    gain=pd.read_csv(OUT/'metrics/gain_bootstrap.csv');absolute=pd.read_csv(OUT/'metrics/holdout_bootstrap.csv');dyn=pd.read_csv(OUT/'metrics/training_safety_dynamics.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,metric,factor,label in zip(axes,['PDMS_gain','feasible_change','EP_change'],[1,100,1],['PDMS gain from own step 0','Feasibility change (pp)','EP change from own step 0']):
        for m in CFG['methods']:
            d=gain[(gain.method==m)&(gain.metric==metric)].sort_values('step');x=d.step.to_numpy();y=d['mean'].to_numpy()*factor
            ax.plot(x,y,'o-',lw=1.5,color=COLORS[m],label=LABELS[m]);ax.fill_between(x,d.ci_low.to_numpy()*factor,d.ci_high.to_numpy()*factor,color=COLORS[m],alpha=.07)
        ax.axhline(0,color='k',lw=.7);ax.set(xlabel='GRPO optimizer step',ylabel=label)
    axes[0].legend(fontsize=8);output(fig,'Fig-R1_Formal_GRPO_gain_and_safety')
    old=pd.read_csv(V3/'metrics/F_gain_bootstrap.csv');fig,ax=plt.subplots(figsize=(12,4.5));x=np.arange(7)
    for j,m in enumerate(CFG['methods']):
        d=gain[(gain.method==m)&(gain.metric=='PDMS_gain')&(gain.step==100)].iloc[0]
        ax.bar(j+.18,d['mean'],width=.34,color=COLORS[m],label='Formal stage3, step 100' if j==0 else None)
        ax.errorbar(j+.18,d['mean'],yerr=[[d['mean']-d.ci_low],[d.ci_high-d['mean']]],fmt='none',color='k',capsize=3)
        if m!='official_il':
            o=old[(old.method==m)&(old.metric=='PDMS_gain')&(old.step==100)].iloc[0]
            ax.bar(j-.18,o['mean'],width=.34,color='lightgray',label='Old V3 wrapper, step 100' if j==1 else None)
    ax.set_xticks(x);ax.set_xticklabels([LABELS[m] for m in CFG['methods']],rotation=15);ax.axhline(0,color='k',lw=.7);ax.set_ylabel('PDMS gain from own step 0');ax.legend();output(fig,'Fig-R2_Old_vs_formal_recipe')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in CFG['methods']:
        d=absolute[absolute.method==m].pivot(index='step',columns='metric',values='mean').sort_index()
        axes[0].plot(d.EP,d.feasible_rate*100,'o-',color=COLORS[m],label=LABELS[m])
        for step,row in d.iterrows():axes[0].annotate(str(step),(row.EP,row.feasible_rate*100),fontsize=7)
        axes[1].plot(d.CRN_policy_change_ADE,d.PDMS-d.PDMS.iloc[0],'o-',color=COLORS[m],label=LABELS[m])
    axes[0].set(xlabel='Holdout EP',ylabel='Holdout feasible rate (%)');axes[1].set(xlabel='CRN policy-change ADE (m)',ylabel='PDMS gain from own step 0');axes[1].axhline(0,color='k',lw=.7);axes[0].legend(fontsize=8);output(fig,'Fig-R3_EP_safety_and_policy_shift')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for m in CFG['methods']:
        d=dyn[dyn.method==m].groupby('step').sum(numeric_only=True)
        # Fixed 11-step epoch-sized window, common to all methods.
        den=d.infeasible_count.rolling(11,min_periods=1).sum();num=d.positive_infeasible_count.rolling(11,min_periods=1).sum()
        axes[0].plot(d.index,num/den.replace(0,np.nan),color=COLORS[m],label=LABELS[m])
        dd=dyn[dyn.method==m].groupby('step').mean(numeric_only=True)
        axes[1].plot(dd.index,dd.feasible_rate.rolling(11,min_periods=1).mean()*100,color=COLORS[m],label=LABELS[m])
    axes[0].set(xlabel='GRPO step',ylabel='Positive infeasible advantage rate');axes[1].set(xlabel='GRPO step',ylabel='Training feasible rollout rate (%)');axes[0].legend(fontsize=8);output(fig,'Fig-R4_Training_safety_and_advantage')
if __name__=='__main__':main()
