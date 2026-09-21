from shared import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'official_il':'Initial IL','original_grpo_11970':'After original GRPO'}
COLORS={'official_il':'#2878B5','original_grpo_11970':'#D35E22'}
def savefig(fig,name,data):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    data.to_csv(folder/f'{name}_data.csv',index=False);plt.close(fig)
def lines(ax,s,column,label,scale=1):
    for m,q in s.groupby('model'):
        ax.plot(q.G,q[column]*scale,'o-',color=COLORS[m],label=LABELS[m])
    ax.set(xlabel='Group size G (nested real samples)',ylabel=label,xticks=[8,16,32,64,128]);ax.set_xscale('log',base=2)
    ax.set_xticks([8,16,32,64,128],labels=['8','16','32','64','128']);ax.grid(alpha=.2)
def main():
    s=pd.read_csv(OUT/'metrics/method_summary.csv');s=s[s.grouping=='prefix']
    fig,axes=plt.subplots(2,3,figsize=(14,7.5))
    for ax,col,lab,scale in zip(axes.flat,['mean_PDMS','max_PDMS','feasible_rate','no_safe_group','safe_EP_opportunity','safe_EP_headroom'],['Mean sampled PDMS (points)','Best sampled PDMS (points)','Feasible samples (%)','No feasible member (%)','Safe EP opportunity groups (%)','Safe EP headroom (normalized points)'],[1,1,100,100,100,100]):lines(ax,s,col,lab,scale)
    axes[0,0].legend();savefig(fig,'Fig1_GroupSize_Quality_Opportunity',s)
    component_rows=[]
    for _,r in s[s.G==16].iterrows():
        for n in ['NC','DAC','EP','TTC','DDC','Comfort']:
            component_rows.append(dict(model=r.model,component=n,mean=r[n+'_mean'],std=r[n+'_std'],all_pass=r[n+'_all_pass'],all_fail=r[n+'_all_fail'],mixed=r[n+'_mixed'],positive_mean=r[n+'_positive_mean'],adv_cov=r['covariance_adv_'+n]))
    c=pd.DataFrame(component_rows);fig,axes=plt.subplots(1,3,figsize=(15,4.5));names=['NC','DAC','EP','TTC','DDC','Comfort'];x=np.arange(6)
    for k,m in enumerate(CFG['models']):
        q=c[c.model==m].set_index('component').loc[names]
        axes[0].bar(x+(k-.5)*.35,q['mean']*100,.35,color=COLORS[m],label=LABELS[m])
        axes[1].bar(x+(k-.5)*.35,q['mixed']*100,.35,color=COLORS[m])
        axes[2].bar(x+(k-.5)*.35,q['adv_cov'],.35,color=COLORS[m])
    for ax in axes:ax.set_xticks(x,labels=names);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Component mean (0-100)');axes[0].legend();axes[1].set_ylabel('Groups with pass and non-pass (%)');axes[2].set_ylabel('Mean Cov(native advantage, component)');axes[2].axhline(0,color='k',lw=.5)
    savefig(fig,'Fig2_Component_Distributions_Credit',c)
    fig,axes=plt.subplots(2,3,figsize=(14,7.5))
    for ax,col,lab,scale in zip(axes.flat,['safety_pattern_disagreement','effective_safety_patterns','component_tradeoff_pair_fraction','zero_advantage_group','positive_infeasible_rate','unsafe_reward_winner_with_safe_alternative'],['Safety-outcome disagreement (%)','Observed effective outcome patterns','Opposing-component pairs (%)','No reward contrast groups (%)','P(positive A | infeasible) (%)','Unsafe winner with safe alternative (%)'],[100,1,100,100,100,100]):lines(ax,s,col,lab,scale)
    axes[0,0].legend();savefig(fig,'Fig3_Outcome_Diversity_Reward_Limitations',s)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,col,lab in zip(axes,['mean_pairwise_ADE','medoid_radius_P90','covariance_effective_rank'],['Mean pairwise ADE (m)','P90 distance to medoid (m)','XY covariance effective rank']):lines(ax,s,col,lab)
    for ax in axes:ax.set_ylim(bottom=0)
    axes[2].set_ylim(0,16)
    axes[0].legend();savefig(fig,'Fig4_Geometric_Width',s)
    paths=[OUT/'metrics'/f'noise_decomposition_{m}.csv' for m in CFG['models']]
    if all(p.exists() for p in paths):
        n=pd.concat([pd.read_csv(p) for p in paths]);means=n.groupby('model').mean(numeric_only=True)
        fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(3);cols=['pairwise_ADE','pre_final_noise_mean_pairwise_ADE','residual_pairwise_ADE']
        for k,m in enumerate(CFG['models']):
            axes[0].bar(x+(k-.5)*.35,means.loc[m,cols].to_numpy(float),.35,color=COLORS[m],label=LABELS[m])
            z=n[n.model==m];axes[1].scatter(z.pre_final_noise_mean_pairwise_ADE,z.pairwise_ADE,alpha=.5,label=LABELS[m],color=COLORS[m])
        axes[0].set_xticks(x,labels=['Sample output','Before final noise','Observed residual']);axes[0].set_ylabel('Pairwise ADE (m)');axes[0].legend()
        axes[1].set(xlabel='Before final noise: pairwise ADE (m)',ylabel='Actual output: pairwise ADE (m)');savefig(fig,'Fig5_FinalStep_Noise_Decomposition',n)
if __name__=='__main__':main()
