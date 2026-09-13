"""Export PNG/PDF figures with CSV plotting inputs; no result-conditioned subsets."""
import argparse
from common_v3 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COLORS={'gt_only':'#777777','score':'#d65c55','pareto':'#8f63ba','gt_distance':'#4e9f80','old_pc':'#c09b3f','conditional_pc':'#2675b9'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
def output(fig,number,title,table):
    name=f'Fig-V3-{number}_{title}';csv(table,f'{name}_plot_data.csv')
    for suffix in ['png','pdf']:fig.savefig(OUT/'figures'/f'{name}.{suffix}',dpi=190)
    plt.close(fig)
def static():
    d=pd.read_csv(OUT/'metrics/A_eligibility_funnel.csv');table=d.groupby('stage')['count'].mean().reindex(['raw','hard_safety','compatible','quality','local','global_front2','conditional_front2']).reset_index();r=pd.read_csv(OUT/'metrics/A_rescued_candidates.csv')
    fig,ax=plt.subplots(1,2,figsize=(14,4.8),layout='constrained');ax[0].barh(np.arange(len(table)),table['count'],color=['#888']*5+['#d65c55','#2675b9']);ax[0].set_yticks(np.arange(len(table)),table.stage);ax[0].invert_yaxis();ax[0].set_xlabel('Unique candidates per scene');ax[0].set_title('Full-1000 eligibility funnel');ax[0].set_xlim(0,210);ax[1].tick_params(axis='y',labelsize=8)
    for j,n in enumerate(table['count']):ax[0].text(n+.5,j,f'{n:.2f}',va='center',fontsize=9)
    c=r.groupby('source').size().sort_values();ax[1].barh(c.index,c.values,color='#2675b9');ax[1].set_xlabel('Rescued candidates');ax[1].set_title(f'{len(r):,} rescued / {r.token.nunique()} scenes\nFar-dominator rate {r.far_dominator_poisoning.mean():.1%}');output(fig,'1','Gate_order',table)
    d=pd.read_csv(OUT/'metrics/B_frontier_bootstrap.csv');d=d[d.feasibility=='conservative'];fig,ax=plt.subplots(1,3,figsize=(13,3.7))
    for a,metric,label in zip(ax,['best_PDMS','compatibility_tax','coverage'],['Retained oracle PDMS','Compatibility tax (points)','Scene coverage']):
        g=d[d.metric==metric];a.plot(g.q_threshold,g['mean'],'o-',color='#2675b9');a.fill_between(g.q_threshold,g.ci_low,g.ci_high,alpha=.2,color='#2675b9');a.set_xlabel('q threshold (calibrated rank, not probability)');a.set_ylabel(label);a.axvline(95,color='#777',linestyle='--',linewidth=.8)
    fig.suptitle('Raw reservoir; NC/DAC/TTC/DDC feasible, Full-1000');fig.tight_layout();output(fig,'2','Quality_compatibility_frontier',d)
    if (OUT/'metrics/C_paired_bootstrap.csv').exists():
        d=pd.read_csv(OUT/'metrics/C_paired_bootstrap.csv');g=d[(d.protocol=='primary')&(d.near_region=='boundary')];fig,ax=plt.subplots(1,3,figsize=(11,3.8))
        for a,metric,label in zip(ax,['diffusion_loss','E_rec','return_rate'],['Native diffusion MSE','Reconstruction ADE (m)','Return-to-candidate rate']):
            r=g[g.metric==metric].iloc[0];a.bar([0,1],[r.near_mean,r.far_mean],color=['#2675b9','#d65c55']);a.set_xticks([0,1],['Boundary','Far']);a.set_ylabel(label);a.set_title(f'Far − Boundary: {r["mean"]:.4g}\ncluster CI [{r.cluster_ci_low:.4g}, {r.cluster_ci_high:.4g}]',fontsize=10)
        fig.suptitle('Same scene, exact source and PDMS gap ≤ 0.25; common noise');fig.tight_layout();output(fig,'3','Matched_learnability',g)
    if (OUT/'metrics/D_method_bootstrap.csv').exists():
        d=pd.read_csv(OUT/'metrics/D_method_bootstrap.csv');matched=pd.read_csv(OUT/'metrics/D_matched_bootstrap.csv');fig,ax=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
        for a,metric,label in zip(ax[0],['gradient_norm','cosine','predicted_delta_L_ref'],['Gradient L2 norm','Alignment with retention reference','Predicted reference loss change']):
            g=d[d.metric==metric].set_index('method').reindex(METHODS);x=np.arange(5);a.bar(x,g['mean'],color=[COLORS[m] for m in METHODS]);a.errorbar(x,g['mean'],yerr=np.stack([g['mean']-g.ci_low,g.ci_high-g['mean']]),fmt='none',color='black',capsize=3);a.set_xticks(x,[LABELS[m] for m in METHODS],rotation=25,ha='right');a.set_ylabel(label)
        ax[0,1].set_title('Method pools: source composition differs')
        for a,metric,label in zip(ax[1],['gradient_norm','cosine','predicted_delta_L_ref'],['Paired gradient norm difference','Paired alignment difference','Paired reference-loss-change difference']):
            r=matched[matched.metric==metric].iloc[0];a.axhline(0,color='#777',linewidth=1,linestyle='--');a.errorbar([0],[r['mean']],yerr=[[r['mean']-r.cluster_ci_low],[r.cluster_ci_high-r['mean']]],fmt='o',color='#2675b9',capsize=6);a.set_xlim(-.6,.6);a.set_xticks([0],['Far minus compatible']);a.set_ylabel(label);a.set_title(f'Same scene/source/PDMS; {int(r.n)} pairs',fontsize=9)
        output(fig,'4','Gradient_interference',pd.concat([d.assign(panel='method_pool'),matched.assign(panel='source_quality_matched')],ignore_index=True))
def training():
    if (OUT/'metrics/E_seed_means.csv').exists():
        d=pd.read_csv(OUT/'metrics/E_seed_means.csv');fig,ax=plt.subplots(1,2,figsize=(12,4))
        for method,g in d.groupby('method'):
            for sd,h in g.groupby('seed'):
                h=h.sort_values('step');gain=h.PDMS-h.PDMS.iloc[0];style='o-' if sd==1701 else 's--';ax[0].plot(h.center_shift,gain,style,color=COLORS[method],alpha=.7,label=LABELS[method] if sd==1701 else None);ax[1].plot(h.step,h.IL_retention_loss/h.IL_retention_loss.iloc[0],style,color=COLORS[method],alpha=.7)
        ax[0].set(xlabel='Center shift from official IL (m)',ylabel='Holdout PDMS gain (points)');ax[1].set(xlabel='Micro-SFT optimizer update',ylabel='IL-native retention loss / step0');ax[0].legend(fontsize=8)
        from matplotlib.lines import Line2D
        ax[1].legend(handles=[Line2D([0],[0],color='#555',marker='o',label='Seed 1701'),Line2D([0],[0],color='#555',marker='s',linestyle='--',label='Seed 2903')],fontsize=8)
        fig.tight_layout();output(fig,'5','Micro_SFT_shift',d)
    if (OUT/'metrics/F_gain_bootstrap.csv').exists():
        gain=pd.read_csv(OUT/'metrics/F_gain_bootstrap.csv');safety=pd.read_csv(OUT/'metrics/F_seed_means.csv');pia=pd.read_csv(OUT/'metrics/F_training_safety_dynamics.csv');fig,ax=plt.subplots(2,2,figsize=(12,8))
        for method in CFG['training']['methods']:
            g=gain[(gain.method==method)&(gain.metric=='PDMS_gain')];c=COLORS[method];ax[0,0].plot(g.step,g['mean'],'o-',color=c,label=LABELS[method]);ax[0,0].fill_between(g.step,g.ci_low,g.ci_high,color=c,alpha=.1)
            g=safety[safety.method==method].groupby('step').mean(numeric_only=True);ax[0,1].plot(g.index,g.feasible_rate,'o-',color=c);ax[1,0].plot(g.EP,g.feasible_rate,'o-',color=c)
            xy=g[['EP','feasible_rate']].to_numpy()
            for i in range(len(xy)-1):ax[1,0].annotate('',xy=xy[i+1],xytext=xy[i],arrowprops=dict(arrowstyle='->',color=c,lw=1,alpha=.8))
            g=pia[pia.method==method].groupby('step')[['positive_infeasible_count','infeasible_count']].sum();num=g.positive_infeasible_count.rolling(10,min_periods=1).sum();den=g.infeasible_count.rolling(10,min_periods=1).sum();ax[1,1].plot(g.index,num/den.replace(0,np.nan),color=c)
        ax[0,0].set(xlabel='GRPO optimizer update',ylabel='Holdout PDMS gain from SFT initialization');ax[0,1].set(xlabel='GRPO optimizer update',ylabel='Holdout feasible rate');ax[1,0].set(xlabel='Holdout EP',ylabel='Holdout feasible rate');ax[1,1].set(xlabel='GRPO optimizer update',ylabel='Positive infeasible advantage rate (10-step window)');ax[0,0].legend(fontsize=8);fig.tight_layout();output(fig,'6','GRPO_gain_safety',gain)
    if (OUT/'metrics/G_promotion_bootstrap.csv').exists():
        d=pd.read_csv(OUT/'metrics/G_promotion_bootstrap.csv');fig,ax=plt.subplots(1,2,figsize=(11,4));labels=['sft_final','grpo_middle','grpo_final']
        for a,metric in zip(ax,['frontier_promotion_rate','newly_eligible_PC_rate']):
            for scope,style in [('Full-1000','-'),('holdout-300','--')]:
                for sd,color in [(1701,'#2675b9'),(2903,'#d65c55')]:
                    g=d[(d.scope==scope)&(d.metric==metric)].set_index('run').reindex([f'{l}_seed{sd}' for l in labels]);a.plot(range(3),g['mean'],style+'o',color=color,label=f'{scope}, seed {sd}')
            if (OUT/'metrics/G_promotion_vs_null.csv').exists():
                null=pd.read_csv(OUT/'metrics/G_promotion_vs_null.csv');nm='null_promotion_rate' if metric=='frontier_promotion_rate' else 'null_newly_eligible_rate';n=null[(null.run=='unchanged_IL')&(null.scope=='holdout-300')&(null.metric==nm)].iloc[0];a.axhline(n['mean'],color='#555',linestyle=':',label='Unchanged IL null, holdout')
            a.set_xticks(range(3),labels,rotation=15);a.set_ylabel(metric);a.set_ylim(bottom=0)
        ax[0].legend(fontsize=8);fig.tight_layout();output(fig,'7','Frontier_promotion',d)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--static',action='store_true');args=a.parse_args();static()
    if not args.static:training()
