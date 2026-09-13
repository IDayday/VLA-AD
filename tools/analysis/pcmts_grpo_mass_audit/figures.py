"""Six figures drawn exclusively from completed real experiment tables."""
from common_mass import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=dict(zip(METHODS,['#d55e00','#cc79a7','#0072b2','#009e73']))
LABELS={'score':'Score','pareto':'Pareto','gt_geometry':'GT geometry','grpo_mass_pc':'GRPO mass PC'}
def export(fig,name,source):
    d=OUT/'figures';d.mkdir(exist_ok=True,parents=True);fig.tight_layout()
    for ext in ['png','svg','pdf']:
        path=d/f'{name}.{ext}';fig.savefig(path,dpi=190,bbox_inches='tight')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    source.to_csv(d/f'{name}_data.csv',index=False);plt.close(fig)
def ecdf(ax,values,method):
    a=np.sort(np.asarray(values,float));a=a[np.isfinite(a)]
    if len(a):ax.plot(a,np.arange(1,len(a)+1)/len(a),label=LABELS[method],color=COLORS[method])
def grouped_bars(ax,df,metric,pool_type='operational',scale=1):
    a=df[df.pool_type==pool_type].set_index('method').reindex(METHODS)
    ax.bar(range(4),a[metric]*scale,color=[COLORS[m] for m in METHODS]);ax.set_xticks(range(4),[LABELS[m] for m in METHODS],rotation=15,ha='right');ax.set_ylabel(metric+(' (%)' if scale==100 else ''))
def main():
    m=OUT/'metrics';selected=pd.read_parquet(m/'selected_pools_4x1000x16.parquet');selected=selected[selected.valid_mask.fillna(False)];summary=pd.read_csv(m/'method_summary.csv');pools=pd.read_csv(m/'pool_scene_metrics.csv');raw=pd.read_csv(m/'raw_source_summary.csv');noil=pd.read_csv(m/'no_il_native_control.csv');matches=pd.read_csv(m/'source_quality_matched.csv');sens=pd.read_csv(m/'sensitivity.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(16,4.3))
    for ax,pt in zip(axs[:2],['operational','strict']):
        # Every scenario is present; points are within-scene candidate means.
        for method in METHODS:
            g=pools[(pools.method==method)&(pools.pool_type==pt)]
            ax.scatter(g.candidate_PDMS,g.mean_candidate_p_B*100,s=8,alpha=.25,color=COLORS[method],label=LABELS[method])
        ax.set(xlabel='Candidate mean PDMS (points)',ylabel='Independent neighborhood mass (%)',title=pt+' pools',xlim=(-1,101),ylim=(-.0001,101));ax.set_yscale('symlog',linthresh=.001);ax.axhline(3,color='grey',ls='--')
        if pt=='strict' and not pools[(pools.method=='grpo_mass_pc')&(pools.pool_type==pt)].candidate_PDMS.notna().any():
            ax.text(.03,.93,'PC: 0 / 1000 nonempty strict pools',transform=ax.transAxes,color=COLORS['grpo_mass_pc'])
    axs[0].legend(fontsize=8);grouped_bars(axs[2],summary,'strict_16_scene_fraction',scale=100);axs[2].set_title('Strict 16 coverage / Full 1000')
    export(fig,'Fig1_quality_sampling_coverage',pools[['token','method','pool_type','candidate_PDMS','mean_candidate_p_B','strict_count','empty_scene']])
    fig,axs=plt.subplots(1,3,figsize=(16,4.3))
    for method in METHODS:
        g=selected[selected.method==method];ecdf(axs[0],g.p_B*100,method);axs[1].scatter(g.p_A*100,g.p_B*100,s=2,alpha=.06,color=COLORS[method])
    axs[0].axvline(3,ls='--',color='grey');axs[0].set(xlabel='Candidate p_B (%)',ylabel='Candidate ECDF',xlim=(-1,101));axs[0].legend(fontsize=8)
    axs[1].plot([0,100],[0,100],'k--',lw=1);axs[1].set(xlabel='Selection p_A (%)',ylabel='Independent p_B (%)',xlim=(-1,101),ylim=(-1,101));grouped_bars(axs[2],summary,'zero_hit_fraction',scale=100)
    axs[0].set_xscale('symlog',linthresh=.1);axs[1].set_xscale('symlog',linthresh=.1);axs[1].set_yscale('symlog',linthresh=.1)
    export(fig,'Fig2_independent_confirmation',selected[['token','method','candidate_id','p_A','p_B','hit_count_B','selection_optimism','strict_qualified']])
    fig,axs=plt.subplots(1,3,figsize=(16,4.3));grouped_bars(axs[0],summary,'strict_16_scene_fraction',scale=100)
    for method in METHODS:ecdf(axs[1],pools[(pools.method==method)&(pools.pool_type=='operational')].strict_count,method)
    axs[1].set(xlabel='Strict candidate count',ylabel='Scene ECDF',xlim=(-.5,16.5));axs[1].legend(fontsize=8)
    a=summary[summary.pool_type=='operational'].set_index('method').reindex(METHODS);bottom=np.zeros(4)
    for level in range(4):
        v=a[f'level_{level}_fraction'].to_numpy()*100;axs[2].bar(range(4),v,bottom=bottom,label=['Strict','Method relaxed','Quality relaxed','Safety relaxed'][level]);bottom+=v
    axs[2].set_xticks(range(4),[LABELS[x] for x in METHODS],rotation=15,ha='right');axs[2].set_ylabel('Operational slots (%)');axs[2].legend(fontsize=8)
    export(fig,'Fig3_strict_coverage_fallback',pools[pools.pool_type=='operational'])
    fig,axs=plt.subplots(1,3,figsize=(16,4.3));sources=['GT','GT_structured','ddv2','drivor','IL_native_C']
    axs[0].bar(raw.source,raw.unique_candidates);axs[0].tick_params(axis='x',rotation=30);axs[0].set_ylabel('Raw unique candidates (exclusive attribution)')
    for ax,table,title in [(axs[1],summary,'Primary shared reservoir'),(axs[2],noil,'C-exclusive candidates removed')]:
        a=table[table.pool_type=='operational'].set_index('method').reindex(METHODS);bottom=np.zeros(4)
        for source in sources:
            v=a[f'source_{source}_fraction'].to_numpy()*100;ax.bar(range(4),v,bottom=bottom,label=source);bottom+=v
        ax.set_xticks(range(4),[LABELS[x] for x in METHODS],rotation=15,ha='right');ax.set(title=title,ylabel='Fractional source attribution (%)')
    axs[2].legend(fontsize=8)
    export(fig,'Fig4_sources_and_no_native',pd.concat([raw.assign(table='raw'),summary.assign(table='primary'),noil.assign(table='no_native')],ignore_index=True))
    fig,axs=plt.subplots(1,3,figsize=(16,4.3));grouped_bars(axs[0],summary,'PoolMass',scale=100);grouped_bars(axs[1],summary,'QualityMatchedPoolMass',scale=100)
    for method in METHODS:
        g=pools[(pools.method==method)&(pools.pool_type=='operational')];ecdf(axs[2],g.candidate_local_gain,method)
    axs[2].set(xlabel='Local expert gain (points; hits >= 10)',ylabel='Scene ECDF')
    if pools.candidate_local_gain.notna().any():
        axs[2].axvline(.5,ls='--',color='grey');axs[2].legend(fontsize=8)
    else:
        axs[2].text(.5,.5,'UNESTIMABLE\n0 candidates with >= 10 B hits\nAll four methods; all 1000 scenes',ha='center',va='center',transform=axs[2].transAxes)
        axs[2].set_xticks([]);axs[2].set_yticks([])
    export(fig,'Fig5_pool_coverage_expert_gain',pools[['token','method','pool_type','PoolMass','QualityMatchedPoolMass','candidate_local_gain','local_gain_defined_candidate_count','local_gain_defined_scene_count','near_duplicate_representatives_0_10m']])
    fig,axs=plt.subplots(1,3,figsize=(16,4.3));g=matches[(matches.pool_type=='operational')&(matches.metric=='p_B')].set_index('comparator').reindex(METHODS[:3])
    defined=g.scene_equal_mean.notna()
    if defined.any():
        h=g[defined];axs[0].errorbar(np.flatnonzero(defined),h.scene_equal_mean*100,yerr=[np.maximum(0,(h.scene_equal_mean-h.scene_equal_ci_low)*100),np.maximum(0,(h.scene_equal_ci_high-h.scene_equal_mean)*100)],fmt='o',capsize=4)
        for j in np.flatnonzero(~defined):
            axs[0].text(j,.08,'NA: no distinct\nmatched candidates',ha='center',fontsize=8,transform=axs[0].get_xaxis_transform())
    else:axs[0].text(.5,.5,'No matched common support',ha='center',transform=axs[0].transAxes)
    axs[0].axhline(0,color='grey');axs[0].set_xticks(range(3),[LABELS[x] for x in METHODS[:3]],rotation=15);axs[0].set_ylabel('Source/quality matched p_B difference (pp)')
    all_s=pd.concat([summary.assign(variant='primary'),sens],ignore_index=True);vlookup={v['name']:v for v in read(OUT/'manifests/selection_frozen.json')['variants']};all_s['epsilon']=all_s.variant.map(lambda x:vlookup[x]['epsilon']);all_s['pmin']=all_s.variant.map(lambda x:vlookup[x]['pmin'])
    pc=all_s[(all_s.method=='grpo_mass_pc')&(all_s.pool_type=='operational')&~all_s.variant.str.startswith('gt')]
    for p in [.01,.03,.05]:
        g=pc[pc.pmin==p].sort_values('epsilon');axs[1].plot(g.epsilon,g.PoolMass*100,'o-',label=f'p_min {p:.0%}');axs[2].plot(g.epsilon,g.candidate_PDMS,'o-',label=f'p_min {p:.0%}')
    axs[1].set(xlabel='ADE neighborhood radius (m)',ylabel='PoolMass (%)',ylim=(-1,101));axs[2].set(xlabel='ADE neighborhood radius (m)',ylabel='Candidate PDMS (points)',ylim=(-1,101));axs[1].legend(fontsize=8)
    export(fig,'Fig6_matching_and_sensitivity',pd.concat([matches.assign(table='matched'),all_s.assign(table='sensitivity')],ignore_index=True))
if __name__=='__main__':main()
