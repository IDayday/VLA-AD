from common import *
from clustered_stats import interval,summarise
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 f=pd.read_parquet(OUT/'trajectory_credit.parquet');rows=[];eps=cfg()['component_equality_atol'];decomp=[]
 for keys,b in f.groupby(['model','protocol','split','token','log','group']):
  ident=dict(zip(['model','protocol','split','token','log','group'],keys));feas=b[b.feasible];pos=np.maximum(b.advantage.to_numpy(),0);mass=pos.sum();active=mass>1e-8
  component_only=len(feas)>=2 and all(np.ptp(feas[x])<=eps for x in ['NC','DAC','DDC','TTC','Comfort']) and np.ptp(feas.EP)>eps
  strict=component_only and all(feas[x].nunique()==1 for x in ['feasible','progress_ok','ttc_ok','pareto','positive_eligible'])
  rec=dict(ident,all_infeasible=len(feas)==0,mixed_feasible=0<len(feas)<16,all_feasible=len(feas)==16,all_full_pdms=bool((b.PDMS==1).all()),all_ep_saturated=bool((b.EP==1).all()),all_components_identical=all(np.ptp(b[x])<=eps for x in ['NC','DAC','TTC','Comfort','DDC','EP']),
   ep_only_components=component_only,ep_only_identical_gates=strict,positive_mass=mass,ep_only_strict_mass=mass*strict,ep_only_components_mass=mass*component_only,information_active=bool(active and (b.advantage< -1e-8).any()),positive_count=int((pos>0).sum()),feasible_comparison=len(feas)>=2)
  worsen=(b.EP>b.reference_ep+eps)&((b.TTC<b.reference_ttc-eps)|(b.Comfort<b.reference_quality-eps))
  rec['positive_ep_up_safety_or_comfort_down_vs_reference_count']=int(((pos>0)&worsen).sum());rec['positive_ep_up_safety_or_comfort_down_vs_reference_mass']=float(pos[worsen].sum())
  for field in ['raw_progress','end_speed']:
   rec['positive_weighted_'+field+'_shift']=float(np.sum(pos*(b[field]-b[field].mean()))/mass) if active else np.nan
  for shadow in ['fixed','full']:
   alt=b['shadow_'+shadow+'_advantage'].to_numpy();rec['shadow_'+shadow+'_mass']=float(np.maximum(alt,0).sum());rec['shadow_'+shadow+'_positive_disappeared']=int(((pos>1e-8)&(alt<=1e-8)).sum());rec['shadow_'+shadow+'_positive_flipped']=int(((pos>1e-8)&(alt< -1e-8)).sum());rec['shadow_'+shadow+'_information_active']=bool((alt>1e-8).any() and (alt< -1e-8).any())
   rec['shadow_'+shadow+'_rank_changed']=not np.array_equal(b.advantage.rank(method='min').to_numpy(),pd.Series(alt).rank(method='min').to_numpy())
   rec['shadow_'+shadow+'_spearman']=float(spearmanr(b.advantage,alt).statistic) if np.ptp(alt)>eps and np.ptp(b.advantage)>eps else np.nan
  rows.append(rec)
  if len(feas)>=2 and np.ptp(feas.NC*feas.DAC)<=eps:
   prod=float((feas.NC*feas.DAC).iloc[0]);sums=np.zeros(len(feas))
   for i,(_,v) in enumerate(feas.iterrows()):
    d=dict(ident,candidate=v.candidate)
    for col,w in [('EP',10),('TTC',5),('Comfort',2)]:d['centered_'+col]=prod*w/17*(v[col]-feas[col].mean())
    d['centered_training_scalar']=v.training_scalar-feas.training_scalar.mean();d['nonlinear_reference_margin_centered']=v.reference_margin-feas.reference_margin.mean()
    assert abs(sum(d['centered_'+x] for x in ['EP','TTC','Comfort'])-d['centered_training_scalar'])<1e-10;decomp.append(d)
 groups=pd.DataFrame(rows);groups.to_parquet(OUT/'group_statistics.parquet',index=False);pd.DataFrame(decomp).to_parquet(OUT/'linear_feasible_scalar_decomposition.parquet',index=False)
 cols=[c for c in groups if c not in ['model','protocol','split','token','log','group']];summarise(groups,cols,['model','protocol','split'],'group_stats').to_csv(OUT/'credit_summary.csv',index=False)
 scene=groups.groupby(['model','protocol','split','token','log'],as_index=False)[cols].mean();scene.to_parquet(OUT/'scene_credit_statistics.parquet',index=False)
 for name in ['fixed','full']:
  den=groups.groupby(['model','protocol','split'])[['positive_mass','positive_count','information_active','shadow_'+name+'_mass','shadow_'+name+'_positive_disappeared','shadow_'+name+'_positive_flipped','shadow_'+name+'_information_active','shadow_'+name+'_rank_changed']].sum().reset_index()
  den['positive_loss_fraction']=den['shadow_'+name+'_positive_disappeared']/den.positive_count;den['positive_flip_fraction']=den['shadow_'+name+'_positive_flipped']/den.positive_count;den['mass_change_fraction']=den['shadow_'+name+'_mass']/den.positive_mass-1;den.to_csv(OUT/f'shadow_{name}_effects.csv',index=False)
 f['failure']=((f.NC<1)|(f.DAC<1)|(f.TTC<1)).astype(float);f['zero_score']=(f.PDMS==0).astype(float);f['pdms_below50']=(f.PDMS<.5).astype(float);f['pdms_below95']=(f.PDMS<.95).astype(float)
 measures=['PDMS','NC','DAC','TTC','Comfort','EP','raw_progress','end_speed','failure','zero_score','pdms_below50','pdms_below95']
 means=f.groupby(['model','protocol','split','token','log'],as_index=False)[measures].mean();means.to_csv(OUT/'scene_checkpoint_metrics.csv',index=False)
 summarise(means,measures,['model','protocol','split'],'checkpoint').to_csv(OUT/'checkpoint_summary.csv',index=False)
 parent=means[means.model=='a5_sft'].drop(columns='model');rl=means[means.model=='a5_grpo_300'].drop(columns='model');pairs=rl.merge(parent,on=['protocol','split','token','log'],suffixes=('_grpo','_sft'))
 for m in measures:pairs[m]=pairs[m+'_grpo']-pairs[m+'_sft']
 pairs.to_csv(OUT/'checkpoint_paired_differences.csv',index=False);summarise(pairs,measures,['protocol','split'],'checkpoint_delta').to_csv(OUT/'checkpoint_paired_ci.csv',index=False)
 # Same-seed outputs are paired scenario draws, not the same behaviour.
 one=f[f.model=='a5_sft'].merge(f[f.model=='a5_grpo_300'],on=['token','log','split','protocol','group','candidate'],suffixes=('_sft','_grpo'))
 one['safe_to_failure']=(one.failure_sft==0)&(one.failure_grpo==1);one['failure_to_safe']=(one.failure_sft==1)&(one.failure_grpo==0)
 one['ep_up_safety_improves']=(one.EP_grpo>one.EP_sft+eps)&(one.failure_grpo<one.failure_sft)
 one[['token','log','split','protocol','group','candidate','safe_to_failure','failure_to_safe','ep_up_safety_improves']].to_parquet(OUT/'paired_safety_transitions.parquet',index=False)
 one.groupby(['protocol','split'])[['safe_to_failure','failure_to_safe','ep_up_safety_improves']].sum().to_csv(OUT/'safety_transition_counts.csv')
 assoc=pairs.merge(scene[scene.model=='a5_sft'][['token','protocol','ep_only_components_mass','positive_mass']],on=['token','protocol']);assoc['ep_component_credit_share']=assoc.ep_only_components_mass/assoc.positive_mass.replace(0,np.nan);assoc.to_csv(OUT/'ep_credit_degradation_association.csv',index=False)
 plot=groups.groupby(['model','protocol'])[['positive_mass','ep_only_strict_mass','ep_only_components_mass']].sum().reset_index();plot['strict_share']=plot.ep_only_strict_mass/plot.positive_mass;plot['component_share']=plot.ep_only_components_mass/plot.positive_mass;plot.to_csv(OUT/'positive_credit_plot_data.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(plot));ax.bar(x-.18,plot.strict_share,.36,label='EP only; all credit gates identical');ax.bar(x+.18,plot.component_share,.36,label='EP only components; EP-dependent gates may differ');ax.set_xticks(x,[a+'\n'+b for a,b in zip(plot.model,plot.protocol)]);ax.set_ylabel('Share of sum(max(A,0))');ax.set_ylim(0,1);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'positive_credit_sources.png',dpi=160);plt.close(fig)
 ci=pd.read_csv(OUT/'checkpoint_paired_ci.csv');data=ci[(ci.split=='confirmation')&ci.metric.isin(['PDMS','NC','DAC','TTC','EP'])];data.to_csv(OUT/'protocol_change_plot_data.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4));metrics=['PDMS','NC','DAC','TTC','EP'];x=np.arange(5)
 for i,p in enumerate(['native_grpo','deployment']):
  q=data[data.protocol==p].set_index('metric').loc[metrics];ax.errorbar(x+(i-.5)*.18,q['mean']*100,yerr=np.maximum(0,np.stack([q['mean']-q.low,q.high-q['mean']]))*100,fmt='o',label=p,capsize=4)
 ax.set_xticks(x,metrics);ax.axhline(0,color='gray',lw=1);ax.set_ylabel('step300 minus SFT (percentage points)');ax.legend();fig.tight_layout();fig.savefig(OUT/'train_deploy_changes.png',dpi=160);plt.close(fig)
 save(OUT/'audits/stage1_analysis.json',dict(status='PASS',scenes=512,logs=read(OUT/'manifests/scenes.json')['logs_by_split'],positive_mass='sum(max(A,0)); neither gradient contribution nor causal attribution',strict_ep_only='feasible subset: other components and ALL actual gates identical; EP varies',component_only='descriptive relaxation allows EP-dependent progress/Pareto gate changes',linear_decomposition='only feasible equal multiplicative gates; nonlinear reference margin reported separately; no forced linear percentage',official_control='descriptive existing reports only; no extra sampling'))
if __name__=='__main__':main()
