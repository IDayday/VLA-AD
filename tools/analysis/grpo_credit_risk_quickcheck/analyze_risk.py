from common import *
from clustered_stats import interval,summarise
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def weights(a,b,larger=True):
 if not np.isfinite(a) or not np.isfinite(b):return None
 if a==b:return np.array([.5,.5])
 return np.array([1.,0.]) if (a>b)==larger else np.array([0.,1.])

def main():
 pairs=pd.read_parquet(OUT/'matched_pairs.parquet')
 if (OUT/'constructed_pairs.parquet').exists():pairs=pd.concat([pairs,pd.read_parquet(OUT/'constructed_pairs.parquet')],ignore_index=True)
 nom=pd.read_parquet(OUT/'nominal_margins.parquet');risk=pd.read_parquet(OUT/'perturbation_results.parquet');measure=['risk','NC_fail','DAC_fail','TTC_fail','hard_fail'];raw=[];excluded=[];audit=[]
 for _,p in pairs.iterrows():
  ids=[p.candidate_a,p.candidate_b];n=nom[(nom.token==p.token)&(nom.source==p.source)].set_index('candidate').loc[ids];r=risk[(risk.token==p.token)&(risk.source==p.source)&risk.candidate.isin(ids)]
  counts=r[r.status=='OK'].groupby(['candidate','stream']).size();complete=all(counts.get((int(i),s),0)==8 for i in ids for s in ['selection_v1','evaluation_v1'])
  if not complete:excluded.append(dict(pair_id=p.pair_id,token=p.token,split=p.split,source=p.source,reason='not_all_16_draws_valid_or_not_run'));continue
  selection=r[r.stream=='selection_v1'].groupby('candidate')[measure].mean().loc[ids];evaluation=r[r.stream=='evaluation_v1'].groupby('candidate')[measure+['raw_progress','end_speed','PDMS']].mean().loc[ids]
  margin=np.fmin(np.minimum(n.object_clearance_m.to_numpy(),10),np.minimum(n.road_margin_m.to_numpy(),10));ws={'A':weights(*n.PDMS.to_numpy()),'B':np.array([.5,.5]),'C':weights(*margin),'D':weights(*selection.risk.to_numpy(),larger=False)}
  base=dict(pair_id=p.pair_id,token=p.token,log=p.log,split=p.split,source=p.source,both_full_score=p.both_full_score,progress_gap_m=p.progress_gap_m,selection_preference_available=bool(selection.risk.iloc[0]!=selection.risk.iloc[1]))
  for selector,w in ws.items():
   if w is None:continue
   v=dict(base,selector=selector,weight_a=w[0],weight_b=w[1])
   for k in measure:v['evaluation_'+k]=float(w@evaluation[k].to_numpy());v['selection_'+k]=float(w@selection[k].to_numpy())
   for k in ['PDMS','raw_progress','end_speed','EP']:v['nominal_'+k]=float(w@n[k].to_numpy())
   for k in ['raw_progress','end_speed','PDMS']:v['evaluation_'+k]=float(w@evaluation[k].to_numpy())
   raw.append(v)
  audit.append(dict(base,pdms_a=n.PDMS.iloc[0],pdms_b=n.PDMS.iloc[1],selection_risk_a=selection.risk.iloc[0],selection_risk_b=selection.risk.iloc[1],evaluation_risk_a=evaluation.risk.iloc[0],evaluation_risk_b=evaluation.risk.iloc[1],object_margin_a=n.object_clearance_m.iloc[0],object_margin_b=n.object_clearance_m.iloc[1],road_margin_a=n.road_margin_m.iloc[0],road_margin_b=n.road_margin_m.iloc[1]))
 pd.DataFrame(excluded,columns=['pair_id','token','split','source','reason']).to_csv(OUT/'excluded_pair_evaluations.csv',index=False)
 f=pd.DataFrame(raw);f.to_parquet(OUT/'selector_pair_results.parquet',index=False);pd.DataFrame(audit).to_csv(OUT/'matched_pair_plot_data.csv',index=False)
 if len(f)==0:
  save(OUT/'audits/signal_gate.json',dict(status='INSUFFICIENT_SIGNAL',optimizer='NOT_RUN',reason='No fully valid independent evaluation pairs',thresholds=cfg()['update_gate']));return
 cols=[x for x in f if x.startswith(('evaluation_','selection_','nominal_')) and x!='selection_preference_available'];diff=[]
 for comparator in ['A','B']:
  baseline=f[f.selector==comparator].set_index('pair_id')
  for selector in ['A','C','D']:
   if selector==comparator:continue
   q=f[f.selector==selector].copy();b=baseline.loc[q.pair_id]
   for k in cols:q[k]=q[k].to_numpy()-b[k].to_numpy()
   q['comparator']=comparator;diff.append(q)
 deltas=pd.concat(diff,ignore_index=True);deltas.to_csv(OUT/'selector_paired_differences.csv',index=False);summary=summarise(deltas,cols,['source','split','selector','comparator'],'independent_risk')
 summary.to_csv(OUT/'selector_comparisons.csv',index=False)
 summarise(deltas[deltas.both_full_score],cols,['source','split','selector','comparator'],'full_pdms_subset').to_csv(OUT/'selector_comparisons_pdms1.csv',index=False)
 # Confirmatory gate checks D against BOTH A and random B. C remains descriptive.
 q=summary[(summary.source=='native_within_group')&(summary.split=='confirmation')&(summary.selector=='D')];cov=read(OUT/'manifests/matching_decision.json');checks={};values={}
 for comp in ['A','B']:
  data=q[q.comparator==comp].set_index('metric')
  for m in ['evaluation_risk','nominal_PDMS','evaluation_NC_fail','evaluation_DAC_fail']:
   if m in data.index:values[comp+'_'+m]={k:float(data.loc[m,k]) for k in ['mean','low','high','scenes','logs']}
  checks[comp+'_risk_ci_upper_lt0']='evaluation_risk' in data.index and data.loc['evaluation_risk','high']<0
  checks[comp+'_nominal_pdms_ci_lower_ge_minus005']='nominal_PDMS' in data.index and data.loc['nominal_PDMS','low']>=-.005
  checks[comp+'_no_clear_NC_DAC_reverse_harm']=all(m in data.index and data.loc[m,'low']<=0 for m in ['evaluation_NC_fail','evaluation_DAC_fail'])
 valid=f[(f.source=='native_within_group')&(f.split=='confirmation')&(f.selector=='D')];checks['at_least50_native_confirmation_scenes']=valid.token.nunique()>=50;checks['high_score_group_coverage_ge10pct']=cov['high_score_group_coverage']>=.1
 checks['progress_constraint_preserved']=len(valid)>0 and valid.progress_gap_m.max()<=.5+1e-12;checks['identity_scoring_pass']=read(OUT/'audits/SMOKE_COMPLETE.json')['status']=='PASS';checks['calibration_freeze_pass']=read(OUT/'perturbation_manifest.json')['calibration_status']=='PASS'
 status='WITHIN_GROUP_SIGNAL' if all(checks.values()) else 'INSUFFICIENT_SIGNAL'
 # Evidence against expected direction is distinct from an imprecise null.
 if any(v['low']>0 for k,v in values.items() if k.endswith('evaluation_risk')):status='NOT_SUPPORTED'
 constructed=summary[(summary.source=='external_constructed')&(summary.split=='confirmation')&(summary.selector=='D')&(summary.comparator=='A')].set_index('metric')
 if status!='WITHIN_GROUP_SIGNAL' and 'evaluation_risk' in constructed.index and constructed.loc['evaluation_risk','high']<0 and constructed.loc['nominal_PDMS','low']>=-.005:status='CONSTRUCTED_ONLY_SIGNAL'
 gate=dict(status=status,checks={k:bool(v) for k,v in checks.items()},values=values,thresholds=cfg()['update_gate'],valid_confirmation_scenes=int(valid.token.nunique()),confirmation_logs=int(valid.log.nunique()),all_group_coverage=cov['all_group_coverage'],high_score_group_coverage=cov['high_score_group_coverage'],optimizer='ELIGIBLE_PENDING_RUN' if status=='WITHIN_GROUP_SIGNAL' else 'NOT_RUN',evaluation_valid_draws=int(((risk.split=='confirmation')&(risk.stream=='evaluation_v1')&(risk.status=='OK')).sum()),NC_DAC_harm_rule='clear reverse harm means paired 95% CI entirely above zero; all mean differences also reported',counts='pairs and perturbations are not independent scene sample sizes')
 save(OUT/'audits/signal_gate.json',gate)
 plot=pd.DataFrame(audit);p=plot[(plot.split=='confirmation')&(plot.source=='native_within_group')]
 fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(p.pdms_a,p.pdms_b,s=8,alpha=.4);axes[0].plot([.95,1],[.95,1],c='gray');axes[0].set(xlabel='Nominal PDMS, candidate A',ylabel='Nominal PDMS, candidate B',xlim=(.948,1.002),ylim=(.948,1.002))
 axes[1].scatter(p.evaluation_risk_a,p.evaluation_risk_b,s=12,alpha=.4);axes[1].plot([0,1],[0,1],c='gray');axes[1].set(xlabel='Independent risk, candidate A (8 draws)',ylabel='Independent risk, candidate B (8 draws)');fig.tight_layout();fig.savefig(OUT/'matched_pdms_independent_risk.png',dpi=160);plt.close(fig)
 coverage=pd.read_csv(OUT/'native_vs_constructed_coverage.csv');v=coverage[(coverage.model=='a5_sft')&(coverage.protocol=='native_grpo')&(coverage.tolerance_m==.5)&(coverage.split=='confirmation')];v.to_csv(OUT/'signal_coverage_plot_data.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,4));labels=['Native G16','Merged G32','External <=6'];counts=[float(v[v.scope=='within_group'].matched_scene.iloc[0]),float(v[v.scope=='merged_two_groups'].matched_scene.iloc[0]),float(f[(f.source=='external_constructed')&(f.split=='confirmation')].token.nunique())];ax.bar(labels,counts);ax.set_ylabel('Confirmation scenes with matched pairs');ax.set_title('Coverage is not independent risk validation')
 if not cov['external_trigger']:ax.text(2,max(5,max(counts)*.05),'NOT_RUN\ntrigger false',ha='center',fontsize=9)
 fig.tight_layout();fig.savefig(OUT/'signal_coverage.png',dpi=160);plt.close(fig)
 print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
