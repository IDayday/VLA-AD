from common import *
import itertools

def eligible(b):
 return (b.NC==1)&(b.DAC==1)&(b.TTC==1)&(b.DDC>=b.reference_gt_ddc-.01)&(b.PDMS>=.95)

def pairs(b,tolerance=.5,allow_cross_group=False,low=False):
 q=b[eligible(b)&(b.low_progress_branch==low)];found=[]
 if len(q)<2:return found
 i,j=np.triu_indices(len(q),1);g=q.group.to_numpy();c=q.candidate.to_numpy();score=q.PDMS.to_numpy();progress=q.raw_progress.to_numpy()
 keep=(np.abs(score[i]-score[j])<=.01+1e-12)&(np.abs(progress[i]-progress[j])<=tolerance+1e-12)
 if not allow_cross_group:keep &= g[i]==g[j]
 a=q.iloc[0];identity=dict(token=a.token,log=a.log,split=a.split,model=a.model,protocol=a.protocol)
 for x,y in zip(i[keep],j[keep]):
  ids=sorted([int(g[x])*16+int(c[x]),int(g[y])*16+int(c[y])])
  found.append(dict(identity,candidate_a=ids[0],candidate_b=ids[1],group_a=ids[0]//16,group_b=ids[1]//16,both_full_score=bool(score[x]==1 and score[y]==1),progress_gap_m=abs(progress[x]-progress[y]),pdms_gap=abs(score[x]-score[y]),low_progress_branch=low))
 return found

def select_pairs(found):
 selected=[];used=set()
 for p in sorted(found,key=lambda x:digest(['fixed_pair_v1',x['token'],x['model'],x['protocol'],x['candidate_a'],x['candidate_b']])):
  if p['candidate_a'] in used or p['candidate_b'] in used:continue
  p=dict(p,pair_id=digest(['pair_v1',p['token'],p['model'],p['protocol'],p['candidate_a'],p['candidate_b']])[:20],source='native_within_group')
  selected.append(p);used|={p['candidate_a'],p['candidate_b']}
  if len(selected)==2:break
 return selected

def main():
 frame=pd.read_parquet(OUT/'trajectory_credit.parquet');coverage=[];chosen=[];diagnostics=[]
 for (model,protocol,token),b in frame.groupby(['model','protocol','token']):
  for tol in [.25,.5,1.]:
   for scope in ['within_group','merged_two_groups']:
    found=pairs(b,tol,scope=='merged_two_groups');groups={p['group_a'] for p in found}|{p['group_b'] for p in found};hi=[g for g,s in b.groupby('group') if eligible(s).sum()>=2]
    coverage.append(dict(token=token,log=b.log.iloc[0],split=b.split.iloc[0],model=model,protocol=protocol,scope=scope,tolerance_m=tol,scenes=1,matched_scene=bool(found),matched_groups=len(groups),total_groups=2,high_score_groups=len(hi),matched_high_score_groups=len(groups&set(hi)),candidate_pairs=len(found),full_score_pairs=sum(p['both_full_score'] for p in found)))
    if model==cfg()['matching']['primary_model'] and protocol=='native_grpo' and tol==.5 and scope=='within_group':chosen+=select_pairs(found)
  low=pairs(b,.5,False,True)
  for g,s in b.groupby('group'):
   safe=(s.NC==1)&(s.DAC==1)&(s.TTC==1)&(s.DDC>=s.reference_gt_ddc-.01);high=safe&(s.PDMS>=.95);normal=high&~s.low_progress_branch
   found=pairs(s)
   reason='matched' if found else ('nominal_safety_lt2' if safe.sum()<2 else 'pdms95_lt2' if high.sum()<2 else 'normal_progress_lt2' if normal.sum()<2 else 'no_joint_pdms_progress_match')
   diagnostics.append(dict(token=token,log=s.log.iloc[0],split=s.split.iloc[0],model=model,protocol=protocol,group=g,reason=reason,safe_candidates=int(safe.sum()),high_score_candidates=int(high.sum()),normal_candidates=int(normal.sum()),low_progress_pair_count=sum(p['group_a']==g for p in low)))
 pd.DataFrame(chosen).to_parquet(OUT/'matched_pairs.parquet',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage_by_scene.csv',index=False);pd.DataFrame(diagnostics).to_csv(OUT/'unmatched_reasons.csv',index=False)
 cov=pd.DataFrame(coverage);summary=cov.groupby(['split','model','protocol','scope','tolerance_m'],as_index=False)[['scenes','matched_scene','matched_groups','total_groups','high_score_groups','matched_high_score_groups','candidate_pairs','full_score_pairs']].sum()
 summary['all_group_coverage']=summary.matched_groups/summary.total_groups;summary['high_score_group_coverage']=summary.matched_high_score_groups/summary.high_score_groups.replace(0,np.nan)
 summary.to_csv(OUT/'native_vs_constructed_coverage.csv',index=False)
 primary=summary[(summary.split=='confirmation')&(summary.model=='a5_sft')&(summary.protocol=='native_grpo')&(summary.scope=='within_group')&(summary.tolerance_m==.5)].iloc[0]
 save(OUT/'manifests/matching_decision.json',dict(native_confirmation_scenes=int(primary.matched_scene),all_group_coverage=float(primary.all_group_coverage),high_score_group_coverage=float(primary.high_score_group_coverage),external_trigger=bool(primary.matched_scene<50 or primary.high_score_group_coverage<.10),pair_manifest_sha256=sha(OUT/'matched_pairs.parquet'),selection_uses_risk=False,high_score_group_definition='at least 2 candidates pass nominal safety/DDC and PDMS>=.95; denominator includes low-progress groups'))
 print('MATCHED',len(chosen),read(OUT/'manifests/matching_decision.json'))
if __name__=='__main__':main()
