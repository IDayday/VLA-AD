import concurrent.futures
from common_v3 import *
from scoring import init,cache,score_arrays,STATE,SCORERS
def work(args):
    run,scene=args;token=scene['token'];path=OUT/'cache/progressive'/run/f'{token}.npz';a=np.load(path);dest=OUT/'cache/progressive_scores'/run/f'{token}.npz';meta=dict(identity=identity(),input_sha256=sha(path))
    if valid(dest,meta):return
    STATE['scorer']=SCORERS['evaluation'];score=score_arrays(cache(scene),a['reference']);npz(dest,meta,reference=score)
def main():
    runs=[f'{label}_seed{s}' for s in CFG['progressive']['seeds'] for label in ['sft_final','grpo_middle','grpo_final']]
    with concurrent.futures.ProcessPoolExecutor(96,initializer=init) as ex:list(ex.map(work,[(run,r) for run in runs for r in scenes()],chunksize=1))
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');records=[];promoted=[];holdout=set(tokens('holdout'))
    for run in runs:
        for token,g in raw.groupby('token',sort=False):
            a=np.load(OUT/'cache/progressive'/run/f'{token}.npz');qnew=a['q_holdout'];rnew=a['r_knn'];ref=float(np.load(OUT/'cache/progressive_scores'/run/f'{token}.npz')['reference'][0,6]*100);oldfar=g.q_holdout.to_numpy()>=95;hq=g.PDMS.to_numpy()>=g.PDMS.quantile(.75);domain=oldfar&hq&g.unique.to_numpy();promotion=domain&(qnew<95);eligible=promotion&g.hard_safe.to_numpy()&(g.PDMS.to_numpy()>=ref-1e-8)
            records.append(dict(run=run,token=token,split='holdout' if token in holdout else 'train',old_high_quality_far=int(domain.sum()),promoted_count=int(promotion.sum()),newly_eligible_count=int(eligible.sum()),frontier_promotion_rate=float(promotion.sum()/domain.sum()) if domain.any() else np.nan,newly_eligible_PC_rate=float(eligible.sum()/domain.sum()) if domain.any() else np.nan,current_reference_PDMS=ref,old_reference_PDMS=g.reference_PDMS.iloc[0],current_reference_gain=ref-g.reference_PDMS.iloc[0]))
            for i in np.flatnonzero(promotion):
                r=g.iloc[i];promoted.append(dict(run=run,token=token,raw_index=int(r.raw_index),source=r.exact_source,PDMS=r.PDMS,q_old=r.q_holdout,q_new=qnew[i],r_knn_old=r.r_knn,r_knn_new=rnew[i],improvement_margin=r.PDMS-ref,newly_eligible=bool(eligible[i]),split='holdout' if token in holdout else 'train'))
    d=pd.DataFrame(records);csv(d,'G_scene_promotion.csv');parquet(pd.DataFrame(promoted),'G_promoted_candidates.parquet');summary=[]
    for run,g in d.groupby('run'):
        for scope in ['Full-1000','holdout-300']:
            sub=g if scope=='Full-1000' else g[g.split=='holdout']
            for metric in ['frontier_promotion_rate','newly_eligible_PC_rate','current_reference_gain']:summary.append(dict(run=run,scope=scope,metric=metric,**bootstrap(sub[metric],f'G_{run}_{scope}_{metric}')))
    csv(pd.DataFrame(summary),'G_promotion_bootstrap.csv')
    stage_audit('G',scenes=1000,checkpoints=6,reference_R=128,calibration_Q=128,raw_bank_unchanged=True,seed_namespaces_independent_of_V2=True,full_and_holdout_reported=True,promoted_candidate_rows=len(promoted),new_training_rounds=0,raw_sha256=sha(OUT/'metrics/A_raw_candidates.parquet'),outcome_conditioned_filtering=False)
    print(pd.DataFrame(summary).query("scope=='holdout-300' and metric=='frontier_promotion_rate'")[['run','mean','ci_low','ci_high']].to_string(index=False),flush=True)
if __name__=='__main__':main()
