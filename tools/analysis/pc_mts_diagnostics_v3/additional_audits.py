"""Definition clarity and diagnostic-holdout provenance without changing any primary gate."""
from common_v3 import *
def main():
    records=[]
    for r in scenes():
        t=r['token'];p=np.load(V2/'candidate_pools/pc_mts'/f'{t}.npz');raw=p['raw_index']>=0;original_primary=raw&(p['fallback_level']==1)&p['qualified'];original_qualified=raw&p['qualified']
        records.append(dict(token=t,original_V2_primary_unique_count=int(original_primary.sum()),original_V2_qualified_unique_count=int(original_qualified.sum()),original_V2_zero_qualified=int(not original_qualified.any()),frozen_16_slot_PDMS=float(p['scores'][:,6].mean()*100),frozen_filler_count=int((~raw).sum())))
    csv(pd.DataFrame(records),'A_original_V2_qualification.csv')
    tr=set(tokens('train'));ho=set(tokens('holdout'));tl={r['log'] for r in scenes() if r['token'] in tr};hl={r['log'] for r in scenes() if r['token'] in ho}
    stage_audit('split_provenance',train_tokens=700,holdout_tokens=300,token_overlap=len(tr&ho),train_logs=len(tl),holdout_logs=len(hl),overlapping_logs=len(tl&hl),holdout_scenes_sharing_a_train_log=sum(r['log'] in tl for r in scenes() if r['token'] in ho),split_changed=False,frozen_GT_distance_threshold_calibration='V1 official-IL rollout-to-GT thresholds used all 1000 diagnostic scenes, including the later V3 holdout. Kept unchanged as requested; this is preexisting calibration exposure.',global_no_leakage_claim_permitted=False,note='Required token-hash split. No new optimizer updates use holdout tokens, but historical pretraining, log overlap and frozen GT-threshold calibration prevent treating this as an independently untouched test set.')
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');cor=[]
    for t,g in raw.groupby('token'):
        cor.append(dict(token=t,spearman_dGT_rknn=g[['d_GT','r_knn']].corr(method='spearman').iloc[0,1]))
    csv(pd.DataFrame(cor),'S_GT_policy_distance_scene_correlations.csv')
    if (OUT/'metrics/C_candidate_measurements.parquet').exists():
        c=pd.read_parquet(OUT/'metrics/C_candidate_measurements.parquet');export=c[['query_id','token','raw_index','protocol','diffusion_loss','E_rec','return_rate']].rename(columns={'diffusion_loss':'epsilon_prediction_mse'});csv(export,'C_native_diffusion_loss.csv')
if __name__=='__main__':main()
