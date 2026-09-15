"""Reuse audited historical logs as evidence, separate from new frozen sampling."""
from common_support import *

def main():
    sft=ROOT/'outputs/five_checkpoint_training_distribution/metrics/historical_training_at_checkpoint.csv'
    grpo=ROOT/'outputs/historical_sft_grpo_longchain/metrics/training_scalars.parquet'
    a=pd.read_csv(sft);fields=['diffusion_loss','selected_target_is_gt_ratio','dpsi_non_gt_target_loss','dpsi_non_gt_to_gt_loss_ratio','dpsi_non_gt_exposure_ratio','dpsi_non_gt_residual_mass_ratio_pre_budget','dpsi_non_gt_residual_mass_ratio_post_budget','dpsi_non_gt_target_weight_ratio_pre_budget','dpsi_non_gt_target_weight_ratio_post_budget','dpsi_gt_weight_ratio']
    a=a[a.metric.isin(['train/'+k+'_epoch' for k in fields])];csv('historical_sft_learning.csv',a)
    b=pd.read_parquet(grpo);fields=['reward','bc_loss','reference_kl_loss','reference_kl_coeff','lfp_positive_advantage_ratio','lfp_zero_advantage_ratio','lfp_negative_advantage_ratio','lfp_positive_advantage_below_ref_scalar_ratio','lfp_positive_advantage_reference_dominated_ratio','lfp_positive_advantage_scalar_delta_mean','lfp_positive_advantage_quality_delta_mean','occupied_support_bucket_count','rankable_support_bucket_count','within_support_score_std','between_support_rep_std','support_concentration','free_bucket_ratio','best_valid_minus_reference','lfp_group_pairwise_ade_m','lfp_group_scalar_range']
    b=b[b.tag.isin(['train/'+k+'_epoch' for k in fields])].sort_values(['run','tag','step']);csv('historical_grpo_credit.csv',b)
    save(OUT/'audits/historical_log_sources.json',dict(sft_path=str(sft),sft_hash=sha(sft),grpo_path=str(grpo),grpo_hash=sha(grpo),scope='Actual historical complete-epoch log summaries. Do not treat as uniformly sampled new1000 scenes or compare KL scales across different implementations.'))

if __name__=='__main__':main()
