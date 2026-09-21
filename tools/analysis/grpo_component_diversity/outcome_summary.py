"""Direct, fully enumerated outcomes and CRN transitions; no chosen scenes."""
from shared import *
from metrics import safe,NAMES
def main():
    rows=[]
    for r in scenes():
        a=arrays(CFG['models'][0],r['token'],True);b=arrays(CFG['models'][1],r['token'],True)
        for name,pa,pb in [('feasible',safe(a),safe(b))]+[(n,a[:,i]>=1-1e-8,b[:,i]>=1-1e-8) for i,n in enumerate(NAMES) if n!='EP']:
            rows.append(dict(token=r['token'],log=r['log'],component=name,both_pass=(pa&pb).mean(),IL_fail_GRPO_pass=(~pa&pb).mean(),IL_pass_GRPO_fail=(pa&~pb).mean(),both_fail=(~pa&~pb).mean()))
    csv('scene_CRN_outcome_transitions.csv',rows)
    csv('CRN_outcome_transitions.csv',pd.DataFrame(rows).groupby('component',as_index=False).mean(numeric_only=True))
    p=pd.read_parquet(OUT/'metrics/scene_outcome_patterns.parquet')
    q=p.groupby(['model','NC','DAC','TTC','DDC'],as_index=False)['count'].sum();q['fraction']=q['count']/(5000*128)
    csv('joint_safety_outcome_distribution.csv',q)
    groups=pd.read_parquet(OUT/'metrics/group_metrics.parquet');groups=groups[groups.block==0]
    regimes=[]
    for (m,g),z in groups.groupby(['model','G']):
        regimes.append(dict(model=m,G=g,scenes=len(z),no_safe=z.no_safe_group.mean(),mixed_safe_unsafe=z.mixed_safety_group.mean(),all_safe_progress_contrast=(z.all_safe_group&z.safe_EP_opportunity).mean(),all_safe_small_EP_range=(z.all_safe_group&~z.safe_EP_opportunity).mean(),zero_reward_contrast_all_safe=(z.zero_advantage_group&z.all_safe_group).mean(),zero_reward_contrast_no_safe=(z.zero_advantage_group&z.no_safe_group).mean(),zero_reward_contrast_mixed_safety=(z.zero_advantage_group&z.mixed_safety_group).mean()))
    csv('group_signal_regimes.csv',regimes)
if __name__=='__main__':main()
