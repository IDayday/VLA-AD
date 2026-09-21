"""Outcome diversity and usable reward contrasts, not semantic mode counts."""
import numpy as np
NAMES=['NC','DAC','EP','TTC','Comfort','DDC']
SAFETY=[0,1,3,5]
TOL=1e-8
def safe(s):return (s[:,SAFETY]>=1-TOL).all(1)
def reward(s,training=True):
    ep=10 if training else 5
    return s[:,0]*s[:,1]*(ep*s[:,2]+5*s[:,3]+2*s[:,4])/(ep+7)
def advantage(s):
    # Torch native reward is FP32; quantile endpoints 0/1 do not change A.
    r=reward(s).astype(np.float32)
    a=(r-r.mean())/(r.std(ddof=1)+np.float32(1e-8))
    # Exactly equal rewards mathematically have no contrast. Native FP32
    # reduction can leave tiny constant A; preserve this in parity audit.
    return a.astype(float)
def divide(n,d):return float(n/d) if d else np.nan
def effective_rank(x):
    eigen=np.linalg.eigvalsh(np.cov(x,rowvar=False));eigen=eigen[eigen>1e-12]
    if not len(eigen):return 0.
    p=eigen/eigen.sum();return float(np.exp(-(p*np.log(p)).sum()))
def group_stats(t,s,native_adv=None):
    t=np.asarray(t,dtype=float);s=np.asarray(s,dtype=float);g=len(s)
    assert t.shape==(g,8,3) and s.shape==(g,7) and np.isfinite(s).all()
    ii,jj=np.triu_indices(g,1);sf=safe(s);pdms=s[:,6]*100;r=reward(s);adv=advantage(s) if native_adv is None else np.asarray(native_adv,float);pos=adv>1e-8
    d=np.linalg.norm(t[:,None,:,:2]-t[None,:,:,:2],axis=-1).mean(-1);pw=d[ii,jj]
    medoid=int(d.mean(1).argmin());rad=d[medoid]
    patterns,counts=np.unique(np.round(s[:,SAFETY],8),axis=0,return_counts=True)
    probs=counts/g;entropy=float(-(probs*np.log(probs)).sum())
    disagree=(np.abs(s[ii][:,SAFETY]-s[jj][:,SAFETY])>TOL).any(1)
    delta=s[ii,:6]-s[jj,:6];tradeoff=(delta>TOL).any(1)&(delta< -TOL).any(1)
    outcome_diff=disagree|(abs(delta[:,2])>=.05)|(abs(delta[:,4])>TOL)
    tied=abs(r[ii]-r[jj])<=.01
    zero_mask=s[:,0]*s[:,1]<=TOL;both_zero=zero_mask[ii]&zero_mask[jj]
    both_safe=sf[ii]&sf[jj]
    cross_safe=sf[ii]!=sf[jj]
    unsafe_index=np.where(sf[ii],jj,ii);safe_index=np.where(sf[ii],ii,jj)
    unsafe_ep_higher=s[unsafe_index,2]>s[safe_index,2]+TOL
    unsafe_reward_higher=r[unsafe_index]>r[safe_index]+TOL
    vector=s[:,:6];dominates=(vector[:,None]>=vector[None,:]-TOL).all(2)&(vector[:,None]>vector[None,:]+TOL).any(2)
    safe_ep=s[sf,2];safe_pdms=pdms[sf]
    result=dict(mean_PDMS=pdms.mean(),min_PDMS=pdms.min(),max_PDMS=pdms.max(),std_PDMS=pdms.std(ddof=1),
        train_reward_mean=r.mean()*100,train_reward_std=r.std(ddof=1)*100,
        feasible_rate=sf.mean(),all_safe_group=sf.all(),no_safe_group=not sf.any(),mixed_safety_group=sf.any() and not sf.all(),
        hard_failure_rate=(s[:,[0,1]]<1-TOL).any(1).mean(),
        mean_pairwise_ADE=pw.mean(),median_pairwise_ADE=np.median(pw),pairwise_P90_ADE=np.quantile(pw,.9),
        medoid_radius_P50=np.quantile(rad,.5),medoid_radius_P90=np.quantile(rad,.9),covariance_effective_rank=effective_rank(t[:,:,:2].reshape(g,16)),
        safe_pairwise_ADE=pw[both_safe].mean() if both_safe.any() else np.nan,
        same_outcome_pair_ADE=pw[~disagree].mean() if (~disagree).any() else np.nan,
        different_outcome_pair_ADE=pw[disagree].mean() if disagree.any() else np.nan,
        safety_pattern_count=len(patterns),safety_pattern_entropy=entropy,effective_safety_patterns=np.exp(entropy),
        safety_pattern_disagreement=disagree.mean(),component_tradeoff_pair_fraction=tradeoff.mean(),pareto_fraction=(~dominates.any(0)).mean(),
        dominated_sample_fraction=dominates.any(0).mean(),
        safe_EP_mean=safe_ep.mean() if len(safe_ep) else np.nan,
        safe_EP_std=safe_ep.std(ddof=1) if len(safe_ep)>1 else np.nan,
        safe_EP_headroom=safe_ep.max()-safe_ep.mean() if len(safe_ep) else np.nan,
        safe_EP_opportunity=len(safe_ep)>1 and np.ptp(safe_ep)>=.05,
        safe_PDMS_headroom=safe_pdms.max()-safe_pdms.mean() if len(safe_pdms) else np.nan,
        best_safe_PDMS=safe_pdms.max() if len(safe_pdms) else np.nan,
        zero_advantage_group=np.ptp(r)<=TOL,
        native_nonzero_advantage_on_constant_reward=(np.ptp(r)<=TOL) and (abs(adv)>1e-8).any(),
        native_mean_advantage=adv.mean(),
        no_reward_contrast_but_outcome_difference=(np.ptp(r)<=TOL) and outcome_diff.any(),
        near_tie_different_outcome_pair_fraction=(tied&outcome_diff).mean(),
        outcome_difference_given_near_tie=divide((tied&outcome_diff).sum(),tied.sum()),
        zero_reward_masked_fraction=zero_mask.mean(),
        masked_pair_outcome_difference=divide((both_zero&outcome_diff).sum(),both_zero.sum()),
        positive_infeasible_rate=divide((pos&~sf).sum(),(~sf).sum()),
        infeasible_among_positive=divide((pos&~sf).sum(),pos.sum()),
        positive_infeasible_adv_mass=divide(adv[pos&~sf].sum(),adv[pos].sum()),
        positive_count=pos.sum(),infeasible_count=(~sf).sum(),positive_infeasible_count=(pos&~sf).sum(),
        positive_reward_winner_infeasible=not sf[int(r.argmax())],
        unsafe_reward_winner_with_safe_alternative=sf.any() and not sf[int(r.argmax())],
        unsafe_progress_tradeoff_fraction=divide((cross_safe&unsafe_ep_higher).sum(),cross_safe.sum()),
        reward_prefers_unsafe_fraction=divide((cross_safe&unsafe_reward_higher).sum(),cross_safe.sum()),
        component_effective_rank=effective_rank(s[:,:6]),
        reward_mean_rank_flip_vs_PDMS=divide(((r[ii]-r[jj])*(pdms[ii]-pdms[jj])< -TOL).sum(),len(ii)))
    # Group-relative reward coefficient association, not parameter gradient.
    for i,name in enumerate(NAMES):
        x=s[:,i];passed=x>=1-TOL
        cov=float(np.mean((adv-adv.mean())*(x-x.mean())))
        result.update({f'{name}_mean':x.mean(),f'{name}_min':x.min(),f'{name}_max':x.max(),f'{name}_std':x.std(ddof=1),f'{name}_P10':np.quantile(x,.1),f'{name}_P50':np.quantile(x,.5),f'{name}_P90':np.quantile(x,.9),f'{name}_all_pass':passed.all(),f'{name}_all_fail':not passed.any(),f'{name}_mixed':passed.any() and not passed.all(),f'{name}_headroom':x.max()-x.mean(),f'covariance_adv_{name}':cov,f'negative_adv_{name}_association':cov< -TOL,f'positive_{name}_failure_rate':divide((pos&~passed).sum(),(~passed).sum()),f'{name}_positive_mean':x[pos].mean() if pos.any() else np.nan})
    # Additive gated contributions: safety multiplier is retained, so do not
    # interpret these as causal contributions of progress vs safety alone.
    for name,i,w in [('EP',2,10),('TTC',3,5),('Comfort',4,2)]:
        result[f'train_gated_{name}_contribution']=float((s[:,0]*s[:,1]*s[:,i]*w/17).mean()*100)
    return result
