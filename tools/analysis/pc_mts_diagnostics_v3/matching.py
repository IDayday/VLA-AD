"""Prespecified, source-exact non-replacement minimum-quality-gap matching."""
from common_v3 import *
from scipy.optimize import linear_sum_assignment

def match(a,b,gap):
    if not len(a) or not len(b):return []
    av=a.PDMS.to_numpy();bv=b.PDMS.to_numpy();cost=abs(av[:,None]-bv[None,:]);valid=cost<=gap+1e-8
    augmented=np.full((len(a),len(b)+len(a)),100.);augmented[:,:len(b)]=np.where(valid,cost,1e6)
    ri,ci=linear_sum_assignment(augmented)
    return [(a.index[i],b.index[j],cost[i,j]) for i,j in zip(ri,ci) if j<len(b) and valid[i,j]]

def main():
    assert (OUT/'manifests/audit_B.json').exists()
    raw=pd.read_parquet(OUT/'metrics/A_raw_candidates.parquet');pairs=[];queries=[]
    for gap in [.25,.5]:
        label='primary' if gap==.25 else 'sensitivity'
        for (token,source,nc,dac),g in raw[raw.unique].groupby(['token','exact_source','NC','DAC'],sort=True):
            used=set();far=g[g.q_holdout>=95]
            for near_region,mask in [('boundary',(g.q_holdout>=50)&(g.q_holdout<95)),('core',g.q_holdout<50)]:
                for ai,bi,d in match(g[mask&~g.index.isin(used)],far[~far.index.isin(used)],gap):
                    used.update([ai,bi]);a=raw.loc[ai];b=raw.loc[bi];pid=f'{label}:{token}:{int(a.raw_index)}:{int(b.raw_index)}'
                    pairs.append(dict(pair_id=pid,protocol=label,gap_limit=gap,token=token,exact_source=source,NC=nc,DAC=dac,near_region=near_region,near_index=int(a.raw_index),far_index=int(b.raw_index),PDMS_gap=d,near_PDMS=a.PDMS,far_PDMS=b.PDMS,near_q=a.q_holdout,far_q=b.q_holdout,near_r_knn=a.r_knn,far_r_knn=b.r_knn,common_noise_key=pid))
                    for side,r in [('near',a),('far',b)]:queries.append(dict(query_id=pid+':'+side,pair_id=pid,protocol=label,token=token,raw_index=int(r.raw_index),side=side,noise_key=pid))
    for token,g in raw[raw.unique].groupby('token',sort=True):
        selected=sorted(g.raw_index,key=lambda i:seed(token,'v3_regression_raw',int(i)))[:8]
        for i in selected:queries.append(dict(query_id=f'regression:{token}:{i}',pair_id='',protocol='regression',token=token,raw_index=i,side='',noise_key=f'regression:{token}:{i}'))
    p=pd.DataFrame(pairs);q=pd.DataFrame(queries);assert q.query_id.is_unique
    for protocol,g in p.groupby('protocol'):
        used=pd.concat([g[['token','near_index']].rename(columns={'near_index':'raw_index'}),g[['token','far_index']].rename(columns={'far_index':'raw_index'})]);assert not used.duplicated().any()
        assert (g.PDMS_gap<=g.gap_limit+1e-8).all()
    csv(p,'C_pair_manifest.csv');parquet(q,'C_query_manifest.parquet')
    stage_audit('C_matching',pairs=len(p),primary_pairs=int((p.protocol=='primary').sum()),primary_scenes=p[p.protocol=='primary'].token.nunique(),query_count=len(q),query_manifest_sha256=sha(OUT/'metrics/C_query_manifest.parquet'),pair_manifest_sha256=sha(OUT/'metrics/C_pair_manifest.csv'),same_exact_source=True,same_hard_class=True,no_replacement_within_protocol=True,common_random_numbers='pair_id keyed t and epsilon; both candidates identical',source_composition=p.groupby(['protocol','exact_source']).size().unstack(0,fill_value=0).to_dict(),result_conditioned_matching=False)
    print(p.groupby(['protocol','near_region']).size().to_string(),flush=True)
if __name__=='__main__':main()
