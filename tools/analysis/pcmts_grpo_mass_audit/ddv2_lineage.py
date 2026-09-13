"""Audit native DDV2 augmentation ancestry without changing any raw candidate."""
from common_mass import *
import pandas as pd
def main():
    rows=[]
    for s in scenes():
        z=np.load(OUT/'cache/external/ddv2'/f'{s["token"]}.npz');full=z['native_proposals'];assert len(full)==800
        for slot,j in enumerate(z['proposal_indices']):
            j=int(j);parent=j%200;aug=j//200;x=full[parent,:,:2].astype(float);y=full[j,:,:2].astype(float);den=(x*x).sum(0);f=np.divide((x*y).sum(0),den,out=np.full(2,np.nan),where=den>1e-20);err=np.max(np.abs(x*f-y)) if np.isfinite(f).all() else np.nan
            if np.isfinite(err):assert err<5e-5,(s['token'],j,err)
            rows.append(dict(token=s['token'],source_model='DDV2',candidate_slot=slot,native_proposal_index=j,parent_candidate_id=f'ddv2:{s["token"]}:native_proposal{parent}',parent_native_proposal_index=parent,perturbation_family='native_multiplicative_XY' if aug else 'none',augmentation_block=aug,configured_std_min=.1 if aug else 0,configured_std_max=.3 if aug else 0,realized_x_multiplier=f[0],realized_y_multiplier=f[1],lineage_reconstruction_max_abs_m=err,native_cache_path=str(OUT/'cache/external/ddv2'/f'{s["token"]}.npz')))
    frame=pd.DataFrame(rows);frame.to_parquet(OUT/'audits/ddv2_native_augmentation_lineage.parquet',index=False)
    save(OUT/'audits/ddv2_lineage_summary.json',dict(candidate_count=len(frame),native_augmented_count=int((frame.augmentation_block>0).sum()),not_GT_perturbations=True,max_lineage_rounding_error=float(frame.lineage_reconstruction_max_abs_m.max()),data_unchanged=True))
    print('DDV2 lineage audit',len(frame),'PASS')
if __name__=='__main__':main()
