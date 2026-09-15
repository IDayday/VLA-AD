"""Recover actual historical teacher tensors, weights and training membership."""
from common_support import *
import torch,lzma,pickle

def main():
    ident=identity();old=ROOT/'outputs/five_checkpoint_training_distribution/metrics/teacher_candidates_scored.csv'
    frame=pd.read_csv(old).fillna({'support_tag':''})
    groups=frame.groupby('token');psid=read(OUT/'manifests/PSI_support_identity.json');assert sha(psid['path'])==psid['sha256']
    psi=torch.load(psid['path'],map_location='cpu',weights_only=False);membership=pd.read_csv(OUT/'metrics/training_membership.csv').set_index('token')
    hashes={};allrows=[]
    for si,s in enumerate(scenes()):
        token=s['token'];values=[];rows=[]
        def add(t,**fields):
            a=np.asarray(t,dtype=np.float32);assert a.shape==(8,3) and np.isfinite(a).all()
            j=len(values);values.append(a);rows.append(dict(token=token,teacher_row=j,teacher_id=f'{token}:{j}',trajectory_hash=digest(a.tolist()),**fields))
        add(s['gt'],origin='official_il',source='gt',is_gt=True,weight=1.,actual_training_target=True,archive_path='scene_manifest',raw_index=-1)
        for _,r in groups.get_group(token).iterrows():
            path=r.archive_path
            if path not in hashes:hashes[path]=sha(path)
            assert hashes[path]==r.archive_sha256
            with lzma.open(path,'rb') as f:record=pickle.load(f)
            assert record['token']==token
            a=np.asarray(record['candidates'][int(r.raw_index)],dtype=np.float32)
            assert digest(a.tolist())==r.trajectory_hash
            origin={'mts_8751':'a5_sft','mts_8692':'v6_sft'}[r.model]
            add(a,origin=origin,source=r.source,is_gt=r.source_code==1,weight=float(r.expected_weight_pre_budget),actual_training_target=True,archive_path=path,archive_sha256=hashes[path],raw_index=int(r.raw_index),historical_reward=float(r.historical_reward),previous_PDMS=float(r.PDMS))
        pr=psi['token_to_row'][token]
        for j in np.flatnonzero(np.asarray(psi['support_mask'][pr])):
            source=psi['support_sources'][pr][j]
            add(psi['support_trajectories'][pr,j],origin='psi_sft',source=source,is_gt=source in ['gt','gt_fallback'],weight=float(psi['support_weights'][pr,j]),actual_training_target=bool(membership.loc[token,'psi_train']),archive_path=psid['path'],archive_sha256=psid['sha256'],raw_index=int(j))
        # Exact duplicates remain linked as distinct supervision identities but
        # are never counted as new output modes. Analyses operate per origin.
        for r in rows:r['common_train']=bool(membership.loc[token,'common_train'])
        npz(OUT/'cache/teachers'/f'{token}.npz',dict(protocol_hash=ident,token=token,rows=rows),trajectories=np.stack(values))
        allrows.extend(rows)
        if si%100==0:print('TEACHERS',si+1,flush=True)
    csv('teachers.csv',allrows)
    save(OUT/'audits/teacher_recovery.json',dict(protocol_hash=ident,scene_count=1000,teacher_rows=len(allrows),old_table_hash=sha(old),archive_hashes=hashes,psi=psid,weight_scope='A5/V6 expected pre-residual-budget mass; PSI native weighted one-target sampling; actual post-budget historical mass is separately logged, not reconstructed'))

if __name__=='__main__':main()
