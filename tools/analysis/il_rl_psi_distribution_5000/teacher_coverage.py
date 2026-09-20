"""Actual PSI supervision coverage on the same 5000 scenes; no new GPU work."""
from common_5000 import *
import torch, argparse, datetime, time
import analyze as historical

CFG_SUP=WORK/'configs/il_rl_psi_distribution_5000/teacher_coverage.yaml'
ARGS=MAIN/'outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/train_args.json'

def freeze():
    identity();dest=OUT/'manifests/teacher_coverage_protocol.json'
    ident=read(OLDOUT/'manifests/PSI_support_identity.json')
    if dest.exists():
        assert read(dest)['config_sha256']==sha(CFG_SUP);return
    assert sha(ident['path'])==ident['sha256']
    save(dest,dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config_sha256=sha(CFG_SUP),scene_hash=sha(OUT/'manifests/scenes.json'),support_identity=ident,train_args_path=str(ARGS),train_args_sha256=sha(ARGS),no_new_diffusion_loss=True))

def main():
    freeze();identity();meta=read(OUT/'manifests/teacher_coverage_protocol.json')
    assert sha(ARGS)==meta['train_args_sha256']
    support=torch.load(meta['support_identity']['path'],map_location='cpu',weights_only=False)
    args=read(ARGS);trainlogs=set(args['train_logs']);vallogs=set(args['val_logs']);assert not trainlogs&vallogs
    detailed=[];aggregates=[];membership=[]
    for si,scene in enumerate(scenes()):
        token=scene['token'];pr=support['token_to_row'][token]
        is_train=scene['log'] in trainlogs;is_val=scene['log'] in vallogs
        assert is_train!=is_val,(token,scene['log'])
        assert is_train==(support['support_sources'][pr][0]!='gt_fallback'),token
        mask=np.asarray(support['support_mask'][pr]).astype(bool)
        weights=np.asarray(support['support_weights'][pr]);ix=np.flatnonzero(mask & (weights>0))
        teachers=np.asarray(support['support_trajectories'][pr])[ix]
        sources=[support['support_sources'][pr][j] for j in ix]
        isgt=np.array([s in ['gt','gt_fallback'] for s in sources]);w=weights[ix]
        membership.append(dict(token=token,log=scene['log'],cohort=scene['cohort'],psi_train=is_train,psi_validation=is_val,positive_weight_targets=len(ix),non_gt_targets=int((~isgt).sum())))
        for model in CFG['primary_models']:
            with np.load(bank_path(model,token)) as z:
                for protocol in CFG['protocols']:
                    traj=z[protocol].reshape(64,8,3);d=distance(teachers,traj)
                    values=dict(nearest_ADE=d.min(1))
                    for a in [.25,.5,1.]:
                        suffix=str(a).replace('.','p');hit=d<=a
                        values[f'hit64_{suffix}']=hit.any(1);values[f'hit16_{suffix}']=hit.reshape(-1,4,16).any(2).mean(1);values[f'mass64_{suffix}']=hit.mean(1)
                    base=dict(token=token,log=scene['log'],cohort=scene['cohort'],psi_train=is_train,model=model,protocol=protocol)
                    for j,raw_index in enumerate(ix):
                        detailed.append(dict(**base,teacher_index=int(raw_index),teacher_hash=digest(teachers[j].tolist()),source=sources[j],is_gt=bool(isgt[j]),weight=float(w[j]),**{k:float(v[j]) for k,v in values.items()}))
                    for kind,keep in [('ALL',np.ones(len(ix),dtype=bool)),('NON_GT',~isgt),('GT',isgt)]:
                        if not keep.any():continue
                        ww=w[keep]/w[keep].sum()
                        aggregates.append(dict(**base,kind=kind,teachers=int(keep.sum()),**{k:float(v[keep].mean()) for k,v in values.items()},**{'weighted_'+k:float(v[keep]@ww) for k,v in values.items()}))
        if si%500==0:print('TEACHER COVERAGE',si+1,flush=True)
    mf=pd.DataFrame(membership);af=pd.DataFrame(aggregates)
    mf.to_csv(OUT/'metrics/psi_training_membership.csv',index=False)
    pd.DataFrame(detailed).to_parquet(OUT/'metrics/psi_teacher_coverage.parquet',index=False)
    af.to_parquet(OUT/'metrics/psi_teacher_scene_coverage.parquet',index=False)
    summaries=[];comparisons=[]
    for scope,data in [('FULL5000',af),('NEW4000',af[af.cohort=='NEW4000']),('OLD1000',af[af.cohort=='OLD1000'])]:
        for membership_name,subset in [('PSI_TRAIN',data[data.psi_train]),('PSI_VALIDATION',data[~data.psi_train])]:
            for (model,protocol,kind),g in subset.groupby(['model','protocol','kind']):
                summaries.append(dict(scope=scope,membership=membership_name,model=model,protocol=protocol,kind=kind,scenes=len(g),teachers=int(g.teachers.sum()),**{k:g[k].mean() for k in values},**{'weighted_'+k:g['weighted_'+k].mean() for k in values}))
            for protocol in CFG['protocols']:
                for kind in ['ALL','NON_GT','GT']:
                    base=subset[(subset.model=='official_il')&(subset.protocol==protocol)&(subset.kind==kind)].set_index('token')
                    if base.empty:continue
                    for model in CFG['primary_models'][1:]:
                        comp=subset[(subset.model==model)&(subset.protocol==protocol)&(subset.kind==kind)].set_index('token').loc[base.index]
                        for metric in ['hit64_0p5','hit16_0p5','mass64_0p5','nearest_ADE','weighted_hit64_0p5']:
                            comparisons.append(dict(scope=scope,membership=membership_name,model=model,protocol=protocol,kind=kind,metric=metric,**historical.bootstrap_difference(comp[metric]-base[metric],base.log,f'teacher/{scope}/{membership_name}/{model}/{protocol}/{kind}/{metric}')))
    csv('psi_teacher_summary.csv',summaries);csv('psi_teacher_comparisons.csv',comparisons)
    save(OUT/'audits/teacher_coverage.json',dict(status='PASS',scenes=5000,training_scenes=int(mf.psi_train.sum()),validation_scenes=int(mf.psi_validation.sum()),training_scenes_with_non_gt=int(((mf.non_gt_targets>0)&mf.psi_train).sum()),support_sha256=meta['support_identity']['sha256'],train_val_labels_checked_against_actual_log_lists=True,teacher_conditioning=False,no_new_native_diffusion_loss=True))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');a.add_argument('--wait',action='store_true');args=a.parse_args()
    if args.freeze:freeze()
    else:
        if args.wait:
            while not (OUT/'audits/gpu_sampling_complete.json').exists():time.sleep(20)
        main()
