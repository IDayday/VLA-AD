"""Scene-paired training outcomes, seed-specific results, and genuine gain trajectories."""
import argparse,concurrent.futures,math
from common_v3 import *
from analyze_policy_distribution import geometry

def scene_record(task):
    spec,scene=task;token=scene['token'];src=spec['source'];relative=Path(src['stage'])/src['run']/f"step{src['step']:04d}"/f'{token}.npz';a=np.load(OUT/'cache/rollouts'/relative);s=np.load(OUT/'cache/scores'/relative)['scores'];traj=a['trajectories'];assert len(traj)==64 and np.isfinite(s).all()
    il=np.load(OUT/'cache/rollouts/sft/official_il/step0000'/f'{token}.npz')['trajectories'];ig,ic=geometry(il);g,c=geometry(traj);f=feasible(s);hard=hard_failure(s);ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6]);positive=f&(s[:,6]>ref);hit=f&(s[:,6]>=ref+.01);nh=int(hit.sum());fit=dict(zip(a['fitting_names'],a['fitting_loss']))
    init=il if spec['stage']=='sft' else np.load(OUT/'cache/rollouts/sft'/f"{spec['method']}_seed{spec['seed']}"/'step0200'/f'{token}.npz')['trajectories']
    result=dict(stage=spec['stage'],method=spec['method'],seed=spec['seed'],step=spec['step'],token=token,PDMS=float(s[:,6].mean()*100),feasible_rate=float(f.mean()),hard_failure=float(hard.mean()),zero_score_rate=float((s[:,6]==0).mean()),EP=float(s[:,2].mean()),NC=float(s[:,0].mean()),DAC=float(s[:,1].mean()),TTC=float(s[:,3].mean()),DDC=float(s[:,5].mean()),pairwise_ADE=g['pairwise_ade'],Spread_AUC=g['spread_auc'],spread_ratio_IL=g['spread_auc']/max(ig['spread_auc'],1e-12),center_shift=float(distance(c[None],ic[None])[0,0]),mean_center_shift=float(distance(traj.mean(0)[None],il.mean(0)[None])[0,0]),CRN_policy_change_ADE=float(np.linalg.norm(traj[...,:2]-init[...,:2],axis=-1).mean()),D_positive=pair_mean(traj[positive]) if positive.sum()>=2 else 0.,positive_count=int(positive.sum()),IL_retention_loss=float(fit['IL_retention']),candidate_fitting_loss=float(fit[spec['method']]))
    for k in [1,8,16,64]:result[f'Hit{k}']=1-math.comb(64-nh,k)/math.comb(64,k) if 64-nh>=k else 1.
    return result

def main(stage):
    manifest=read(OUT/'manifests'/f'evaluation_{stage}.json');aliases=manifest['aliases'];rows=[r for r in scenes() if r['token'] in set(tokens('holdout'))]
    with concurrent.futures.ProcessPoolExecutor(48) as ex:records=list(ex.map(scene_record,[(sp,r) for sp in aliases for r in rows],chunksize=8))
    d=pd.DataFrame(records);assert len(d)==6*2*4*300;tag='E' if stage=='sft' else 'F';csv(d,f'{tag}_holdout_scene.csv');stats=[];metrics=[k for k in d.columns if k not in ['stage','method','seed','step','token']]
    # Average seeds within a scene for primary paired inference; report both seeds separately.
    av=d.groupby(['method','step','token'])[metrics].mean().reset_index();csv(d.groupby(['method','seed','step'])[metrics].mean().reset_index(),f'{tag}_seed_means.csv');csv(av,f'{tag}_seed_averaged_scene.csv')
    for (m,step),g in av.groupby(['method','step']):
        for metric in metrics:stats.append(dict(method=m,step=step,metric=metric,**bootstrap(g[metric],f'{tag}_{m}_{step}_{metric}')))
    csv(pd.DataFrame(stats),f'{tag}_holdout_bootstrap.csv');comparisons=[]
    final=max(d.step)
    for m in [m for m in CFG['training']['methods'] if m!='conditional_pc']:
        for metric in metrics:
            t=av[av.step==final].pivot(index='token',columns='method',values=metric);comparisons.append(dict(method=m,step=final,metric=metric,contrast='conditional_minus_baseline',**bootstrap(t.conditional_pc-t[m],f'{tag}_paired_{m}_{metric}')))
    csv(pd.DataFrame(comparisons),f'{tag}_final_paired.csv')
    gains=[]
    for (m,sd,token),g in d.groupby(['method','seed','token']):
        g=g.sort_values('step');zero=g[g.step==0].iloc[0]
        for _,r in g.iterrows():gains.append(dict(method=m,seed=sd,token=token,step=int(r.step),PDMS_gain=r.PDMS-zero.PDMS,feasible_change=r.feasible_rate-zero.feasible_rate,EP_change=r.EP-zero.EP,center_shift_change=r.center_shift-zero.center_shift))
    gain=pd.DataFrame(gains);csv(gain,f'{tag}_gain_scene.csv');gainav=gain.groupby(['method','step','token']).mean(numeric_only=True).reset_index();gainstats=[]
    for (m,step),g in gainav.groupby(['method','step']):
        for metric in ['PDMS_gain','feasible_change','EP_change','center_shift_change']:gainstats.append(dict(method=m,step=step,metric=metric,**bootstrap(g[metric],f'{tag}_gain_{m}_{step}_{metric}')))
    csv(pd.DataFrame(gainstats),f'{tag}_gain_bootstrap.csv')
    if stage=='grpo':
        audit_rollouts=[];dynamics=[]
        for m in CFG['training']['methods']:
            for sd in CFG['training']['seeds']:
                r=pd.read_parquet(OUT/'metrics'/f'F_rollouts_{m}_{sd}.parquet');assert len(r)==100*8*8 and set(r.token)<=set(tokens('train'));assert not set(r.token)&set(tokens('holdout'))
                for step,g in r.groupby('step'):
                    inf=~g.feasible;pia=((g.advantage>0)&inf).sum()/inf.sum() if inf.any() else np.nan
                    dynamics.append(dict(method=m,seed=sd,step=step,PDMS=g.score.mean()*100,feasible_rate=g.feasible.mean(),hard_failure=g.hard_failure.mean(),zero_score_rate=(g.score==0).mean(),EP=g.ego_progress.mean(),NC=g.no_at_fault_collisions.mean(),DAC=g.drivable_area_compliance.mean(),TTC=g.time_to_collision_within_bound.mean(),DDC=g.driving_direction_compliance.mean(),PIA_rate=pia,infeasible_count=int(inf.sum()),positive_infeasible_count=int(((g.advantage>0)&inf).sum())))
                audit_rollouts.append(dict(method=m,seed=sd,rollouts=len(r),group_count=r.group.nunique(),tokens=r.token.nunique(),sha256=sha(OUT/'metrics'/f'F_rollouts_{m}_{sd}.parquet')))
        csv(pd.DataFrame(dynamics),'F_training_safety_dynamics.csv');eff=[]
        for (m,sd,token),g in gain.groupby(['method','seed','token']):
            g=g.sort_values('step');eff.append(dict(method=m,seed=sd,token=token,gain_AUC=np.trapz(g.PDMS_gain,g.step)/100,gain_per_100_steps=g.iloc[-1].PDMS_gain,gain_per_1000_steps_linear_scale_only=g.iloc[-1].PDMS_gain*10))
        eff=pd.DataFrame(eff);csv(eff,'F_learning_efficiency_scene.csv');csv(eff.groupby(['method','seed']).mean(numeric_only=True).reset_index(),'F_seed_gain_means.csv');ee=eff.groupby(['method','token']).mean(numeric_only=True).reset_index();rr=[]
        for metric in ['gain_AUC','gain_per_100_steps']:
            t=ee.pivot(index='token',columns='method',values=metric)
            for m in CFG['training']['methods']:
                if m!='conditional_pc':rr.append(dict(method=m,metric=metric,**bootstrap(t.conditional_pc-t[m],f'F_efficiency_{m}_{metric}')))
        csv(pd.DataFrame(rr),'F_efficiency_paired.csv')
    audits=[]
    for m in CFG['training']['methods']:
        for sd in CFG['training']['seeds']:audits.append(read(OUT/'manifests'/f'train_{stage}_{m}_{sd}.json'))
    assert all(r['steps']==(200 if stage=='sft' else 100) for r in audits)
    stage_audit(tag,holdout_scenes=300,train_scenes=700,methods=6,seeds=2,snapshots=4,rollouts_per_scene=64,holdout_scene_rows=len(d),all_fixed_snapshots_included=True,no_best_checkpoint_selection=True,holdout_used_for_updates=False,initialization_hashes={f"{r['method']}_{r['seed']}":r['initial_checkpoint_sha256'] for r in audits},results_sha256=sha(OUT/'metrics'/f'{tag}_holdout_scene.csv'),CRN_namespace='v3_diagnostic_holdout64',new_update_holdout_only_not_unseen_official_pretraining=True,explicit_GT_fallback_updates={f"{r['method']}_{r['seed']}":r['explicit_GT_fallback_updates'] for r in audits})
    print(d[d.step==final].groupby('method')[['PDMS','feasible_rate','center_shift','Spread_AUC','Hit8','IL_retention_loss']].mean().to_string(),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['sft','grpo']);main(a.parse_args().stage)
