"""All fixed endpoints, scene-paired inference, seed-resolved gains and training safety."""
from common_r import *
import concurrent.futures,math,pandas as pd
from evaluate_r import source_paths
from analyze_policy_distribution import geometry
def scene_record(item):
    method,sd,step,token=item
    rp,sp=source_paths(method,sd,step,token);a=np.load(rp);s=np.load(sp)['scores'];traj=a['trajectories']
    assert traj.shape==(64,8,3) and s.shape==(64,7) and np.isfinite(s).all()
    il=np.load(source_paths('official_il',1701,0,token)[0])['trajectories'];init=np.load(source_paths(method,sd,0,token)[0])['trajectories']
    ig,ic=geometry(il);g,c=geometry(traj);f=v3.feasible(s);ref=float(np.load(V1/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6])
    positive=f&(s[:,6]>ref);nh=int((f&(s[:,6]>=ref+.01)).sum());fit=dict(zip(a['fitting_names'],a['fitting_loss']))
    fm='gt_only' if method=='official_il' else method
    d=dict(method=method,seed=sd,step=step,token=token,PDMS=float(s[:,6].mean()*100),feasible_rate=float(f.mean()),hard_failure=float(v3.hard_failure(s).mean()),
        zero_score_rate=float((s[:,6]==0).mean()),EP=float(s[:,2].mean()),NC=float(s[:,0].mean()),DAC=float(s[:,1].mean()),TTC=float(s[:,3].mean()),DDC=float(s[:,5].mean()),Comfort=float(s[:,4].mean()),
        pairwise_ADE=g['pairwise_ade'],Spread_AUC=g['spread_auc'],center_shift=float(v3.distance(c[None],ic[None])[0,0]),
        CRN_policy_change_ADE=float(np.linalg.norm(traj[...,:2]-init[...,:2],axis=-1).mean()),D_positive=v3.pair_mean(traj[positive]) if positive.sum()>=2 else 0.,
        IL_retention_loss=float(fit['IL_retention']),candidate_fitting_loss=float(fit[fm]))
    for k in [1,8,16,64]:d[f'Hit{k}']=1-math.comb(64-nh,k)/math.comb(64,k) if 64-nh>=k else 1.
    return d
def write(df,name):
    dest=guard(OUT/'metrics'/name);dest.parent.mkdir(parents=True,exist_ok=True);df.to_csv(dest,index=False)
def main():
    identity();items=[(m,sd,s,t) for m in CFG['methods'] for sd in CFG['seeds'] for s in CFG['snapshots'] for t in tokens('holdout')]
    with concurrent.futures.ProcessPoolExecutor(48) as pool:records=list(pool.map(scene_record,items,chunksize=12))
    df=pd.DataFrame(records);assert len(df)==21000
    write(df,'holdout_scene.csv');cols=[x for x in df if x not in ['method','seed','step','token']]
    write(df.groupby(['method','seed','step'])[cols].mean().reset_index(),'seed_means.csv')
    av=df.groupby(['method','step','token'])[cols].mean().reset_index();write(av,'seed_averaged_scene.csv')
    stats=[]
    for (m,step),g in av.groupby(['method','step']):
        for metric in cols:stats.append(dict(method=m,step=step,metric=metric,**v3.bootstrap(g[metric],f'R_{m}_{step}_{metric}')))
    write(pd.DataFrame(stats),'holdout_bootstrap.csv')
    gains=[]
    for (m,sd,t),g in df.groupby(['method','seed','token']):
        z=g[g.step==0].iloc[0]
        for _,r in g.iterrows():gains.append(dict(method=m,seed=sd,step=r.step,token=t,PDMS_gain=r.PDMS-z.PDMS,feasible_change=r.feasible_rate-z.feasible_rate,EP_change=r.EP-z.EP,retention_loss_change=r.IL_retention_loss-z.IL_retention_loss))
    gd=pd.DataFrame(gains);write(gd,'gain_scene.csv');gm=gd.groupby(['method','step','token']).mean(numeric_only=True).reset_index();write(gd.groupby(['method','seed','step']).mean(numeric_only=True).reset_index(),'gain_seed_means.csv')
    gs=[]
    for (m,step),g in gm.groupby(['method','step']):
        for metric in ['PDMS_gain','feasible_change','EP_change','retention_loss_change']:gs.append(dict(method=m,step=step,metric=metric,**v3.bootstrap(g[metric],f'R_gain_{m}_{step}_{metric}')))
    write(pd.DataFrame(gs),'gain_bootstrap.csv')
    contrasts=[]
    for end in [100,110]:
        for metric in ['PDMS_gain','feasible_change','EP_change']:
            table=gm[gm.step==end].pivot(index='token',columns='method',values=metric)
            for ref in ['official_il','gt_only','conditional_pc']:
                for m in CFG['methods']:
                    if m==ref:continue
                    contrasts.append(dict(endpoint=end,method=m,reference=ref,metric=metric,**v3.bootstrap(table[m]-table[ref],f'R_contrast_{m}_{ref}_{end}_{metric}')))
    write(pd.DataFrame(contrasts),'paired_gain_contrasts.csv')
    final_contrasts=[]
    for end in [100,110]:
        for metric in ['PDMS','feasible_rate','hard_failure','EP','TTC','CRN_policy_change_ADE','IL_retention_loss','Hit8']:
            table=av[av.step==end].pivot(index='token',columns='method',values=metric)
            for ref in ['gt_only','conditional_pc']:
                for m in CFG['methods']:
                    if m==ref:continue
                    final_contrasts.append(dict(endpoint=end,method=m,reference=ref,metric=metric,**v3.bootstrap(table[m]-table[ref],f'R_final_{m}_{ref}_{end}_{metric}')))
    write(pd.DataFrame(final_contrasts),'paired_final_contrasts.csv')
    old=pd.read_csv(V3/'metrics/F_gain_scene.csv');oldav=old.groupby(['method','step','token']).mean(numeric_only=True).reset_index();oldcontrast=[]
    for end in [10,50,100]:
        for m in CFG['methods'][1:]:
            x=gm[(gm.method==m)&(gm.step==end)].set_index('token');y=oldav[(oldav.method==m)&(oldav.step==end)].set_index('token')
            for metric in ['PDMS_gain','feasible_change','EP_change']:
                oldcontrast.append(dict(method=m,step=end,metric=metric,**v3.bootstrap(x[metric]-y[metric],f'R_old_{m}_{end}_{metric}')))
    write(pd.DataFrame(oldcontrast),'formal_minus_old_recipe_gain.csv')
    dynamics=[];audit=[];ledgers={}
    for m in CFG['methods']:
        for sd in CFG['seeds']:
            run=run_name(m,sd);directory=OUT/'metrics'/run
            r=pd.concat([pd.read_parquet(directory/f'rollouts_rank{k}.parquet') for k in range(8)],ignore_index=True)
            assert len(r)==110*64*8 and set(r.token)==set(tokens('train')) and not set(r.token)&set(tokens('holdout'))
            assert (r.groupby(['step','rank','group']).size()==8).all()
            updates=[u for rank in range(8) for u in read(directory/f'updates_rank{rank}.json')]
            assert len(updates)==110*8
            ledgers[(m,sd)]={(u['step'],u['rank']):(u['tokens'],u['rng_after_sampling']) for u in updates}
            for step,g in r.groupby('step'):
                inf=~g.feasible;u=[z for z in updates if z['step']==step]
                dynamics.append(dict(method=m,seed=sd,step=step,PDMS=g.score.mean()*100,feasible_rate=g.feasible.mean(),hard_failure=g.hard_failure.mean(),zero_score_rate=(g.score==0).mean(),
                    EP=g.ego_progress.mean(),NC=g.no_at_fault_collisions.mean(),DAC=g.drivable_area_compliance.mean(),TTC=g.time_to_collision_within_bound.mean(),DDC=g.driving_direction_compliance.mean(),
                    PIA_rate=((g.advantage>0)&inf).sum()/inf.sum() if inf.any() else np.nan,infeasible_count=int(inf.sum()),positive_infeasible_count=int(((g.advantage>0)&inf).sum()),
                    lr_after_step=u[0]['lr'],lr_used_from_frozen_epoch_schedule=1e-4*.5*(1+math.cos(math.pi*u[0]['epoch']/10)),amp_skipped_update=any(z['amp_nonfinite_gradient'] for z in u),gradient_norm=np.mean([z['gradient_norm'] for z in u if z['gradient_norm'] is not None])))
            audit.append(dict(method=m,seed=sd,rollouts=len(r),unique_tokens=r.token.nunique(),groups=r.group.nunique(),checkpoint_hashes={str(step):sha(checkpoint(m,sd,step)) for step in CFG['snapshots']}))
    for sd in CFG['seeds']:
        reference=ledgers[('official_il',sd)]
        for m in CFG['methods']:assert ledgers[(m,sd)]==reference,('Training CRN/sampler mismatch',m,sd)
    write(pd.DataFrame(dynamics),'training_safety_dynamics.csv')
    eff=[]
    for (m,sd,t),g in gd.groupby(['method','seed','token']):
        g=g.sort_values('step');eff.append(dict(method=m,seed=sd,token=t,gain_AUC=np.trapz(g.PDMS_gain,g.step)/110,gain_per_100_updates=g.iloc[-1].PDMS_gain/110*100))
    write(pd.DataFrame(eff),'efficiency_scene.csv')
    save(OUT/'manifests/final_metrics_audit.json',dict(identity(),runs=audit,scene_rows=21000,new_evaluation_rollouts=1075200,all_fixed_endpoints=True,training_CRN_and_order_identical_across_methods=True,
        holdout_tokens_used_for_updates=False,holdout_scope='diagnostic holdout from new updates; not claimed unseen in original IL pretraining',previous_results_modified=False))
    print(pd.DataFrame(gs).query('metric=="PDMS_gain" and step==110').to_string(index=False),flush=True)
if __name__=='__main__':main()
