"""Read-only analysis of June/July long runs. Never loads a model or trains."""
from pathlib import Path
import csv, hashlib, json, re, sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path('/mnt/project/VLA-AD')
OLD = Path('/mnt/project/VLA-AD_last_vla_dev')
BACK = ROOT.parent / 'container_backups/container_root_backup_20260709T073417Z/workspace/recogdrive_stage3_rl'
OUT = ROOT / 'outputs/historical_sft_grpo_longchain'
CONFIG = ROOT / 'configs/historical_sft_grpo_longchain/analysis.yaml'
CFG = yaml.load(CONFIG.read_text(), Loader=yaml.CSafeLoader)
A5 = OLD / 'outputs/stage3_lfp_grpo_v1_exact_b8g16_epoch155_r4_20260712T083012Z'
V6 = OLD / 'outputs/stage3_lfp_grpo_v1_v6_epoch165_formal_20260715T145650Z'
V6E = OLD / 'outputs/stage3_epoch165_common_noise_seed0_navtest_zt3_20260716T0221Z/eval'
PS = ROOT / 'outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z'
PR = ROOT / 'outputs/psi_drive_stage3_sr_pgrpo_epoch011_b2acc4_20e_20260627T135941Z'
IL = ROOT / 'outputs/recogdrive_official_2b_il_navtest_dual_eval_seed0_20260724/v1'
SOURCE_FILES = {}
MAP = {'score':'PDMS', 'no_at_fault_collisions':'NC', 'drivable_area_compliance':'DAC',
       'time_to_collision_within_bound':'TTC', 'ego_progress':'EP',
       'driving_direction_compliance':'DDC', 'comfort':'Comfort'}
METRICS = ['PDMS','NC','DAC','TTC','EP','DDC','Comfort']

def historical_summary_points(row):
    """The historical watcher TSV stores PDMS in [0,1], despite its column name."""
    values={m:float(row[m if m!='Comfort' else 'comfort']) for m in METRICS}
    assert all(np.isfinite(v) and 0<=v<=1 for v in values.values())
    return {m:v*100 for m,v in values.items()}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def record(path, role):
    p = Path(path).resolve()
    assert '/pc_mts_diagnostics' not in str(p), p
    assert '/progressive_pcmts' not in str(p), p
    SOURCE_FILES[str(p)] = {'sha256':sha(p), 'role':role, 'bytes':p.stat().st_size}
    return p

def save_json(name, obj):
    p = OUT/name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str, allow_nan=False)+'\n')

def save_csv(name, obj):
    p = OUT/'metrics'/name; p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(obj).to_csv(p, index=False)

def normalize_rows(frame):
    """Only genuine valid per-token rows; preserve explicit missing counts elsewhere."""
    d = frame.copy()
    assert not d['token'].duplicated().any(), 'duplicate benchmark tokens'
    for m in METRICS:
        d[m] = pd.to_numeric(d[m], errors='coerce')
        assert np.isfinite(d[m]).all(), (m, 'unknown score cannot become a zero/failure')
        assert d[m].dropna().between(-1e-7, 1.0000001).all(), (m, 'not fractional NAVSIM metric')
    d['PDMS'] *= 100
    d['feasible'] = (d[['NC','DAC','TTC','DDC']] == 1).all(axis=1).astype(float)
    d['zero_score'] = (d['PDMS']==0).astype(float)
    return d.set_index('token')

def prediction_bank(directory):
    rows=[]; invalid=[]
    files=sorted(Path(directory).glob('shard_*/predictions.json'))
    assert files, directory
    for p in files:
        a=json.loads(record(p,'historical_prediction_and_score').read_text())
        for r in a:
            token=r['sample_token']
            if not r.get('pdm_valid', False): invalid.append(token); continue
            q=r['pdm']; row={'token':token,'scene_token':r['scene_token']}
            row.update({dst:q[src] for src,dst in MAP.items()})
            traj=np.asarray(r['pred_traj'],dtype=float)
            assert traj.shape==(8,3) and np.isfinite(traj).all()
            row['endpoint_x']=traj[-1,0];row['endpoint_y']=traj[-1,1]
            row['trajectory']=traj
            rows.append(row)
    result=normalize_rows(pd.DataFrame(rows))
    result.attrs['missing_or_invalid']=len(invalid)
    result.attrs['total_manifest_rows']=len(rows)+len(invalid)
    return result

def csv_scores(path):
    d=pd.read_csv(record(path,'historical_per_token_score'))
    assert 'token' in d
    d=d[d['token'].astype(str).str.fullmatch('[0-9a-f]{16,64}')].copy()
    if 'valid' in d:
        d=d[d['valid'].astype(str).str.lower().isin(['true','1'])]
    d=d.rename(columns=MAP)
    return normalize_rows(d[['token']+METRICS])

def remap(path):
    p=Path(path)
    if p.exists(): return p
    if str(p).startswith('/workspace/recogdrive_stage3_rl/'):
        p=BACK/str(p).split('/workspace/recogdrive_stage3_rl/',1)[1]
    assert p.exists(), p
    return p

def summary_record(run, checkpoint_id):
    inv=pd.read_csv(record(run/'checkpoint_store/inventory.tsv','historical_checkpoint_inventory'),sep='\t')
    row=inv[inv.checkpoint_id==checkpoint_id].iloc[0]
    summ=pd.read_csv(record(run/'eval/navtest/summary.tsv','historical_evaluation_index'),sep='\t')
    match=summ[summ.checkpoint_id==row.sha256].iloc[0]
    return csv_scores(remap(match.csv_path)), row

def boot_mean(delta, groups, reps=3000, seed=2026091401):
    """Ratio-of-sums cluster bootstrap preserves benchmark sample weighting."""
    v=np.asarray(delta,float); v=v[:,None] if v.ndim==1 else v
    _,idx=np.unique(np.asarray(groups,str),return_inverse=True)
    n=idx.max()+1; sums=np.zeros((n,v.shape[1]));counts=np.bincount(idx)
    np.add.at(sums,idx,v)
    rng=np.random.default_rng(seed)
    weights=rng.multinomial(n,np.full(n,1/n),size=reps)
    means=(weights@sums)/(weights@counts)[:,None]
    return np.quantile(means,[.025,.975],axis=0)

def compare(base, after, label, clusters, seed_protocol):
    keys=base.index.intersection(after.index).sort_values()
    a=base.loc[keys];b=after.loc[keys];cols=METRICS+['feasible','zero_score']
    d=b[cols]-a[cols]
    groups=clusters.reindex(keys)
    assert groups.notna().all(), 'missing historical scene clusters'
    ci=boot_mean(d.to_numpy(), groups, CFG['bootstrap_replicates'], CFG['bootstrap_seed'])
    table=[]
    for k,m in enumerate(cols):
        scale=1 if m=='PDMS' else 100
        table.append(dict(comparison=label,metric=m,unit='points' if m=='PDMS' else 'percentage_points',
            baseline_mean=float(a[m].mean()*scale),after_mean=float(b[m].mean()*scale),
            mean_difference=float(d[m].mean()*scale),median_difference=float(d[m].median()*scale),
            ci_low=float(ci[0,k]*scale),ci_high=float(ci[1,k]*scale),
            token_win_fraction=float((d[m]>1e-10).mean()),
            scene_win_fraction=float((d[m].groupby(groups).mean()>1e-10).mean()),
            n_tokens=len(keys),n_scene_clusters=groups.nunique(),seed_protocol=seed_protocol))
    regress=(b[['NC','DAC','TTC']]<a[['NC','DAC','TTC']]-1e-10).any(axis=1)
    delta=d['PDMS'];tail_count=max(1,int(np.ceil(.05*len(keys))))
    tail=dict(comparison=label,n_tokens=len(keys),safety_regression_fraction=float(regress.mean()),
        safety_regression_contribution=float(delta[regress].sum()/len(keys)),
        nonregression_contribution=float(delta[~regress].sum()/len(keys)),
        total_delta=float(delta.mean()),worst5_contribution=float(delta.nsmallest(tail_count).sum()/len(keys)),
        zero_score_repairs=int(((a.PDMS==0)&(b.PDMS>0)).sum()),
        new_zero_scores=int(((a.PDMS>0)&(b.PDMS==0)).sum()))
    for m in ['NC','DAC','TTC']:
        tail[m+'_repairs']=int(((a[m]<1)&(b[m]==1)).sum())
        tail[m+'_regressions']=int(((a[m]==1)&(b[m]<1)).sum())
    if 'trajectory' in a and 'trajectory' in b:
        ta=np.stack(a.trajectory);tb=np.stack(b.trajectory)
        tail['mean_trajectory_shift_ADE_m']=float(np.linalg.norm(tb[:,:,:2]-ta[:,:,:2],axis=-1).mean())
        tail['mean_endpoint_x_shift_m']=float((tb[:,-1,0]-ta[:,-1,0]).mean())
        tail['mean_endpoint_y_shift_m']=float((tb[:,-1,1]-ta[:,-1,1]).mean())
    assert abs(tail['safety_regression_contribution']+tail['nonregression_contribution']-tail['total_delta'])<1e-8
    return table,tail

def curve_row(family, step, frame, protocol, label):
    return dict(family=family,step=step,checkpoint=label,protocol=protocol,n_valid=len(frame),
        **{m:float(frame[m].mean()*(1 if m=='PDMS' else 100)) for m in METRICS+['feasible','zero_score']})

def training_events():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    original=BACK/'outputs/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z/navsim_exp/training_recogdrive_agent/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z'
    roots={'official_original':original/'lightning_logs','psi_sr':PR/'lightning_logs','a5_lfp':A5/'hydra','v6_lfp':V6/'hydra'}
    records=[];tags_by_run={}
    useful=['reward','bc_loss','bc_coeff','reference_kl','lfp_scalar','lfp_ref_scalar','lfp_delta_scalar',
        'lfp_ep_mean','lfp_ttc_mean','lfp_nc_mean','lfp_dac_mean','lfp_feasible','lfp_exact_kl',
        'lfp_group_pairwise','lfp_group_scalar','lfp_ttc_fail','lfp_positive','lfp_negative','lfp_zero',
        'lfp_advantage','lfp_all_infeasible','lfp_sampler','core_pareto_pdms_formula','safe_ratio',
        'support_relative_enabled','support_missing_ratio','final_positive_invalid','final_positive_slow',
        'within_support_score_std','occupied_support','rankable_support','frontier_gain','lfp_quality_reference']
    for name,root in roots.items():
        files=sorted(root.rglob('events.out.tfevents*'));assert files,(name,root)
        for f in files:
            a=EventAccumulator(str(record(f,'historical_long_training_tensorboard')),size_guidance={'scalars':0});a.Reload()
            tags=a.Tags()['scalars'];tags_by_run[name]=tags
            for tag in tags:
                if tag not in ['epoch','lr-AdamW'] and not any(s in tag for s in useful):continue
                for e in a.Scalars(tag):records.append(dict(run=name,tag=tag,step=e.step,value=e.value,wall_time=e.wall_time))
        print('events extracted',name,flush=True)
    d=pd.DataFrame(records).sort_values('wall_time').drop_duplicates(['run','tag','step'],keep='last')
    d.to_parquet(OUT/'metrics/training_scalars.parquet',index=False)
    result=[]
    for (run,tag),g in d.groupby(['run','tag']):
        if tag.endswith('_epoch'):
            for r in g.itertuples():result.append(dict(run=run,tag=tag,phase='epoch_observation',step=r.step,mean=r.value,n=1))
        else:
            for label,lo,hi in [('steps_0_300',0,300),('steps_301_1613',301,1613),('steps_1614_plus',1614,float('inf'))]:
                h=g[(g.step>=lo)&(g.step<=hi)]
                if len(h):result.append(dict(run=run,tag=tag,phase=label,step=int(h.step.max()),mean=float(h.value.mean()),n=len(h)))
    save_csv('training_phase_summary.csv',result)
    save_json('manifests/training_event_tags.json',tags_by_run)

def main():
    (OUT/'metrics').mkdir(parents=True,exist_ok=True)
    record(CONFIG,'analysis_specification')
    banks={};curves=[];comparisons=[];tails=[]
    base_dir=A5/'navtest_stable_seed0_paired_20260712/v1'
    for label,name,step in [('base','epoch_155.ckpt',0),('s300','step-step_300.ckpt',300),
        ('e0','epoch_0-step_1614.ckpt',1614),('e1','epoch_1-step_3228.ckpt',3228),('e2','epoch_2-step_4842.ckpt',4842)]:
        banks['a5_'+label]=prediction_bank(base_dir/name)
        curves.append(curve_row('a5_lfp',step,banks['a5_'+label],'historical_seed0_common_noise',name))
    clusters=banks['a5_base']['scene_token']
    assert clusters.nunique()==1203
    for label in ['s300','e0','e1','e2']:
        t,z=compare(banks['a5_base'],banks['a5_'+label],'a5_'+label,clusters,'CRN_seed0');comparisons+=t;tails.append(z)
    for f in sorted((A5/'navtest_epoch_eval_v1_v2_top3').glob('epoch_*/navsim_v1/navtest_summary.tsv')):
        row=pd.read_csv(record(f,'historical_non_CRN_watcher'),sep='\t').iloc[0]
        curves.append(dict(family='a5_watcher',step=(int(f.parts[-3].split('_')[-1])+1)*1614,
            checkpoint=row.checkpoint,protocol='historical_watcher_not_CRN',n_valid=int(row.num_pdm_valid),
            **historical_summary_points(row)))
    for label,name,step in [('base','eval_baseline-epoch_165',0),('s300','eval_lfp-step_300',300),
                           ('s2100','eval_lfp-step_2100',2100),('s3300','eval_lfp-step_3300',3300)]:
        f=next((V6E/name).glob('hydra/*/*/*.csv'));banks['v6_'+label]=csv_scores(f)
        curves.append(curve_row('v6_lfp',step,banks['v6_'+label],'historical_seed0_common_noise',name))
    for label in ['s300','s2100','s3300']:
        t,z=compare(banks['v6_base'],banks['v6_'+label],'v6_'+label,clusters,'CRN_seed0');comparisons+=t;tails.append(z)
    banks['psi_base'],_=summary_record(PS,'epoch_011')
    curves.append(curve_row('psi_sr',0,banks['psi_base'],'historical_independent_eval_seeds','sft_epoch11'))
    inv=pd.read_csv(record(PR/'checkpoint_store/inventory.tsv','historical_checkpoint_inventory'),sep='\t')
    for row in inv.itertuples():
        f,_=summary_record(PR,row.checkpoint_id)
        curves.append(curve_row('psi_sr',int(row.train_step),f,'historical_independent_eval_seeds',row.checkpoint_id))
        if int(row.train_step) in [300,3000,12000,21000,26400]:
            t,z=compare(banks['psi_base'],f,'psi_'+str(row.train_step),clusters,'not_CRN_historical');comparisons+=t;tails.append(z)
    il_dir=next(p for p in IL.iterdir() if p.is_dir() and p.name.startswith('ReCogDrive_Diffusion'))
    banks['official_base']=prediction_bank(il_dir)
    curves.append(curve_row('official_original',0,banks['official_base'],'July_seed0_IL_vs_June_unmatched_RL','released_IL'))
    f=BACK/'outputs/pdms_eval_stage3_ckpt_watcher_20260609T093852Z/pdms_summary.csv'
    table=pd.read_csv(record(f,'historical_original_GRPO_curve'))
    for row in table.itertuples():
        if row.status!='done':continue
        frame=csv_scores(remap(row.csv));step=int(re.search(r'step(\d+)',row.ckpt_name).group(1))
        curves.append(curve_row('official_original',step,frame,'historical_unmatched_eval_seeds',row.ckpt_name))
        if step in [1330,11970,13300]:
            t,z=compare(banks['official_base'],frame,'official_'+str(step),clusters,'not_CRN_July_IL_June_RL');comparisons+=t;tails.append(z)
    save_csv('historical_curves.csv',curves);save_csv('paired_comparisons.csv',comparisons);save_csv('tail_decomposition.csv',tails)
    save_csv('scene_clusters.csv',clusters.rename('scene_token').reset_index())
    save_json('manifests/dataset_counts.json',{k:dict(n_valid=len(v),**v.attrs) for k,v in banks.items()})
    training_events()
    previous=OUT/'manifests/inputs.json'
    merged=json.loads(previous.read_text()) if previous.exists() else {}
    merged.update(SOURCE_FILES)
    save_json('manifests/inputs.json',merged)
    print('DONE historical analysis',len(curves),'curve rows',len(comparisons),'paired metrics',flush=True)

if __name__=='__main__': main()
