"""Configuration/identity audit, historical control and scale calculation. No inference."""
import subprocess
from analyze import *

def read_config(p):
    return yaml.load(record(p,'historical_resolved_config').read_text(),Loader=yaml.CSafeLoader)

def read_args(p):
    return json.loads(record(p,'historical_train_args').read_text())

def agent_fields(a):
    keys=['checkpoint_path','reference_policy_checkpoint','vlm_path','lr','grpo','stage3_algorithm',
          'allow_random_init','stage2_target_source','stage2_scheduler_epochs','stage2_scheduler_warmup_epochs',
          'stage2_scheduler_min_lr','freeze_base_action_head','cache_hidden_state','use_planning_token_adapter',
          'use_fs_norm','fs_norm_stats_path','fs_norm_output_clip_mode','fs_norm_target_clip','use_trajectory_anchor',
          'trajectory_aux_weight','feasibility_aux_weight','diffusion_loss_weight','metric_cache_path',
          'bc_coeff','bc_coeff_start','bc_coeff_end','bc_anneal','bc_anneal_epochs','reference_kl_coeff',
          'grpo_sample_time','grpo_min_sampling_denoising_std','grpo_min_logprob_denoising_std',
          'grpo_gamma_denoising','grpo_randn_clip_value','grpo_reward_mode','grpo_scheduler_epochs',
          'grpo_scheduler_min_lr','grpo_scheduler_warmup_epochs','lfp_grpo_cfg','support_index_path']
    return {k:v for k,v in a.items() if k in keys or k.startswith('offline_rl_dpsi_') or k.startswith('stage2_support')}

def audits():
    configs={};weights=[];scale=[]
    orig=BACK/'outputs/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z/navsim_exp/training_recogdrive_agent/stage3_rl_2b_official_chunkcache_retry_20260608T225423Z/code/hydra/config.yaml'
    sources={'official_original':orig,'a5_lfp':A5/'hydra/training_recogdrive_agent'/A5.name/'code/hydra/config.yaml',
             'v6_lfp':V6/'hydra/training_recogdrive_agent'/V6.name/'code/hydra/config.yaml'}
    for name,p in sources.items():
        d=read_config(p);configs[name]={'path':p,'agent':agent_fields(d['agent']),
                                      'trainer':d.get('trainer'),'dataloader':d.get('dataloader')}
    archive=BACK/'official_recogdrive/navsim/agents/recogdrive'
    original_source={str(record(archive/f,'historical_original_GRPO_archived_source')):sha(archive/f)
        for f in ['recogdrive_diffusion_planner.py','recogdrive_agent.py']}
    save_json('manifests/original_archived_code_defaults.json',{'source_files':original_source,
        'group_size_default':8,'BC_coefficient_default':0.1,'sampling_floor_default':0.04,
        'logprob_floor_default':0.1,'scheduler_epochs':10,'scheduler_min_lr':0,'warmup_epochs':0,
        'note':'These values are read from the historical archived implementation, not absent Hydra fields.'})
    for name,run in [('psi_sft',PS),('psi_sr',PR),('a5_sft',OLD/'outputs/stage2_pta_fs_dit_a5_full103k_standardv2_clean_20260710T091344Z'),
                     ('v6_sft',OLD/'outputs/stage2_pta_fs_dit_v6_full103k_from_random_20260713T1706Z')]:
        p=run/'train_args.json';d=read_args(p)
        configs[name]={'path':p,'agent':agent_fields(d['agent']),
                      'trainer':d.get('trainer'),'dataloader':d.get('dataloader')}
    for name in ['a5_lfp','v6_lfp']:
        a=configs[name]['agent'];p=Path(a['checkpoint_path'])
        assert a['checkpoint_path']==a['reference_policy_checkpoint']
        weights.append(dict(run=name,role='SFT_parent_and_GRPO_reference',path=str(p),exists=p.is_file(),
                            sha256=sha(p) if p.is_file() else None))
        p=Path(a['fs_norm_stats_path']);s=np.load(record(p,'historical_FS_statistics'))
        rms=np.sqrt(np.sum(s['std']**2,axis=0))
        for j,c in enumerate(['x_m','y_m','heading_rad']):
            scale.append(dict(run=name,coordinate=c,normalized_unit_endpoint_rms=float(rms[j]),
                              floor=.04,single_transition_endpoint_rms_at_floor=float(rms[j]*.04)))
    for j,c in enumerate(['x_m','y_m','heading_rad']):
        scale.append(dict(run='legacy_absolute',coordinate=c,normalized_unit_endpoint_rms=[33.37,21,1.765][j],
                          floor=.04,single_transition_endpoint_rms_at_floor=[33.37,21,1.765][j]*.04))
    save_csv('normalization_scale.csv',scale)
    save_json('manifests/training_configs.json',configs)
    save_json('manifests/parent_checkpoint_identity.json',weights)
    # Reconstruct relevant source from the saved commit AND uncommitted patch.
    patch=record(V6/'source_diff.patch','historical_uncommitted_code_patch').read_text()
    headers=re.findall(r'^diff --git a/(.*?) b/',patch,flags=re.M)
    snapshot=OUT/'code_snapshot';snapshot.mkdir(exist_ok=True)
    codes=[]
    for name,run,commitfile in [('a5',A5,'git_commit.txt'),('v6',V6,'source_commit.txt')]:
        commit=record(run/commitfile,'historical_source_commit').read_text().strip()
        for fname in ['recogdrive_diffusion_planner.py','stage3_lfp_grpo.py','stage3_metric_adapter.py','recogdrive_agent.py','fs_norm.py']:
            rel='navsim/agents/recogdrive/'+fname
            data=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
            p=snapshot/(name+'_'+fname);p.write_bytes(data)
            codes.append(dict(run=name,commit=commit,path=rel,sha256=sha(p),
                              patched_in_saved_diff=(rel in headers) if name=='v6' else 'not_proven_byte_exact'))
    save_json('manifests/code_provenance.json',{'files':codes,'v6_patch_files':headers,
        'note':'A5 commit alone cannot prove absence of uncommitted runtime edits. PSI recoverable source caveat remains.'})
    import ast
    tree=snapshot/'v6_patched';tree.mkdir(exist_ok=True)
    names=['recogdrive_diffusion_planner.py','stage3_lfp_grpo.py','stage3_metric_adapter.py','recogdrive_agent.py','fs_norm.py']
    for fname in names:
        dest=tree/'navsim/agents/recogdrive'/fname;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((snapshot/('v6_'+fname)).read_bytes())
    blocks=re.split(r'(?=^diff --git )',patch,flags=re.M)
    selected=''.join(b for b in blocks if any(b.startswith('diff --git a/navsim/agents/recogdrive/'+n+' ') for n in names))
    r=subprocess.run(['patch','-p1','--forward','--batch'],cwd=tree,input=selected,text=True,capture_output=True)
    assert r.returncode==0,r.stdout+r.stderr
    check={}
    for fname,func in [('recogdrive_diffusion_planner.py','forward_lfp_grpo'),('stage3_lfp_grpo.py','compute_lfp_advantages')]:
        def node_hash(p):
            node=next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name==func)
            return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        before=node_hash(snapshot/('v6_'+fname));after=node_hash(tree/'navsim/agents/recogdrive'/fname)
        check[func]={'AST_hash_before':before,'AST_hash_after':after,'unchanged':before==after}
    save_json('manifests/v6_patch_reconstruction.json',{'patch_sha256':sha(V6/'source_diff.patch'),
        'patch_applied':True,'active_loss_and_credit':check,
        'patched_files':{n:sha(tree/'navsim/agents/recogdrive'/n) for n in names}})
    record(V6/'TERMINATED_BY_REVIEW.md','historical_formal_stop_record')
    return configs

def available_weights():
    import torch
    rows=[]
    for name,run,wanted in [('a5',A5,[300,4842,8100]),('v6',V6,[300,2100,3300])]:
        paths=sorted((run/'hydra/training_recogdrive_agent'/run.name).rglob('*.ckpt'))
        for p in paths:
            match=re.search(r'step=(\d+)',p.name)
            if not match:continue
            step=int(match.group(1));row=dict(run=name,path=str(p),step_from_name=step,bytes=p.stat().st_size)
            if step in wanted:
                state=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
                row.update(sha256=sha(p),epoch=state.get('epoch'),global_step=state.get('global_step'),
                           state_dict_key_count=len(state['state_dict']))
                assert row['global_step']==step
                del state
            rows.append(row)
    save_csv('available_historical_grpo_checkpoints.csv',rows)

def wrong_vlm():
    run=ROOT/'outputs/sft_only_navsim_stage3_original_grpo_from_stage2_epoch200_20260708T014559Z'
    manifest=read_args(run/'manifest.json');rows=[]
    train=read_config(next((run/'train/hydra').glob('*/*/code/hydra/config.yaml')))
    for kind,dirname in [('wrong_vlm','navtest_eval_local_pair8'),('correct_vlm','navtest_eval_local_pair8_correct_vlm_epoch6_9')]:
        for epoch in range(6,10):
            p=next((run/dirname).glob(f'epoch_{epoch}-*/*/aggregate_summary.json'))
            d=json.loads(record(p,'historical_VLM_identity_evaluation').read_text())
            c=next(p.parent.glob('shard_0/hydra/*/*/code/hydra/config.yaml'));a=read_config(c)['agent']
            scores=csv_scores(remap(d['output_csv']))
            assert abs(scores.PDMS.mean()-d['mean_score']*100)<1e-7
            rows.append(dict(epoch=epoch,evaluation=kind,PDMS=scores.PDMS.mean(),n_valid=len(scores),
                checkpoint_path=a.get('checkpoint_path'),vlm_path=a.get('vlm_path'),
                cache_hidden_state=a.get('cache_hidden_state'),train_vlm_path=train['agent'].get('vlm_path'),
                summary_path=str(p),config_path=str(c)))
    save_csv('wrong_vlm_false_collapse.csv',rows)
    save_json('manifests/wrong_vlm_run_identity.json',manifest)

def historical_control():
    root=OLD/'outputs/stage3_core_pareto_v2_historical_seed0_paired_20260712/v1'
    names=[('ReCogDrive_Diffusion_Planner_2B_IL.ckpt',0),('step-step_300.ckpt',300),
           ('epoch_0-step_1330.ckpt',1330),('epoch_1-step_2660.ckpt',2660),('epoch_2-step_3990.ckpt',3990)]
    base=prediction_bank(root/names[0][0]);curves=[];comparisons=[];tails=[]
    for name,step in names:
        b=base if step==0 else prediction_bank(root/name)
        curves.append(curve_row('official_core_pareto',step,b,'historical_seed0_common_noise',name))
        if step:
            t,z=compare(base,b,'official_core_pareto_'+str(step),base.scene_token,'CRN_seed0_historical_long_run')
            comparisons+=t;tails.append(z)
    save_csv('official_core_pareto_historical_control.csv',curves)
    save_csv('official_core_pareto_paired.csv',comparisons)
    save_csv('official_core_pareto_tail.csv',tails)

if __name__=='__main__':
    audits();available_weights();wrong_vlm();historical_control()
    old=json.loads((OUT/'manifests/inputs.json').read_text());old.update(SOURCE_FILES)
    save_json('manifests/inputs.json',old)
    print('DONE supplementary historical audit',flush=True)
