"""Small reviewable tables; raw data, weights and full logs stay outside git."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,shutil,importlib.metadata
from scripts.opd.common import *
from scripts.opd.analyze import bank,paired_quantities


def csvwrite(path,records):
    path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in records for k in r))
    with open(path,'w') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(records)


def main():
    dest=REPORTS/'tables';dest.mkdir(parents=True,exist_ok=True)
    splits=[]
    for split in ['train','cal','dev','navtest']:
        for r in rows(split):splits.append(dict(split=split,token=r['token'],log_name=r['log_name'],parent_driving_record=r.get('driver_record',r['log_name'].rsplit('_',2)[0]),capture_timestamp=r['log_name'].split('_')[0]))
    csvwrite(dest/'split_membership.csv',splits)
    g=read(ROOT/'gates/static.json');csvwrite(dest/'train_static_gate.csv',[dict(token=k,**v) for k,v in g['scenes'].items()])
    checkpoints=[]
    for p in sorted((ROOT/'runs').glob('*/complete.json')):
        r=read(p);cfg=r['config'];checkpoints.append(dict(run=p.parent.name,experiment=cfg['experiment'],stage=cfg['stage'],seed=cfg['seed'],updates=r['optimizer_updates'],
            checkpoint=str(p.parent/'latest.pt'),checkpoint_sha256=r['checkpoint_sha256'],seconds=r['seconds'],peak_GPU_GiB=r['peak_gpu_bytes']/2**30,initial_student_sha256=r.get('initial_student_sha256'),
            source_code_sha=r['owner']['code_sha'],config_sha256=cfg.get('config_sha256')))
    csvwrite(dest/'checkpoint_index.csv',checkpoints)
    seeds=[]
    for split in ['dev','navtest']:
        p=ROOT/'analysis'/f'formal_{split}.json'
        if not p.exists():continue
        data=read(p);base=bank(split,'E0','IL');stable=(bank(split,'E0','IL_screen')[...,-1]>=.8).all(-1)
        base_mean=base[...,-1].mean(-1);teacher=bank(split,'E0','RL');teacher_mean=teacher[...,-1].mean(-1);failures={}
        for e in data['methods']:
            failures[e]={}
            for s in config()['train_seeds']:
                v=bank(split,f'train_{e}_{s}');j=v[...,-1].mean(-1)
                counts=dict(stable_high_to_low_count=int((stable&(j<.8)).sum()),stable_high_to_zero_count=int((stable&(j<=1e-12)).sum()),
                    paired_stable_high_to_low_count=int((stable&(base_mean>=.8)&(j<.8)).sum()),
                    paired_stable_high_material_drop_count=int((stable&(j-base_mean<-.02)).sum()),
                    baseline_stable_group_low_count=int((stable&(base_mean<.8)).sum()),baseline_stable_group_zero_count=int((stable&(base_mean<=1e-12)).sum()),
                    positive_to_zero_count=int(((base_mean>1e-12)&(j<=1e-12)).sum()),zero_to_positive_count=int(((base_mean<=1e-12)&(j>1e-12)).sum()),
                    zero_output_count=int((v[...,-1]<=1e-12).sum()),teacher_positive_student_zero_count=int(((teacher_mean>1e-12)&(j<=1e-12)).sum()),
                    stable_high_zero_output_count=int((stable[:,None]&(v[...,-1]<=1e-12)).sum()),
                    paired_positive_output_to_zero_count=int(((base[...,-1]>1e-12)&(v[...,-1]<=1e-12)).sum()),
                    teacher_zero_student_positive_count=int(((teacher_mean<=1e-12)&(j>1e-12)).sum()),
                    both_zero_student_positive_count=int(((base_mean<=1e-12)&(teacher_mean<=1e-12)&(j>1e-12)).sum()),
                    scene_zero_count=int((j<=1e-12).sum()),scene_low_positive_through_point6_count=int(((j>1e-12)&(j<=.6)).sum()),
                    scene_high_count=int((j>=.8).sum()),median_scene_PDMS=float(np.median(j)),median_scene_PDMS_difference=float(np.median(j-base_mean)))
                for component,index in [('NC',0),('DAC',1),('TTC',3)]:
                    counts['new_'+component+'_failed_output_count']=int(((base[...,index]>1e-12)&(v[...,index]<=1e-12)).sum())
                    counts['new_'+component+'_violation_output_count']=int(((base[...,index]>=1-1e-6)&(v[...,index]<1-1e-6)).sum())
                    counts[component+'_worsened_output_count']=int((v[...,index]<base[...,index]-1e-6).sum())
                seeds.append(dict(split=split,experiment=e,training_seed=s,**paired_quantities(v[None],base,stable),**counts))
                failures[e][str(s)]=dict(**counts,stable_high_to_zero_tokens=[r['token'] for r,m in zip(rows(split),stable&(j<=1e-12)) if m])
        write(ROOT/'analysis'/f'failure_counts_{split}.json',dict(scope='Counts per trained policy, not independent scene repeats; zero scene means five-draw mean<=1e-12; output counters refer to single paired draws. New component violation means baseline≈1 and new<1, including fractional NC penalties; worsened includes further partial degradation.',methods=failures))
        shutil.copyfile(ROOT/'analysis'/f'failure_counts_{split}.json',dest/f'failure_counts_{split}.json')
    if seeds:csvwrite(dest/'individual_training_seed_results.csv',seeds)
    for path in ['audit/final_provenance.json','audit/official_model_sources.json','audit/VLM_ancillary_release_comparison.json','audit/official_forward_parity.json','audit/correction_registration.json','audit/IL_RL_tensor_comparison.json','audit/RL_reference_comparison.json','audit/formal_model_certificate.json','audit/evaluation_certificate_dev.json','audit/evaluation_certificate_navtest.json','audit/hardware_hosts.json','tests/final_matrix.json','tests/real_parallel_inference.json','tests/real_new_host.json','tests/real_grpo_resume.json','resources/navtest_routing_lock.json',
                 'selection/tuning.json','selection/navtest.json','selection/E6_admission.json','selection/E6_budget.json','calibration/gradient_scales.json',
                 'diagnostics/chain_distributions.json','diagnostics/policy_refresh_probes.json','diagnostics/gradients.json','diagnostics/gate_results.json',
                 'analysis/secondary_anchor_vs_GT_dev.json','analysis/teacher_comparisons_dev.json','analysis/teacher_comparisons_navtest.json','analysis/reporting_scope_registration.json',
                 'tests/real_split_bundle.json','tests/real_fresh_manifest.json','tests/real_statistics_projection.json','audit/online_inference_cost.json','audit/resource_summary.json',
                 'audit/metric_cache_version.json','analysis/case_rule_registration.json','calibration/registration.json','protocol.lock.json',
                 'analysis/parent_record_registration_dev.json','analysis/parent_record_registration_navtest.json',
                 'analysis/parent_record_sensitivity_dev.json','analysis/parent_record_sensitivity_navtest.json']:
        p=ROOT/path
        if p.exists():shutil.copyfile(p,dest/(path.replace('/','_')))
    versions=sorted({f"{d.metadata.get('Name','UNKNOWN')}=={d.version}" for d in importlib.metadata.distributions()})
    (REPORTS/'requirements.lock.txt').write_text('\n'.join(versions)+'\n')
    write(dest/'artifact_index.json',dict(external_root=str(ROOT),protocol_sha256=sha(SOURCE/'configs/opd/protocol.yaml'),
        structured_evaluation='evaluation/<split>/<run>/{predictions,scores,complete_*}',actual_commands='jobs/*/spec.json',
        model_source_provenance='jobs/*/owner.json:source_hashes and runs/*/config.json',
        cache='observations/<split>/<token>.pt: independently rebuilt pinned official fixed2800 conditioning; no teacher-forced future tokens',
        invalid_previous_attempt=dict(root=str(ROOT.parent),status_file=str(ROOT.parent/'INVALID.json'),reports=str(ROOT.parent/'reports_invalid_snapshot'),protocol_file='configs/opd/protocol_v1_invalid.yaml',not_pooled=True),
        raw_artifacts_excluded_from_git=['Model weights','Original sensor images and logs','Raw VLM tokens','Complete optimizer states','Large training logs'],
        git_files={str(p.relative_to(REPORTS)):sha(p) for p in dest.glob('*') if p.name!='artifact_index.json'}))


if __name__=='__main__':main()
