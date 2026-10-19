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
        for r in rows(split):splits.append(dict(split=split,token=r['token'],log_name=r['log_name'],parent_driving_record=r['log_name'].split('_')[0]))
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
        base_mean=base[...,-1].mean(-1);failures={}
        for e in data['methods']:
            failures[e]={}
            for s in config()['train_seeds']:
                v=bank(split,f'train_{e}_{s}');j=v[...,-1].mean(-1)
                counts=dict(stable_high_to_low_count=int((stable&(j<.8)).sum()),stable_high_to_zero_count=int((stable&(j<=1e-12)).sum()),
                    positive_to_zero_count=int(((base_mean>1e-12)&(j<=1e-12)).sum()),zero_to_positive_count=int(((base_mean<=1e-12)&(j>1e-12)).sum()),
                    zero_output_count=int((v[...,-1]<=1e-12).sum()))
                seeds.append(dict(split=split,experiment=e,training_seed=s,**paired_quantities(v[None],base,stable),**counts))
                failures[e][str(s)]=dict(**counts,stable_high_to_zero_tokens=[r['token'] for r,m in zip(rows(split),stable&(j<=1e-12)) if m])
        write(ROOT/'analysis'/f'failure_counts_{split}.json',dict(scope='Counts per trained policy, not independent scene repeats; zero means five-draw scene mean<=1e-12',methods=failures))
        shutil.copyfile(ROOT/'analysis'/f'failure_counts_{split}.json',dest/f'failure_counts_{split}.json')
    if seeds:csvwrite(dest/'individual_training_seed_results.csv',seeds)
    for path in ['audit/final_provenance.json','audit/IL_RL_tensor_comparison.json','audit/RL_reference_comparison.json','audit/formal_model_certificate.json','audit/evaluation_certificate_dev.json','audit/evaluation_certificate_navtest.json','audit/hardware_hosts.json','tests/final_matrix.json','tests/real_parallel_inference.json',
                 'selection/tuning.json','selection/navtest.json','selection/E6_admission.json','selection/E6_budget.json','calibration/gradient_scales.json',
                 'diagnostics/chain_distributions.json','diagnostics/gradients.json','diagnostics/gate_results.json']:
        p=ROOT/path
        if p.exists():shutil.copyfile(p,dest/(path.replace('/','_')))
    versions=sorted({f"{d.metadata.get('Name','UNKNOWN')}=={d.version}" for d in importlib.metadata.distributions()})
    (REPORTS/'requirements.lock.txt').write_text('\n'.join(versions)+'\n')
    write(dest/'artifact_index.json',dict(external_root=str(ROOT),protocol_sha256=sha(SOURCE/'configs/opd/protocol.yaml'),
        structured_evaluation='evaluation/<split>/<run>/{predictions,scores,complete_*}',actual_commands='jobs/*/spec.json',
        model_source_provenance='jobs/*/owner.json:source_hashes and runs/*/config.json',
        cache='observations/navtest and previously bound frozen VLM cache; no teacher-forced future tokens',
        raw_artifacts_excluded_from_git=['Model weights','Original sensor images and logs','Raw VLM tokens','Complete optimizer states','Large training logs'],
        git_files={str(p.relative_to(REPORTS)):sha(p) for p in dest.glob('*') if p.name!='artifact_index.json'}))


if __name__=='__main__':main()
