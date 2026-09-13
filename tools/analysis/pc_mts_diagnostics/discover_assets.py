"""Audit actual checkpoints and read-only historical code/data assets."""
import json, subprocess
from pathlib import Path
import yaml
from common import *

def main():
    old={m['id']:m for m in read(OLD/'models.json')}
    keys=['official_stage2_il','candidate_sft_v6_epoch165','candidate_sft_a5_epoch155','local_original_grpo_epoch8','final_apr_9145']
    rows=[]
    for name,key,label in zip(MODELS,keys,LABELS):
        m=old[key];assert Path(m['path']).is_file() and sha(m['path'])==m['sha256']
        root=Path(m['code_root'])
        try:commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        except subprocess.CalledProcessError:commit='archived source; file hashes recorded'
        category='gt_sft' if name=='official_il' else ('multi_sft' if name.startswith('mts') else 'rl_optimized')
        rows.append(dict(name=name,label=label,category=category,checkpoint_path=m['path'],sha256=m['sha256'],config_path=m.get('training_config_path'),reported_pdms=m['pdms_points'],training_method={'gt_sft':'released GT-only IL','multi_sft':'multi-trajectory supervision; no mapping to reconstructed M1-M4','rl_optimized':'original GRPO' if name.startswith('grpo') else 'APR / residual-merged policy'}[category],git_commit=commit,metadata_source=m.get('training_config_path',m['evidence']),notes='Native learned normalization and adapter retained; common output is XY metres, heading radians.',code_root=str(root),config=m['config']))
    for dest in [ROOT/'configs/pc_mts_diagnostics/checkpoints.yaml',OUT/'manifests/checkpoint_manifest.yaml']:
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(yaml.safe_dump(rows,sort_keys=False,allow_unicode=True))
    assets=[CODE/'navsim/evaluate/pdm_score_batch.py',CODE/'navsim/agents/recogdrive/pareto_support/pareto_archive.py',CODE/'navsim/agents/recogdrive/pareto_support/operators.py',ARCHIVE/'official_recogdrive/navsim/agents/recogdrive/recogdrive_diffusion_planner.py']
    for m in rows:
        assets+=[Path(m['code_root'])/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py']
        if m['config'].get('fs_norm_stats_path'):assets.append(Path(m['config']['fs_norm_stats_path']))
    historical=Path('/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_v6_full_training_20260713T1648Z')
    example=next((historical/'support_v6').glob('*.pkl.xz'))
    save(OUT/'manifests/assets.json',dict(files=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(assets)],historical_candidates=dict(found=True,root=str(historical/'support_v6'),example=str(example),example_sha256=sha(example),validation=read(historical/'validation.json'),use='audit only; reconstructed common reservoir is generated independently'),metric_cache=str(METRIC_CACHE),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('Audited five checkpoint SHA256 values and historical candidate/code assets',flush=True)

if __name__=='__main__':main()
