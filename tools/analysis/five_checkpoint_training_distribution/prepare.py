"""Freeze an exploratory follow-up and record immutable input provenance."""
import datetime, subprocess
from common_fd import *

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifests/protocol_frozen.json'
    if path.exists():
        config_identity()
        return
    examples=[]
    for cmd in sorted({s['command'] for s in SCENES}):
        examples += [s['token'] for s in sorted([s for s in SCENES if s['command']==cmd],key=lambda s:digest([CFG['seed'],s['token']]))[:2]]
    tracked=subprocess.check_output(['git','ls-files','-z','outputs/pc_mts_diagnostics','outputs/pc_mts_diagnostics_v2','outputs/pc_mts_diagnostics_v3','outputs/pcmts_grpo_mass_audit','reports'],cwd=ROOT).decode().split('\0')
    protected={p:sha(ROOT/p) for p in tracked if p and (ROOT/p).is_file()}
    save(OUT/'manifests/protected_files.json',protected)
    models=[]
    for model, m in MODEL_INFO.items():
        actual=sha(m['checkpoint_path'])
        assert actual==m['sha256']
        runtime=Path(m['code_root'])/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'
        row=dict(m,verified_checkpoint_sha256=actual,runtime_code_path=str(runtime),runtime_code_sha256=sha(runtime))
        if model in ARCHIVES:
            cfg=read(m['config_path'])
            row.update(training_agent=cfg['agent'],training_seed=cfg.get('seed'),archive=str(ARCHIVES[model]),train_args_sha256=sha(m['config_path']))
        models.append(row)
    save(OUT/'manifests/models_and_training.json',models)
    save(OUT/'manifests/scenes_1000.json',v1.manifest())
    save(path,dict(config_sha256=sha(CONFIG),timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        scene_manifest_sha256=sha(ROOT/CFG['scene_manifest']),scene_digest=digest(v1.manifest()),
        scene_count=len(SCENES),examples=examples,native_teacher_helper_path=str(NATIVE),native_teacher_helper_sha256=sha(NATIVE),
        status='Exploratory follow-up: V1 outcomes already known; new analyses fixed before their execution',
        unchanged_weights=True,optimizer_updates=0,protected_file_count=len(protected)))
    print('FROZEN',sha(CONFIG),len(SCENES),flush=True)

if __name__=='__main__':main()
