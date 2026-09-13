"""Freeze all science choices before sampling; verify fixed scene identities."""
from common_mass import *
import shutil, subprocess, concurrent.futures

def check_scene(r):
    token=r['token'];f=V1/'features'/f'{token}.pt';m=Path(r['metric_cache_path'])
    log=Path('/mnt/navsim/trainval_navsim_logs/trainval')/(r['log']+'.pkl')
    return dict(token=token,log=r['log'],frame_start=r['frame_start'],feature_path=str(f),feature_sha256=sha(f) if f.exists() else None,metric_cache_path=str(m),metric_cache_sha256=sha(m) if m.exists() else None,log_exists=log.exists(),status='OK' if f.exists() and m.exists() and log.exists() else 'MISSING')

def main():
    dest=OUT/'manifests/protocol_frozen.json'
    if dest.exists():assert read(dest)['sha256']==protocol();return
    original=ROOT/cfg()['scene_manifest_source'];manifest=read(original)
    assert len(manifest['scenes'])==1000 and len({r['token'] for r in manifest['scenes']})==1000
    save(OUT/'manifests/scenes_1000.json',manifest)
    with concurrent.futures.ThreadPoolExecutor(16) as ex:checks=list(ex.map(check_scene,manifest['scenes']))
    save(OUT/'audits/scene_cache_identity.json',dict(source_path=str(original),source_sha256=sha(original),checks=checks))
    old={}
    # All tracked protected artifacts plus local cache content manifests.
    names=subprocess.check_output(['git','ls-files','outputs/pc_mts_diagnostics','outputs/pc_mts_diagnostics_v2','outputs/pc_mts_diagnostics_v3','reports'],cwd=ROOT,text=True).splitlines()
    for n in names:
        if (ROOT/n).is_file():old[n]=sha(ROOT/n)
    save(OUT/'audits/protected_artifact_hashes.json',old)
    save(dest,dict(sha256=protocol(),frozen_at=utc(),config=str(CONFIG),choices=cfg(),actual_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scene_manifest_sha256=sha(original),thresholds_are_prespecified_working_settings=True))
    print('FROZEN',protocol(),'scenes',len(checks),'missing',sum(x['status']!='OK' for x in checks),flush=True)
if __name__=='__main__':main()
