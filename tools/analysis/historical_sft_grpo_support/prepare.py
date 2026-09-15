"""Freeze historical model identities, observations, splits and exact source trees."""
from common_support import *
import io,tarfile,subprocess,re,copy,shutil

def archive(name,commit,patch=None):
    dest=OUT/'runtime'/name;dest.mkdir(parents=True,exist_ok=True)
    marker=dest/'identity.json'
    ident={'commit':commit,'patch_sha256':sha(patch) if patch else None}
    if marker.exists():assert read(marker)==ident;return dest
    data=subprocess.check_output(['git','archive',commit,'navsim'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for member in tf.getmembers():
            assert not member.name.startswith('/') and '..' not in Path(member.name).parts
        tf.extractall(dest)
    if patch:
        blocks=re.split(r'(?=^diff --git )',Path(patch).read_text(),flags=re.M)
        selected=''.join(b for b in blocks if b.startswith('diff --git a/navsim/'))
        r=subprocess.run(['patch','-p1','--forward','--batch'],cwd=dest,input=selected,text=True,capture_output=True)
        assert r.returncode==0,r.stdout+r.stderr
    save(marker,ident);return dest

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=v1.manifest();assert original['scene_count']==1000
    sources=[]
    for s in original['scenes']:
        f=v1.OUT/'features'/f"{s['token']}.pt"
        assert f.is_file() and Path(s['metric_cache_path']).is_file()
        sources.append(dict(token=s['token'],feature_path=str(f),feature_sha256=sha(f),metric_cache_sha256=sha(s['metric_cache_path'])))
    original['factorial_tokens']=[s['token'] for s in sorted(original['scenes'],key=lambda s:digest(['factorial',s['token']]))[:CFG['factorial_scene_count']]]
    save(OUT/'manifests/scenes.json',original);save(OUT/'manifests/observations.json',sources)
    a5run=OLD/'outputs/stage3_lfp_grpo_v1_exact_b8g16_epoch155_r4_20260712T083012Z'
    v6run=OLD/'outputs/stage3_lfp_grpo_v1_v6_epoch165_formal_20260715T145650Z'
    psi=ROOT/'outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z'
    psirun=ROOT/'outputs/psi_drive_stage3_sr_pgrpo_epoch011_b2acc4_20e_20260627T135941Z'
    runtimes={'a5':archive('a5','f350e56408584c0d375b1ff10744da2750442580'),
              'v6':archive('v6','7ced3afdfd2d45ab52b08066380b9651d8e18388',v6run/'source_diff.patch'),
              'psi':archive('psi','870d7167b17889dbdc89d1517f428cd8d8ee083d')}
    dependency=Path('/mnt/project/VLA-AD-worktrees/v6/navsim/agents/recogdrive/trajectory_anchors.py')
    destination=runtimes['v6']/'navsim/agents/recogdrive/trajectory_anchors.py'
    if not destination.exists():
        assert sha(dependency)=='3c8b132a7a756d18697b127a6d039ddfc454b6092899e11ec0a4cfd84f59e912'
        shutil.copy2(dependency,destination)
    assert sha(destination)=='3c8b132a7a756d18697b127a6d039ddfc454b6092899e11ec0a4cfd84f59e912'
    oldmodels={m['name']:m for m in v1.models()};out={}
    for name,base,family in [('official_il','official_il','original'),('a5_sft','mts_8751','a5'),('v6_sft','mts_8692','v6'),('original_grpo_11970','grpo_9041','original')]:
        m=copy.deepcopy(oldmodels[base]);m['name']=name;m['family']=family;m['v1_cache_name']=base
        if family in runtimes:m['code_root']=str(runtimes[family])
        if family in ['a5','v6']:
            run=a5run if family=='a5' else v6run
            c=run/'hydra/training_recogdrive_agent'/run.name/'code/hydra/config.yaml'
            m['grpo_config_path']=str(c);m['grpo_config']=yaml.load(c.read_text(),Loader=yaml.CSafeLoader)['agent']
        out[name]=m
    ck=psi/'checkpoint_store/objects/.051ac3312a848b36eeeaccc3935e03582f9acc89737a99672349a5887faa0aff.3941396.tmp'
    args=read(psi/'train_args.json')['agent']
    out['psi_sft']=dict(name='psi_sft',family='psi',checkpoint_path=str(ck),sha256=sha(ck),code_root=str(runtimes['psi']),
        config=args,config_path=str(psi/'train_args.json'),grpo_config=read(psirun/'train_args.json')['agent'],
        grpo_config_path=str(psirun/'train_args.json'),v1_cache_name=None)
    for family,run,steps in [('a5',a5run,[300,4842]),('v6',v6run,[300,3300])]:
        ckroot=run/'hydra/training_recogdrive_agent'/run.name
        for step in steps:
            paths=list(ckroot.rglob(f'*step={step}.ckpt'));assert len(paths)==1,paths
            name=f'{family}_grpo_{step}';m=copy.deepcopy(out[family+'_sft']);m.update(name=name,checkpoint_path=str(paths[0]),sha256=sha(paths[0]),v1_cache_name=None);out[name]=m
    for m in out.values():
        assert sha(m['checkpoint_path'])==m['sha256']
        code=Path(m['code_root'])/'navsim/agents/recogdrive/recogdrive_diffusion_planner.py'
        m['runtime_planner_path']=str(code);m['runtime_planner_sha256']=sha(code)
    save(OUT/'manifests/models.json',out)
    import torch
    support_path=ROOT/'outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt'
    supports=torch.load(support_path,map_location='cpu',weights_only=False)
    membership=[]
    for scene in original['scenes']:
        token=scene['token'];r=supports['token_to_row'][token]
        train=supports['support_sources'][r][0]!='gt_fallback'
        membership.append(dict(token=token,log=scene['log'],psi_train=train,a5_train=True,v6_train=True,
                               common_train=train))
    csv('training_membership.csv',membership)
    save(OUT/'manifests/PSI_support_identity.json',dict(path=str(support_path),sha256=sha(support_path),
         common_train_scenes=sum(r['common_train'] for r in membership),full1000=1000))
    frozen=dict(config_sha256=sha(CFG_PATH),scene_sha256=sha(OUT/'manifests/scenes.json'),models_sha256=sha(OUT/'manifests/models.json'),
                parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),timestamp=time.time(),
                no_new_training=True,source_reconstruction_caveat='PSI code is recoverable later commit; A5 runtime uncommitted changes not proven absent; V6 saved patch applied.')
    target=OUT/'manifests/protocol.json'
    if target.exists():assert read(target)['config_sha256']==frozen['config_sha256']
    else:save(target,frozen)
    print('PREPARED',len(out),'models; common training scenes',sum(r['common_train'] for r in membership),flush=True)

if __name__=='__main__':main()
