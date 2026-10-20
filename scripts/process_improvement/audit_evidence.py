"""Pin evidence, public reference commits, source hashes, and compact populations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import gzip
import subprocess
from scripts.process_improvement.common import *


def main():
    cfg=config();refs=ROOT/'references'
    inspected={
        'DiffusionOPD':['scripts/train_sd3_opd.py'],
        'VIDD':['finetune_reward.py','diffusion_gosai.py','losses/rl_loss.py'],
        'SVDD-image':['sd_pipeline.py','inference_decoding_nonp.py','diffusers_patch/ddim_with_kl.py'],
        'dppo':['model/diffusion/diffusion_vpg.py','model/diffusion/diffusion_ppo.py','model/diffusion/eta.py']}
    urls=dict(DiffusionOPD='https://github.com/ali-vilab/DiffusionOPD',VIDD='https://github.com/divelab/VIDD',
              **{'SVDD-image':'https://github.com/masa-ue/SVDD-image','dppo':'https://github.com/irom-princeton/dppo'})
    records=[]
    for name,files in inspected.items():
        p=refs/name
        records.append(dict(name=name,url=urls[name],commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
              license='Apache-2.0' if name=='DiffusionOPD' else 'MIT' if name=='dppo' else 'No license file found; no source code copied',
              inspected_files={f:sha(p/f) for f in files},reuse='mathematical_adaptation_only_no_third_party_runtime_import'))
    papers=[('DiffusionOPD','2605.15055v1'),('VIDD','2507.00445v3'),('SVDD','2408.08252v5'),('MPO','1806.06920v1'),('DPPO','2409.00588v3')]
    import requests
    for name,arxiv in papers:
        path=refs/(name+'.html');url='https://arxiv.org/html/'+arxiv
        if not path.exists():
            r=requests.get(url,timeout=45);r.raise_for_status();path.write_bytes(r.content)
        records.append(dict(name=name+'_paper',url=url,local_html_sha256=sha(path),role='primary_method_reference'))
    write(SOURCE/'configs/process_improvement/references.lock.json',records)
    checkpoints={}
    for name,path in [('IL',cfg['student_init']),('RL',cfg['teacher_init']),('VLM',str(Path(cfg['vlm'])/'model.safetensors'))]:
        checkpoints[name]=dict(path=path,bytes=Path(path).stat().st_size,sha256=sha(path))
    write(ROOT/'audit/checkpoints.json',checkpoints)
    expected=dict(IL='4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443',
                  RL='23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20',VLM=cfg['vlm_sha256'])
    assert all(checkpoints[k]['sha256']==v for k,v in expected.items())
    freeze=subprocess.check_output([PYTHON,'-m','pip','freeze'],text=True)
    (REPORTS/'environment.lock.txt').write_text(freeze)
    index=SOURCE/'configs/process_improvement/splits';index.mkdir(parents=True,exist_ok=True)
    for path in (ROOT/'manifests').glob('*.json'):
        # Deterministic gzip is a small, versioned index, not a feature cache.
        target=index/(path.name+'.gz')
        target.write_bytes(gzip.compress(path.read_bytes(),compresslevel=9,mtime=0))
    write(REPORTS/'tables/population_summary.json',read(ROOT/'manifests/identity.json'))
    write(REPORTS/'tables/checkpoint_identity.json',checkpoints)
    write(REPORTS/'tables/numerical_parity.json',read(ROOT/'tests/parity.json'))
    write(REPORTS/'tables/batch_profile.json',read(ROOT/'tests/batch_profile.json'))
    write(REPORTS/'tables/distributed_resume.json',read(ROOT/'tests/distributed_resume.json'))
    write(REPORTS/'tables/protocol_lock.json',read(ROOT/'protocol.lock.json'))
    provenance={str(p.relative_to(SOURCE)):sha(p) for folder in ['navsim/agents/recogdrive/opd','navsim/agents/recogdrive/process_improvement','scripts/opd','scripts/process_improvement']
                for p in (SOURCE/folder).glob('*.py')}
    for name in ['navsim/evaluate/pdm_score.py','navsim/agents/recogdrive/recogdrive_diffusion_planner.py','navsim/agents/recogdrive/recogdrive_dit.py',
                 'navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py','navsim/planning/metric_caching/metric_cache_processor.py']:
        provenance[name]=sha(SOURCE/name)
    write(ROOT/'audit/source_lock.json',dict(base_commit=BASE,files=provenance))
    print('AUDIT_EVIDENCE_PINNED',checkpoints,flush=True)


if __name__=='__main__':main()
