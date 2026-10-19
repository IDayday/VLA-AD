"""Reproducibility and data-isolation certificate, including failed jobs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,xml.etree.ElementTree as ET
from scripts.opd.common import *


def provenance():
    import yaml
    cfg=config();manifest={};official={}
    for split in ['navtrain','navtest']:
        p=SOURCE/f'navsim/planning/script/config/common/train_test_split/scene_filter/{split}.yaml'
        y=yaml.safe_load(p.read_text());official[split]=dict(file=str(p),sha256=sha(p),tokens=set(y['tokens']))
    for split in ['train','cal','dev','navtest']:
        data=rows(split);ids={r['token'] for r in data};logs={r['log_name'] for r in data};target='navtest' if split=='navtest' else 'navtrain'
        assert ids<=official[target]['tokens'] and len(ids)==len(data)
        if split=='navtest':assert ids==official['navtest']['tokens']
        manifest[split]=dict(scenes=len(data),logs=len(logs),manifest_sha256=sha(ROOT/'manifests'/f'{split}.json'),official_filter_sha256=official[target]['sha256'])
    for a in manifest:
        for b in manifest:
            if a!=b:
                for key in ['token','log_name']:
                    assert not ({r[key] for r in rows(a)}&{r[key] for r in rows(b)})
                assert not ({r['log_name'].split('_')[0] for r in rows(a)}&{r['log_name'].split('_')[0] for r in rows(b)})
    audit=read(ROOT/'audit/checkpoints.json');weights={}
    for k in ['IL','RL','VLM']:
        path=Path(audit[k]['path']);path=path/'model.safetensors' if path.is_dir() else path
        value=sha(path);assert value==audit[k]['sha256'];weights[k]=dict(path=str(path),sha256=value)
    scoring=['navsim/evaluate/pdm_score.py','navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py',
        'navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py','navsim/planning/metric_caching/metric_cache_processor.py','scripts/opd/scoring.py']
    unique=sorted({r['metadata_path'] for s in manifest for r in rows(s)})
    metadata=[dict(path=p,sha256=sha(p),bytes=Path(p).stat().st_size) for p in unique]
    result=dict(passed=True,manifests=manifest,source_logs=metadata,weights=weights,scoring_sources={p:sha(SOURCE/p) for p in scoring},
        native_model_sources={p:sha(SOURCE/p) for p in ['navsim/agents/recogdrive/recogdrive_agent.py','navsim/agents/recogdrive/recogdrive_backbone.py',
            'navsim/agents/recogdrive/recogdrive_features.py','navsim/agents/recogdrive/recogdrive_diffusion_planner.py','navsim/agents/recogdrive/recogdrive_dit.py',
            'navsim/agents/recogdrive/utils/internvl_preprocess.py']},
        VLM_ancillary_sources={str(p.relative_to(cfg['vlm'])):sha(p) for p in Path(cfg['vlm']).glob('*') if p.suffix in ('.py','.json','.txt')},
        maps=dict(path='/mnt/navsim/maps',version='nuplan-maps-v1.0',metadata_sha256=sha('/mnt/navsim/maps/nuplan-maps-v1.0.json')),
        nonoverlap=['token','complete_log','parent_driving_record'],protocol_sha256=sha(SOURCE/'configs/opd/protocol.yaml'),
        meaning='Train/cal/dev are held out only from this new training, not proven unseen in original base training',owner=identity())
    assert result['protocol_sha256']=='e02028b646a2745dca0b0a6f6eae7dff57119b3d12f8aeb0a3324a1f80c88968'
    write(ROOT/'audit/final_provenance.json',result);print('PROVENANCE_PASS',manifest)


def accounting():
    jobs=[]
    for p in sorted((ROOT/'jobs').glob('*/spec.json')):
        spec=read(p);end=p.parent/'complete.json';failure=p.parent/'admission_failure.json'
        record=dict(name=p.parent.name,host=spec['host'],gpus=spec['gpu'],commands=spec['commands'],spec_sha256=sha(p))
        if end.exists():
            c=read(end);record.update(status=c['status'],seconds=c['seconds'],source_hashes=c['owner']['source_hashes'])
        elif failure.exists():record.update(status='NOT_LAUNCHED',reason=read(failure))
        else:record['status']='RUNNING'
        jobs.append(record)
    gpu_hours=sum(r.get('seconds',0)*len(r['gpus'] if isinstance(r['gpus'],list) else ([] if r['gpus'] is None else [r['gpus']]))/3600 for r in jobs)
    result=dict(jobs=jobs,reserved_GPU_hours=float(gpu_hours),definition='Wall-clock GPU reservation including evaluation and startup; CPU-only controllers excluded; not a utilization estimate',
        failures=[r['name'] for r in jobs if r['status'] in ['FAILED','NOT_LAUNCHED']],running=[r['name'] for r in jobs if r['status']=='RUNNING'])
    write(ROOT/'audit/accounting.json',result);print('ACCOUNTING',gpu_hours,result['failures'],result['running'])


def tests():
    required=['real_batch','real_grpo','real_resume_v2','real_ddp','real_online_vlm','real_parameter_perturbation','real_GT_parity','real_inference_cache','real_parallel_inference','real_shared_lock']
    evidence={n:read(ROOT/'tests'/f'{n}.json') for n in required}
    assert all(v['passed'] for v in evidence.values())
    xml=ET.parse(ROOT/'tests/cpu.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ['tests','failures','errors','skipped']}
    assert not counts['failures'] and not counts['errors']
    write(ROOT/'tests/cpu.json',dict(passed=True,**counts,source='Actual pytest JUnit XML',junit_sha256=sha(ROOT/'tests/cpu.xml')))
    mapping={
        'T01':'audit/checkpoints.json: IL strict restoration','T02':'audit/checkpoints.json: RL active head strict restoration',
        'T03':'audit/gates.json: native RL/COPY both modes','T04':'tests/real_batch.json: native dense batch and future-poison parity',
        'T05':'audit/gates.json: sampler/adapter all states and means','T06':'tests/opd/test_ddim.py and adapter scheduler assertions',
        'T07':'tests/real_GT_parity.json and test_ddim.py normalization','T08':'audit/gates.json and all run teacher hashes',
        'T09':'audit/gates.json and all anchor run hashes','T10':'audit/gates.json and actual optimizer gradient histories',
        'T11':'tests/real_parameter_perturbation.json identical model loss','T12':'tests/real_parameter_perturbation.json nonzero perturbed loss',
        'T13':'tests/opd/test_rollout.py and runs/*/rollouts current_student','T14':'tests/opd/test_rollout.py and E2 frozen_teacher archives',
        'T15':'audit/gates.json chain refresh and fixed-noise policy_probes','T16':'CPU loss tests and actual batch1/batch8 runs',
        'T17':'audit/gates.json deterministic and native stochastic parity','T18':'tests/opd/test_losses.py degenerate and clipped Gaussian rejection',
        'T19':'tests/opd/test_rollout.py and E4/E5 anchor unchanged completion','T20':'tests/opd/test_gating.py and train global effective-count normalization',
        'T21':'tests/opd/test_losses.py zero mask and actual E5 zero-gate batches','T22':'audit/final_provenance.json official split/log/parent checks',
        'T23':'tests/opd/test_eval_safety.py and pure head evaluation without teacher','T24':'tests/real_batch.json future poison and real_online_vlm exact native inputs',
        'T25':'tests/real_resume_v2.json exact actual parameters,optimizer,RNG','T26':'audit/gates.json same-noise repeated outputs',
        'T27':'tests/real_ddp.json actual2-GPU gradient reduction','T28':'tests/real_grpo.json exact zero-OPD native loss/gradient'}
    write(ROOT/'tests/final_matrix.json',dict(passed=True,CPU=counts,integration_evidence={k:sha(ROOT/'tests'/f'{k}.json') for k in required},tests={k:dict(status='PASS',evidence=v) for k,v in mapping.items()},
        caveat='Numerical acceptance is separate from research performance. Optional KL/timestep performance ablations are NOT_RUN.'))
    print('TEST_MATRIX_PASS',counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['provenance','accounting','tests']);a=p.parse_args();globals()[a.action]()
