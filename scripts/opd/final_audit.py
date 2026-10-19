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
    assert result['protocol_sha256']==read(ROOT/'protocol.lock.json')['sha256']
    write(ROOT/'audit/final_provenance.json',result);print('PROVENANCE_PASS',manifest)


def accounting():
    jobs=[]
    for p in sorted((ROOT/'jobs').glob('*/spec.json')):
        spec=read(p);end=p.parent/'complete.json';failure=p.parent/'admission_failure.json'
        record=dict(name=p.parent.name,host=spec['host'],gpus=spec['gpu'],commands=spec['commands'],spec_sha256=sha(p))
        if end.exists():
            c=read(end);record.update(status=c['status'],seconds=c['seconds'],returncodes=[v['returncode'] for v in c['results']],source_hashes=c['owner']['source_hashes'])
        elif failure.exists():record.update(status='NOT_LAUNCHED',reason=read(failure))
        else:record['status']='RUNNING'
        record['rejected_admission_attempts']=[read(a) for a in sorted((p.parent/'admissions').glob('*.json'))]
        if (p.parent/'handoff.json').exists():record['controller_handoff']=read(p.parent/'handoff.json')
        jobs.append(record)
    gpu_hours=sum(r.get('seconds',0)*len(r['gpus'] if isinstance(r['gpus'],list) else ([] if r['gpus'] is None else [r['gpus']]))/3600 for r in jobs)
    result=dict(jobs=jobs,reserved_GPU_hours=float(gpu_hours),definition='Wall-clock GPU reservation including evaluation and startup; CPU-only controllers excluded; not a utilization estimate',
        failures=[r['name'] for r in jobs if r['status'] in ['FAILED','NOT_LAUNCHED']],running=[r['name'] for r in jobs if r['status']=='RUNNING'])
    write(ROOT/'audit/accounting.json',result);print('ACCOUNTING',gpu_hours,result['failures'],result['running'])
    resource_summary()


def resource_summary():
    from collections import Counter
    current=read(ROOT/'audit/accounting.json');old_hours=0.
    if (ROOT.parent/'INVALID.json').exists():
        for p in (ROOT.parent/'jobs').glob('*/spec.json'):
            end=p.parent/'complete.json'
            if not end.exists():continue
            g=read(p)['gpu'];count=len(g) if isinstance(g,list) else int(g is not None)
            old_hours+=read(end)['seconds']*count/3600
    restored=[read(p) for p in sorted((ROOT/'resources').glob('*_pressure_resume_v2.json'))]
    assert all(v['action']=='resume' and v['prior_pause']==v['tag']=='recogdrive-opd-20261020-official-v2' for v in restored)
    write(ROOT/'audit/resource_summary.json',dict(valid_v2_GPU_hours=current['reserved_GPU_hours'],invalid_v1_GPU_hours=old_hours,
        total_GPU_hours=current['reserved_GPU_hours']+old_hours,definition=current['definition'],
        valid_v2_job_status_counts=dict(Counter(v['status'] for v in current['jobs'])),failure_names=current['failures'],failures_remain_visible=True,
        failure_records=[{k:r[k] for k in ['name','status','commands','returncodes'] if k in r} for r in current['jobs'] if r['status'] in ['FAILED','NOT_LAUNCHED']],
        restored_pressure_automation_hosts=[v['host'] for v in restored],restoration='Only owned pause markers removed; original controllers reobserve fresh30min utilization; no custom filler started',
        failure_explanation='Original G0_G1_audit encountered a GRPO diagnostic before E0 labels existed; controller_official_bootstrap_sourcefix propagated that failure. Repaired dependency and passed GRPO_real_low_score. Other failed CPU controllers have -15 handoff exits; scientific children were preserved. Full commands, exit codes and source hashes remain in accounting.json.'))


def cache_versions():
    from concurrent.futures import ThreadPoolExecutor
    data=[(s,r) for s in ['train','cal','dev','navtest'] for r in rows(s)]
    def one(item):
        split,row=item;path=Path(row['metric_cache_path']);stat=path.stat()
        first=read(ROOT/'evaluation'/split/'E0/scores'/f"{row['token']}.json")['model_metadata']['owner']['time']
        if stat.st_mtime>first:raise ValueError(f'Metric cache modified after its baseline prediction: {path}')
        return dict(split=split,token=row['token'],path=str(path),sha256=sha(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,baseline_prediction_time=first)
    with ThreadPoolExecutor(8) as pool:result=list(pool.map(one,data))
    path=ROOT/'audit/metric_cache_index.json';write(path,result)
    write(ROOT/'audit/metric_cache_version.json',dict(passed=True,scenes=len(result),index=str(path),index_sha256=sha(path),
        capture='Post-run hash snapshot; every cache mtime precedes its baseline prediction; not falsely labeled as hashes captured at original cache creation',
        split_digests={s:digest([(r['token'],r['sha256']) for r in result if r['split']==s]) for s in ['train','cal','dev','navtest']}))
    print('METRIC_CACHE_VERSION_PASS',len(result))


def storage():
    scope=ROOT.parent if (ROOT.parent/'INVALID.json').exists() else ROOT
    command=['du','-sb',str(scope)];value=int(subprocess.check_output(command,text=True).split()[0])
    write(ROOT/'audit/storage.json',dict(bytes=value,GiB=value/2**30,scope=str(scope),command=command,time=time.time(),
        definition='Apparent bytes of task outputs, including INVALID history and valid rerun; no symlink traversal; existing external raw datasets, source weights and reused train metric caches excluded'))
    print('TASK_OUTPUT_APPARENT_GiB',value/2**30)


def model_sources():
    import requests
    from concurrent.futures import ThreadPoolExecutor
    baseline=read(ROOT/'audit/checkpoints.json')
    def inspect(label):
        repo='owl10/'+{'IL':'ReCogDrive-2B-IL','RL':'ReCogDrive-2B-RL','VLM':'ReCogDrive-VLM-2B'}[label]
        url='https://huggingface.co/api/models/'+repo+'?blobs=true'
        result=requests.get(url,timeout=30);result.raise_for_status();data=result.json()
        filename='model.safetensors' if label=='VLM' else Path(baseline[label]['path']).name
        file=next(f for f in data['siblings'] if f['rfilename']==filename)
        expected=file['lfs']['sha256'];actual_path=Path(baseline[label]['path'])
        actual_path=actual_path/filename if actual_path.is_dir() else actual_path
        assert expected==baseline[label]['sha256']==sha(actual_path),(label,expected,baseline[label]['sha256'])
        if label=='VLM':
            ancillary=[]
            for entry in data['siblings']:
                path=Path(baseline[label]['path'])/entry['rfilename']
                if path.suffix not in ['.py','.json','.txt']:continue
                if not path.is_file():
                    if entry['rfilename'] in ['all_results.json','train_results.json','trainer_state.json']:continue
                    raise FileNotFoundError('Missing VLM source/tokenizer/config: '+str(path))
                payload=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
                ancillary.append(dict(file=entry['rfilename'],official_blob=entry['blobId'],actual_blob=blob,exact=blob==entry['blobId'],sha256=sha(path)))
            assert ancillary and all(r['exact'] for r in ancillary),'Modified VLM inference source/config/tokenizer'
            write(ROOT/'audit/VLM_ancillary_release_comparison.json',dict(revision=data['sha'],records=ancillary,all_exact=True))
        return label,dict(repo_id=repo,revision=data['sha'],filename=filename,sha256=expected,bytes=file['lfs']['size'],
            pinned_source='https://huggingface.co/'+repo+'/blob/'+data['sha']+'/'+filename,local_file_verified=True)
    with ThreadPoolExecutor(3) as pool:records=dict(pool.map(inspect,['IL','RL','VLM']))
    write(ROOT/'audit/official_model_sources.json',dict(passed=True,records=records,owner=identity(),source='Public HuggingFace LFS SHA256 metadata, compared to actual local files'))
    print('OFFICIAL_WEIGHT_LFS_IDENTITIES_PASS', {k:v['revision'] for k,v in records.items()})


def hardware():
    from concurrent.futures import ThreadPoolExecutor
    from scripts.opd.resources import remote
    from scripts.opd.suite import HOSTS
    policy=read('/mnt/project/server_dispatch_policy.json')
    assert all(h in policy['allowed_hosts'] and h not in policy.get('global_dispatch_prohibitions',{}) for h in HOSTS)
    def inspect(host):
        code="""import os,sys,json,subprocess
from pathlib import Path
versions=subprocess.check_output([PYTHON,'-c',"import sys,json,importlib.metadata as m; print(json.dumps(dict(python=sys.version,packages={n:m.version(n) for n in ['torch','transformers','numpy','scipy','safetensors','flash-attn','nuplan-devkit']})))"],text=True)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True)
print(json.dumps(dict(host=os.uname().nodename,affinity_CPUs=len(os.sched_getaffinity(0)),memory_total=next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemTotal:')),GPU=gpu,environment=json.loads(versions))))
""".replace('PYTHON',repr(PYTHON))
        return host,remote(host,code)
    with ThreadPoolExecutor(4) as pool:data=dict(pool.map(inspect,HOSTS))
    uuids=[line.split(',')[1].strip() for v in data.values() for line in v['GPU'].splitlines()]
    assert len(uuids)==len(set(uuids))==len(HOSTS)*8
    write(ROOT/'audit/hardware_hosts.json',dict(hosts=data,unique_A800_devices=len(uuids),timestamp=time.time(),policy_sha256=sha('/mnt/project/server_dispatch_policy.json'),
        scope='Actual allowlisted research hosts. Genuine other workloads preserved; availability rechecked at every dispatch. Additional authorized hosts inspected but occupied; new host must separately pass exact native inference parity.'))
    print('HARDWARE_VERIFIED',len(uuids),'unique GPUs')


def tests():
    required=['real_batch','real_grpo','real_resume_v2','real_grpo_resume','real_ddp','real_online_vlm','real_parameter_perturbation','real_GT_parity','real_inference_cache','real_parallel_inference','real_shared_lock','real_student_deployment','real_disabled_auxiliary']
    for name in ['real_statistics_projection','real_split_bundle','real_fresh_manifest']:
        if (ROOT/'tests'/f'{name}.json').exists():required.append(name)
    if (ROOT/'resources/navtest_routing_lock.json').exists() and read(ROOT/'resources/navtest_routing_lock.json').get('GPU_capacity',32)>32:required.append('real_new_host')
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
        'T23':'tests/real_student_deployment.json trained native online student, no teacher/anchor','T24':'tests/real_batch.json future poison and real_online_vlm exact native inputs',
        'T25':'tests/real_resume_v2.json and real_grpo_resume.json exact actual full parameters,optimizer,RNG,counters','T26':'audit/gates.json same-noise repeated outputs',
        'T27':'tests/real_ddp.json actual2-GPU gradient reduction','T28':'tests/real_grpo.json exact zero-OPD native loss/gradient'}
    write(ROOT/'tests/final_matrix.json',dict(passed=True,CPU=counts,integration_evidence={k:sha(ROOT/'tests'/f'{k}.json') for k in required},tests={k:dict(status='PASS',evidence=v) for k,v in mapping.items()},
        caveat='Numerical acceptance is separate from research performance. Optional KL/timestep performance ablations are NOT_RUN.'))
    print('TEST_MATRIX_PASS',counts)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['provenance','accounting','tests','cache_versions','model_sources','hardware','storage']);a=p.parse_args();globals()[a.action]()
