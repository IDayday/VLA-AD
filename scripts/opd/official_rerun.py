"""Uniform engineering-correction rerun, with unchanged registered experiments."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.opd.common import *
from scripts.opd.suite import run_queue,formal,EXPS
from scripts.opd.resources import launch

def call(*args):subprocess.run([PYTHON,*args],cwd=SOURCE,check=True)

def wait_job(name):
    p=ROOT/'jobs'/name/'complete.json'
    while not p.exists():time.sleep(10)
    if read(p)['status']!='COMPLETE':raise RuntimeError('Failed '+name)

def bundle(name,commands):
    path=ROOT/'resources/bundles'/f'{name}.json';write(path,commands)
    return ['scripts/opd/parallel.py','--spec',str(path),'--concurrency','4','--receipt',str(path.with_suffix('.receipt.json'))]

def qualify():
    for i in range(32):wait_job(f'official_inputs_{i}')
    call('scripts/opd/prepare_data.py')
    run_queue([('G0_G1_audit',[
        ['scripts/opd/audit.py'],['scripts/opd/validate.py','--mode','batch'],['scripts/opd/validate.py','--mode','grpo'],
        ['scripts/opd/validate.py','--mode','perturb'],['scripts/opd/verify_integrations.py','auxiliary']])])
    serial=['scripts/opd/evaluate.py','--exp','E0','--name','E0_serial_profile','--split','cal','--limit','16','--serial','--workers','1']
    parallel=bundle('profile_parallel',[['scripts/opd/evaluate.py','--exp','E0','--name','E0_parallel_profile','--split','cal',
        '--limit','16','--rank',str(i),'--shards','4','--workers','1'] for i in range(4)])
    run_queue([
        ('profile_serial',[serial]),('profile_parallel',[parallel]),
        ('online_official',[['scripts/opd/inputs.py','verify'],['scripts/opd/official_parity.py']]),
        ('native_cache_parity',[['scripts/opd/validate.py','--mode','cache']]),
        ('E6_real_integration',[['scripts/opd/train_grpo.py','--stage','smoke','--variant','IL_JOINT','--name','E6_IL_JOINT_42_smoke']]),
        ('resume_v2_full',[['scripts/opd/train.py','--exp','E3','--stage','smoke','--seed','73','--name','resume_v2_full']]),
        ('resume_v2_split',[['scripts/opd/train.py','--exp','E3','--stage','smoke','--seed','73','--name','resume_v2_split','--stop-after','2'],
            ['scripts/opd/train.py','--exp','E3','--stage','smoke','--seed','73','--name','resume_v2_split','--resume']]),
        ('ddp_single',[['scripts/opd/train.py','--exp','E3','--stage','smoke','--seed','72','--name','ddp_smoke_single']]),
    ])
    launch('ddp_actual_two','training-rl-zt2',[0,1],[['-m','torch.distributed.run','--standalone','--nproc_per_node','2',
        'scripts/opd/train.py','--exp','E3','--stage','smoke','--seed','72','--name','ddp_smoke_fixed']])
    wait_job('ddp_actual_two')
    for action in ['parallel','resume','ddp','gt_parity','shared_lock']:call('scripts/opd/verify_integrations.py',action)
    smoke=read(ROOT/'runs/E6_IL_JOINT_42_smoke/complete.json');per=smoke['seconds']/smoke['optimizer_updates']
    fitting=[n for n in [100,250,500] if n*per*9/3600<=64]
    if not fitting:raise RuntimeError('No registered E6 budget fits measured throughput')
    write(ROOT/'selection/E6_budget.json',dict(status='LOCKED',steps=max(fitting),measured_seconds_per_update=per,estimated_training_GPU_hours=max(fitting)*per*9/3600,rule='Largest registered100/250/500 fitting64GPUh before E6 outcomes'))
    write(ROOT/'suite/G0_G1_complete.json',dict(status='COMPLETE',official_conditioning=True,time=time.time()))

def banks():
    tasks=[]
    for rank in range(16):
        commands=[['scripts/opd/collect_teacher_outputs.py','--rank',str(rank),'--shards','16']]
        commands.append(bundle(f'fit_E0_{rank}',[['scripts/opd/evaluate.py','--exp','E0','--split','train','--rank',str(rank*4+k),'--shards','64','--workers','1'] for k in range(4)]))
        tasks.append((f'baseline_teacher_{rank}',commands))
    for split,groups in [('cal',4),('dev',8)]:
        for rank in range(groups):
            tasks.append((f'baseline_{split}_{rank}',[bundle(f'{split}_E0_{rank}',[['scripts/opd/evaluate.py','--exp','E0','--split',split,
                '--rank',str(rank*4+k),'--shards',str(groups*4),'--workers','1'] for k in range(4)])]))
    run_queue(tasks)
    call('scripts/opd/build_advantage_gate.py')

def lock_protocol():
    import yaml
    p=read(ROOT/'protocol.registration.json');p['splits']=read(ROOT/'manifests/identity.json')
    p['tuning']['auxiliary_weights']=read(ROOT/'calibration/gradient_scales.json')['weights']
    p['tuning']['auxiliary_calibration_record_sha256']=sha(ROOT/'calibration/gradient_scales.json')
    p['qualification']['measured_cost']=dict(audit=read(ROOT/'audit/gates.json')['seconds'],native_batch=read(ROOT/'tests/real_batch.json'),parallel=read(ROOT/'tests/real_parallel_inference.json'))
    text=yaml.safe_dump(p,sort_keys=False,allow_unicode=True);dest=SOURCE/'configs/opd/protocol.yaml'
    if (ROOT/'protocol.lock.json').exists():raise ValueError('Protocol already locked')
    dest.write_text(text);(ROOT/'protocol.yaml').write_text(text)
    write(ROOT/'protocol.lock.json',dict(status='LOCKED',sha256=sha(dest),time=time.time(),registration_sha256=sha(ROOT/'protocol.registration.json'),owner=identity()))

def controlled():
    tasks=[]
    for exp in EXPS:
        name=f'tiny_{exp}_42'
        tasks.append((name,[['scripts/opd/train.py','--exp',exp,'--stage','tiny'],
            ['scripts/opd/evaluate.py','--exp',exp,'--name',name,'--split','cal']]))
    run_queue(tasks)
    call('scripts/opd/analyze.py','--phase','tiny')
    if read(ROOT/'analysis/G2.json')['status']!='PASS':raise RuntimeError('G2 failed; stop scientific expansion')
    run_queue([('gradient_calibration',[['scripts/opd/calibrate.py']])]);lock_protocol()
    tasks=[]
    for exp in EXPS:
        for slot,lr in enumerate(config()['lr_candidates']):
            name=f'tune_{exp}_{slot}'
            tasks.append((name,[['scripts/opd/train.py','--exp',exp,'--stage','tune','--seed','41','--lr',str(lr),'--name',name],
                ['scripts/opd/evaluate.py','--exp',exp,'--name',name,'--split','cal']]))
    run_queue(tasks);formal()

if __name__=='__main__':
    qualify();banks();controlled()
