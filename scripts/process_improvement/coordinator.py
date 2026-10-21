"""Resumable first-stage real experiment orchestration over owned GPU leases.

Only predefined protocols/selection rules are used. Empty devices are checked
again atomically by dispatch.launch. Completed and failed artifacts are never
overwritten; failures require explicit review, never automatic success/retry.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import fcntl
import time
import re
from scripts.process_improvement.common import *
from scripts.process_improvement.dispatch import remote,launch

# All currently user-authorized canonical hosts are continuously considered.
# Admission still requires an empty physical device and the live policy; an
# unrelated or invisible CUDA owner is never preempted. Local stays available
# for short calibration/diagnostic work until a later explicit routing handoff.
SINGLE_HOSTS=['training-vla-zt','training-vlawm-tsinghua-zt2',
              'training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt6',
              'training-rl-zt2','training-vlawm-tsinghua-zt',
              'training-vlawm-tsinghua-zt3','training-vlawm-tsinghua-zt4']
RL_HOSTS=['training-rl-zt2','training-vlawm-tsinghua-zt5','training-vlawm-tsinghua-zt6',
          'training-vlawm-tsinghua-zt','training-vlawm-tsinghua-zt3',
          'training-vlawm-tsinghua-zt4','training-vlawm-tsinghua-zt2']


def evaluation_task(name,checkpoint=None,baseline=None,split='cal_tune',priority=0):
    dest=ROOT/'evaluation'/split/name
    if all((dest/f'complete_{r}.json').exists() for r in range(3)):return None
    commands=[]
    for rank in range(3):
        cmd=['scripts/process_improvement/evaluate.py','--name',name,'--split',split,
             '--cuda-graph','--rank',str(rank),'--shards','3']
        cmd+=['--checkpoint',str(checkpoint)] if checkpoint else ['--baseline',baseline]
        commands.append(cmd)
    path=ROOT/'suite'/f'auto_eval_{split}_{name}.json'
    if path.exists():assert read(path)==commands
    else:write(path,commands)
    worker_commands=[]
    gate=ROOT/'tests/trained_graph'/f'{name}.json'
    if checkpoint and not gate.exists():
        worker_commands.append(['scripts/process_improvement/qualify_trained_graph.py',
                                '--checkpoint',str(checkpoint),'--name',name])
    worker_commands.append(['scripts/process_improvement/parallel.py','--spec',str(path),
                           '--receipt',str(path.with_suffix('.receipt.json'))])
    return dict(name=f'auto_eval_{split}_{name}',priority=priority,gpus=1,hosts=SINGLE_HOSTS,
                commands=worker_commands)


def eval_done(name,split='cal_tune'):
    dest=ROOT/'evaluation'/split/name
    return all((dest/f'complete_{rank}.json').exists() for rank in range(3))


def ready_tasks(include_formal=False):
    from scripts.process_improvement import calibrate
    extra=ROOT/'suite/extra_queue.json'
    tasks=read(extra) if extra.exists() else []
    suffix_flags=['--frozen-suffix-cache'] if frozen_suffix_cache_available() else []
    # Fixed-hyperparameter safety monitoring baselines, no test data access.
    for base in ['IL','RL']:
        task=evaluation_task('B0_'+base,baseline=base,split='cal_confirm',priority=25)
        if task:tasks.append(task)
    names=['tune_'+m+'_'+lr for m in ['OPD','ANCHOR'] for lr in ['1e-5','3e-5']]
    rates=ROOT/'selection/learning_rates.json'
    if not rates.exists() and all(eval_done(name) for name in names):calibrate.rates()
    if rates.exists():
        selected=read(rates)['selected']
        for method in ['OPD','ANCHOR']:
            for seed in [42,43,44]:
                name=f'full_{method}_{seed}';run=ROOT/'runs'/name
                if not (run/'paused.json').exists() and not (run/'complete.json').exists():
                    tasks.append(dict(name='auto_'+name,priority=10,gpus=1,hosts=SINGLE_HOSTS,
                        commands=[['scripts/process_improvement/run_suite.py','--phase','scaling','--method',method,'--seed',str(seed)]]))
                for epoch,step in [(1,2341),(2,4681)]:
                    checkpoint=run/f'head_{step}.pt'
                    if checkpoint.exists():
                        task=evaluation_task(f'{name}_epoch{epoch}',checkpoint,priority=1)
                        if task:tasks.append(task)
        for population in ['legacy_4k','train_25k']:
            name=f'scaling_{population}_OPD_42';run=ROOT/'runs'/name
            if not (run/'complete.json').exists():
                tasks.append(dict(name='auto_'+name,priority=11,gpus=1,hosts=SINGLE_HOSTS,
                    commands=[['scripts/process_improvement/run_suite.py','--phase','scaling','--population',population]]))
        warm=ROOT/'selection/warm_start.json';extension_file=ROOT/'selection/four_epoch_extension.json'
        two_epoch_cal=all(eval_done(f'full_{m}_{s}_epoch{e}') for m in ['OPD','ANCHOR'] for s in [42,43,44] for e in [1,2])
        if two_epoch_cal and not extension_file.exists():calibrate.extension()
        if extension_file.exists():
            decision=read(extension_file)['decisions'];additional_ready=True
            for method in ['OPD','ANCHOR']:
                if not decision[method]['extend_to_four_epochs']:
                    if not all((ROOT/'runs'/f'full_{method}_{seed}'/'complete.json').exists() for seed in [42,43,44]):
                        subprocess.run([PYTHON,'scripts/process_improvement/finalize_epoch.py','--method',method],cwd=SOURCE,check=True)
                    continue
                for seed in [42,43,44]:
                    name=f'full_{method}_{seed}';run=ROOT/'runs'/name
                    if not (run/'complete.json').exists():
                        host=read(run/'config.json')['owner']['host']
                        tasks.append(dict(name='auto_extend_'+name,priority=10,gpus=1,hosts=[host],
                            commands=[['scripts/process_improvement/run_suite.py','--phase','scaling','--method',method,
                                       '--seed',str(seed),'--resume','--extend-four-epochs']]))
                    checkpoint=run/'head_9361.pt'
                    if checkpoint.exists():
                        task=evaluation_task(f'{name}_epoch4',checkpoint,priority=1)
                        if task:tasks.append(task)
                    additional_ready=additional_ready and eval_done(f'{name}_epoch4')
            if not warm.exists() and two_epoch_cal and additional_ready:calibrate.warm()
        if warm.exists():
            chosen=read(warm)['chosen']
            # Audit every fixed full-training endpoint after warm selection is
            # locked. No dev result enters the calibration/extension decisions.
            for method in ['OPD','ANCHOR']:
                epochs=[(1,2341),(2,4681)]
                if read(extension_file)['decisions'][method]['extend_to_four_epochs']:
                    epochs.append((4,9361))
                for seed in [42,43,44]:
                    for epoch,step in epochs:
                        name=f'full_{method}_{seed}'
                        task=evaluation_task(f'{name}_epoch{epoch}',ROOT/'runs'/name/f'head_{step}.pt',split='dev',priority=15)
                        if task:tasks.append(task)
            scaling=[('legacy_4k',[(2,258),(8,1029),('64k',2000)]),
                     ('train_25k',[(1,812),(2,1623),(4,3245),('64k',2000)])]
            for population,endpoints in scaling:
                for epoch,step in endpoints:
                    name=f'scaling_{population}_OPD_42';checkpoint=ROOT/'runs'/name/f'head_{step}.pt'
                    if checkpoint.exists():
                        task=evaluation_task(f'{name}_epoch{epoch}',checkpoint,split='dev',priority=16)
                        if task:tasks.append(task)
            task=evaluation_task('full_OPD_42_exposure64000',ROOT/'runs/full_OPD_42/head_2000.pt',split='dev',priority=16)
            if task:tasks.append(task)
            for seed in [42,43,44]:
                name=f"selected_full_{chosen['method']}_{seed}_epoch{chosen['epoch']}"
                if not (ROOT/'exports'/name/'verification.json').exists():
                    tasks.append(dict(name='auto_export_'+name,priority=14,gpus=1,hosts=SINGLE_HOSTS,
                        commands=[['scripts/process_improvement/export_student.py','--name',name,
                                   '--checkpoint',chosen['checkpoints'][str(seed)]['path'],'--verify']]))
            plan=diagnostic_plan()
            suffix=ROOT/'diagnostics/suffix_mismatch/selected_full_seed42/results.json'
            if (ROOT/'selection'/plan['choice_file']).exists() and not suffix.exists():
                tasks.append(dict(name='auto_selected_full_suffix_mismatch',priority=18,gpus=1,hosts=SINGLE_HOSTS,
                    commands=[['scripts/process_improvement/diagnose_suffix.py','--name','selected_full_seed42',
                               '--checkpoint',chosen['checkpoints']['42']['path'],'--config',plan['config'],'--limit','128']]))
    # The original native algorithm is independent of every process gate.
    for initial in ['IL','RL']:
        for lr in ['1e-5','3e-5']:
            name=f'tune_GRPO_{initial}_{lr}';run=ROOT/'runs'/name
            if (run/'complete.json').exists():
                task=evaluation_task(name,run/'head_812.pt')
                if task:tasks.append(task)
            elif name!='tune_GRPO_IL_1e-5':
                tasks.append(dict(name='auto_'+name,priority=5,gpus=8,hosts=RL_HOSTS,
                    commands=[['scripts/process_improvement/verify_inputs.py'],
                              ['-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                               'scripts/process_improvement/train_grpo.py','--name',name,'--init',initial,
                               '--lr',lr,'--split','train_25k','--steps','812','--score-budget','207872']]))
    grpo_names=[f'tune_GRPO_{initial}_{lr}' for initial in ['IL','RL'] for lr in ['1e-5','3e-5']]
    if not (ROOT/'selection/grpo_rates.json').exists() and all(eval_done(n) for n in grpo_names):calibrate.grpo_rates()
    # Calibration selection and confirmation occupy disjoint physical records.
    plan=diagnostic_plan();local=ROOT/'diagnostics'/plan['cal_name']
    choice=ROOT/'selection'/plan['choice_file']
    if (ROOT/'selection/diagnostic_protocol_correction.json').exists():
        for rank in range(24):
            parts=[]
            for purpose in ['train','cal']:
                name=plan[purpose+'_name'];split=plan[purpose+'_split']
                if not (ROOT/'diagnostics'/name/f'complete_{rank}.json').exists():
                    parts.append(['scripts/process_improvement/diagnose.py','--split',split,'--rank',str(rank),'--shards','24',
                                  '--grid','--name',name,'--scoring-workers','4',*suffix_flags])
            if parts:tasks.append(dict(name=f'corrected_parent_diagnostics_{rank}',priority=12,gpus=1,
                                       hosts=SINGLE_HOSTS+RL_HOSTS,commands=parts))
    if not choice.exists() and all((local/f'complete_{r}.json').exists() for r in range(24)):calibrate.mechanism()
    if choice.exists():
        phase=plan['confirm_name'];expansion=False
        if all((ROOT/'diagnostics'/phase/f'complete_{r}.json').exists() for r in range(8)):
            if not (ROOT/'selection'/f'process_gate_{phase}.json').exists():calibrate.process_gate()
            decision=read(ROOT/'selection'/f'process_gate_{phase}.json')
            if decision['expansion_allowed']:
                phase=plan['expansion_name'];expansion=True
            else:phase=None
        if phase:
            for rank in range(8):
                if (ROOT/'diagnostics'/phase/f'complete_{rank}.json').exists():continue
                command=['scripts/process_improvement/diagnose.py','--name',phase,'--split',plan['confirm_split'],
                         '--config',plan['config'],
                         '--rank',str(rank),'--shards','8','--scoring-workers','4',*suffix_flags]
                if expansion:command+=['--test','32']
                # Complete the independent confirmation before optional local
                # ablations/suffix probes can occupy its last pending shard.
                tasks.append(dict(name=f'auto_{phase}_{rank}',priority=17,gpus=1,hosts=SINGLE_HOSTS,commands=[command]))
            if expansion and all((ROOT/'diagnostics'/phase/f'complete_{r}.json').exists() for r in range(8)):
                if not (ROOT/'selection'/f'process_gate_{phase}.json').exists():calibrate.process_gate()
    if include_formal:
        from scripts.process_improvement.formal_controls import tasks as formal_tasks
        tasks.extend(formal_tasks())
    return sorted([t for t in tasks if t and all(Path(p).exists() for p in t.get('requires',[]))],
                  key=lambda t:(t['priority'],t['name']))


def available(host):
    occupied=set()
    for spec in (ROOT/'jobs').glob('*/spec.json'):
        if (spec.parent/'launch.json').exists() and not (spec.parent/'complete.json').exists():
            value=read(spec)
            if value['host']==host:occupied.update(value['gpus'])
    code="""import json,subprocess,time
def idle():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 return {int(r.split(',')[0]) for r in rows if int(r.split(',')[1])<=100 and int(r.split(',')[2])<=5}
first=idle()
if first:
 time.sleep(1)
 first &= idle()
print(json.dumps(sorted(first)))
"""
    return [gpu for gpu in remote(host,code) if gpu not in occupied]


def local_rl_handoff_ready(tasks):
    """Release the short-work pool only after its actual prerequisites drain.

    A supported process route retains its single-GPU capacity for real process
    training. Closed negative/inconclusive routes leave independent GRPO free
    to use the local eight GPUs. Resume jobs retain their original host below.
    """
    gate=ROOT/'selection/process_gate.json'
    warm=ROOT/'selection/warm_start.json'
    if not gate.exists() or not warm.exists():return False
    decision=read(gate)
    if decision['status'] not in ['NOT_SUPPORTED','INCONCLUSIVE'] or decision['expansion_allowed']:
        return False
    # Mechanism/qualification short work must actually finish. Periodic formal
    # GRPO endpoint evaluations are independent control work, not renewed
    # mechanism prerequisites: a qualified new four-rank job may use the other
    # idle cards while those evaluations run. Their failures still stay visible
    # in blocked_ready_tasks; this does not change their completion or results.
    for task in tasks:
        if task['gpus']!=1:continue
        if re.fullmatch(r'(?:reviewed_retry\d+_)*auto_eval_(?:dev|cal_confirm)_GRPO_(?:IL|RL|WARM)_(?:42|43|44)_step(?:977|3907|15625)',task['name']):
            continue
        done=ROOT/'jobs'/task['name']/'complete.json'
        if not done.exists() or read(done)['status']!='COMPLETE':return False
    return True


def fully_research_owned_or_idle(host,free):
    """Only reserve a gang while all missing cards run our own finite jobs.

    Foreign/unknown occupancy never turns seven idle GPUs into a permanently
    reserved host. The live physical recheck remains mandatory at admission.
    """
    covered=set(free)
    for path in (ROOT/'jobs').glob('*/spec.json'):
        if not (path.parent/'launch.json').exists() or (path.parent/'complete.json').exists():continue
        spec=read(path)
        if spec['host']==host:covered.update(spec['gpus'])
    return set(range(8))<=covered


def local_partition_task(task, host, idle, handoff):
    """Admit new formal GRPO on the two audited A800 hosts at global32.

    This cannot preempt short work, migrate/resume an existing lineage, change
    its global batch, or qualify itself from a unit-test fixture in production.
    An entire available host keeps the preferred eight-rank execution.
    """
    if (not handoff or host not in ['training-vla-zt','training-rl-zt2'] or not 4<=len(idle)<8 or
            task['gpus']!=8 or not re.fullmatch(r'(?:reviewed_retry\d+_)*formal_GRPO_(?:IL|RL|WARM)_\d+_to\d+',task['name'])):
        return task
    commands=task['commands']
    if any('--resume' in c for c in commands):return task
    path=ROOT/'tests/native_GRPO_partition32.json'
    if not path.exists():return task
    gate=read(path)
    if (gate.get('status')!='PASS' or gate.get('world_sizes')!=[4,8] or
            gate.get('global_batch')!=32 or gate.get('actual_optimizer_updates')!=4 or
            gate.get('validated_initializations')!=['IL','RL','WARM'] or
            not gate.get('exact_resume_optimizer')):return task
    for field,source in [
        ('train_source_sha256','scripts/process_improvement/train_grpo.py'),
        ('planner_source_sha256','navsim/agents/recogdrive/recogdrive_diffusion_planner.py'),
        ('rng_source_sha256','navsim/agents/recogdrive/process_improvement/rng_plan.py'),
        ('qualification_source_sha256','scripts/process_improvement/qualify_grpo_partition32.py')]:
        if gate.get(field)!=sha(SOURCE/source):return task
    rewritten=[]
    for command in commands:
        if ('scripts/process_improvement/train_grpo.py' not in command or
                '--nproc_per_node=8' not in command or '--name' not in command):return task
        if '--global-batch' in command and command[command.index('--global-batch')+1]!='32':return task
        name=command[command.index('--name')+1]
        if (ROOT/'runs'/name/'config.json').exists():return task
        rewritten.append(['--nproc_per_node=4' if value=='--nproc_per_node=8' else value for value in command])
    return dict(task,gpus=4,commands=rewritten)


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--formal-controls',action='store_true');a=p.parse_args()
    folder=ROOT/'controller';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lease.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(folder/'owner.json',identity())
        while True:
            identity();assert sha(ROOT/'protocol.yaml')==read(ROOT/'protocol.lock.json')['sha256']
            write(folder/'phase.json',dict(phase='scheduling',owner=identity()))
            tasks=ready_tasks(a.formal_controls);free={};started=[];errors=[];blocked=[];reserved=[]
            retry_path=ROOT/'controller/admission_retry_map.json'
            if retry_path.exists():
                retries=read(retry_path)['current']
                tasks=[dict(t,name=retries.get(t['name'],t['name'])) for t in tasks]
            handoff=local_rl_handoff_ready(tasks)
            for task in tasks:
                job=ROOT/'jobs'/task['name']
                if job.exists():
                    # Persistent failed admissions must remain visible after
                    # the transient launch-error iteration has passed.
                    done=read(job/'complete.json') if (job/'complete.json').exists() else None
                    if done and done['status']!='COMPLETE':
                        blocked.append(dict(name=task['name'],reason=done['status']))
                    elif not (job/'launch.json').exists():
                        blocked.append(dict(name=task['name'],reason='Admission needs explicit review; no launch receipt'))
                    continue # No implicit retry/overwrite of failed jobs.
                hosts=list(task['hosts'])
                if handoff and task['gpus']==8 and not any('--resume' in command for command in task['commands']):
                    if 'training-vla-zt' not in hosts:hosts.append('training-vla-zt')
                for host in hosts:
                    if host not in free:
                        try:free[host]=available(host)
                        except Exception as exc:errors.append(dict(host=host,error=repr(exc)));free[host]=[]
                    execution=local_partition_task(task,host,free[host],handoff)
                    if len(free[host])<execution['gpus']:
                        if (execution['gpus']>1 and free[host] and
                                fully_research_owned_or_idle(host,free[host])):
                            reserved.append(dict(host=host,for_task=task['name'],
                                                 idle_gpus=free[host],requested=execution['gpus']))
                            free[host]=[] # Let owned jobs drain; stop lower-priority backfill.
                        continue
                    gpus=free[host][:execution['gpus']]
                    try:
                        launch(task['name'],host,gpus,execution['commands']);free[host]=free[host][execution['gpus']:]
                        started.append(dict(name=task['name'],host=host,gpus=gpus))
                    except Exception as exc:
                        errors.append(dict(name=task['name'],error=repr(exc)))
                        # The live atomic recheck supersedes this loop's earlier
                        # idle snapshot. Do not cascade admissions to that card.
                        free[host]=[]
                    break
            state=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),ready_tasks=[t['name'] for t in tasks],
                       launched=started,errors=errors,blocked_ready_tasks=blocked,reserved_for_gangs=reserved,
                       owner=identity(),local_eight_GPU_handoff_ready=handoff)
            write(folder/'state.json',state)
            write(folder/'phase.json',dict(phase='idle',owner=identity()))
            print('COORDINATOR',state,flush=True)
            if a.once:return
            time.sleep(30)


if __name__=='__main__':main()
