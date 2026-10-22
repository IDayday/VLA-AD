"""One registered safe save/resume of owned native jobs for host qualification.

Never SIGKILL, never signal a process selected only by its executable name, and
never reset or repartition scientific state. The CPU coordinator must already
be stopped. Intermediate maintenance heads are neither evaluated nor selected.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import ctypes
import signal
import time
from scripts.process_improvement.common import *

JOBS={'GRPO_IL_44':'formal_GRPO_IL_44_to3907',
      'GRPO_WARM_43':'formal_GRPO_WARM_43_to3907'}


def verify_command(command,name,world):
    if (name not in JOBS or world!=4 or command.count('--name')!=1 or
            command[command.index('--name')+1]!=name or '--resume' not in command or
            command.count('scripts/process_improvement/train_grpo.py')!=1 or
            '--nproc_per_node=4' not in command or '--stop-after' not in command or
            command[command.index('--stop-after')+1]!='3907' or '--score-budget' not in command or
            command[command.index('--score-budget')+1]!='4000000'):
        raise ValueError('Not the registered owned native continuation')


def verify_final_world8_command(command,name,world):
    if (name!='GRPO_IL_42' or world!=8 or '--resume' not in command or
            '--nproc_per_node=8' not in command or '--stop-after' in command or '--steps' in command or
            command.count('scripts/process_improvement/train_grpo.py')!=1 or command.count('--name')!=1 or
            command[command.index('--name')+1]!=name or '--score-budget' not in command or
            command[command.index('--score-budget')+1]!='4000000'):
        raise ValueError('Not the registered final-budget world8 continuation')


def verify_live_hosts(registration,policy):
    permitted=set(policy['allowed_hosts'])-set(policy.get('global_dispatch_prohibitions',{}))
    if not {registration['source_host'],registration['destination_host']}<=permitted:
        raise PermissionError('Registered host withdrawn; no native pause or migration permitted')


def main():
    import psutil
    from scripts.process_improvement.controller_control import owned
    from scripts.process_improvement.calibrate import immutable
    from scripts.process_improvement.endpoint_state import freeze_endpoint_state
    from scripts.process_improvement.formal_controls import freeze_endpoint_cost
    from scripts.process_improvement.dispatch import launch
    from scripts.process_improvement.coordinator import available
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pause','resume'])
    p.add_argument('--current-world8',action='store_true');a=p.parse_args()
    here=identity()['host'];assert here=='training-vla-zt'
    assert owned() is None,'Stop only the owned CPU coordinator before maintenance'
    registration=read(ROOT/'selection'/('host8_execution_amendment.json' if a.current_world8 else 'host8_execution_registration.json'))
    verify_live_hosts(registration,read(PROJECT.parent/'server_dispatch_policy.json'))
    assert registration['destination_host']==here
    jobs={'GRPO_IL_42':'formal_GRPO_IL_42_to15625'} if a.current_world8 else JOBS
    world=8 if a.current_world8 else 4
    verify=verify_final_world8_command if a.current_world8 else verify_command
    source=read(ROOT/'tests/native_GRPO_host6_world8_source.json')
    assert source['status']=='PASS_SOURCE_ONLY_NOT_MIGRATION_ADMISSION'
    from scripts.process_improvement.qualify_grpo_host8 import sources
    assert source['qualification_source_hashes']==sources()
    folder=ROOT/'maintenance'/('host8_current_qualification' if a.current_world8 else 'host8_qualification')
    folder.mkdir(parents=True,exist_ok=True)
    if a.action=='pause':
        if (folder/'pause_complete.json').exists():raise FileExistsError('Maintenance pause already recorded')
        selected=[]
        for name,job in jobs.items():
            run=ROOT/'runs'/name;cfg=read(run/'config.json');spec=read(ROOT/'jobs'/job/'spec.json')
            assert cfg['owner']['host']==spec['host']==here and cfg['world']==world
            assert not (ROOT/'jobs'/job/'complete.json').exists()
            assert len(spec['commands'])==1;verify(spec['commands'][0],name,cfg['world'])
            if a.current_world8:assert spec['commands'][0]==registration['original_command']
            parent=read(ROOT/'jobs'/job/'child_0.json')['pid']
            matches=[]
            for child in psutil.Process(parent).children(recursive=False):
                try:
                    args=child.cmdline();env=child.environ()
                    if ('scripts/process_improvement/train_grpo.py' in args and '--name' in args and
                            args[args.index('--name')+1]==name and env.get('RANK')=='0' and env.get('WORLD_SIZE')==str(world)):
                        assert child.uids().real==os.getuid() and Path(child.cwd()).resolve()==SOURCE.resolve()
                        matches.append(child.pid)
                except psutil.NoSuchProcess:continue
            assert len(matches)==1,'Native rank0 process identity is ambiguous'
            proc=Path('/proc')/str(matches[0])
            selected.append(dict(name=name,job=job,pid=matches[0],parent_pid=parent,
                start_ticks=(proc/'stat').read_text().split()[21],argv=(proc/'cmdline').read_bytes().hex(),
                config_sha256=sha(run/'config.json'),spec_sha256=sha(ROOT/'jobs'/job/'spec.json')))
        immutable(folder/'signal_plan.json',dict(action='NATIVE_SIGUSR1_ATOMIC_SAVE',records=selected))
        for row in selected:
            proc=Path('/proc')/str(row['pid']);libc=ctypes.CDLL(None,use_errno=True)
            fd=int(libc.syscall(434,row['pid'],0))
            if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open; no signal sent')
            try:
                assert (proc/'stat').read_text().split()[21]==row['start_ticks']
                assert (proc/'cmdline').read_bytes().hex()==row['argv'] and proc.stat().st_uid==os.getuid()
                assert psutil.Process(row['pid']).ppid()==row['parent_pid']
                if libc.syscall(424,fd,int(signal.SIGUSR1),0,0)<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
            finally:os.close(fd)
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            if all((ROOT/'jobs'/job/'complete.json').exists() for job in jobs.values()):break
            time.sleep(.5)
        else:raise TimeoutError('Native jobs have not completed their safe save; no force signal permitted')
        saved=[]
        for name,job in jobs.items():
            assert read(ROOT/'jobs'/job/'complete.json')['status']=='COMPLETE'
            run=ROOT/'runs'/name;pause=read(run/'paused.json')
            assert (3907<pause['step']<15625) if a.current_world8 else (977<pause['step']<3907)
            saved.append(dict(name=name,original_job=job,pause=pause,
                state=freeze_endpoint_state(run,pause['step']),cost=freeze_endpoint_cost(run,pause['step'])))
        immutable(folder/'pause_complete.json',dict(status='ACTUAL_NATIVE_ATOMIC_STATES_PRESERVED',records=saved,
            interpretation='Administrative save only; same full native lineages and original targets; no evaluation or selection of these intermediate heads'))
        print('OWNED_NATIVE_MAINTENANCE_SAVE_COMPLETE',[(x['name'],x['pause']['step']) for x in saved],flush=True);return
    paused=read(folder/'pause_complete.json');submitted=[]
    for item in paused['records']:
        name=item['name'];run=ROOT/'runs'/name;job=jobs[name];spec=read(ROOT/'jobs'/job/'spec.json')
        verify(spec['commands'][0],name,read(run/'config.json')['world'])
        assert read(run/'paused.json')==item['pause']
        assert sha(run/'latest.pt')==item['state']['full_state_sha256']
        host=here
        if a.current_world8:
            gate=ROOT/'tests/native_GRPO_cross_host_resume8.json'
            if gate.exists():
                value=read(gate)
                assert value['status']=='PASS' and value['world']==8 and value['qualification_source_hashes']==sources()
                assert value['destination_host']==here and value['source_host']==registration['source_host']
                host=value['source_host']
        assert set(spec['gpus'])<=set(available(host))
        new=('maintenance_host8_current_resume_'+name+'_to15625') if a.current_world8 else ('maintenance_host8_resume_'+name+'_to3907')
        receipt=launch(new,host,spec['gpus'],spec['commands'])
        submitted.append(dict(name=name,new_job=new,launch=receipt,paused_step=item['pause']['step'],
                              host=host,world=world,original_full_state_sha256=item['state']['full_state_sha256']))
    immutable(folder/'resume_submitted.json',dict(records=submitted))
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        if all(not (ROOT/'runs'/x['name']/'paused.json').exists() and
               read(ROOT/'runs'/x['name']/'progress.json')[-1]['step']>x['paused_step'] for x in submitted):break
        for row in submitted:
            done=ROOT/'jobs'/row['new_job']/'complete.json'
            if done.exists():raise RuntimeError('Continuation ended before acknowledgement: '+row['new_job'])
        time.sleep(.5)
    else:raise TimeoutError('Await actual native update before restarting coordinator')
    immutable(folder/'resume_acknowledged.json',dict(status='ACTUAL_SAME_WORLD_NATIVE_UPDATES_RESUMED',records=submitted,
        interpretation='Native strict config/full optimizer/per-rank RNG restoration; original targets unchanged; no science result inferred from maintenance'))
    print('OWNED_NATIVE_MAINTENANCE_RESUME_ACKNOWLEDGED',flush=True)


if __name__=='__main__':main()
