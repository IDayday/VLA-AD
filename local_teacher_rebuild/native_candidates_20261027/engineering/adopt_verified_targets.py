"""One controlled publication after exact owned CPU/GPU producer exits.

This script never signals a process identified by a discovered PID alone: the
registered invocation, start ticks, host, and immutable spec must all agree.
"""
import ctypes,fcntl,json,os,select,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
PY='/mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python'
sys.path[:0]=[str(ROOT/'code'),str(CODE)]
from gt_numeric import atomic,read,sha,stamp
from cached_target_publication import reference,validate_publication
from preparation_identity import validate_current_output

def stop_registered_cpu():
    pid=582673;proc=Path(f'/proc/{pid}')
    expected=[PY,'-B',str(ROOT/'code/round1_prepare_and_run.py'),'--root',str(ROOT),'--poll-seconds','30',
              '--resources',str(RESULTS/'resources/ROUND1_FIXED_32GPU_WITH_TSINGHUA_ALLOCATION.json')]
    actual=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
    ticks=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    if actual!=expected or ticks!='1453379771' or (proc/'task'/str(pid)/'children').read_text().strip():
        raise RuntimeError('Exact registered preparation owner identity or child boundary differs')
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,pid,0)
    if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open')
    try:
        if (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=ticks:raise RuntimeError('PID reused')
        if libc.syscall(424,fd,signal.SIGTERM,0,0)<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
        if not select.select([fd],[],[],20)[0]:raise RuntimeError('Exact prep owner has not exited')
    finally:os.close(fd)
    with (RESULTS/'preparation_owner/OWNER.lock').open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return {'pid':pid,'start_ticks':ticks,'argv':actual,'kernel_exit_observed':True,'OWNER_lock_free':True,'time':stamp()}

REMOTE=r'''
import ctypes,fcntl,json,os,select,signal,socket,sys,time
from pathlib import Path
attempt=Path(sys.argv[1]);s=json.loads((attempt/'STATUS.json').read_text());spec=json.loads((attempt/'SPEC.json').read_text())
if socket.gethostname()!='training-vlawm-tsinghua-zt3-worker-0' or spec['host']!='training-vlawm-tsinghua-zt3':raise RuntimeError('Host differs')
if s['status']!='RUNNING' or s['child_pid']!=1516456 or str(s['child_start_ticks'])!='1454465927':raise RuntimeError('Registered child differs')
if s['pid']!=1474145 or str(s['process_start_ticks'])!='1454465741':raise RuntimeError('Registered wrapper differs')
def identity(pid,ticks,argv=None):
 p=Path('/proc')/str(pid);actual=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 if (p/'stat').read_text().rsplit(')',1)[1].split()[19]!=str(ticks):raise RuntimeError('Process start differs')
 if argv is not None and actual!=argv:raise RuntimeError('Process command differs')
 return actual
identity(s['child_pid'],s['child_start_ticks'],spec['command'])
wrapper=identity(s['pid'],s['process_start_ticks'])
if '--execute-spec' not in wrapper or str(attempt/'SPEC.json') not in wrapper:raise RuntimeError('Wrapper command differs')
libc=ctypes.CDLL(None,use_errno=True)
fds=[libc.syscall(434,pid,0) for pid in (s['child_pid'],s['pid'])]
if min(fds)<0:raise OSError(ctypes.get_errno(),'pidfd_open')
try:
 identity(s['child_pid'],s['child_start_ticks'],spec['command']);identity(s['pid'],s['process_start_ticks'])
 if libc.syscall(424,fds[0],signal.SIGTERM,0,0)<0:raise OSError(ctypes.get_errno(),'pidfd_send_signal')
 for fd in fds:
  if not select.select([fd],[],[],45)[0]:raise RuntimeError('Owned process exit not observed')
finally:
 for fd in fds:os.close(fd)
end=json.loads((attempt/'STATUS.json').read_text())
if end['status']!='EXITED' or end['returncode']!=-15:raise RuntimeError('Actual signal exit not recorded')
for p in (attempt.parent/'FAMILY_LEASE.lock',attempt/'LEASE.lock'):
 with p.open('a+') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(json.dumps({'exact_pid_start_command_verified':True,'pidfd_exit_observed':True,
 'wrapper_exit_observed':True,'family_and_attempt_leases_free':True,'child_pid':s['child_pid'],
 'child_start_ticks':s['child_start_ticks'],'actual_child_returncode':end['returncode'],
 'wrapper_argv':wrapper,'hostname':socket.gethostname()}))
'''

def main():
    out=ROOT/'engineering/cached_actual_target_adoption_20261027';out.mkdir(exist_ok=True)
    if (out/'CPU_STOP.json').exists():raise RuntimeError('A controlled transition was already recorded; inspect before resuming')
    candidate=RESULTS/'targets_verified_native_cache_stage/train';manifest=candidate/'MANIFEST.json'
    paths=read(RESULTS/'protocol/PATHS.json')
    assert validate_current_output('targets_train',manifest,paths=paths,results=RESULTS,code=CODE,world_size=4)
    producer=ROOT/'engineering/cached_actual_target_probe/FULL_TRAIN_EXIT.json'
    if read(producer)['actual_returncode']!=0:raise RuntimeError('CPU candidate producer has not exited zero')
    atomic(out/'CANDIDATE_ADMISSION.json',{'status':'PASS_ALL_85813_CURRENT_IDENTITY','manifest':reference(manifest),'time':stamp()})
    atomic(out/'CPU_STOP.json',stop_registered_cpu())
    attempt=RESULTS/'preparation_owner/remote_execution/jobs/targets_train/attempt_00000'
    # The remote script verifies both recorded process identities before the only GPU signal.
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=6','training-vlawm-tsinghua-zt3',
        'env ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+PY+' -B - '+str(attempt)],
        input=REMOTE,text=True,capture_output=True,timeout=100,check=True)
    atomic(out/'SUPERSESSION.json',dict(json.loads(result.stdout),time=stamp()))
    old=RESULTS/'targets/train';preserved=out/'superseded_cuda_partial_train'
    if preserved.exists():raise RuntimeError('Preservation destination already exists')
    old.rename(preserved);candidate.rename(old)
    value={'schema':'nrkd_cached_actual_target_publication_v1','status':'COMPLETE','time':stamp(),
      'published_manifest':reference(old/'MANIFEST.json'),'producer_exit':reference(producer),
      'candidate_acceptance':reference(ROOT/'engineering/cached_actual_target_probe/FULL_TRAIN_CANDIDATE_ACCEPTANCE.json'),
      'superseded_status':reference(attempt/'STATUS.json'),'superseded_spec':reference(attempt/'SPEC.json'),
      'supersession_proof':reference(out/'SUPERSESSION.json'),'preserved_partial_directory':str(preserved),
      'all_previous_writers_exited':True,'family_and_attempt_leases_free':True,'old_partial_preserved':True}
    atomic(old/'ADOPTION.json',value)
    validate_publication(old/'MANIFEST.json',RESULTS)
    assert validate_current_output('targets_train',old/'MANIFEST.json',paths=paths,results=RESULTS,code=CODE,world_size=4)
    atomic(out/'COMPLETE.json',dict(value,status='CURRENT_CANONICAL_TARGETS_PUBLISHED_AND_VALIDATED'))
    print(json.dumps({'status':'PUBLISHED','manifest':reference(old/'MANIFEST.json'),'proof':reference(out/'COMPLETE.json')}))

if __name__=='__main__':main()
