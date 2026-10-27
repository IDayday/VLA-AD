"""CPU byte-equivalence, interruption and actual completed-pilot queue checks."""
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import fit_library as fitter
import queue_transport as transport
original_queue=fitter._FITTER._original_source.queue_request


def fast(folder,request,chunk):
    return transport.queue_request(folder,request,chunk,approved_root=ROOT,
        reference=fitter.ref,digest=fitter.digest,atomic=fitter.atomic)


def compare(a,b):
    files=['QUEUE.json']+[str(p.relative_to(a)) for p in sorted((a/'tasks').glob('*.json'))]
    for rel in files:assert (a/rel).read_bytes()==(b/rel).read_bytes(),rel
    return len(files)


def rejected(function):
    try:function()
    except ValueError:return
    raise AssertionError('Invalid existing task/queue or ROOT escaped its guard')


def main():
    folder=ROOT/'selftests/queue_transport'/str(time.time_ns());folder.mkdir(parents=True)
    compared=0
    for i,(actions,chunk) in enumerate(((1,1),(9,8),(17,9),(17,8))):
        case=folder/str(i);request=case/'REQUEST.json'
        fitter.atomic(request,{'indices':[11,37,91],'actions':[{'choice':k} for k in range(actions)],'noise_indices':[0,1,2]})
        serial=case/'serial';parallel=case/'parallel';original_queue(serial,request,chunk);fast(parallel,request,chunk)
        compared+=compare(serial,parallel)
        before={str(p.relative_to(parallel)):p.stat().st_mtime_ns for p in parallel.rglob('*.json') if p.name!='QUEUE_PERSISTENCE.json'}
        fast(parallel,request,chunk)
        assert before=={str(p.relative_to(parallel)):p.stat().st_mtime_ns for p in parallel.rglob('*.json') if p.name!='QUEUE_PERSISTENCE.json'}
        partial=case/'partial';(partial/'tasks').mkdir(parents=True)
        for task in sorted((serial/'tasks').glob('*.json'))[::2]:shutil.copy2(task,partial/'tasks'/task.name)
        fast(partial,request,chunk);compared+=compare(serial,partial)
        task=next((partial/'tasks').glob('*.json'));saved=task.read_bytes();task.write_bytes(saved.replace(b'"index": 11',b'"index": 999'))
        # Choose an unambiguous corruption regardless of the first task index.
        task.write_bytes(saved+b' ');rejected(lambda:fast(partial,request,chunk));task.write_bytes(saved)
        q=partial/'QUEUE.json';body=q.read_bytes();q.write_bytes(body+b' ');rejected(lambda:fast(partial,request,chunk));q.write_bytes(body)
    request=folder/'interrupt/REQUEST.json';fitter.atomic(request,{'indices':list(range(40)),'actions':list(range(17)),'noise_indices':[0,1]})
    target=request.parent/'output';real_fsync=transport.os.fsync;calls=[0];lock=threading.Lock()
    def interrupt(fd):
        with lock:
            calls[0]+=1
            fail=calls[0]==7
        if fail:raise OSError('PRIVATE_CPU_INJECTED_QUEUE_WRITE_FAILURE')
        return real_fsync(fd)
    with patch.object(transport.os,'fsync',interrupt):
        try:fast(target,request,8)
        except OSError:pass
        else:raise AssertionError('Injected write failure did not interrupt publication')
    assert not (target/'QUEUE.json').exists()
    completed={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (target/'tasks').glob('*.json')}
    fast(target,request,8)
    for name,(data,mtime) in completed.items():assert (target/'tasks'/name).read_bytes()==data and (target/'tasks'/name).stat().st_mtime_ns==mtime
    expected=request.parent/'serial';original_queue(expected,request,8);compared+=compare(expected,target)
    rejected(lambda:fast(Path('/tmp/PRIVATE_QUEUE_MUST_NOT_BE_CREATED'),request,8))
    unsafe=folder/'unsafe';unsafe.mkdir();(unsafe/'tasks').symlink_to('/tmp',target_is_directory=True)
    rejected(lambda:fast(unsafe,request,8))
    pilot=ROOT/'requests/pilot';assert fitter.read(pilot/'COMPLETE.json')['status']=='COMPLETE'
    shadow=folder/'actual_completed_pilot';begin=time.monotonic();fast(shadow,pilot/'REQUEST.json',8);seconds=time.monotonic()-begin
    pilot_files=compare(pilot,shadow);assert pilot_files==2913
    cloned=fitter._ADAPTER.cloned_fitter(ROOT)
    assert cloned.queue_request.func is transport.queue_request
    for name in ('fit_inner','build_space','fit_payload','mlp_fit','coefficient','pack','collect','refit','freeze'):
        assert getattr(cloned,name).__code__ is getattr(cloned._original_source,name).__code__,name
    value={'status':'PASS_CPU_ORIGINAL_QUEUE_BYTES_DURABLE_RESUME_AND_ACTUAL_PILOT','created_at':fitter.stamp(),
        'source_sha256':fitter.sha(ROOT/'code/queue_transport.py'),'fixture':str(folder),
        'synthetic_byte_identical_files':compared,'actual_completed_pilot_tasks':2912,'actual_pilot_byte_identical_files':pilot_files,
        'actual_pilot_parallel_persistence_seconds':seconds,'resume_keeps_existing_file_bytes_and_mtime':True,
        'interrupted_file_write_leaves_no_QUEUE_then_resumes':True,'corrupt_task_or_queue_rejected':True,
        'outside_ROOT_or_symlinked_tasks_outside_ROOT_rejected':True,'original_task_IDs_order_actions_noises_and_queue_bytes_preserved':True,
        'GPU_or_model_dispatch_or_live_progress_reads':False,'CLI_cloned_fitter_queue_adapter_installed':True,
        'original_numerical_function_code_objects_unchanged':True,'running_old_pack_cal_process_not_interrupted':True}
    fitter.atomic(ROOT/'engineering/QUEUE_TRANSPORT_CPU_CHECK.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
