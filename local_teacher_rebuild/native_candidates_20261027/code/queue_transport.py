"""Concurrent persistence of the unchanged Native scoring queue bytes.

Every original task ID, task body, queue order and logical evaluation remains
unchanged. Completed task files are verified and reused. Each new file retains
the original fsync followed by atomic replace, and QUEUE is published last.
"""
from concurrent.futures import ThreadPoolExecutor
import functools
import json
import os
from pathlib import Path
import time
import uuid


def _bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def queue_request(folder,request,action_chunk=8,*,approved_root,reference,digest,atomic,workers=16):
    if type(action_chunk) is not int or action_chunk<1:raise ValueError('Positive action chunk required')
    folder=Path(folder).resolve();approved_root=Path(approved_root).resolve()
    if not folder.is_relative_to(approved_root):raise ValueError('Queue write outside approved new root')
    folder.mkdir(parents=True,exist_ok=True)
    task_folder=folder/'tasks';task_folder.mkdir(parents=True,exist_ok=True);task_folder=task_folder.resolve()
    if not task_folder.is_relative_to(approved_root):raise ValueError('Task directory resolves outside approved new root')
    reference_value=reference(request);value=json.loads(Path(request).read_text());tasks=[];records=[]
    for row_index,index in enumerate(value['indices']):
        for first in range(0,len(value['actions']),action_chunk):
            positions=list(range(first,min(first+action_chunk,len(value['actions']))))
            identity={'request':reference_value,'row_index':row_index,'index':index,'positions':positions}
            record=dict(id=digest(identity),**identity);tasks.append(record['id']);records.append(record)
    existing={entry.name for entry in os.scandir(task_folder)}
    def persist(record):
        filename=record['id']+'.json';path=task_folder/filename;raw=_bytes(record)
        if filename in existing:
            if path.is_symlink() or path.read_bytes()!=raw:raise ValueError('Existing scoring task bytes differ: '+str(path))
            return 0
        temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
        return 1
    began=time.monotonic();written=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,len(records),256):
            written+=sum(pool.map(persist,records[first:first+256]))
    queue={'status':'READY_NATIVE_SCORING','request':reference_value,'tasks':tasks,'total_tasks':len(tasks),
           'scene_count':len(value['indices']),'noise_indices':value['noise_indices'],
           'logical_expert_trajectories':len(value['indices'])*len(value['actions'])*len(value['noise_indices']),
           'fitting_allowed':False}
    path=folder/'QUEUE.json'
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=_bytes(queue):raise ValueError('Existing scoring queue bytes differ')
    else:atomic(path,queue)
    report=folder/'QUEUE_PERSISTENCE.json'
    if not report.exists():
        atomic(report,{'status':'COMPLETE_IDENTICAL_ORIGINAL_QUEUE_TRANSPORT','tasks':len(tasks),
               'new_task_files':written,'verified_existing_task_files':len(tasks)-written,
               'workers':workers,'persistence_seconds':time.monotonic()-began,
               'file_fsync_and_atomic_replace_retained':True,'QUEUE_published_after_all_tasks':True,
               'task_IDS_bytes_queue_order_and_logical_evaluations_unchanged':True})
    return {'queue':str(path),'tasks':len(tasks)}


def install(module):
    module.queue_request=functools.partial(queue_request,approved_root=module.ROOT,
        reference=module.ref,digest=module.digest,atomic=module.atomic)
    return module
