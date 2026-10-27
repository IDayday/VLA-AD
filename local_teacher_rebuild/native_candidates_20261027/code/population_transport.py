"""Shared population queue/dispatch leases; no scoring or numerical changes."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
from gt_numeric import read, sha, digest

COUNTS={'train':85813,'test':12146,'validation':17475}

@contextmanager
def transport_lock(root, relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Transport lease escapes active root')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield

def validate_population_queue(root,population,features,*,expected_count=None):
    root=Path(root).resolve();folder=root/'requests'/population
    count=COUNTS[population] if expected_count is None else expected_count
    queue=read(folder/'QUEUE.json');request_path=folder/'REQUEST.json'
    if queue['request']!={'path':str(request_path),'sha256':sha(request_path)}:
        raise ValueError('Population queue request identity changed')
    request=read(request_path);bundle_path=root/'frozen_teacher/BUNDLE.json';bundle=read(bundle_path)
    if (request.get('scope')!='full_population' or request.get('population')!=population or
        request.get('bundle')!={'path':str(bundle_path),'sha256':sha(bundle_path)} or
        request.get('feature_manifest')!={'path':str(Path(features).resolve()),'sha256':sha(features)} or
        request.get('noise_indices')!=[0,1,2,3,4] or len(request.get('indices',[]))!=count or
        len(request.get('tokens',[]))!=count or len(set(request['indices']))!=count or len(set(request['tokens']))!=count or
        request.get('fitting_allowed') is not False or request.get('score_units')!='fraction' or
        request.get('diagnostic_only') is not (population!='train')):
        raise ValueError('Population queue differs from the full frozen teacher/assets/budget')
    actions=[{'choice':i,'action_id':digest({'bundle':bundle['teacher_identity'],'choice':i})} for i in range(9)]
    if request.get('actions')!=actions or len(queue['tasks'])!=count or len(set(queue['tasks']))!=count:
        raise ValueError('Full Base-first nine-branch population coverage is required')
    if (queue.get('status')!='READY_NATIVE_SCORING' or queue.get('total_tasks')!=count or
        queue.get('scene_count')!=count or queue.get('noise_indices')!=list(range(5)) or
        queue.get('logical_expert_trajectories')!=count*9*5 or queue.get('fitting_allowed') is not False):
        raise ValueError('Published Native queue totals/status/noise budget differ')
    seen=set()
    for identity in queue['tasks']:
        task=read(folder/'tasks'/(identity+'.json'));row=task['row_index']
        fields={key:task[key] for key in ('request','row_index','index','positions')}
        if (task.get('id')!=identity or digest(fields)!=identity or task.get('request')!=queue['request'] or
            not isinstance(row,int) or not 0<=row<count or row in seen or
            task.get('index')!=request['indices'][row] or task.get('positions')!=list(range(9))):
            raise ValueError('Existing population task coverage/identity changed')
        seen.add(row)
    return queue

def coordinated_queue(root,population,features,create,*,expected_count=None):
    """Publish once, or validate/adopt exact already-published immutable tasks."""
    with transport_lock(root,Path('requests')/population/'POPULATION_QUEUE.lock'):
        path=Path(root)/'requests'/population/'QUEUE.json'
        adopted=path.exists()
        if not adopted:create()
        validate_population_queue(root,population,features,expected_count=expected_count)
        return adopted

def population_writer(root,population,features,create,*,expected_count=None):
    """The producer child retains this lease even if its CPU owner exits."""
    with transport_lock(root,Path('requests')/population/'POPULATION_WRITER.lock'):
        if not (Path(root)/'requests'/population/'QUEUE.json').exists():create()
        return validate_population_queue(root,population,features,expected_count=expected_count)

def install(module):
    original=module.queue_population
    def queue_population(args):
        return population_writer(args.root,args.population,args.features,lambda:original(args))
    module.queue_population=queue_population
    # Cloned functions share a private globals dictionary; CLI dispatch must
    # use the same wrapper without modifying the archived source module.
    original.__globals__['queue_population']=queue_population
    return module
