"""Archive cancelled scope and launch a four-run owner without touching training."""
import fcntl,shutil,sys
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[2]
RESULTS=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'engineering/permanent_fullflow')]
from gt_numeric import read,atomic,sha,stamp
from permanent_dispatch import install,grant,PYTHON

def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}

def main():
    grant();stopped=read(HERE/'ALL_FOUR_SEED28_STOPPED.json');exited=read(HERE/'OLD_CONTROLLERS_EXITED.json')
    if len(stopped['runs'])!=4 or not exited['both_CPU_leases_free']:raise RuntimeError('Old scope teardown is incomplete')
    for item in exited['processes']:
        if Path('/proc',str(item['pid'])).exists():raise RuntimeError('An old controller remains alive')
    for path in (RESULTS/'SUITE_CONTROLLER.lock',RESULTS/'preparation_owner/OWNER.lock'):
        with path.open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (HERE/'LAUNCH.json').exists():raise RuntimeError('Four-run owner was already launched')
    instruction=ROOT/'engineering/single_seed_20261027/USER_INSTRUCTION.json'
    registry_path=RESULTS/'EXPERIMENT_REGISTRY.json';old=read(registry_path)
    archive=HERE/'prior_eight_run_registry.json'
    if archive.exists():raise RuntimeError('Scope archival already started; inspect the saved transaction')
    shutil.copyfile(registry_path,archive)
    new=deepcopy(old);cancelled={k:v for k,v in new['runs'].items() if k.endswith('_seed20261028')}
    if len(cancelled)!=4:raise RuntimeError('Expected four retained and four cancelled run records')
    for key,value in cancelled.items():
        value['status']='CANCELLED_BY_USER';value['cancelled_at']=stamp();value['all_existing_outputs_preserved']=True
        value['actual_stop_receipt']=ref(HERE/key/'COMPLETE.json')
    new.setdefault('cancelled_runs',{}).update(cancelled)
    new['runs']={k:v for k,v in new['runs'].items() if k.endswith('_seed20261027')}
    if len(new['runs'])!=4:raise RuntimeError('Active scope is not four runs')
    cancelled_tasks={k:v for k,v in new.get('tasks',{}).items() if 'seed20261028' in k}
    new.setdefault('cancelled_tasks',{}).update({k:dict(v,status='CANCELLED_BY_USER') for k,v in cancelled_tasks.items()})
    new['tasks']={k:v for k,v in new.get('tasks',{}).items() if 'seed20261028' not in k}
    new.update(status='ADOPTING_FOUR_RETAINED_SINGLE_SEED_RUNS',user_seed_scope={'active':[20261027],
      'cancelled':[20261028],'active_run_count':4,'user_instruction':ref(instruction)},updated=stamp())
    original=read(RESULTS/'resources/ROUND1_PERMANENT_READY.json');resource=deepcopy(original)
    resource['allocations']={k:v for k,v in original['allocations'].items() if k.endswith('_seed20261027')}
    if len(resource['allocations'])!=4 or len({u for x in resource['allocations'].values() for u in x['gpu_uuids']})!=16:
        raise RuntimeError('Retained resource scope must be sixteen original GPUs')
    resource.update(registered_training_seeds=[20261027],cancelled_training_seeds=[20261028],
      user_seed_scope=ref(instruction),prior_eight_run_resources=ref(RESULTS/'resources/ROUND1_PERMANENT_READY.json'))
    resources=RESULTS/'resources/ROUND1_SINGLE_SEED_16GPU_READY_20261027.json';atomic(resources,resource)
    command=read(RESULTS/'preparation_owner/full_suite_dispatch/COMMAND.json')
    arguments=command[command.index('--config-dir'):];arguments[arguments.index('--resources')+1]=str(resources)
    arguments+=['--training-seeds','20261027']
    original_specs={}
    for run in new['runs']:
        path=RESULTS/'jobs'/('train__'+run+'__attempt00')/'attempt_00000/SPEC.json'
        state=read(path.parent/'STATUS.json')
        if state['status']!='RUNNING':raise RuntimeError('Retained child is not running; inspect before adoption')
        original_specs[run]={'spec':ref(path),'child_pid':state['child_pid'],
          'child_start_ticks':state['child_start_ticks'],'gpu_uuids':state['gpu_uuids']}
    atomic(HERE/'OWNER_SPEC.json',{'schema':'nrkd_single_seed_owner_v1','user_instruction':ref(instruction),
      'arguments':arguments,'original_retained_training_specs':original_specs,'source_HEAD_must_remain':'0a58bd8e6271e26103512add2c5cb7816d7eb8f1'})
    atomic(registry_path,new)
    install();import dispatch
    name='single_seed_four_run_owner';directory=ROOT/'dispatch'/'single_seed_four_run_owner_20261027'
    if (directory/'launcher.pid').exists():raise RuntimeError('A replacement launcher already exists')
    owner_command=[PYTHON,'-B',str(HERE/'single_seed_owner.py')]
    launch=dispatch.launch('training-vla-zt',directory,[(name,'-',owner_command)])
    value={'status':'FOUR_RETAINED_RUN_OWNER_LAUNCHED_FOR_LIVE_ADOPTION','time':stamp(),'command':owner_command,
      'launch':launch,'actual_status':str(directory/'status'/(name+'.json')),'prior_registry':ref(archive),
      'resources':ref(resources),'source_files':{p.name:ref(p) for p in HERE.glob('*.py')},
      'no_training_child_command_config_source_or_HEAD_changes':True,'seed28_retries_evaluations_and_report_requirements_cancelled':True}
    atomic(HERE/'LAUNCH.json',value);print(value)

if __name__=='__main__':main()
