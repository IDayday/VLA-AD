"""Full original GT -> local D48/C_to_P/K8/double on permanent resources.

Only resource/transport methods are adapted. All original GT stages and final
locally screened candidate exports are required; abandoned shortened engineering is unused.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'code')]
from permanent_dispatch import admit, install, PYTHON, LEGACY
from gt_numeric import atomic, read, sha, stamp, STAGES
from fit_library import ref
from population_transport import coordinated_queue,transport_lock

SPEC=importlib.util.spec_from_file_location('_permanent_original_teacher_suite',ROOT/'code/local_teacher_suite.py')
original=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=original;SPEC.loader.exec_module(original)

def candidate_definition(root):
    path=root/'inputs/LOCAL_CANDIDATES.json';value=read(path)
    rows=value.get('candidates',value.get('records',[]))
    count=len(rows)
    if count<1 or value.get('count',count)!=count:
        raise ValueError('A verified nonempty local candidate manifest is required')
    if len({r['token'] for r in rows})!=count or len({int(r['index']) for r in rows})!=count:
        raise ValueError('Local candidate IDs must be unique')
    return count,{int(r['index']) for r in rows},ref(path)

def full_gt_acceptance(root):
    count,indices,candidates=candidate_definition(root)
    definition=read(candidates['path'])
    rows=definition.get('candidates',definition.get('records',[]))
    expected={(int(r['index']),r['token']) for r in rows}
    gt=root/'native_gt';search=read(gt/'SEARCH_COMPLETE.json');queue=read(gt/'runs/QUEUE.json')
    if (search.get('status')!='FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE' or search.get('stages')!=STAGES or
        search.get('candidate_count',search.get('train_failures'))!=count):
        raise ValueError('All original GT stages must complete for the local candidate set')
    if queue.get('active') or queue.get('stages')!=STAGES+['export_post']:
        raise ValueError('Full local GT queue and final exports must finish')
    evidence=[]
    for stage in STAGES:
        path=gt/'runs'/stage/'ACCEPTED.json';value=read(path);summary=gt/value['summary']
        if value['status']!='PASS_COMPLETE' or value['scenes']!=count or sha(summary)!=value['sha256']:
            raise ValueError('Original stage lacks a full local cumulative summary')
        cases=read(summary)['cases']
        covered={(int(r['index']),r['token']) for r in cases}
        if len(cases)!=count or covered!=expected:
            raise ValueError('Cumulative stage summary does not cover the exact local candidate set')
        evidence.append(ref(path))
    exports=read(gt/'GT_POST_COMPLETE.json')
    if (exports.get('status')!='ALL_LOCAL_GT_POST_CANDIDATES_EXPORTED' or
        exports.get('actual_exports')!=count or exports.get('candidate_count',exports.get('train_failures'))!=count):
        raise ValueError('All local candidates need final same-execution post export records')
    if sha(gt/exports['search_summary'])!=exports['search_summary_sha256']:
        raise ValueError('Final source summary changed')
    actual=list((gt/'exports/post').glob('*/COMPLETE.json'))
    if {p.parent.name for p in actual}!={r['token'] for r in rows} or len(actual)!=count:
        raise ValueError('Final export directories differ from the exact local candidate set')
    for path in actual:
        value=read(path)
        if (value.get('status')!='PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR' or
            (int(value['global_index']),value['token']) not in expected or
            value['token']!=path.parent.name or value.get('population')!='train' or
            value.get('same_execution_interface_and_trajectory') is not True):
            raise ValueError('Final candidate export lacks its actual same-execution Train identity')
    return {'status':'FULL_LOCAL_CANDIDATE_GT_AND_EXPORTS_ACCEPTED','candidate_count':count,
            'candidate_manifest':candidates,'source_search':ref(gt/'SEARCH_COMPLETE.json'),
            'final_exports':ref(gt/'GT_POST_COMPLETE.json'),'all15_original_stages':evidence,
            'candidate_count_is_not_successful_pair_count':True,'shortened_cut_or_FAST_exports_used':False}

class Owner(original.Owner):
    def __init__(self,args):
        self.args=args
        if args.root.resolve()!=ROOT:raise ValueError('This adapter serves the verified local-candidate full-flow root')
        self.allocation=read(args.allocation);admit(self.allocation)
        self.validation_allocation=list(self.allocation)
        self.test_allocation=read(args.test_allocation) if getattr(args,'test_allocation',None) else self.allocation
        admit(self.test_allocation)
        self.folder=args.root/('teacher_test_prefetch' if getattr(args,'test_prefetch_only',False) else 'teacher_suite');self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=(self.folder/'OWNER.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.registry_path=self.folder/'REGISTRY.json'
        self.registry=read(self.registry_path) if self.registry_path.exists() else {'schema':'nrkd_local_teacher_suite_v1','stages':{},'history':[]}
        self.registry.update(owner_pid=os.getpid(),owner_host=socket.gethostname(),
               owner_start_ticks=original.process_start_ticks(os.getpid()),allocation=ref(args.allocation),
               resource_adapter=ref(__file__),permanent_only=True,full_original_GT_flow_required=True,
               validation_uses_all_registered_GPUs=True,shortened_cut_or_FAST_data_used=False)
        install();self.write('WAITING_FULL_ORIGINAL_GT')

    def cpu(self,name,arguments,complete_path=None):
        if name in ('queue_train','queue_test','queue_validation'):
            population=name.removeprefix('queue_')
            adopted=coordinated_queue(self.args.root,population,self.args.features,
                lambda:super(Owner,self).cpu(name,arguments,complete_path))
            if adopted:
                entry=self.registry['stages'].setdefault(name,{})
                entry.update(status='COMPLETE',artifact=ref(complete_path),adopted_existing_identical_population_queue=True)
                self.write()
        else:
            super().cpu(name,arguments,complete_path)
        if name in ('collect_train','collect_test','collect_validation'):
            self.build_repaired_cache_index(name.removeprefix('collect_'))

    def build_repaired_cache_index(self,population):
        """Run the lossless CPU indexer directly after actual collection."""
        folder=self.args.root/'repaired_cache_index_watcher';folder.mkdir(parents=True,exist_ok=True)
        paths=folder/'PATHS.json'
        values={'teacher_bundle':str(self.args.root/'frozen_teacher/BUNDLE.json'),
                'feature_manifest':str(self.args.features),'results_root':str(self.args.results)}
        if paths.exists() and read(paths)!=values:
            raise ValueError('Registered repaired-cache reader paths changed')
        if not paths.exists():atomic(paths,values)
        command=[self.args.python,'-B',str(self.args.round1_repo/'scripts/student_distillation/read_teacher_cache.py'),
                 '--paths',str(paths),'--teacher-root',str(self.args.root),'--population',population,
                 '--build-index','--workers','4']
        complete=self.args.root/'requests'/population/'REPAIRED_CACHE_INDEX.COMPLETE.json'
        self.run_process('index_'+population,command,complete)

    def wait_for_gt(self):
        count,indices,candidates=candidate_definition(self.args.root)
        while True:
            required,requirements=original.required_GT_jobs(self.args.gt_exit_requirements,self.allocation)
            actors=original.actor_summary(required)
            marker=self.args.root/'native_gt/GT_POST_COMPLETE.json'
            exported=list((self.args.root/'native_gt/exports/post').glob('*/COMPLETE.json'))
            if actors['all_exited'] and not actors['all_exit_zero']:
                raise RuntimeError('A registered local GT worker failed; preserve and recover it')
            if marker.exists() and len(exported)==count and actors['all_exit_zero']:
                proof=full_gt_acceptance(self.args.root)
                atomic(self.folder/'FULL_ORIGINAL_GT_ACCEPTED.json',proof)
                atomic(self.folder/'GT_EXIT_ACCEPTANCE.json',{'status':'PASS','GT_export':ref(marker),
                    'workers':actors,'requirements':requirements,'candidate_manifest':candidates,
                    'actual_candidate_export_count':count,'all_required_native_candidate_records_present':True,
                    'no_GT_teacher_GPU_overlap':True,'candidate_count_not_successful_pair_count':True})
                return
            self.wait_tick('WAITING_FULL_LOCAL_CANDIDATE_GT',candidate_count=count,GT_exports=len(exported),
                GT_actors_done=actors['done'],GT_actors_running=actors['pending_or_running'],GT_actors_failed=actors['failed'])

    def live_headroom(self,allocation=None):
        selected=self.allocation if allocation is None else allocation;admit(selected)
        jobs,_=original.required_GT_jobs(self.args.gt_exit_requirements,self.allocation)
        if not original.actor_summary(jobs)['all_exit_zero']:
            raise RuntimeError('Full-flow GT actors are still live or failed; do not overlap teacher')
        grouped=defaultdict(list)
        for row in selected:grouped[row['host']].append(row)
        proof={}
        for host,items in grouped.items():
            query=['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits']
            command=query if socket.gethostname() in (host,host+'-worker-0') else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(query)]
            result=subprocess.run(command,capture_output=True,text=True,timeout=30)
            if result.returncode:raise RuntimeError('Cannot verify permanent resource UUID/headroom: '+host)
            actual={line.split(',')[0].strip():int(line.split(',')[1].strip()) for line in result.stdout.splitlines() if ',' in line}
            if any(actual.get(a['uuid'],-1)<int(a.get('minimum_free_mb',self.args.minimum_free_mb)) for a in items):
                raise RuntimeError('Capacity changed; preserve all other real/unknown jobs')
            proof[host]={a['uuid']:actual[a['uuid']] for a in items}
        return proof

    def launch_attempt(self,name,queue_folder,attempt,population):
        relative=Path('dispatch/local_teacher')/name/f'attempt_{attempt:02d}'/'POPULATION_LAUNCH.lock'
        with transport_lock(self.args.root,relative):
            return self._launch_attempt_locked(name,queue_folder,attempt,population)

    def _launch_attempt_locked(self,name,queue_folder,attempt,population):
        stage=self.args.root/'dispatch/local_teacher'/name/f'attempt_{attempt:02d}'
        path=stage/'JOBS.json'
        allocation=self.validation_allocation if population=='validation' else self.test_allocation if population=='test' else self.allocation
        jobs=read(path) if path.exists() else []
        if not jobs:
            cohort='navtest' if population=='test' else 'navtrain'
            allocated_actors=[a for a in allocation for _ in range(int(a.get('actors_per_gpu',self.args.actors_per_gpu)))]
            for rank,item in enumerate(allocated_actors):
                worker=f'{name}_attempt{attempt:02d}_rank{rank:03d}'
                arguments=[self.args.python,'-B',str(self.args.root/'code/teacher_eval_worker.py'),
                     '--root',str(self.args.root),'--queue',str(queue_folder/'QUEUE.json'),
                     '--round1-repo',str(self.args.round1_repo),'--paths',str(self.args.paths),
                     '--noise-protocol',str(self.args.noise_protocol),'--noise-index',str(self.args.noise_root/'manifests'/f'{cohort}_scenes.json'),
                     '--noise-bank',str(self.args.noise_root/'noise'/f'{cohort}_noise.npy'),
                     '--worker-id',worker,'--max-attempts',str(self.args.engineering_retries+1),'--fail-fast']
                if self.args.inference_graph_receipt:arguments+=['--inference-graph-receipt',str(self.args.inference_graph_receipt)]
                if self.args.deadline_unix:arguments+=['--deadline-unix',str(self.args.deadline_unix)]
                jobs.append({'name':worker,'host':item['host'],'gpu':item['gpu'],'uuid':item['uuid'],
                             'arguments':arguments,'status':str(stage/item['host']/'status'/(worker+'.json')),
                             'worker_receipt':str(queue_folder/'workers'/worker/'COMPLETE.json')})
            atomic(path,jobs)
        install();import dispatch
        by_host=defaultdict(list)
        for job in jobs:by_host[job['host']].append((job['name'],job['uuid'],job['arguments']))
        for host,items in by_host.items():
            directory=stage/host
            if not (directory/'launcher.pid').exists():
                self.live_headroom([a for a in allocation if a['host']==host]);dispatch.launch(host,directory,items)
        return jobs

    def run(self):
        super().run()
        atomic(self.args.root/'repaired_cache_index_watcher/COMPLETE.json',{'status':'COMPLETE',
               'populations':['test','train','validation'],
               'artifacts':{population:ref(self.args.root/'requests'/population/'REPAIRED_CACHE_INDEX.COMPLETE.json')
                            for population in ('test','train','validation')},
               'reader_source':ref(self.args.round1_repo/'scripts/student_distillation/read_teacher_cache.py'),
               'execution':'direct_durable_CPU_completion_callback_after_each_actual_population_collection',
               'no_model_GPU_or_scoring_calls':True})
        atomic(self.folder/'PERMANENT_TEACHER_GPU_RELEASED.json',{'status':'ALL_TEACHER_ACTORS_EXITED',
               'time':stamp(),'allocation':ref(self.args.allocation),'teacher_complete':ref(self.folder/'COMPLETE.json'),
               'informational_not_student_prerequisite':True,'full_original_GT_flow':True})

class TestPrefetchOwner(Owner):
    """Only Native Test production; the main owner retains all collection/indexing."""
    def run(self):
        bundle=self.args.root/'frozen_teacher/BUNDLE.json'
        if read(bundle).get('status')!='FROZEN_LOCAL_TEACHER':
            raise ValueError('Concurrent Test requires the already-frozen teacher')
        bundle_identity=ref(bundle)
        self.cpu('queue_test',['queue-population','--population','test','--features',str(self.args.features)],
                 self.args.root/'requests/test/QUEUE.json')
        for attempt in range(self.args.engineering_retries+1):
            jobs=self.launch_attempt('test',self.args.root/'requests/test',attempt,'test')
            while True:
                if ref(bundle)!=bundle_identity:raise ValueError('Frozen teacher changed during Test prefetch')
                actors=original.actor_summary(jobs)
                if actors['all_exited']:
                    atomic(self.folder/f'attempt_{attempt:02d}_EXIT.json',actors)
                    progress=original.queue_progress(self.args.root/'requests/test')
                    if actors['all_exit_zero'] and progress['pending']==0:
                        value={'status':'FULL_NATIVE_TEST_PREFETCH_COMPLETE','time':stamp(),
                            'teacher_bundle':bundle_identity,'scenes':12146,'branches':9,'noise_indices':[0,1,2,3,4],
                            'jobs':ref(self.args.root/'dispatch/local_teacher/test'/f'attempt_{attempt:02d}'/'JOBS.json'),
                            'all_actual_actor_exits_zero':True,'collectors_selections_and_indexes_owned_by_main_owner':True,
                            'student_Test_or_parameter_fit_performed':False}
                        atomic(self.folder/'COMPLETE.json',value);self.write('NATIVE_TEST_PREFETCH_COMPLETE');return
                    break
                self.wait_tick('NATIVE_TEST_PREFETCH',actors_running=actors['pending_or_running'],
                               actors_done=actors['done'],actors_failed=actors['failed'])
        raise RuntimeError('Full Native Test prefetch exhausted bounded engineering attempts')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--round1-repo',type=Path,default=original.ROUND1)
    parser.add_argument('--results',type=Path,default=original.RESULTS)
    parser.add_argument('--python',default=PYTHON)
    parser.add_argument('--paths',type=Path,default=LEGACY/'paths.local.json')
    parser.add_argument('--features',type=Path,default=original.RESULTS/'features/MANIFEST.ALL_POPULATIONS.json')
    parser.add_argument('--allocation',type=Path,default=HERE/'PERMANENT_TEACHER_ALLOCATION.json')
    parser.add_argument('--test-allocation',type=Path)
    parser.add_argument('--test-prefetch-only',action='store_true')
    parser.add_argument('--gt-exit-requirements',type=Path,default=ROOT/'dispatch/GT_EXIT_REQUIREMENTS.json')
    parser.add_argument('--noise-protocol',type=Path,default=original.OLD/'reference/noise/PROTOCOL.json')
    parser.add_argument('--noise-root',type=Path,default=original.OLD/'reference')
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--engineering-retries',type=int,default=2)
    parser.add_argument('--minimum-free-mb',type=int,default=20000)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--fit-workers',type=int,choices=range(1,9),default=8)
    parser.add_argument('--actors-per-gpu',type=int,choices=(1,2),default=2)
    parser.add_argument('--inference-graph-receipt',type=Path,default=original.RESULTS/'qualification/NATIVE_INFERENCE_GRAPH_ADMISSION.json')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();args.root=args.root.resolve();args.targets_manifest=None;args.diagnostics_out=None
    if not 0<args.poll_seconds<=60:parser.error('poll-seconds must be in (0,60]')
    if args.preflight_only:
        allocation=read(args.allocation);admit(allocation)
        value={'status':'PERMANENT_FULLFLOW_CPU_PREFLIGHT_PASS','physical_GPUs':len(allocation),
               'actors':sum(int(a.get('actors_per_gpu',args.actors_per_gpu)) for a in allocation),
               'original_owner':ref(ROOT/'code/local_teacher_suite.py'),'source':ref(__file__),
               'unchanged_original_methods':['run_process','verify_base_qualification','scoring','targets','costs'],
               'CPU_extension':'lossless_repaired_cache_index_after_actual_population_collection',
               'strict_full_original_GT_required':True,'no_production_registry_or_GPU_dispatch_modified':True}
        atomic(HERE/'CPU_PREFLIGHT.json',value);print(json.dumps(value),flush=True);return
    try:(TestPrefetchOwner(args) if args.test_prefetch_only else Owner(args)).run()
    except BaseException:
        folder='teacher_test_prefetch' if args.test_prefetch_only else 'teacher_suite'
        atomic(args.root/folder/'INCOMPLETE.json',{'status':'INCOMPLETE','time':stamp(),'traceback':traceback.format_exc(),
               'resume_argv':[sys.executable,__file__,*sys.argv[1:]],'full_original_GT_flow_required':True})
        raise

if __name__=='__main__':main()
