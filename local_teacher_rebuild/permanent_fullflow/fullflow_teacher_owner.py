"""Full original GT -> local D48/C_to_P/K8/double on permanent resources.

Only resource/transport methods are adapted. All original GT stages and final
1850 actual post exports are required; abandoned shortened engineering is unused.
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

SPEC=importlib.util.spec_from_file_location('_permanent_original_teacher_suite',ROOT/'code/local_teacher_suite.py')
original=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=original;SPEC.loader.exec_module(original)

def full_gt_acceptance(root):
    gt=root/'native_gt';search=read(gt/'SEARCH_COMPLETE.json');queue=read(gt/'runs/QUEUE.json')
    if search.get('status')!='FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE' or search.get('stages')!=STAGES or search.get('train_failures')!=1850:
        raise ValueError('Every original finite GT stage must actually complete')
    if queue.get('active') or queue.get('stages')!=STAGES+['export_post']:
        raise ValueError('Original full GT queue and final export must actually finish')
    evidence=[]
    for stage in STAGES:
        path=gt/'runs'/stage/'ACCEPTED.json';value=read(path);summary=gt/value['summary']
        if value['status']!='PASS_COMPLETE' or value['scenes']!=1850 or sha(summary)!=value['sha256']:
            raise ValueError('An original GT stage lacks a complete checksum-verified cumulative summary')
        evidence.append(ref(path))
    exports=read(gt/'GT_POST_COMPLETE.json')
    if exports.get('status')!='ALL1850_LOCAL_GT_POST_PAIRS_EXPORTED' or exports.get('actual_exports')!=1850 or exports.get('train_failures')!=1850:
        raise ValueError('All1850 final same-execution full-flow GT post exports are required')
    if sha(gt/exports['search_summary'])!=exports['search_summary_sha256']:
        raise ValueError('Final post_progress source summary changed')
    return {'status':'FULL_ORIGINAL_GT_AND_FINAL1850_EXPORTS_ACCEPTED','source_search':ref(gt/'SEARCH_COMPLETE.json'),
            'final_exports':ref(gt/'GT_POST_COMPLETE.json'),'all15_original_stages':evidence,
            'shortened_cut_or_FAST_exports_used':False}

class Owner(original.Owner):
    def __init__(self,args):
        self.args=args
        if args.root.resolve()!=ROOT:raise ValueError('This adapter serves only the preserved formal full-flow root')
        self.allocation=read(args.allocation);admit(self.allocation)
        self.validation_allocation=list(self.allocation)
        self.folder=args.root/'teacher_suite';self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=(self.folder/'OWNER.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.registry_path=self.folder/'REGISTRY.json'
        self.registry=read(self.registry_path) if self.registry_path.exists() else {'schema':'nrkd_local_teacher_suite_v1','stages':{},'history':[]}
        self.registry.update(owner_pid=os.getpid(),owner_host=socket.gethostname(),
               owner_start_ticks=original.process_start_ticks(os.getpid()),allocation=ref(args.allocation),
               resource_adapter=ref(__file__),permanent_only=True,full_original_GT_flow_required=True,
               validation_uses_all_registered_GPUs=True,shortened_cut_or_FAST_data_used=False)
        install();self.write('WAITING_FULL_ORIGINAL_GT')

    def wait_for_gt(self):
        super().wait_for_gt()
        proof=full_gt_acceptance(self.args.root)
        atomic(self.folder/'FULL_ORIGINAL_GT_ACCEPTED.json',proof)

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
        stage=self.args.root/'dispatch/local_teacher'/name/f'attempt_{attempt:02d}'
        path=stage/'JOBS.json';allocation=self.validation_allocation if population=='validation' else self.allocation
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
        atomic(self.folder/'PERMANENT_TEACHER_GPU_RELEASED.json',{'status':'ALL_TEACHER_ACTORS_EXITED',
               'time':stamp(),'allocation':ref(self.args.allocation),'teacher_complete':ref(self.folder/'COMPLETE.json'),
               'informational_not_student_prerequisite':True,'full_original_GT_flow':True})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--round1-repo',type=Path,default=original.ROUND1)
    parser.add_argument('--results',type=Path,default=original.RESULTS)
    parser.add_argument('--python',default=PYTHON)
    parser.add_argument('--paths',type=Path,default=LEGACY/'paths.local.json')
    parser.add_argument('--features',type=Path,default=original.RESULTS/'features/MANIFEST.ALL_POPULATIONS.json')
    parser.add_argument('--allocation',type=Path,default=HERE/'PERMANENT_TEACHER_ALLOCATION.json')
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
               'unchanged_original_methods':['cpu','run_process','verify_base_qualification','scoring','targets','costs'],
               'strict_full_original_GT_required':True,'no_production_registry_or_GPU_dispatch_modified':True}
        atomic(HERE/'CPU_PREFLIGHT.json',value);print(json.dumps(value),flush=True);return
    try:Owner(args).run()
    except BaseException:
        atomic(args.root/'teacher_suite/INCOMPLETE.json',{'status':'INCOMPLETE','time':stamp(),'traceback':traceback.format_exc(),
               'resume_argv':[sys.executable,__file__,*sys.argv[1:]],'full_original_GT_flow_required':True})
        raise

if __name__=='__main__':main()
