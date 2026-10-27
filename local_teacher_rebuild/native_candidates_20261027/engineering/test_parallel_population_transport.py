"""Private real-process queue/launch race and canonical main adoption checks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import multiprocessing as mp
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'engineering/permanent_fullflow')]
from gt_numeric import atomic,read,sha,digest,stamp
from population_transport import coordinated_queue,validate_population_queue
spec=importlib.util.spec_from_file_location('parallel_population_test_owner',ROOT/'engineering/permanent_fullflow/fullflow_teacher_owner.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def fixture(root):
    atomic(root/'frozen_teacher/BUNDLE.json',{'status':'FROZEN_LOCAL_TEACHER','teacher_identity':'fixture'})
    atomic(root/'FEATURES.json',{'private_fixture':True})

def publish(root):
    with (root/'queue_create_calls').open('a') as f:f.write('create\n')
    time.sleep(.1);folder=root/'requests/test';p=folder/'REQUEST.json'
    request={'scope':'full_population','population':'test','bundle':{'path':str(root/'frozen_teacher/BUNDLE.json'),'sha256':sha(root/'frozen_teacher/BUNDLE.json')},
        'feature_manifest':{'path':str(root/'FEATURES.json'),'sha256':sha(root/'FEATURES.json')},'noise_indices':list(range(5)),
        'indices':[13,29,67],'tokens':['a','b','c'],'fitting_allowed':False,
        'actions':[{'choice':i,'action_id':digest({'bundle':'fixture','choice':i})} for i in range(9)]}
    atomic(p,request);r={'path':str(p),'sha256':sha(p)};ids=[]
    for row,index in enumerate(request['indices']):
        t={'request':r,'row_index':row,'index':index,'positions':list(range(9))};identity=digest(t);t['id']=identity;ids.append(identity)
        atomic(folder/'tasks'/(identity+'.json'),t)
    atomic(folder/'QUEUE.json',{'request':r,'tasks':ids})

def queue_child(root,out):
    adopted=coordinated_queue(root,'test',root/'FEATURES.json',lambda:publish(root),expected_count=3)
    atomic(out,{'adopted':adopted})

def owner(root):
    o=m.Owner.__new__(m.Owner)
    o.args=SimpleNamespace(root=root,python=sys.executable,round1_repo=root/'round1',paths=root/'paths.json',
        noise_protocol=root/'noise.json',noise_root=root/'noise',engineering_retries=2,
        inference_graph_receipt=None,deadline_unix=None,actors_per_gpu=2)
    o.allocation=[{'host':'training-vla-zt','gpu':0,'uuid':'fixtureGPU','actors_per_gpu':2}]
    o.test_allocation=[dict(o.allocation[0],actors_per_gpu=1)];o.validation_allocation=o.allocation
    o.live_headroom=lambda allocation:None
    return o

def fake_launch(root,host,directory,jobs):
    with (root/'native_launch_calls').open('a') as f:f.write(host+'\n')
    time.sleep(.1);directory.mkdir(parents=True,exist_ok=True);(directory/'launcher.pid').write_text('12345\n')

def launch_child(root,out):
    m.install=lambda:None
    sys.modules['dispatch']=SimpleNamespace(launch=lambda host,directory,jobs:fake_launch(root,host,directory,jobs))
    jobs=owner(root).launch_attempt('test',root/'requests/test',0,'test')
    atomic(out,{'names':[j['name'] for j in jobs]})

def parallel(fn,root):
    ctx=mp.get_context('fork');ps=[ctx.Process(target=fn,args=(root,root/f'result{i}.json')) for i in range(2)]
    for p in ps:p.start()
    for p in ps:p.join(20);assert p.exitcode==0,p.exitcode

def main():
    with tempfile.TemporaryDirectory(prefix='private_parallel_teacher_') as d:
        root=Path(d);fixture(root);parallel(queue_child,root)
        assert (root/'queue_create_calls').read_text().splitlines()==['create']
        assert sorted(read(root/f'result{i}.json')['adopted'] for i in range(2))==[False,True]
        p=root/'requests/test/QUEUE.json';before=(p.read_bytes(),p.stat().st_mtime_ns)
        assert coordinated_queue(root,'test',root/'FEATURES.json',lambda:(_ for _ in ()).throw(AssertionError('recreated')),expected_count=3)
        assert (p.read_bytes(),p.stat().st_mtime_ns)==before
        request=root/'requests/test/REQUEST.json';saved=request.read_bytes();v=read(request);v['noise_indices']=[0];atomic(request,v)
        try:validate_population_queue(root,'test',root/'FEATURES.json',expected_count=3)
        except ValueError:pass
        else:raise AssertionError('Changed queue request was adopted')
        request.write_bytes(saved)
        parallel(launch_child,root)
        assert (root/'native_launch_calls').read_text().splitlines()==['training-vla-zt']
        jobs_path=root/'dispatch/local_teacher/test/attempt_00/JOBS.json';jobs=read(jobs_path)
        assert len(jobs)==1 and jobs[0]['name']=='test_attempt00_rank000'
        assert read(root/'result0.json')==read(root/'result1.json')
        prior=(jobs_path.read_bytes(),jobs_path.stat().st_mtime_ns)
        launch_child(root,root/'adopt.json')
        assert (jobs_path.read_bytes(),jobs_path.stat().st_mtime_ns)==prior
        assert len((root/'native_launch_calls').read_text().splitlines())==1
        # Registered intent survives an interrupted pre-launch transport; restart
        # launches the same jobs once, without rewriting their immutable manifest.
        (root/'dispatch/local_teacher/test/attempt_00/training-vla-zt/launcher.pid').unlink()
        launch_child(root,root/'resumed.json')
        assert (jobs_path.read_bytes(),jobs_path.stat().st_mtime_ns)==prior
        assert len((root/'native_launch_calls').read_text().splitlines())==2
        main_owner=owner(root);main_owner.test_allocation=main_owner.allocation
        adopted=main_owner.launch_attempt('test',root/'requests/test',0,'test')
        assert adopted==jobs and len(adopted)==1 # never force old default two actors
        assert len((root/'native_launch_calls').read_text().splitlines())==2
    value={'status':'PASS_CPU_REAL_PROCESS_PARALLEL_QUEUE_LAUNCH_AND_MAIN_ADOPTION','time':stamp(),
       'queue_creator_calls_under_two_process_race':1,'native_launcher_calls_under_two_process_race':1,
       'existing_request_queue_tasks_and_JOBS_not_rewritten':True,'changed_request_rejected':True,
       'interrupted_published_intent_resumes_same_JOBS':True,'main_adopts_smaller_Test_roster_without_relaunch':True,
       'Train_GPU_or_scoring_results_read':False,'real_GPU_or_model_dispatch':False,
       'source_sha256':{str(p):sha(p) for p in [ROOT/'code/population_transport.py',ROOT/'engineering/permanent_fullflow/fullflow_teacher_owner.py',Path(__file__)]}}
    path=ROOT/'engineering/parallel_test_20261027/CPU_TRANSPORT_CHECK.json';atomic(path,value);print(json.dumps(value))

if __name__=='__main__':main()
