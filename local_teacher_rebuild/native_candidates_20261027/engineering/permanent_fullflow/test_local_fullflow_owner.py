"""Private CPU transport gates; no teacher qualification or GPU execution."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
OLD=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026')
sys.path[:0]=[str(HERE),str(HERE.parents[1]/'code'),str(OLD/'code')]
spec=importlib.util.spec_from_file_location('nrkd_local_fullflow_owner_tested',HERE/'fullflow_teacher_owner.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

def main():
    with tempfile.TemporaryDirectory(prefix='local_gt_owner_gate_') as directory:
        root=Path(directory);gt=root/'native_gt';rows=[{'index':i,'token':f'token{i}'} for i in range(3)]
        put(root/'inputs/LOCAL_CANDIDATES.json',{'count':3,'records':rows})
        search={'status':'FULL_ORIGINAL_FINITE_GT_SEARCH_COMPLETE','candidate_count':3,'stages':m.STAGES}
        put(gt/'SEARCH_COMPLETE.json',search);queue={'active':False,'stages':m.STAGES+['export_post']};put(gt/'runs/QUEUE.json',queue)
        for stage in m.STAGES:
            summary=gt/'analysis'/stage/'SUMMARY.json';put(summary,{'cases':rows})
            put(gt/'runs'/stage/'ACCEPTED.json',{'status':'PASS_COMPLETE','scenes':3,'summary':str(summary.relative_to(gt)),'sha256':m.sha(summary)})
        summary=gt/'analysis/post_progress/SUMMARY.json'
        final={'status':'ALL_LOCAL_GT_POST_CANDIDATES_EXPORTED','candidate_count':3,'actual_exports':3,'search_summary':str(summary.relative_to(gt)),'search_summary_sha256':m.sha(summary)}
        put(gt/'GT_POST_COMPLETE.json',final)
        for row in rows:
            put(gt/'exports/post'/row['token']/'COMPLETE.json',
                {'status':'PASS_LOCAL_FROZEN_BASE_GT_POST_PAIR','global_index':row['index'],
                 'token':row['token'],'population':'train','same_execution_interface_and_trajectory':True})
        assert m.full_gt_acceptance(root)['candidate_count']==3
        def rejected():
            try:m.full_gt_acceptance(root)
            except (ValueError,KeyError,FileNotFoundError):return
            raise AssertionError('Invalid local candidate/phase proof accepted')
        put(gt/'SEARCH_COMPLETE.json',dict(search,candidate_count=1850));rejected();put(gt/'SEARCH_COMPLETE.json',search)
        put(gt/'SEARCH_COMPLETE.json',dict(search,status='SHORTENED_LOCAL_GT_SEARCH_CUT'));rejected();put(gt/'SEARCH_COMPLETE.json',search)
        put(gt/'runs/QUEUE.json',dict(queue,active=True));rejected();put(gt/'runs/QUEUE.json',queue)
        put(gt/'GT_POST_COMPLETE.json',dict(final,actual_exports=2));rejected();put(gt/'GT_POST_COMPLETE.json',final)
        export=gt/'exports/post/token0/COMPLETE.json';correct=json.loads(export.read_text())
        put(export,dict(correct,token='wrongtoken'));rejected();put(export,correct)
        p=gt/'analysis/baseline/SUMMARY.json';correct_summary=json.loads(p.read_text())
        a=gt/'runs/baseline/ACCEPTED.json';accepted=json.loads(a.read_text())
        put(p,{'cases':rows+[rows[0]]});put(a,dict(accepted,sha256=m.sha(p)));rejected()
        put(p,{'cases':[dict(r,token='wrongtoken') if r['index']==0 else r for r in rows]})
        put(a,dict(accepted,sha256=m.sha(p)));rejected()
        put(p,correct_summary);put(a,dict(accepted,sha256=m.sha(p)))
        p=gt/'analysis/baseline/SUMMARY.json';put(p,{'cases':[{'index':i,'token':f'token{i}'} for i in [0,1,99]]})
        a=gt/'runs/baseline/ACCEPTED.json';value=json.loads(a.read_text());value['sha256']=m.sha(p);put(a,value);rejected()
        # The actual population collector triggers exactly one durable reader
        # phase. It uses this teacher root and a one-shot CPU index command.
        owner=m.Owner.__new__(m.Owner)
        owner.args=SimpleNamespace(root=root,round1_repo=Path('/private/round1'),python=sys.executable,
            features=root/'FEATURES.json',results=root/'results')
        calls=[];owner.run_process=lambda *args:calls.append(args)
        owner.build_repaired_cache_index('train')
        command=calls[0][1]
        assert calls[0][0]=='index_train' and '--build-index' in command and '--watch-build-index' not in command
        assert command[command.index('--teacher-root')+1]==str(root)
        reader_paths=json.loads((root/'repaired_cache_index_watcher/PATHS.json').read_text())
        assert reader_paths['teacher_bundle']==str(root/'frozen_teacher/BUNDLE.json')
        assert calls[0][2]==root/'requests/train/REPAIRED_CACHE_INDEX.COMPLETE.json'
    result={'status':'PASS_DYNAMIC_LOCAL_CANDIDATE_FULL_GT_GATES','CPU_only':True,'count_not_hardcoded1850':True,'shortened_scope_rejected':True,'active_queue_rejected':True,'incomplete_export_rejected':True,'wrong_candidate_coverage_rejected':True,'candidate_token_and_duplicate_summary_mismatch_rejected':True,'wrong_export_identity_rejected':True,'direct_new_root_CPU_index_callback':True,'formal_teacher_qualification':False}
    p=HERE/'TEACHER_OWNER_CPU_CHECK.json';put(p,result);print(json.dumps(result))

if __name__=='__main__':main()
