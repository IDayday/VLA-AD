"""Publish an exact review snapshot of the running local-candidate source.

The snapshot preserves source bytes and records operational absolute bindings;
it does not rewrite paths or copy any trained weights, feature arrays or scores.
"""
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026')
CODE=Path('/mnt/project/recogdrive-no-routing-round1-20261026')
sys.path.insert(0,str(ROOT/'code'))
from gt_numeric import atomic, read, sha, stamp


def main():
    destination=CODE/'local_teacher_rebuild/native_candidates_20261027'
    sources={}
    for path in sorted((ROOT/'code').glob('*.py')):
        sources[Path('code')/path.name]=path
    for name in ('launch_local_owners.py','restart_waiting_local_owners.py','launch_fixed_tsinghua_gt.py',
                 'snapshot_native_scope.py','test_round1_preparation_owner.py','test_preparation_identity.py','test_queue_transport.py',
                 'NATIVE_SCOPE_INTEGRATION.md','report_teacher_physical_costs.py','report_round1_paired_comparisons.py'):
        sources[Path('engineering')/name]=ROOT/'engineering'/name
    for name in ('fullflow_teacher_owner.py','permanent_dispatch.py','finalize_round1_physical_costs.py',
                 'test_local_fullflow_owner.py','PERMANENT_TEACHER_ALLOCATION.json','PERMANENT_TEACHER_40GPU_ALLOCATION.json'):
        sources[Path('engineering/permanent_fullflow')/name]=ROOT/'engineering/permanent_fullflow'/name
    metadata=OLD/'engineering/local_candidate_recheck_20261027'
    for name in ('fit_library_local.py','recheck_candidates.py','recheck_panel.py','test_fit_library_local.py'):
        sources[Path('metadata_adapter')/name]=metadata/name
    for name in ('LOCAL_CANDIDATES.json','LOCAL_CANDIDATE_RECHECK.json','LOCAL_ORDINARY_PANEL.json',
                 'LOCAL_PANEL_RECHECK.json','LOCAL_FINAL_SPLIT.json','LOCAL_PILOT32.json'):
        sources[Path('inputs')/name]=metadata/name
    sources[Path('evidence/FITTER_IMPLEMENTATION_RECEIPT.json')]=metadata/'FITTER_IMPLEMENTATION_RECEIPT.json'
    gt_tests=OLD/'engineering/local_gt_candidates_20261027'
    for name in ('install_gt.py','test_transport.py','test_all_stage_generators.py','INSTALLATION.json','TRANSPORT_CPU_TESTS.xml'):
        sources[Path('gt_transport_tests')/name]=gt_tests/name
    sources[Path('gt_transport_tests/ALL_STAGE_GENERATORS_COMPLETE.json')]=gt_tests/'ALL_STAGE_GENERATORS_PRIVATE_FIXTURES/COMPLETE.json'
    sources[Path('resources/ROUND1_FIXED_32GPU_WITH_TSINGHUA_ALLOCATION.json')]=Path('/mnt/project/recogdrive-no-routing-round1-results-20261026/resources/ROUND1_FIXED_32GPU_WITH_TSINGHUA_ALLOCATION.json')
    for target,source in sources.items():
        path=destination/target;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,path)
        if sha(path)!=sha(source):raise ValueError('Snapshot source changed during copy')
    records=[{'path':str(target),'active_source':str(source),'sha256':sha(source)}
             for target,source in sorted(sources.items(),key=lambda x:str(x[0]))]
    atomic(destination/'SOURCE_MANIFEST.json',{
        'schema':'nrkd_native_candidate_production_source_v1','created_at':stamp(),
        'teacher_root':str(ROOT),'files':records,'source_bytes_preserved':True,
        'full_original_GT_stages':15,'final_export_required':True,
        'candidate_manifest':{'path':str(ROOT/'inputs/LOCAL_CANDIDATES.json'),
                              'sha256':sha(ROOT/'inputs/LOCAL_CANDIDATES.json')},
        'external_original_fitter':{'path':str(OLD/'code/fit_library.py'),'sha256':sha(OLD/'code/fit_library.py')},
        'original_source_commit':'f85d4e557b12a7d05f5deb8d1640834cacd94d32',
        'student_core_math_modules_changed':False,'formal_teacher_qualification_claimed':False,
        'scientific_completion_claimed':False})
    print(destination/'SOURCE_MANIFEST.json')


if __name__=='__main__':main()
