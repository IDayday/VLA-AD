"""Install candidate-count-generic GT transport without editing original sources."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]
NEW=OLD.parent/'recogdrive_dc8_native_candidates_20261027'

INITIALIZE='''def initialize(args, root):
    from local_gt_initialize import initialize_local
    initialize_local(args, root, configuration_tasks)


'''

def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def install():
    code=NEW/'code';code.mkdir(parents=True,exist_ok=True)
    numeric=(OLD/'code/gt_numeric.py').read_text()
    old="'scope': 'original 1850 Train failure IDs only; no Validation/Test optimization'"
    assert numeric.count(old)==1
    numeric=numeric.replace(old,"'scope': 'locally screened Train failure IDs only; no Validation/Test optimization',\n                         'candidate_count': len(read(self.root / 'manifests/SCENES.json')),\n                         'local_candidates_sha256': read(self.root / 'PROTOCOL.json')['local_candidates']['sha256']")
    (code/'gt_numeric.py').write_text(numeric)
    worker=(OLD/'code/gt_opt_worker.py').read_text()
    block="if len(rows) != 1850 or any(row['population'] != 'train' or row['subset'] != 'exploration' for row in rows.values()):\n        raise ValueError('GT worker must preserve all 1850 original Train failure IDs')"
    assert worker.count(block)==1
    worker=worker.replace(block,"protocol = read(root / 'PROTOCOL.json')\n    if len(rows) != protocol['candidate_count'] or any(row['population'] != 'train' or row['subset'] != 'exploration' for row in rows.values()):\n        raise ValueError('GT worker must preserve the complete verified local candidate manifest')\n    if sha(protocol['local_candidates']['path']) != protocol['local_candidates']['sha256']:\n        raise ValueError('Local candidate manifest changed')")
    needle="    candidates = list((root / 'runs' / stage / 'configuration_tasks').glob('*.json'))"
    assert worker.count(needle)==1
    worker=worker.replace(needle,needle+"\n    pending = root / 'runs' / stage / 'REUSE_PENDING.json'\n    if pending.exists() and not (pending.parent / 'REUSE_IMPORT_COMPLETE.json').exists():\n        blocked = set(read(pending)['blocked_old_intersection_indices'])\n        candidates = [p for p in candidates if int(p.stem.split('__')[0].replace('scene_', '')) not in blocked]")
    (code/'gt_opt_worker.py').write_text(worker)
    controller=(OLD/'code/gt_opt_controller.py').read_text()
    start=controller.index('def initialize(');end=controller.index('def aggregate_receipts(',start)
    controller=controller[:start]+INITIALIZE+controller[end:]
    controller=controller.replace('from gt_numeric import (','from local_gt_reuse import begin_reuse, reuse_completed\nfrom gt_numeric import (',1)
    needle="    tasks = read(root / 'runs' / stage / 'TASKS.json')\n    for scene in tasks:"
    assert controller.count(needle)==1
    controller=controller.replace(needle,"    tasks = read(root / 'runs' / stage / 'TASKS.json')\n    blocked = begin_reuse(root, stage)\n    tasks = sorted(tasks, key=lambda scene: int(scene['index']) in blocked)\n    ready_written = False\n    for scene in tasks:\n        if stage == 'baseline' and int(scene['index']) in blocked and not ready_written:\n            atomic(root / 'BASELINE_LAUNCH_READY.json', {'status': 'BASELINE_NEW_LOCAL_SCENES_LIVE', 'candidate_count': len(read(root / 'manifests/SCENES.json')), 'old_intersection_claims_blocked_until_import': True})\n            print(json.dumps({'event': 'BASELINE_LAUNCH_READY', 'path': str(root / 'BASELINE_LAUNCH_READY.json')}), flush=True)\n            ready_written = True")
    needle="            'generated_at': stamp()})"
    assert controller.count(needle)==1
    controller=controller.replace(needle,needle+"\n    reuse_completed(root, stage)")
    controller=controller.replace("'scope': '1850 original Train failure IDs'","'scope': 'complete locally screened Train failure population'")
    controller=controller.replace("'status': 'ALL1850_LOCAL_GT_POST_PAIRS_EXPORTED'","'status': 'ALL_LOCAL_GT_POST_CANDIDATES_EXPORTED'")
    controller=controller.replace("'train_failures': 1850","'train_failures': len(read(root / 'manifests/SCENES.json')), 'candidate_count': len(read(root / 'manifests/SCENES.json')), 'local_candidates': read(root / 'PROTOCOL.json')['local_candidates'], 'full_original_finite_search_complete': True")
    needle="    parser.add_argument('--train-manifest')"
    controller=controller.replace(needle,needle+"\n    parser.add_argument('--candidates', default=str(Path(__file__).resolve().parents[1] / 'inputs/LOCAL_CANDIDATES.json'))\n    parser.add_argument('--base-root', default=str(Path(__file__).resolve().parents[1] / 'native_base'))")
    (code/'gt_opt_controller.py').write_text(controller)
    shutil.copyfile(OLD/'code/gt_controller_fast.py',code/'gt_controller_fast.py')
    for name in ('local_gt_reuse.py','local_gt_initialize.py'):shutil.copyfile(HERE/name,code/name)
    for name in ('gt_opt_controller.py','gt_opt_worker.py','gt_numeric.py','gt_controller_fast.py','local_gt_reuse.py','local_gt_initialize.py'):
        compile((code/name).read_text(),str(code/name),'exec')
    # Every numerical method other than constructor metadata is byte-for-byte
    # original source; original archived kernels are never copied/modified.
    old_tree=ast.parse((OLD/'code/gt_numeric.py').read_text());new_tree=ast.parse(numeric)
    old_class=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalGTNumeric')
    new_class=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='LocalGTNumeric')
    names=[]
    for left,right in zip(old_class.body,new_class.body):
        if isinstance(left,ast.FunctionDef) and left.name!='__init__':
            assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False),left.name
            names.append(left.name)
    receipt={'status':'GENERIC_LOCAL_CANDIDATE_GT_TRANSPORT_INSTALLED','root':str(NEW),
             'sources':{n:ref(code/n) for n in ('gt_opt_controller.py','gt_opt_worker.py','gt_numeric.py','gt_controller_fast.py','local_gt_reuse.py','local_gt_initialize.py')},
             'unchanged_numeric_methods':names,'original_archived_math_sources_modified':False,
             'installer':ref(Path(__file__))}
    (HERE/'INSTALLATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':install()
