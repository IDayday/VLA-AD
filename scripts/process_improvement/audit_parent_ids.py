"""Check actual metadata log_token/scene_token identifiers across whole-record splits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import pickle
from scripts.process_improvement.common import *


def inspect(path):
    frames=pickle.load(open(path,'rb'))
    return dict(path=path,parent=Path(path).stem.rsplit('_',2)[0],
                log_tokens=sorted({f['log_token'] for f in frames}),scene_tokens=sorted({f['scene_token'] for f in frames}))


def main():
    purposes=read(ROOT/'manifests/parent_splits.json')
    purpose_by_parent={parent:name for name,parents in purposes.items() for parent in parents}
    versions=read(ROOT/'manifests/source_versions.json')
    # Test identifiers are audited without reading any test predictions/rewards.
    tests=membership('navtest')
    for row in tests:
        previous=purpose_by_parent.setdefault(row['driver_record'],'navtest')
        assert previous=='navtest','Test parent overlaps a development purpose'
    versions=versions+[dict(path=path) for path in sorted({r['metadata_path'] for r in tests})]
    log_to_parents=defaultdict(set);scene_to_purposes=defaultdict(set);log_to_purposes=defaultdict(set)
    records=[]
    with ProcessPoolExecutor(4) as pool:
        for item in pool.map(inspect,[v['path'] for v in versions],chunksize=1):
            records.append(item);purpose=purpose_by_parent[item['parent']]
            for token in item['log_tokens']:log_to_parents[token].add(item['parent']);log_to_purposes[token].add(purpose)
            for token in item['scene_tokens']:scene_to_purposes[token].add(purpose)
    conflicts=dict(log_tokens={k:sorted(v) for k,v in log_to_purposes.items() if len(v)>1},
                   scene_tokens={k:sorted(v) for k,v in scene_to_purposes.items() if len(v)>1})
    write(ROOT/'audit/parent_identifiers_including_navtest.json',dict(status='PASS' if not any(conflicts.values()) else 'INVALID',
          raw_log_tokens=len(log_to_parents),raw_scene_tokens=len(scene_to_purposes),
          prefixes_with_shared_actual_log_id={k:sorted(v) for k,v in log_to_parents.items() if len(v)>1},
          conflicts=conflicts,records=records))
    assert not any(conflicts.values()),'Higher-level driving identifier crosses purposes'
    print('PARENT_IDENTIFIER_ISOLATION_PASS',len(log_to_parents),len(scene_to_purposes),flush=True)


if __name__=='__main__':main()
