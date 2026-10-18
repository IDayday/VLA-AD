"""Content identities for the actual selected sensors, logs and metric caches."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from .common import *


def main():
    cfg,identity=protocol();rows=[r for s in ['D_fit','D_cal','D_audit'] for r in scene_rows(s)]
    paths={}
    for row in rows:
        for key in ['metadata_path','front_image_path','metric_cache_path']:
            paths.setdefault(row[key],set()).add(key)
    def inspect(item):
        path,roles=item;st=Path(path).stat()
        return path,dict(sha256=sha(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,roles=sorted(roles))
    with ThreadPoolExecutor(8) as pool:values=dict(pool.map(inspect,paths.items()))
    lock_time=datetime.fromisoformat(cfg['locked_utc']).timestamp()
    late=[p for p,v in values.items() if v['mtime_ns']/1e9>lock_time]
    atomic_json(ROOT/'manifests/input_content_identity.json',dict(protocol_sha256=identity,files=values,
        collected_after_run=True,files_modified_since_protocol_lock=late,
        note='Content hashes supplement the locked membership manifests. Historical mtimes are checked; labels still retain input-trajectory and protocol hashes.'))
    print('INPUT_IDENTITIES',len(values),'modified_since_lock',len(late))
    if late:raise RuntimeError('Selected input changed since protocol lock; investigate affected labels')


if __name__=='__main__':main()
