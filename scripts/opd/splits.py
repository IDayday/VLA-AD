"""Small versioned scene indexes; raw logs, sensors and metric caches stay external."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *

BUNDLE=SOURCE/'configs/opd/splits'
INPUT_KEYS=['token','log_name','metadata_path','start_idx','frame_count','sensor_root','split','observation_seconds','metric_cache_path']

def population(split):
    data=read(BUNDLE/f'{split}.json');lookup={r['log_name']:r for r in data['logs']};result=[]
    for token,log_name,start,cache in data['scenes']:
        context=lookup[log_name]
        result.append(dict(token=token,log_name=log_name,metadata_path=context['metadata_path'],sensor_root=context['sensor_root'],
            driver_record=context['driver_record'],start_idx=start,frame_count=data['frame_count'],split=data['official_split'],
            observation_seconds=data['observation_seconds'],metric_cache_path=cache))
    return result

def export():
    for split in ['train','cal','dev','navtest']:
        data=rows(split);logs={}
        for r in data:
            context=dict(log_name=r['log_name'],metadata_path=r['metadata_path'],sensor_root=r['sensor_root'],
                driver_record=r.get('driver_record',r['log_name'].rsplit('_',2)[0]))
            if r['log_name'] in logs:assert logs[r['log_name']]==context
            logs[r['log_name']]=context
        assert len({r['frame_count'] for r in data})==len({r['split'] for r in data})==len({r['observation_seconds'] for r in data})==1
        value=dict(version=1,source_manifest_sha256=sha(ROOT/'manifests'/f'{split}.json'),
            official_split=data[0]['split'],frame_count=data[0]['frame_count'],observation_seconds=data[0]['observation_seconds'],
            logs=list(logs.values()),scene_fields=['token','log_name','start_idx','metric_cache_path'],
            scenes=[[r['token'],r['log_name'],r['start_idx'],r['metric_cache_path']] for r in data])
        path=BUNDLE/f'{split}.json'
        if path.exists():assert read(path)==value,'Bundled population is immutable'
        else:write(path,value)
    p=BUNDLE/'source_versions.json';value=read(ROOT/'audit/final_provenance.json')['source_logs']
    if p.exists():assert read(p)==value
    else:write(p,value)
    verify()

def verify():
    counts={}
    for split in ['train','cal','dev','navtest']:
        reconstructed=population(split);actual=rows(split)
        assert len(actual)==len(reconstructed)
        assert all(all(a[k]==b[k] for k in INPUT_KEYS) for a,b in zip(actual,reconstructed)),split
        assert all(a.get('driver_record',a['log_name'].rsplit('_',2)[0])==b['driver_record'] for a,b in zip(actual,reconstructed))
        counts[split]=len(actual)
    write(ROOT/'tests/real_split_bundle.json',dict(passed=True,counts=counts,
        exact_order_and_all_used_inputs=True,excluded='Only unused descriptive fields and run-specific observation-cache outputs',
        source_version_manifest_sha256=sha(BUNDLE/'source_versions.json')))
    print('BUNDLED_SCENE_INDEX_PARITY',counts)

def verify_versions():
    from concurrent.futures import ThreadPoolExecutor
    def check(row):
        path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],str(path)
    with ThreadPoolExecutor(4) as pool:list(pool.map(check,read(BUNDLE/'source_versions.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','verify','verify_versions']);a=p.parse_args();globals()[a.action]()
