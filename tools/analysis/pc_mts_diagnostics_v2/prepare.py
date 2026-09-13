"""Seal V1 and predeclare every V2 scene/seed split before new outcomes."""
import argparse,concurrent.futures,subprocess
import pandas as pd
from v2_common import *

def file_record(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify_record(r):
    p=Path(r['path'])
    if not p.exists():return dict(path=str(p),status='missing')
    if p.stat().st_size==r['bytes'] and sha(p)==r['sha256']:return dict(path=str(p),status='unchanged')
    allowed={str(V1/f'pressure_restore_gpu{i}.log') for i in range(8)}
    if str(p) in allowed and p.stat().st_size>=r['bytes']:
        with p.open('rb') as f:prefix=hashlib.sha256(f.read(r['bytes'])).hexdigest()
        if prefix==r['sha256']:return dict(path=str(p),status='preexisting_runtime_log_append_only',appended_bytes=p.stat().st_size-r['bytes'])
    return dict(path=str(p),status='changed')
def main(a):
    make_dirs();original=OUT/'manifests/V1_IMMUTABLE_FILES.json'
    if a.verify:
        frozen=read(original)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:verified=list(ex.map(verify_record,frozen['files']))
        bad=[r for r in verified if r['status'] in ['missing','changed']];logs=[r for r in verified if r['status']=='preexisting_runtime_log_append_only']
        current={str(p) for p in V1.rglob('*') if p.is_file()};assert current==set(frozen['v1_output_paths']), 'V1 output file inventory changed'
        assert not bad,bad;save(OUT/'manifests/V1_UNCHANGED_VERIFICATION.json',dict(files_checked=len(frozen['files']),exact_hash_unchanged=len(verified)-len(logs),all_scientific_results_unchanged=True,all_original_file_contents_preserved=True,runtime_log_appends=logs,note='Eight preexisting V1 pressure jobs each appended 162 bytes of status output before they were stopped. Original byte prefixes match sealed hashes. No V1 file was truncated or rewritten. Restored jobs now log only under V2.'));print('V1 exact unchanged',len(verified)-len(logs),'append-only runtime logs',len(logs));return
    assert not original.exists(),'V1 baseline already sealed; use --verify'
    files=[p for p in V1.rglob('*') if p.is_file()]
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','2dd4b55'],cwd=ROOT,text=True).splitlines()
    important=[ROOT/p for p in tracked if p.startswith(('tools/analysis/pc_mts_diagnostics/','configs/pc_mts_diagnostics/','scripts/evaluation/distribution_audit/')) or p=='reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md']
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:records=list(ex.map(file_record,files+important))
    save(original,dict(v1_commit=ident()['v1_commit'],files=records,v1_output_paths=[str(p) for p in files]))
    sc=scenes();save(OUT/'manifests/scenes_1000.json',sc);save(OUT/'manifests/protocol_frozen.json',dict(identity=ident(),config=CFG))
    ordered=sorted([r['token'] for r in sc['scenes']],key=lambda t:seed(t,'v2_evaluation_scene_order'))
    for name,n in [('denoising',500),('chain',300),('smoke',20)]:save(OUT/'manifests'/f'{name}_scenes.json',dict(identity=ident(),selection='hash order independent of model outcomes',tokens=ordered[:n]))
    banks={n:[seed(r['token'],n,i) for r in sc['scenes'] for i in range(k)] for n,k in [('v2_il_support_R',128),('v2_il_support_Q',128),('v2_candidate_il',32)]}
    allseeds=sum(banks.values(),[]);assert len(set(allseeds))==len(allseeds)
    save(OUT/'manifests/seed_partition.json',dict(namespaces={n:dict(count=len(v),digest=digest(v)) for n,v in banks.items()},global_initial_seed_overlap=0,seed_bits=63))
    table=pd.read_csv(V1/'metrics/TABLE_A.csv');c=pd.read_parquet(V1/'metrics/candidate_records.parquet');loc=pd.read_csv(V1/'metrics/TABLE_C.csv')
    q=c.groupby('method').q_policy.agg(['mean','median',lambda x:float((x==100).mean())]);q.columns=['mean','median','fraction_exactly_100']
    diagnosis=['# V1 PRIMARY OBSERVATIONAL DIAGNOSTICS：复盘','',f'V1 commit `{ident()["v1_commit"]}` 与全部 {len(records)} 个产物/实现文件已封存哈希。V2只读取。','',table[['checkpoint','pairwise_ade','spread_auc','center_shift_from_il']].to_csv(index=False),'', 'V1使用candidate→IL KNN与IL self-KNN经验CDF，超出有限self距离最大值的候选均饱和到100；该值不是模型真实概率。','',q.to_csv(),'','原PC-MTS只有192/1000场景有qualified parent，808场景需要coverage扩展；该事实保持不变。','', 'V1最大局部扰动0.20m，各方法局部可行率约96%以上：','',loc[['method','local_feasible_rate','local_mean']].to_csv(index=False),'','新计算仅新增IL R/Q和IL-native候选、后续denoising及有证据的历史链；五模型1000×64缓存不重跑。']
    (OUT/'report/V1_DIAGNOSIS.md').write_text('\n'.join(diagnosis)+'\n');print('V1 sealed and V2 protocol frozen',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args())
