"""Freeze all choices before A; seal every preexisting V1/V2 file, including byte prefixes of live logs."""
import concurrent.futures, subprocess, argparse
from common_v3 import *
def record(p):
    size=p.stat().st_size
    with p.open('rb') as f:
        h=hashlib.sha256();remaining=size
        while remaining:
            b=f.read(min(8*1024**2,remaining));assert b;h.update(b);remaining-=len(b)
    return dict(path=str(p.relative_to(ROOT)),bytes=size,sha256=h.hexdigest(),append_only_runtime=p.suffix=='.log')
def main(verify=False):
    p=OUT/'manifests/V1_V2_IMMUTABLE_FILES.json'
    if verify:
        frozen=read(p);errors=[];appends=[]
        def check(r):
            path=ROOT/r['path'];current=path.stat().st_size
            with path.open('rb') as f:
                h=hashlib.sha256();remaining=r['bytes']
                while remaining:
                    b=f.read(min(8*1024**2,remaining))
                    if not b: break
                    h.update(b);remaining-=len(b)
            okay=remaining==0 and h.hexdigest()==r['sha256'] and (current==r['bytes'] or r['append_only_runtime'] and current>=r['bytes'])
            return r['path'],okay,current-r['bytes']
        with concurrent.futures.ThreadPoolExecutor(16) as ex:
            for path,okay,extra in ex.map(check,frozen['files']):
                if not okay:errors.append(path)
                if extra:appends.append(dict(path=path,bytes=extra))
        assert not errors,errors
        stage_audit('immutability',files=len(frozen['files']),errors=errors,preexisting_runtime_log_appends=appends,all_original_byte_prefixes_preserved=True)
        print('Immutability PASS',len(frozen['files']),flush=True);return
    assert not p.exists(),'Already frozen; use --verify'
    files=set()
    for root in [V1,V2]:files.update(x for x in root.rglob('*') if x.is_file())
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',CFG['base_commit']],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        if any(s in name for s in ['pc_mts_diagnostics/','pc_mts_diagnostics_v2/','PC_MTS_POLICY_','scripts/evaluation/distribution_audit/']):files.add(ROOT/name)
    with concurrent.futures.ThreadPoolExecutor(16) as ex:records=list(ex.map(record,sorted(files)))
    save(p,dict(base_commit=CFG['base_commit'],files=records))
    ordered=sorted([r['token'] for r in scenes()],key=lambda t:seed(t,'v3_scene_split'))
    gradient=sorted(ordered,key=lambda t:seed(t,'v3_gradient_scenes'))[:256]
    save(OUT/'manifests/splits.json',dict(train=ordered[:700],holdout=ordered[700:],gradient=gradient,all=ordered,split_is_new_update_holdout_only=True,note='Navtrain scenes may already occur in historical official/external model training; this is not Navtest or an independently unseen benchmark.'))
    save(OUT/'manifests/protocol_frozen.json',dict(identity=identity(),yaml_sha256=sha(CONFIG_PATH),config=CFG,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_v3_outcomes_exist=not list((OUT/'metrics').glob('*'))))
    print('FROZEN',sha(CONFIG_PATH),'sealed files',len(records),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');main(a.parse_args().verify)
