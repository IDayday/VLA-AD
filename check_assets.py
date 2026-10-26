"""Check caller-supplied checkpoint/VLM paths without downloads or installs."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json
ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def check(paths,full=True):
    local=json.loads(Path(paths).read_text());cfg=json.loads((ROOT/'configs/repair_final_baseline.json').read_text())
    if not Path(local['checkpoint']).is_file():raise FileNotFoundError(local['checkpoint'])
    if full and sha(local['checkpoint'])!=cfg['checkpoint_sha256']:raise AssertionError('Different RL Base checkpoint: old choices are not calibrated for these weights')
    for name,expected in cfg['vlm_sha256'].items():
        p=Path(local['vlm'])/name
        if name in ('all_results.json','train_results.json','trainer_state.json') and not p.exists():continue
        if not p.is_file():raise FileNotFoundError(p)
        if (full or p.stat().st_size<10_000_000) and sha(p)!=expected:raise AssertionError(f'Different VLM file: {name}')
    versions={}
    reference=json.loads((ROOT/'reference/environment.json').read_text())
    for name in reference:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='MISSING'
    result=dict(status='PASS_ASSET_IDENTITIES',large_weights_hashed=full,versions=versions,reference_versions=reference,
        version_differences=[k for k in reference if reference[k]!=versions[k]],
        dataset_roots_exist={k:Path(v).exists() for k,v in local.items() if k not in ('checkpoint','vlm')})
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--fast',action='store_true');a=p.parse_args();check(a.paths,not a.fast)
