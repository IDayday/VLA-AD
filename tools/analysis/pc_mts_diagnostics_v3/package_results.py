"""Stage a bounded, auditable V3 publication; never add weights or rollout caches."""
import argparse
import subprocess
from common_v3 import *

REPORT = ROOT / 'reports/PC_MTS_POLICY_DIAGNOSTICS_V3_20260913.md'
LOCAL_ONLY = {'V1_V2_IMMUTABLE_FILES.json', 'CACHE_INDEX.json',
              'active_process.json', 'ROOT_RESUME_STATUS.json',
              'READY_FOR_REVIEW.json', 'GPU_PRESSURE_RESTORED.json'}

def allowed(path):
    rel = path.relative_to(ROOT).as_posix()
    return (rel.startswith('tools/analysis/pc_mts_diagnostics_v3/') or
            rel.startswith('configs/pc_mts_diagnostics_v3/') or
            rel.startswith('outputs/pc_mts_diagnostics_v3/') or
            path == REPORT)

def main(stage=False, interim=False):
    if interim:
        manifest_name='interim_report_identity.json'
        assert read(OUT/'manifests'/manifest_name)['final_report_pending']
    else:
        manifest_name='report_identity.json'
        assert read(OUT / 'manifests/READY_FOR_REVIEW.json')['all_experiments_and_audits_complete']
    assert read(OUT / 'manifests'/manifest_name)['sha256'] == sha(REPORT)
    identity()
    paths = [REPORT]
    for directory in [ROOT / 'tools/analysis/pc_mts_diagnostics_v3',
                      ROOT / 'configs/pc_mts_diagnostics_v3']:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    omitted = []
    for directory in ['metrics', 'figures', 'manifests', 'report']:
        for p in (OUT / directory).rglob('*'):
            if not p.is_file() or p.name == 'PUBLICATION_INDEX.json':
                continue
            record = dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                          sha256=sha(p))
            if p.name in LOCAL_ONLY or p.stat().st_size > 20_000_000:
                omitted.append(record)
            else:
                paths.append(p)
    paths = sorted(set(paths))
    assert all(allowed(p) and p.suffix not in ['.pt', '.ckpt', '.npz'] for p in paths)
    records = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                    sha256=sha(p)) for p in paths]
    manifest = OUT / 'manifests/PUBLICATION_INDEX.json'
    save(manifest, dict(identity=identity(), interim=interim, files=records,
                        published_bytes=sum(r['bytes'] for r in records),
                        excluded_small_file_inventory=omitted,
                        local_only_roots=['cache', 'checkpoints', 'logs'],
                        large_artifact_hash_index='manifests/CACHE_INDEX.json',
                        publication_rule='All V3 code/config/report/figures and scientific tables <=20MB each; no outcome filtering.'))
    if stage:
        prior = subprocess.check_output(['git', 'diff', '--cached', '--name-only'],
                                        cwd=ROOT, text=True).splitlines()
        assert all(allowed(ROOT / p) for p in prior), prior
        subprocess.run(['git', 'add', '-f', '--'] +
                       [str(p.relative_to(ROOT)) for p in paths + [manifest]],
                       cwd=ROOT, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT, check=True)
        staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'],
                                         cwd=ROOT, text=True).splitlines()
        assert all(allowed(ROOT / p) for p in staged), staged
    print(json.dumps(dict(files=len(paths), bytes=sum(r['bytes'] for r in records),
                          staged=stage, manifest=str(manifest)), indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--interim', action='store_true')
    args=parser.parse_args()
    main(args.stage, args.interim)
