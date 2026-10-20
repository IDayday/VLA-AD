"""Lock explicit, already developed candidates before the final full Navtest.

This script does not rank candidates or read Navtest scores. The resulting
small JSON must be committed before evaluate.py admits any Navtest worker.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse

from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions
from scripts.process_improvement.calibrate import immutable


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidates-file', required=True,
                   help='JSON list: name, checkpoint, optional dev_name and screening')
    p.add_argument('--calibration-locks', nargs='+', required=True)
    a = p.parse_args()
    cfg = config()
    assert sha(ROOT/'protocol.yaml') == read(ROOT/'protocol.lock.json')['sha256']
    if len(membership('navtest')) != 12146:
        raise ValueError('Final Navtest must use all 12146 locked scenes')
    calibration = {}
    for name in a.calibration_locks:
        path = ROOT/'selection'/name
        if not path.is_file():
            raise ValueError('Unfinished calibration prerequisite: '+name)
        calibration[name] = dict(sha256=sha(path), decision=read(path))
    candidates = []
    names = set()
    for item in read(a.candidates_file):
        name = item['name']
        if name in names:
            raise ValueError('Duplicate candidate name: '+name)
        names.add(name)
        path = Path(item['checkpoint']).resolve(strict=True)
        dev_name = item.get('dev_name', name)
        # Completed dev artifacts are checked, never used for automated ranking.
        values, rows = predictions(dev_name, 'dev')
        meta = read(ROOT/'evaluation/dev'/dev_name/'config_0.json')['spec']
        if meta['checkpoint_sha256'] != sha(path):
            raise ValueError('Dev policy does not match final checkpoint: '+name)
        screening = bool(item.get('screening', False))
        seeds = cfg['screening_seeds' if screening else 'evaluation_seeds']
        if meta['evaluation_seeds'] != seeds or bool(meta['screening']) != screening:
            raise ValueError('Dev and final inference role/seed mismatch: '+name)
        candidates.append(dict(name=name, checkpoint=str(path), checkpoint_sha256=sha(path),
            seeds=seeds, screening=screening, dev_name=dev_name,
            dev_scenes=len(rows), dev_evaluation_identity=digest(meta)))
    if not candidates:
        raise ValueError('Final evaluation cannot lock an empty experiment')
    value = dict(status='LOCKED', study=STUDY, base_commit=BASE,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        protocol_sha256=sha(ROOT/'protocol.yaml'),
        navtest_manifest_sha256=sha(ROOT/'manifests/navtest.json'),
        navtest_scenes=12146, navtest_previously_viewed=True,
        calibration_locks=calibration, candidates=candidates,
        inference='native stochastic DDIM5; one trajectory per scene/seed; no selection',
        candidate_selection='Explicit developed candidates; no Navtest outcome read by this script')
    committed = SOURCE/'configs/process_improvement/final_eval_lock.json'
    immutable(committed, value)
    immutable(ROOT/'selection/final.lock.json', value)
    print('FINAL_CONFIG_LOCKED_COMMIT_REQUIRED', str(committed), sha(committed), flush=True)


if __name__ == '__main__':
    main()
