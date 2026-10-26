"""CPU-only private fixtures for every original GT stage transition.

No live task/controller/source/cache is changed. Scores below are synthetic
branch-control fixtures, never reported as measured driving results.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import gt_opt_controller as controller
from gt_numeric import (atomic, bind_function, descriptor, digest, read, sha,
                        stamp, STAGES, G5)


def fixture_root(root, source):
    rows = []
    cases = []
    baseline = []
    exceptions = []
    scenarios = [(0.4, .7, 95.), (.95, .4, 95.), (.4, .95, 95.),
                 (.8, .7, 80.), (.95, .95, 95.), (.9, .9, 95.), (.9, .95, 95.), (.95, .9, 95.)]
    for index, (pre_score, post_score, gt_score) in enumerate(scenarios, 1):
        token = f'private_fixture_{index:03d}'
        rows.append({'global_index': index, 'token': token, 'legacy_best': None, 'legacy_configurations': {}})
        perblock = {}
        results = []
        configs = []
        for block, score in [('interface_pre', pre_score), ('interface_post', post_score)]:
            config = {'id': block + '_0.1', 'block': block, 'cap': .1, 'steps': 24, 'mode': 'shared', 'lr_multiplier': 1.}
            configs.append(config)
            folder = root / 'analysis/baseline' / f'scene_{index:06d}' / config['id']
            folder.mkdir(parents=True, exist_ok=True)
            arrays = folder / 'CANDIDATES.npz'
            with arrays.open('wb') as stream:
                np.savez(stream, delta=np.zeros((2, 384), np.float32), candidates=np.zeros((2, 2, 384), np.float32))
            records = [{'iteration': 0, 'surrogate_loss': .3, 'score': {'score': score}, 'trajectory': np.zeros((8, 3)).tolist()},
                       {'iteration': 1, 'surrogate_loss': .01, 'score': {'score': max(0., score - .1)}, 'trajectory': np.zeros((8, 3)).tolist()}]
            result = {'block': block, 'config': config, 'arrays': {'path': str(arrays.relative_to(root)), 'sha256': sha(arrays)},
                      'records': records, 'selected_index': 0, 'transfer': [], 'synthetic_only': True}
            filename = folder / 'RESULT.json'; atomic(filename, result)
            perblock[block] = descriptor(root, filename)
            results.append({'path': str(filename.relative_to(root)), 'sha256': sha(filename)})
        selection = perblock['interface_pre'] if pre_score >= post_score else perblock['interface_post']
        cases.append({'index': index, 'token': token, 'subset': 'exploration', 'base': .2,
                      'selection': selection, 'perblock': perblock})
        scene = {'id': f'scene_{index:06d}', 'kind': 'baseline', 'index': index, 'configs': configs}
        baseline.append(scene)
        atomic(root / 'runs/baseline/receipts' / (scene['id'] + '.json'), {'result': {'results': results}})
        exceptions.append({'index': index, 'GT_PDMS': gt_score})
    atomic(root / 'manifests/SCENES.json', rows)
    atomic(root / 'analysis/summary_baseline/SUMMARY.json', {'cases': cases, 'synthetic_only': True})
    atomic(root / 'analysis/GT_exception_interim.json', {'cases': exceptions})
    atomic(root / 'runs/baseline/TASKS.json', baseline)
    atomic(root / 'runs/QUEUE.json', {'stages': ['baseline'], 'active': True})
    return scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT / 'engineering/GT_STAGE_GENERATOR_PREFLIGHT')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'source_snapshot/ReCogDrive')
    args = parser.parse_args()
    run = args.root / ('fixture_' + str(time.time_ns()))
    scenarios = fixture_root(run, args.source_root)
    source = args.source_root / G5 / 'g5_next_stage.py'
    common = {'ROOT': run, 'Path': Path, 'read': read, 'sha': sha, 'atomic': atomic, 'json': json, 'B2': run}
    _, expected_main = bind_function(source, 'main', dict(common, minimum_original_GT_loss=lambda row: None))
    _, expected_minimum = bind_function(source, 'minimum_original_GT_loss', common)
    stages = []
    previous = 'baseline'
    for stage in STAGES[1:]:
        controller.generate_next(run, args.source_root, stage, previous)
        bindings = read(run / 'runs' / stage / 'ORIGINAL_GENERATOR_BINDINGS.json')
        if bindings['main'] != expected_main or bindings['minimum_original_GT_loss'] != expected_minimum:
            raise AssertionError('Production generator code binding changed')
        tasks = read(run / 'runs' / stage / 'TASKS.json')
        seen = set()
        for task in tasks:
            for config in task['configs']:
                assert config['block'] in ('interface_pre', 'interface_post')
                assert config['mode'] in ('shared', 'image_text')
                assert config['steps'] in (64, 128, 256)
                assert config['lr_multiplier'] in (.5, 1., 2.)
                key = (task['index'], digest({key: value for key, value in config.items() if key != 'id'}))
                assert key not in seen, 'Same scene/config duplicated in a stage'
                seen.add(key)
                if stage == 'pre_hard_gt':
                    assert config['initial']['candidate_index'] == 1
                    assert config['initial']['criterion'].startswith('Minimum original-GT surrogate')
                    assert task['index'] != 4, 'GT<=90 fixture incorrectly entered hard-GT retry'
                    assert config['loss_variant']['name'] in ('original', 'xy_tight', 'early_xy_tight', 'motion_matched')
                if stage.startswith('post_'):
                    pre, post, _ = scenarios[task['index'] - 1]
                    assert pre > .9 and post <= .9
                if stage == 'post_from_pre':
                    assert 'project_pre' in config and config['steps'] == 64
                if stage == 'pre_lift':
                    assert 'lift_post' in config
                if stage in ('pre_progress', 'post_progress'):
                    assert config['target_variant']['kind'] in ('identity', 'distance_scale', 'tangent_progress')
        configuration_files = list((run / 'runs' / stage / 'configuration_tasks').glob('*.json'))
        assert len(configuration_files) == sum(len(task['configs']) for task in tasks)
        # Keep the original synthetic scenario states unchanged across stages:
        # this deliberately exercises exact history de-duplication and empty
        # stages when previous configurations failed to improve their score.
        summary = copy.deepcopy(read(run / 'analysis' / ('summary_' + previous) / 'SUMMARY.json'))
        atomic(run / 'analysis' / ('summary_' + stage) / 'SUMMARY.json', summary)
        stages.append({'stage': stage, 'scenes': len(tasks), 'configurations': len(configuration_files),
                       'bindings': bindings, 'tasks_sha256': sha(run / 'runs' / stage / 'TASKS.json')})
        previous = stage
    # The live controller wraps original export selections into configuration
    # tasks. Exercise that special planning branch without any GPU or real GT.
    tasks = [{'id': f"scene_{case['index']:06d}", 'kind': 'export_post', 'index': case['index'],
              'configs': [case['perblock']['interface_post']]} for case in read(run / 'analysis/summary_post_progress/SUMMARY.json')['cases']]
    atomic(run / 'runs/export_post/TASKS.json', tasks)
    controller.configuration_tasks(run, 'export_post')
    assert len(list((run / 'runs/export_post/configuration_tasks').glob('*.json'))) == len(scenarios)
    # Check the original source's three export planning APIs too. They carry
    # selections outside config[]; transport adaptation happens only above.
    exports = []
    for name in ('export', 'export_pre', 'export_post'):
        target = run / ('original_' + name)
        fixture_root(target, args.source_root)
        controller.generate_next(target, args.source_root, name, 'baseline')
        exported = read(target / 'runs' / name / 'TASKS.json')
        assert len(exported) == len(scenarios) and all(task['kind'] == 'export' and 'selection' in task for task in exported)
        exports.append({'name': name, 'scenes': len(exported), 'tasks_sha256': sha(target / 'runs' / name / 'TASKS.json')})
    value = {'status': 'PASS_ALL_ORIGINAL_GT_STAGE_GENERATORS_PRIVATE_CPU_FIXTURES', 'time': stamp(),
             'fixture_root': str(run), 'synthetic_only': True, 'live_ROOT_mutated': False,
             'GPU_model_or_scoring_calls': 0, 'all_14_transitions': stages, 'export_planning': exports,
             'minimum_original_GT_loss_uses_actual_state_candidate_index': True,
             'original_main_binding': expected_main, 'original_minimum_binding': expected_minimum,
             'source_sha256': sha(source), 'controller_source_sha256': sha(Path(controller.__file__)),
             'no_live_source_hot_edit': True}
    atomic(args.root / 'COMPLETE.json', value)
    print(json.dumps({'status': value['status'], 'stages': [{key: row[key] for key in ('stage', 'scenes', 'configurations')} for row in stages],
                      'export_planning': exports, 'evidence': str(args.root / 'COMPLETE.json')}), flush=True)


if __name__ == '__main__':
    main()
