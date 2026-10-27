"""CPU tests for the explicitly Test-selected protocol; no real GPU/model job."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import test_pdms_selector as selector


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.control = self.root / 'control'
        self.control.mkdir()
        selector.atomic_json(self.control / 'REQUEST.json', {'user_instruction': '使用测试集的PDMS结果'})
        self.noise = self.root / 'noise.json'
        entry = {'path': str(self.root / 'noise.npy'), 'sha256': 'fixed-noise-bytes',
                 'tokens_path': str(self.root / 'noise-tokens.json'), 'tokens_sha256': 'fixed-tokens',
                 'noise_indices': list(range(10)), 'source': 'explicit reference'}
        selector.atomic_json(self.noise, {'schema': 'native_explicit_evaluation_noise_v1',
                                         'populations': {'test': entry}})
        self.identities = {'qualification': {'execution': {'world_size': 4}},
                           'protocol_sha256': 'unchanged-training-protocol', 'files': {}}
        for name in ('feature_manifest', 'teacher_bundle', 'targets_manifest', 'calibration'):
            p = self.root / (name + '.json')
            p.write_text('{}\n')
            self.identities['files'][name] = {'path': str(p), 'sha256': selector.sha256(p)}
        selector.atomic_json(self.root / 'EXPERIMENT_REGISTRY.json',
                             {'runs': {name: {} for name in selector.RUNS}})
        self.scores = {8: 90., 16: 96., 24: 96., 32: 95., 40: 94.}
        for name in selector.RUNS:
            folder = self.root / name
            folder.mkdir()
            selector.atomic_json(folder / 'ASSET_IDENTITIES.json', self.identities)
            selector.atomic_json(folder / 'TRAIN_COMPLETE.json', {
                'status': 'TRAIN_COMPLETE', 'global_step': 53640,
                'real_scenes_seen': 3432520, 'epoch_index': 40,
                'qualification': self.identities['qualification']})
            for epoch in selector.ENDPOINTS:
                self.write_checkpoint(folder, epoch)
                ev = folder / 'test' / f'epoch_{epoch:03d}'
                ev.mkdir(parents=True)
                for artifact in ('SUMMARY.json', 'PROTOCOL.json', 'COMPLETE.json'):
                    selector.atomic_json(ev / artifact, {'fixture': True, 'epoch': epoch})
                # Deliberately opposite Validation ranking. Selection must not read it.
                old = folder / 'validation' / f'epoch_{epoch:03d}'
                old.mkdir(parents=True)
                selector.atomic_json(old / 'SUMMARY.json', {'behavior': {'pdms_points': 100 - epoch}})

    def write_checkpoint(self, folder, epoch, progress=None, identities=None):
        p = folder / 'checkpoints' / f'epoch_{epoch:03d}.pt'
        p.parent.mkdir(exist_ok=True)
        state = {'schema': 1, 'progress': progress or {'global_step': 1341 * epoch,
            'real_scenes_seen': 85813 * epoch, 'epoch_index': epoch, 'global_batch_cursor': 0},
            'world_size': 4, 'rng_by_rank': [{}, {}, {}, {}],
            'identities': self.identities if identities is None else identities,
            'config': {'paths': {'evaluation_noise_manifest': str(self.noise)}}}
        torch.save(state, p)
        selector.atomic_json(str(p) + '.COMPLETE.json', {
            'status': 'COMPLETE', 'sha256': selector.sha256(p),
            'global_step': state['progress']['global_step'], 'progress': state['progress']})
        return p

    def fake_verify(self, folder, expected):
        folder = Path(folder)
        assert folder.parent.name == 'test'
        assert expected['population'] == 'test' and expected['scene_count'] == 12146
        assert expected['noise_indices'] == [0, 1, 2, 3, 4]
        assert expected['full_population'] is True
        assert expected['mode'] == 'student' and expected['artifact_kind'] == 'behavior'
        assert expected['noise_identity']['seed_fallback'] is False
        return {'behavior': {'pdms_points': self.scores[int(folder.name.split('_')[1])]}}


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nrkd-Test-selector-')
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)
        self.root_patch = patch.object(selector, 'ROOT', self.f.control)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.run = self.f.root / selector.RUNS[0]

    def test_endpoint_before_training_complete(self):
        (self.run / 'TRAIN_COMPLETE.json').unlink()
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            value = selector.endpoint_candidate(self.run, 8)
        self.assertEqual(value['epoch'], 8)
        self.assertEqual(value['pdms_points'], 90.)
        self.assertFalse((self.run / 'checkpoints' / selector.BEST_NAME).exists())

    def test_partial_checkpoint_is_rejected(self):
        self.f.write_checkpoint(self.run, 8, progress={
            'global_step': 10728, 'epoch_index': 8, 'real_scenes_seen': 686503,
            'global_batch_cursor': 0})
        with self.assertRaisesRegex(ValueError, 'exposure'):
            selector.endpoint_checkpoint(self.run, 8)

    def test_checksum_corruption_is_rejected(self):
        p = self.run / 'checkpoints/epoch_008.pt'
        p.write_bytes(p.read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            selector.endpoint_checkpoint(self.run, 8)

    def test_teacher_or_asset_changed_is_rejected(self):
        p = Path(self.f.identities['files']['teacher_bundle']['path'])
        p.write_text('changed teacher')
        with self.assertRaisesRegex(ValueError, 'Frozen training asset'):
            selector.endpoint_checkpoint(self.run, 8)

    def test_checkpoint_identity_changed_is_rejected(self):
        identities = copy.deepcopy(self.f.identities)
        identities['protocol_sha256'] = 'other-model'
        self.f.write_checkpoint(self.run, 8, identities=identities)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            selector.endpoint_checkpoint(self.run, 8)

    def test_test_pdms_selects_ties_earlier_not_validation(self):
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            value = selector.select(self.run)
        self.assertEqual(value['best']['epoch'], 16)
        self.assertEqual(value['last']['epoch'], 40)
        self.assertEqual(value['selection_population'], 'test')
        self.assertFalse(value['independent_held_out_test'])
        self.assertFalse(value['validation_used_for_model_selection'])
        self.assertTrue(value['test_used_for_model_selection'])
        self.assertNotIn('validation_summary_sha256', value['best'])
        self.assertFalse((self.run / 'checkpoints/validation_best.json').exists())

    def test_final_selection_still_requires_full_budget(self):
        p = self.run / 'TRAIN_COMPLETE.json'
        value = json.loads(p.read_text())
        value['global_step'] -= 1
        selector.atomic_json(p, value)
        with self.assertRaisesRegex(ValueError, '40-epoch'):
            selector.select(self.run)

    def test_four_run_lock_and_legacy_evidence_preserved(self):
        legacy = self.f.root / 'CHECKPOINT_SELECTION_LOCK.json'
        legacy.write_text('preserve old Validation selection\n')
        old_bytes = legacy.read_bytes()
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            value = selector.lock_suite(self.f.root)
            checked = selector.verify_selection_lock(self.f.root, full=True)
        self.assertEqual(checked, value)
        self.assertEqual(len(value['runs']), 4)
        self.assertTrue(value['all_registered_train_and_full_test_complete'])
        self.assertEqual(legacy.read_bytes(), old_bytes)
        self.assertEqual(selector.verify_selection_lock(self.f.root), value)

    def test_changed_summary_is_rejected_after_lock(self):
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            selector.lock_suite(self.f.root)
        p = self.run / 'test/epoch_008/SUMMARY.json'
        p.write_text('changed Test score')
        with self.assertRaisesRegex(ValueError, 'selection evidence changed'):
            selector.verify_selection_lock(self.f.root)

    def test_lock_digest_tamper_is_rejected(self):
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            selector.lock_suite(self.f.root)
        p = self.f.root / selector.LOCK_NAME
        value = json.loads(p.read_text())
        value['registered_run_count'] = 8
        selector.atomic_json(p, value)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            selector.verify_selection_lock(self.f.root)

    def test_cancelled_seed_endpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four retained'):
            selector.endpoint_checkpoint(self.f.root / 'H_full_seed20261028', 8)

    def test_nan_pdms_is_rejected(self):
        self.f.scores[8] = float('nan')
        with patch.object(selector, 'verify_evaluation', self.f.fake_verify):
            with self.assertRaisesRegex(ValueError, 'finite PDMS'):
                selector.endpoint_candidate(self.run, 8)

    def test_full_real_12146_scene_evidence_and_noise_coverage(self):
        endpoint = selector.endpoint_checkpoint(self.run, 8)
        ev = self.run / 'test/epoch_008'
        tokens = [f'{i:016x}' for i in range(12146)]
        protocol = {'population': 'test', 'scene_count': 12146, 'noise_indices': list(range(5)),
                    'full_population': True, 'checkpoint_sha256': endpoint['sha256'],
                    'artifact_kind': 'behavior', 'mode': 'student', 'target_version': None,
                    'representation_only': False, 'paired_scene_noise_base': True,
                    'targets_manifest_sha256': None, 'tokens_sha256': selector.digest(tokens),
                    'teacher_bundle_sha256': endpoint['teacher_bundle_sha256'],
                    'feature_manifest_sha256': endpoint['feature_manifest_sha256'],
                    'calibration_sha256': endpoint['calibration_sha256'],
                    'noise_identity': selector._noise_identity(endpoint)}
        protocol_sha = selector.digest(protocol)
        selector.atomic_json(ev / 'PROTOCOL.json', {**protocol, 'protocol_sha256': protocol_sha})
        scores = {key: 1. for key in ('score', 'no_at_fault_collisions',
            'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound',
            'comfort', 'driving_direction_compliance')}
        score_pair = {'trajectory': [[0., 0., 0.]] * 8,
                      'base_trajectory': [[0., 0., 0.]] * 8,
                      'score': scores, 'base_score': scores}
        noise = [dict(score_pair, noise_index=j) for j in range(5)]
        with (ev / 'scene_scores.jsonl').open('w') as handle:
            for token in tokens:
                handle.write(json.dumps({'token': token, 'protocol_sha256': protocol_sha,
                                         'noise': noise}, separators=(',', ':')) + '\n')
        summary = {key: protocol[key] for key in ('checkpoint_sha256', 'population',
            'noise_indices', 'mode', 'artifact_kind', 'target_version', 'full_population')}
        summary.update(protocol_sha256=protocol_sha, scenes=12146,
                       behavior={'scenes': 12146, 'pdms_points': 100.})
        selector.atomic_json(ev / 'SUMMARY.json', summary)
        selector.atomic_json(ev / 'COMPLETE.json', {'status': 'COMPLETE',
            'protocol_sha256': protocol_sha,
            'artifacts': {name: selector.sha256(ev / name) for name in
                          ('scene_scores.jsonl', 'SUMMARY.json', 'PROTOCOL.json')}})
        value = selector.endpoint_candidate(self.run, 8)
        self.assertEqual(value['pdms_points'], 100.)
        # Re-seal a scientific error, proving rejection is not only a byte hash check.
        rows = (ev / 'scene_scores.jsonl').read_text().splitlines()
        broken = json.loads(rows[0])
        broken['noise'].pop()
        rows[0] = json.dumps(broken, separators=(',', ':'))
        (ev / 'scene_scores.jsonl').write_text('\n'.join(rows) + '\n')
        marker = json.loads((ev / 'COMPLETE.json').read_text())
        marker['artifacts']['scene_scores.jsonl'] = selector.sha256(ev / 'scene_scores.jsonl')
        selector.atomic_json(ev / 'COMPLETE.json', marker)
        with self.assertRaisesRegex(ValueError, 'missing required behavior noise'):
            selector.endpoint_candidate(self.run, 8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
