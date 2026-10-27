"""CPU-only fixture exercises complete report checks and Test-selection labels."""
import ast
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import report_scope


def atomic_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selections = {}
        self.qualification = {'mode': 'fixture'}
        self.runs = {f'{arm}_seed20261027': {'experiment': arm, 'training_seed': 20261027,
                    'status': 'TRAIN_COMPLETE'} for arm in ('H_full', 'H_gain1', 'H_mu_teacher', 'H_mu_student')}
        self.registry = {'runs': self.runs, 'tasks': {}}
        atomic_json(self.root / 'EXPERIMENT_REGISTRY.json', self.registry)
        for name, run in self.runs.items():
            folder = self.root / name
            arm = run['experiment']
            target = 'full' if arm == 'H_full' else 'gain1'
            atomic_json(folder / 'ASSET_IDENTITIES.json', {'qualification': self.qualification,
                        'files': {'teacher_bundle': {'sha256': 'same-teacher'}}})
            (folder / 'effective_config.yaml').write_text('teacher_target: ' + target + '\nfixture: true\n')
            atomic_json(folder / 'TRAIN_COMPLETE.json', {'wall_seconds_total': 100., 'world_size': 4,
                        'global_batch_size': 64, 'microbatch_per_device': 1, 'gradient_accumulation_steps': 16})
            selection = {'selection_population': 'test', 'test_used_for_model_selection': True,
                         'independent_held_out_test': False,
                         'best': {'epoch': 8, 'sha256': 'sha8'}, 'last': {'epoch': 40, 'sha256': 'sha40'}}
            self.selections[name] = selection
            atomic_json(folder / 'checkpoints/test_pdms_best.json', selection)
            # Deliberately conflicting old Validation choice proves it is ignored.
            atomic_json(folder / 'checkpoints/validation_best.json', {'best': {'epoch': 16}})
            for label, epoch in [('best', 8), ('last', 40)]:
                checksum = 'sha' + str(epoch)
                summary = {'checkpoint_sha256': checksum, 'behavior': {'pdms_points': 94. + epoch / 100,
                           'delta_points': -.2, 'positive_gain_sum_points': 10., 'negative_loss_sum_points': 11.},
                           'components_points': {'score': 94.}, 'events': {'new_zero_score': 1,
                           'paired_base_gt90_student_zero': 1}, 'groups': {'log': {}}}
                atomic_json(folder / 'test' / label / 'SUMMARY.json', summary)
                # Validation has unrelated values and cannot alter checkpoint choice.
                atomic_json(folder / 'validation' / f'epoch_{epoch:03d}' / 'SUMMARY.json',
                            {'behavior': {'pdms_points': 99. if epoch == 40 else 1.}})
                repr_summary = {'representation': {'all': {'h_normalized': .2}, 'keep': {}, 'repair': {}}}
                for population in ('train', 'validation'):
                    atomic_json(folder / 'diagnostics' / f'{population}_representation_{label}' / 'SUMMARY.json',
                                dict(repr_summary, checkpoint_sha256=checksum))
                    atomic_json(folder / 'diagnostics' / f'{population}_functional_{label}' / 'SUMMARY.json',
                                {'checkpoint_sha256': checksum, 'functional': {source: [
                                 {'normalized_mse': .3} for _ in range(5)] for source in ('teacher', 'student')}})
                student_dir = folder / 'diagnostics' / f'train_panel_{label}'
                atomic_json(student_dir / 'SUMMARY.json', {'checkpoint_sha256': checksum, 'scenes': 8192,
                            'noise_indices': [0, 1, 2, 3, 4], 'behavior': {'pdms_points': 95.}})
                atomic_json(student_dir / 'PROTOCOL.json', {'tokens_sha256': 'cohort'})
                teacher_dir = self.root / 'train_panel_teachers' / target
                atomic_json(teacher_dir / 'SUMMARY.json', {'scenes': 8192, 'noise_indices': [0, 1, 2, 3, 4],
                            'behavior': {'pdms_points': 97.}})
                atomic_json(teacher_dir / 'PROTOCOL.json', {'tokens_sha256': 'cohort'})
                atomic_json(folder / 'diagnostics' / f'capture_{label}.json',
                            {'target_version': target, 'noise_indices': [0, 1, 2, 3, 4],
                             'tokens_sha256': 'cohort', 'student_summary_sha256': digest(student_dir / 'SUMMARY.json'),
                             'teacher_summary_sha256': digest(teacher_dir / 'SUMMARY.json'), 'gain_capture': .1})
            atomic_json(folder / 'diagnostics/validation_new_noise/SUMMARY.json', {'checkpoint_sha256': 'sha8'})

        module = types.ModuleType('student_distillation.aggregate')
        module.__package__ = 'student_distillation'
        module.__dict__.update(Path=Path, json=json, csv=csv, defaultdict=defaultdict, np=np, atomic_json=atomic_json,
            _summary=lambda p: json.loads(p.read_text()) if p.exists() else None, make_figures=lambda *a: None)
        self.report = report_scope.build_report(module)
        self.verify_calls = []

        def verify(path, expected):
            self.verify_calls.append((str(path), expected))
            value = json.loads((Path(path) / 'SUMMARY.json').read_text())
            if value.get('checkpoint_sha256') is not None and value['checkpoint_sha256'] != expected.get('checkpoint_sha256'):
                raise ValueError('checkpoint mismatch')
            return value

        self.modules = {
            'student_distillation.evaluate': types.SimpleNamespace(verify_evaluation=verify),
            'student_distillation.select_checkpoint': types.SimpleNamespace(
                verify_selection_lock=lambda out, full: {'runs': self.selections}),
            'student_distillation.run_suite': types.SimpleNamespace(verify_validation_teacher=lambda *a: None),
            'student_distillation.experiment_registry': types.SimpleNamespace(
                EXPERIMENTS=('H_full', 'H_gain1', 'H_mu_teacher', 'H_mu_student'), SEEDS=(20261027,)),
            'student_distillation.train': types.SimpleNamespace(verify_qualification=lambda cfg: self.qualification),
            'student_distillation.checkpointing': types.SimpleNamespace(sha256=digest),
        }

    def run_report(self, require_complete=True):
        with patch.dict('sys.modules', self.modules):
            return self.report(self.root, require_complete=require_complete)

    def test_complete_report_uses_test_choice_and_discloses_selection(self):
        result = self.run_report()
        self.assertEqual(result['rows'], 8)
        self.assertEqual(result['incomplete'], [])
        self.assertTrue(result['test_used_for_model_selection'])
        self.assertFalse(result['independent_held_out_test'])
        with (self.root / 'ROUND1_RESULTS.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({x['epoch'] for x in rows if x['checkpoint'] == 'best'}, {'8'})
        self.assertTrue(all(x['selection_population'] == 'test' for x in rows))
        self.assertTrue(all(x['validation_role'] == 'diagnostic_only_not_a_selection_metric' for x in rows))
        report = (self.root / 'ROUND1_REPORT.md').read_text()
        self.assertIn('Test is used for model selection', report)
        self.assertIn('not an independent held-out final', report)
        self.assertIn('Val PDMS (diagnostic)', report)
        self.assertIn('sole retained training seed', report)
        self.assertNotIn('two independent training seeds', report)

    def test_old_validation_selection_cannot_substitute_for_test(self):
        name = next(iter(self.runs))
        (self.root / name / 'checkpoints/test_pdms_best.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'checkpoint selection'):
            self.run_report()

    def test_missing_required_diagnostic_still_blocks_complete_report(self):
        name = next(iter(self.runs))
        (self.root / name / 'diagnostics/train_functional_best/SUMMARY.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'all-five functional diagnostics'):
            self.run_report()

    def test_selection_identity_cannot_claim_independent_test(self):
        name = next(iter(self.runs))
        value = dict(self.selections[name], independent_held_out_test=True)
        atomic_json(self.root / name / 'checkpoints/test_pdms_best.json', value)
        with self.assertRaisesRegex(RuntimeError, 'invalid Test-PDMS selection identity'):
            self.run_report()

    def test_selection_file_must_equal_verified_test_lock(self):
        name = next(iter(self.runs))
        value = dict(self.selections[name], best={'epoch': 40, 'sha256': 'sha40'})
        atomic_json(self.root / name / 'checkpoints/test_pdms_best.json', value)
        with self.assertRaisesRegex(RuntimeError, 'differs from the verified Test-PDMS lock'):
            self.run_report()


class SourceTransform(unittest.TestCase):
    def test_overlay_rejects_unknown_source_shape(self):
        with self.assertRaises(ValueError):
            report_scope.transform_report_source('def report(out): return {}')

    def test_only_report_function_is_compiled(self):
        source = (report_scope.CODE / 'student_distillation/aggregate.py').read_text()
        tree = report_scope.transform_report_source(source)
        self.assertEqual(len(tree.body), 1)
        self.assertEqual(tree.body[0].name, 'report')
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertTrue(any(n.module == 'select_checkpoint' for n in imports))


if __name__ == '__main__':
    unittest.main()
