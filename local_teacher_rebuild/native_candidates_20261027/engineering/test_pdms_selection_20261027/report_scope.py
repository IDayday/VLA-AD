"""Test-PDMS report selection overlay; the frozen training tree is read-only.

Install after ``single_seed_scope.install`` and the Test selector overlay. Only
the report function is recompiled. Per-scene scoring, statistics and plotting
helpers retain their original implementations.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path

CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
SOURCE_SHA256 = '133f1605b205c5c2d147d38e2fa540a515924c4dc96ed18b92a08f024d9d1035'
SELECTION_METADATA = {
    'selection_population': 'test',
    'selection_metric': 'full_Test_PDMS_points',
    'selection_scene_count': 12146,
    'selection_noise_indices': [0, 1, 2, 3, 4],
    'selection_epochs': [8, 16, 24, 32, 40],
    'test_used_for_model_selection': True,
    'independent_held_out_test': False,
    'validation_role': 'diagnostic_only_not_a_selection_metric',
    'active_training_seeds': [20261027],
}

DISCLOSURE = (
    'Checkpoint and arm comparisons use full Test PDMS over 12,146 scenes and '
    'S0-S4 by the latest user instruction. The best checkpoint maximizes the '
    'mean Test PDMS at epochs 8/16/24/32/40; exact ties use the earlier epoch. '
    'Test is used for model selection and is not an independent held-out final '
    'evaluation. Validation PDMS and representation/function measurements are '
    'diagnostics only and never rank checkpoints or arms. This change does not '
    'put Test examples or teacher targets into gradient training or loss scaling.'
)

_REPLACEMENTS = {
    'checkpoints/validation_best.json': 'checkpoints/test_pdms_best.json',
    'The registry must contain exactly all eight registered run IDs':
        'The registry must contain exactly the four user-retained run IDs',
    'Actual all-eight training/full-Validation selection lock: ':
        'Actual four-run training/full-Test PDMS selection lock: ',
    'All eight runs must share exactly one frozen local teacher':
        'All four retained runs must share exactly one frozen local teacher',
    '# Round1 continuous no-routing distillation':
        '# Round1 continuous no-routing distillation: Test-PDMS selection',
    '| Run | Checkpoint | Epoch | Train H | Val H | Val PDMS | Test PDMS | Test delta | New zero | Paired >90 to zero |':
        '| Run | Checkpoint | Epoch | Train H | Val H (diagnostic) | Val PDMS (diagnostic) | Test PDMS (selection) | Test delta | New zero | Paired >90 to zero |',
    'Each scene receives equal weight after averaging its five action-noise repetitions. S0–S4 are sampling repetitions, while 20261027 and 20261028 are the two independent training seeds. Intervals in each SUMMARY.json resample entire logs 2,000 times with paired Base/student draws.':
        'Each scene receives equal weight after averaging its five action-noise repetitions. S0–S4 are sampling repetitions. The sole retained training seed is 20261027; partial seed20261028 outputs are archived and excluded. Intervals in each SUMMARY.json resample entire logs 2,000 times with paired Base/student draws; they do not correct for Test-based checkpoint selection.',
    'The per-run summaries report full-Train representation error, held-out-log validation representation error, direction and amplitude, zero-target leakage, and source groups. Train behavior is restricted to the fixed 8,192-scene log × Base-score panel; it is not full-Train PDMS. Each behavior summary separates positive gains from negative losses, all native PDM components, per-log changes and paired severe events.':
        'The per-run summaries report full-Train representation error, diagnostic Validation representation error, direction and amplitude, zero-target leakage, and source groups. Train behavior is restricted to the fixed 8,192-scene log × Base-score panel; it is not full-Train PDMS. Each behavior summary separates positive gains from negative losses, all native PDM components, per-log changes and paired severe events.',
    'The additional S5–S9 validation run occurs after the selection lock and never changes the selected checkpoint. Teacher validation choices are post-hoc diagnostic upper bounds and never enter training or calibration. Two training seeds provide limited information about training randomness. A failure of this adapter does not establish that all representation distillation is impossible.':
        'The additional S5–S9 Validation run occurs after the Test-PDMS selection lock and never changes the selected checkpoint. Teacher Validation choices are post-hoc diagnostic upper bounds and never enter training or calibration. One retained training seed provides no estimate of variability across training seeds. A failure of this adapter does not establish that all representation distillation is impossible.',
}


def transform_report_source(source: str) -> ast.Module:
    """Transform the exact frozen report function, retaining all artifact checks."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'report']
    if len(functions) != 1:
        raise ValueError('Expected exactly one original report function')
    node = copy.deepcopy(functions[0])
    seen = {value: 0 for value in _REPLACEMENTS}

    class Overlay(ast.NodeTransformer):
        def visit_Constant(self, value):
            if isinstance(value.value, str) and value.value in _REPLACEMENTS:
                seen[value.value] += 1
                value.value = _REPLACEMENTS[value.value]
            return value

        def visit_For(self, value):
            value = self.generic_visit(value)
            if (isinstance(value.target, ast.Name) and value.target.id == 'seed' and
                    isinstance(value.iter, ast.Tuple) and
                    [getattr(x, 'value', None) for x in value.iter.elts] == [20261027, 20261028]):
                value.iter = ast.Tuple(elts=[ast.Constant(20261027)], ctx=ast.Load())
            return value

        def visit_Assign(self, value):
            value = self.generic_visit(value)
            if len(value.targets) != 1 or not isinstance(value.targets[0], ast.Name):
                return value
            name = value.targets[0].id
            if name == 'selection':
                checks = ast.parse('''
if selection and (selection.get('selection_population') != 'test' or selection.get('test_used_for_model_selection') is not True or selection.get('independent_held_out_test') is not False):
    required.append(f'{run_id}: invalid Test-PDMS selection identity')
if selection and locked.get('runs', {}).get(run_id) != selection:
    required.append(f'{run_id}: selection file differs from the verified Test-PDMS lock')
''').body
                return [value, *checks]
            if name == 'row':
                return [value, *ast.parse('row.update(_TEST_SELECTION_METADATA)').body]
            if name == 'text' and isinstance(value.value, ast.List):
                value.value.elts[4:4] = [ast.Constant(DISCLOSURE), ast.Constant('')]
            return value

    node = Overlay().visit(node)
    if any(count != 1 for count in seen.values()):
        raise ValueError(f'Frozen report wording/layout changed: {seen}')
    return ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))


def build_report(original_module=None):
    """Return a report callable; does not mutate or write production source."""
    original_module = original_module or importlib.import_module('student_distillation.aggregate')
    path = CODE / 'student_distillation/aggregate.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Frozen aggregate source changed before report overlay')
    namespace = dict(original_module.__dict__)
    namespace['_TEST_SELECTION_METADATA'] = SELECTION_METADATA
    exec(compile(transform_report_source(source.decode()), str(path) + ':test_selection_overlay', 'exec'), namespace)
    adapted = namespace['report']

    def report(out, require_complete=True):
        result = adapted(out, require_complete=require_complete)
        metadata = dict(SELECTION_METADATA, report_overlay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        frozen_aggregate_sha256=SOURCE_SHA256)
        request = Path(__file__).with_name('REQUEST.json')
        if request.is_file():
            metadata['user_instruction'] = {'path': str(request), 'sha256': hashlib.sha256(request.read_bytes()).hexdigest()}
        namespace['atomic_json'](Path(out) / 'REPORT_SELECTION_PROTOCOL.json', metadata)
        return dict(result, **metadata)

    report.__name__ = 'report'
    report.__doc__ = 'Complete four-run report selected by full Test PDMS; Validation is diagnostic.'
    return report


def install():
    """Install after the single-seed registry and Test selector overlays."""
    module = importlib.import_module('student_distillation.aggregate')
    if not getattr(module, '_test_pdms_report_scope_installed', False):
        module.report = build_report(module)
        module._test_pdms_report_scope_installed = True
    return module.report
