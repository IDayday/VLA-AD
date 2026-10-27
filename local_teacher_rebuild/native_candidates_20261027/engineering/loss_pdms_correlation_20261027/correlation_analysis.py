"""A-arm checkpoint loss versus complete paired Test PDMS, without GPU work.

Input is a frozen manifest, not a watched log. Missing evidence produces an
explicit INCOMPLETE report and no correlation estimate. Five checkpoints from
one training trajectory are descriptive observations, not independent seeds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

CODE = Path('/mnt/project/recogdrive-no-routing-round1-20261026')
DEFAULT_RESULTS = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
RUN_ID = 'H_full_seed20261027'
TEST_SCENES = 12146
PANEL_SCENES = 8192
TRAIN_SCENES = 85813
UPDATES_PER_EPOCH = 1341
NOISE_INDICES = [0, 1, 2, 3, 4]
LOSS_COLUMNS = ('online_epoch_loss', 'online_epoch_h', 'fixed_panel_h_normalized')
FIXED_LOSS_LABEL = '固定Train8192面板H（冻结checkpoint）'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_reference(value, name):
    if not isinstance(value, dict) or not value.get('path') or not value.get('sha256'):
        raise ValueError(f'{name} must contain frozen path and SHA256')
    path = Path(value['path']).resolve()
    if sha256(path) != value['sha256']:
        raise ValueError(f'{name} frozen source changed')
    return path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.analysis.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' must be a finite number')
    return float(value)


def close(actual, expected, label, atol=1e-9):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=atol):
        raise ValueError(f'{label} summary differs from actual per-scene records: {actual} != {expected}')


def average_ranks(values):
    """One-based average ranks, including exact ties, without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.
        start = end
    return ranks


def correlation(values, scores, method='pearson'):
    x, y = np.asarray(values, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        return {'value': None, 'reason': 'at_least_three_paired_checkpoints_required'}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Correlation input is nonfinite')
    if method == 'spearman':
        x, y = average_ranks(x), average_ranks(y)
    elif method != 'pearson':
        raise ValueError('Unknown correlation method')
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return {'value': None, 'reason': 'constant_loss_or_test_pdms'}
    return {'value': float(np.clip(np.dot(x, y) / denominator, -1., 1.)), 'reason': None}


def native_tools():
    sys.path.insert(0, str(CODE))
    from student_distillation.checkpointing import read_checkpoint
    from student_distillation.evaluate import verify_evaluation
    return read_checkpoint, verify_evaluation


def expected_noise_identity(cfg):
    manifest = Path(cfg['paths']['evaluation_noise_manifest']).resolve()
    value = read_json(manifest)
    if value.get('schema') != 'native_explicit_evaluation_noise_v1':
        raise ValueError('Expected explicit native evaluation noise manifest')
    entry = value['populations']['test']
    resolve = lambda text: str((Path(text) if Path(text).is_absolute() else manifest.parent / text).resolve())
    return {'manifest': str(manifest), 'manifest_sha256': sha256(manifest), 'population': 'test',
            'path': resolve(entry['path']), 'sha256': entry['sha256'],
            'tokens_path': resolve(entry['tokens_path']), 'tokens_sha256': entry['tokens_sha256'],
            'noise_indices': list(entry['noise_indices']), 'source': entry.get('source'), 'seed_fallback': False}


def parse_epoch_statistics(path, expected_epochs):
    data = read_json(path)
    rows = {}
    for row in data['epochs']:
        epoch = row['epoch']
        if epoch not in expected_epochs:
            continue
        if epoch in rows:
            raise ValueError('Duplicate epoch statistics')
        if (row.get('status') != 'COMPLETE' or row.get('real_samples') != TRAIN_SCENES or
                row.get('observed_updates') != UPDATES_PER_EPOCH or
                row.get('nonfinite_gradients_reported') != 0 or row.get('missing_global_steps')):
            raise ValueError(f'Incomplete actual training epoch statistics: {epoch}')
        loss = finite(row['metrics']['loss']['mean'], 'online epoch loss')
        h = finite(row['metrics']['h']['mean'], 'online epoch H')
        mu = finite(row['metrics']['mu']['mean'], 'online epoch mu')
        if mu != 0. or not math.isclose(loss, h, rel_tol=0., abs_tol=1e-12):
            raise ValueError('A must have representation loss only')
        checkpoint = row.get('checkpoint', {})
        if checkpoint.get('status') != 'PASS' or not checkpoint.get('checkpoint_sha256'):
            raise ValueError('Epoch loss lacks the matching completed checkpoint evidence')
        rows[epoch] = {'online_epoch_loss': loss, 'online_epoch_h': h,
                       'checkpoint_sha256': checkpoint['checkpoint_sha256']}
    if set(rows) != set(expected_epochs):
        raise ValueError('Frozen epoch statistics do not cover every selected checkpoint')
    return rows


def test_records(folder, summary):
    """Recompute scene-equal five-noise scores and log sums from real evidence."""
    tokens, logs, student, baseline = [], [], [], []
    paired_base_hash = hashlib.sha256()
    by_log = {}
    with (Path(folder) / 'scene_scores.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            noise = record['noise']
            if [x['noise_index'] for x in noise] != NOISE_INDICES:
                raise ValueError('Each Test scene requires exactly S0-S4')
            score = math.fsum(finite(x['score']['score'], 'Test score') for x in noise) * 100. / 5.
            base = math.fsum(finite(x['base_score']['score'], 'paired Base score') for x in noise) * 100. / 5.
            token, log = record['token'], str(record['log_id'])
            tokens.append(token); logs.append(log); student.append(score); baseline.append(base)
            values = by_log.setdefault(log, [0., 0., 0])
            values[0] += score; values[1] += base; values[2] += 1
            base_evidence = {'token': token, 'log_id': log,
                             'noise': [{'noise_index': x['noise_index'], 'base_score': x['base_score'],
                                        'base_trajectory': x['base_trajectory']} for x in noise]}
            paired_base_hash.update(json.dumps(base_evidence, sort_keys=True, separators=(',', ':')).encode())
            paired_base_hash.update(b'\n')
    if len(tokens) != TEST_SCENES or len(set(tokens)) != TEST_SCENES:
        raise ValueError('Incomplete full Test cohort')
    pdms, base_pdms = math.fsum(student) / len(student), math.fsum(baseline) / len(baseline)
    close(pdms, finite(summary['behavior']['pdms_points'], 'summary Test PDMS'), 'Test PDMS')
    close(base_pdms, finite(summary['behavior']['base_pdms_points'], 'summary Base PDMS'), 'Base PDMS')
    close(pdms - base_pdms, finite(summary['behavior']['delta_points'], 'summary delta'), 'Test delta')
    return {'tokens': tokens, 'logs': logs, 'by_log': by_log, 'test_pdms_points': pdms,
            'base_pdms_points': base_pdms, 'test_delta_points': pdms - base_pdms,
            'paired_base_evidence_sha256': paired_base_hash.hexdigest()}


def panel_records(folder, summary):
    tokens, normalized, raw = [], [], []
    with (Path(folder) / 'scene_scores.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            tokens.append(row['token'])
            normalized.append(finite(row['representation']['h_normalized'], 'panel normalized H'))
            raw.append(finite(row['representation']['h_mse'], 'panel raw H'))
    if len(tokens) != PANEL_SCENES or len(set(tokens)) != PANEL_SCENES:
        raise ValueError('Incomplete fixed Train panel')
    h = math.fsum(normalized) / len(normalized)
    mse = math.fsum(raw) / len(raw)
    close(h, finite(summary['representation']['all']['h_normalized'], 'panel H summary'), 'Panel H')
    close(mse, finite(summary['representation']['all']['h_mse'], 'panel MSE summary'), 'Panel MSE')
    return {'tokens': tokens, 'fixed_panel_h_normalized': h, 'fixed_panel_h_mse': mse}


def load_point(entry, run, statistics, read_checkpoint, verify_evaluation):
    epoch = entry['epoch']
    checkpoint = Path(entry['checkpoint']).resolve()
    if checkpoint != (run / 'checkpoints' / f'epoch_{epoch:03d}.pt').resolve():
        raise ValueError('A checkpoint path must match its registered epoch')
    identities = read_json(run / 'ASSET_IDENTITIES.json')
    for name in ('teacher_bundle', 'feature_manifest', 'targets_manifest', 'calibration'):
        checked_reference(identities['files'][name], 'checkpoint asset ' + name)
    state = read_checkpoint(checkpoint, identities)
    if state.get('world_size') != identities['qualification']['execution']['world_size']:
        raise ValueError('Checkpoint world size differs from its registered execution')
    expected = {'epoch_index': epoch, 'global_batch_cursor': 0,
                'global_step': epoch * UPDATES_PER_EPOCH, 'real_scenes_seen': epoch * TRAIN_SCENES}
    if any(state['progress'].get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint has incorrect actual training exposure')
    cfg = state['config']
    if cfg.get('experiment') != 'H_full' or cfg.get('training_seed') != 20261027 or cfg.get('teacher_target') != 'full':
        raise ValueError('Correlation task accepts the retained A/full run only')
    checksum = sha256(checkpoint)
    if statistics[epoch]['checkpoint_sha256'] != checksum:
        raise ValueError('Online epoch loss and evaluated checkpoint do not match')
    common = {'checkpoint_sha256': checksum, 'noise_indices': NOISE_INDICES, 'mode': 'student',
              'teacher_bundle_sha256': identities['files']['teacher_bundle']['sha256'],
              'feature_manifest_sha256': identities['files']['feature_manifest']['sha256'],
              'calibration_sha256': identities['files']['calibration']['sha256']}
    test = Path(entry['test_evaluation']).resolve()
    panel = Path(entry['train_panel_representation']).resolve()
    test_summary = verify_evaluation(test, dict(common, population='test', scene_count=TEST_SCENES,
                    full_population=True, artifact_kind='behavior', target_version=None,
                    noise_identity=expected_noise_identity(cfg)))
    panel_summary = verify_evaluation(panel, dict(common, population='train', scene_count=PANEL_SCENES,
                    full_population=False, artifact_kind='representation', target_version='full',
                    targets_manifest_sha256=identities['files']['targets_manifest']['sha256']))
    test_data, panel_data = test_records(test, test_summary), panel_records(panel, panel_summary)
    protocol = read_json(test / 'PROTOCOL.json')
    panel_protocol = read_json(panel / 'PROTOCOL.json')
    row = {'epoch': epoch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checksum,
           **statistics[epoch],
           'fixed_panel_h_normalized': panel_data['fixed_panel_h_normalized'],
           'fixed_panel_h_mse': panel_data['fixed_panel_h_mse'],
           'test_pdms_points': test_data['test_pdms_points'],
           'base_pdms_points': test_data['base_pdms_points'],
           'test_delta_points': test_data['test_delta_points'],
           'test_scene_count': TEST_SCENES, 'train_panel_scene_count': PANEL_SCENES,
           'test_evaluation': str(test), 'train_panel_representation': str(panel),
           'test_complete_sha256': sha256(test / 'COMPLETE.json'),
           'test_summary_sha256': sha256(test / 'SUMMARY.json'),
           'panel_complete_sha256': sha256(panel / 'COMPLETE.json'),
           'panel_summary_sha256': sha256(panel / 'SUMMARY.json'),
           'new_zero_scene_noise_events': test_summary.get('events', {}).get('new_zero_score'),
           'paired_Base_gt90_student_zero_events': test_summary.get('events', {}).get('paired_base_gt90_student_zero'),
           'test_improved_scenes': test_summary['behavior'].get('improved'),
           'test_degraded_scenes': test_summary['behavior'].get('degraded')}
    identity = {'test_tokens_sha256': digest(test_data['tokens']),
                'test_token_log_sha256': digest(list(zip(test_data['tokens'], test_data['logs']))),
                'paired_base_evidence_sha256': test_data['paired_base_evidence_sha256'],
                'test_noise_identity': protocol.get('noise_identity'), 'test_scoring': protocol.get('scoring'),
                'panel_tokens_sha256': digest(panel_data['tokens']),
                'panel_selection_sha256': panel_protocol.get('tokens_sha256'),
                'feature_manifest_sha256': common['feature_manifest_sha256'],
                'teacher_bundle_sha256': common['teacher_bundle_sha256'],
                'calibration_sha256': common['calibration_sha256']}
    return row, identity, test_data['by_log']


def bootstrap_correlations(rows, logs_by_checkpoint, replicates=2000, seed=20261027):
    logs = sorted(logs_by_checkpoint[0])
    if any(sorted(value) != logs for value in logs_by_checkpoint):
        raise ValueError('Test log population differs across checkpoints')
    counts = np.asarray([logs_by_checkpoint[0][key][2] for key in logs], dtype=np.float64)
    if any([value[key][2] for key in logs] != counts.tolist() for value in logs_by_checkpoint):
        raise ValueError('Test per-log scene population differs across checkpoints')
    sums = np.asarray([[value[key][0] for key in logs] for value in logs_by_checkpoint], dtype=np.float64)
    random = np.random.default_rng(seed)
    sampled = random.integers(len(logs), size=(replicates, len(logs)))
    totals = sums[:, sampled].sum(axis=2).T
    scores = totals / counts[sampled].sum(axis=1)[:, None]
    output = {'resampling_unit': 'Test log', 'replicates': replicates, 'seed': seed, 'n_logs': len(logs),
              'interpretation': 'Conditional on these five correlated checkpoints and fixed Train losses; '
                  'resamples paired Test logs only, not independent training seeds or new checkpoint trajectories.',
              'intervals': {}}
    for column in LOSS_COLUMNS:
        losses = [row[column] for row in rows]
        output['intervals'][column] = {}
        for method in ('pearson', 'spearman'):
            values = [correlation(losses, draw, method)['value'] for draw in scores]
            valid = [value for value in values if value is not None]
            output['intervals'][column][method] = {
                'finite_replicates': len(valid), 'undefined_replicates': replicates - len(valid),
                'conditional_Test_log_95ci': np.quantile(valid, [.025, .975]).tolist() if valid else None}
    return output


def summarize_correlations(rows):
    scores = [row['test_pdms_points'] for row in rows]
    return {column: {method: correlation([row[column] for row in rows], scores, method)
                     for method in ('pearson', 'spearman')} for column in LOSS_COLUMNS}


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, column, label in zip(axes, ('online_epoch_loss', 'fixed_panel_h_normalized'),
            ('Online epoch mean H (weights change within epoch)', 'Frozen checkpoint H on fixed Train8192 panel')):
        x = [row[column] for row in rows]
        y = [row['test_pdms_points'] for row in rows]
        axis.plot(x, y, 'o-', color='#167d8d', linewidth=1.4)
        for row in rows:
            axis.annotate(f"e{row['epoch']}\n{row['checkpoint_sha256'][:8]}",
                          (row[column], row['test_pdms_points']), xytext=(5, 5), textcoords='offset points', fontsize=8)
        axis.set(xlabel=label, ylabel='Full Test PDMS points (12146 scenes x S0-S4)')
        axis.grid(alpha=.2)
    fig.suptitle('A / H_full / seed20261027: descriptive checkpoint-loss association')
    fig.tight_layout()
    fig.savefig(Path(out) / 'LOSS_VS_TEST_PDMS.png', dpi=180)
    fig.savefig(Path(out) / 'LOSS_VS_TEST_PDMS.svg')
    plt.close(fig)


def write_report(result, out):
    out = Path(out)
    rows = result['rows']
    lines = ['# A组：checkpoint loss与完整Test PDMS', '', '状态：' + result['status'], '',
             '预先按训练进度和loss跨度选取epoch1/4/8/12/16，不按Test高低挑点。'
             '每个Test点覆盖12,146场景、相同S0–S4及配对Base。', '',
             '整轮在线loss是在该epoch内不断更新权重时的85,813场景加权均值，并非保存权重本身的固定误差。'
             '另一指标使用每个冻结checkpoint在同一Train8192面板上的H，适合直接比较固定权重。'
             '该面板H不是完整Train H，也不是Train行为PDMS。', '',
             '| Epoch | checkpoint SHA | 整轮在线loss | 固定Train8192 H | 完整Test PDMS | 配对Base | 差值 |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['epoch']} | {row['checkpoint_sha256'][:12]} | {row['online_epoch_loss']:.8f} | "
                     f"{row['fixed_panel_h_normalized']:.8f} | {row['test_pdms_points']:.6f} | "
                     f"{row['base_pdms_points']:.6f} | {row['test_delta_points']:+.6f} |")
    if result['status'] != 'COMPLETE':
        lines += ['', '完整证据尚未齐全，因此没有计算或推断相关系数。', '',
                  *['- ' + item for item in result['incomplete']]]
    else:
        lines += ['', '| loss指标 | Pearson r | Spearman ρ |', '|---|---:|---:|']
        labels = {'online_epoch_loss': '整轮在线loss', 'online_epoch_h': '整轮在线H（A中与loss相同）',
                  'fixed_panel_h_normalized': FIXED_LOSS_LABEL}
        for name, methods in result['correlations'].items():
            fmt = lambda method: '未定义' if methods[method]['value'] is None else f"{methods[method]['value']:+.6f}"
            lines.append(f"| {labels[name]} | {fmt('pearson')} | {fmt('spearman')} |")
        lines += ['', '负相关表示这些checkpoint中loss较低时Test PDMS较高；正相关表示loss降低时分数反而降低。'
                  '这只是同一训练轨迹上五个权重的描述性关联，不能解释为因果关系、五个独立训练种子或普遍规律。', '',
                  'JSON中的95%区间按Test log成对重采样，保留各checkpoint相同的测试人群；'
                  '它仅反映给定这五个权重和固定Train损失时的Test人群采样变化，未估计训练随机性。'
                  '未报告把五个checkpoint视作独立样本的显著性p值。', '',
                  'Test已按用户要求参与权重及方法比较，不能再称为独立held-out最终测试。'
                  'S0–S4是动作噪声重复，不是五个training seed。', '',
                  '图：`LOSS_VS_TEST_PDMS.png`；逐点与关联数值：`CORRELATION_ANALYSIS.json`、`CHECKPOINT_LOSS_TEST.csv`。']
    (out / 'LOSS_PDMS_CORRELATION_REPORT.md').write_text('\n'.join(lines) + '\n')


def analyze(manifest_path, out, *, bootstrap_replicates=2000, make_plot=True):
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    if manifest.get('schema') != 'nrkd_loss_pdms_correlation_input_v1' or manifest.get('run_id') != RUN_ID:
        raise ValueError('Wrong correlation manifest or registered run')
    epochs = manifest['epochs']
    if epochs != [1, 4, 8, 12, 16] or len({x['epoch'] for x in manifest['points']}) != len(epochs):
        raise ValueError('Selected epochs must be frozen before Test scores are read')
    if set(x['epoch'] for x in manifest['points']) != set(epochs):
        raise ValueError('Selected epoch points do not match the frozen plan')
    checked_reference(manifest['request'], 'user request')
    epoch_path = checked_reference(manifest['epoch_statistics'], 'epoch statistics')
    statistics = parse_epoch_statistics(epoch_path, epochs)
    run = Path(manifest.get('results_root', DEFAULT_RESULTS)).resolve() / RUN_ID
    read_checkpoint, verify_evaluation = native_tools()
    rows, identities, logs, incomplete = [], [], [], []
    for entry in sorted(manifest['points'], key=lambda x: x['epoch']):
        try:
            row, identity, by_log = load_point(entry, run, statistics, read_checkpoint, verify_evaluation)
            if identities and identity != identities[0]:
                raise ValueError('Fixed Test/Train-panel population, noise, paired Base or assets differ across checkpoints')
            rows.append(row); identities.append(identity); logs.append(by_log)
        except (OSError, ValueError, KeyError, RuntimeError, EOFError) as exc:
            incomplete.append(f"epoch{entry['epoch']}: {exc}")
    result = {'schema': 'nrkd_A_loss_Test_PDMS_correlation_v1',
              'status': 'COMPLETE' if not incomplete else 'INCOMPLETE',
              'created_at': datetime.now(timezone.utc).isoformat(), 'run_id': RUN_ID,
              'manifest': {'path': str(manifest_path), 'sha256': sha256(manifest_path)},
              'selected_epochs': epochs, 'n_checkpoints': len(rows), 'rows': rows,
              'test_used_for_model_selection': True, 'independent_held_out_test': False,
              'checkpoint_observations_are_independent_training_seeds': False,
              'cohort_identity': identities[0] if identities else None, 'incomplete': incomplete,
              'correlations': None, 'bootstrap': None}
    if not incomplete:
        result['correlations'] = summarize_correlations(rows)
        result['bootstrap'] = bootstrap_correlations(rows, logs, bootstrap_replicates)
        if make_plot:
            plot(rows, out)
    atomic_json(out / 'CORRELATION_ANALYSIS.json', result)
    if rows:
        with (out / 'CHECKPOINT_LOSS_TEST.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_report(result, out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error('--bootstrap-replicates must be positive')
    result = analyze(args.manifest, args.out, bootstrap_replicates=args.bootstrap_replicates,
                     make_plot=not args.no_plot)
    print(json.dumps({'status': result['status'], 'n_checkpoints': result['n_checkpoints'],
                      'incomplete': result['incomplete'], 'out': str(args.out)}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 2)


if __name__ == '__main__':
    main()
