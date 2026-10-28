#!/usr/bin/env python3
"""Archive existing completed results only. No inference, scoring, or training."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
N = Path('/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027')
F = N / 'engineering/full_train_loss_pdms_generalization_20261028'
STOP = N / 'engineering/distillation_stop_cleanup_20261028'
R = Path('/mnt/project/recogdrive-no-routing-round1-results-20261026')
EPOCHS = [4, 8, 16, 24, 32]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def copy(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def verified_summary(folder):
    complete = read(folder / 'COMPLETE.json')
    assert complete['status'] == 'COMPLETE', folder
    for filename in ['SUMMARY.json', 'PROTOCOL.json']:
        assert sha(folder / filename) == complete['artifacts'][filename], folder
    return read(folder / 'SUMMARY.json'), {
        'complete': ref(folder / 'COMPLETE.json'),
        'summary': ref(folder / 'SUMMARY.json'),
        'protocol': ref(folder / 'PROTOCOL.json'),
        'scene_scores_sha256_as_recorded_by_COMPLETE': complete['artifacts']['scene_scores.jsonl'],
    }


def main():
    cleanup = read(STOP / 'CLEANUP_COMPLETE.json')
    assert cleanup['status'] == 'COMPLETE_USER_REQUESTED_DISTILLATION_STOP_AND_INTERMEDIATE_CHECKPOINT_CLEANUP'
    assert cleanup['all_A_assets']['status'] == 'ALL_A_FILES_PRESERVED_UNCHANGED'
    assert read(STOP / 'STOP_COMPLETE.json')['owned_live_processes_after_stop'] == 0
    assert read(STOP / 'A_PROTECTED_AFTER.json')['changed_or_deleted_A_files'] == 0
    source = F / 'analysis'
    summary = read(source / 'SUMMARY.json')
    assert summary['Train_is_complete_population'] is True
    for filename in ['FULL_TRAIN_H_PDMS_GENERALIZATION_REPORT.md', 'CHECKPOINT_CURVES.csv',
                     'PANEL_REPRESENTATIVENESS.csv', 'FULL_TRAIN_TEST_H_PDMS_CURVES.png',
                     'ENDPOINT_EVIDENCE.json', 'SUMMARY.json', 'SOURCE_PROVENANCE.json', 'COMPLETE.json']:
        copy(source / filename, HERE / 'A_full_train' / filename)
    corr, adjacent, groups = [], [], []
    for cohort, v in summary['cohorts'].items():
        for comparison, values in v['checkpoint_curve'].items():
            if isinstance(values, dict) and 'pearson' in values:
                corr.append({'cohort': cohort, 'comparison': comparison, **values})
        for a in v['adjacent_same_scene_differences']:
            adjacent.append({'cohort': cohort, 'from_epoch': a['from_epoch'], 'to_epoch': a['to_epoch'],
                'scenes': a['delta_H']['count'], 'mean_delta_H': a['delta_H']['mean'],
                'mean_delta_PDMS_points': a['delta_PDMS_points']['mean'],
                'mean_delta_H_CI95_low': a['mean_delta_H_cluster_CI95'][0],
                'mean_delta_H_CI95_high': a['mean_delta_H_cluster_CI95'][1],
                'mean_delta_PDMS_CI95_low': a['mean_delta_PDMS_cluster_CI95'][0],
                'mean_delta_PDMS_CI95_high': a['mean_delta_PDMS_cluster_CI95'][1],
                'same_scene_delta_Pearson': a['deltaH_vs_deltaPDMS']['pearson'],
                'same_scene_delta_Spearman': a['deltaH_vs_deltaPDMS']['spearman'],
                'H_down_scene_pct': a['H_down_scene_pct'],
                'H_down_PDMS_down_scene_pct': a['H_down_PDMS_down_scene_pct'],
                'PDMS_down_among_H_down_pct': a['PDMS_down_among_H_down_pct']})
        for group, g in v['groups'].items():
            for row in g['epochs']:
                groups.append({'cohort': cohort, 'group': group, 'scenes': g['scenes'], **row})
    write_csv(HERE / 'A_full_train/CORRELATIONS.csv', corr)
    write_csv(HERE / 'A_full_train/ADJACENT_DIFFERENCES.csv', adjacent)
    write_csv(HERE / 'A_full_train/KEEP_REPAIR_CURVES.csv', groups)

    # Canonical bounded, newline-complete records; latest recovery duplicate wins.
    helper = N / 'engineering/epoch_loss_statistics_20261027/epoch_loss_statistics.py'
    spec = importlib.util.spec_from_file_location('epoch_stats', helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inventory = read(STOP / 'inventory/SUMMARY.json')
    runs = {r['run_id']: r for r in inventory['runs']}
    online, behavior, source_refs, run_status = [], [], {}, []
    for group, name in [('B', 'H_gain1'), ('C', 'H_mu_teacher'), ('D', 'H_mu_student')]:
        run_id = name + '_seed20261027'
        run = R / run_id
        assert read(run / 'TARGET_IDENTITY.json')['target'] == 'gain1'
        records, logmeta = mod.load_records(run / 'logs/train_metrics.jsonl')
        assert not logmeta['invalid_lines'] and not logmeta['ignored_trailing_bytes']
        latest = runs[run_id]['latest_complete_epoch']['epoch']
        completed = []
        epochs = sorted(set([8, 16, latest] + ([24] if group == 'B' else [])))
        online_by_epoch = {}
        for epoch in epochs:
            rows = [records[s] for s in range((epoch-1)*1341+1, epoch*1341+1)]
            assert sum(x['real_scenes'] for x in rows) == 85813
            for j, row in enumerate(rows):
                assert row['real_scenes'] == (53 if j == 1340 else 64)
            means = {k: math.fsum(x[k] * x['real_scenes'] for x in rows) / 85813 for k in ['h', 'mu', 'loss']}
            entry = {'group': group, 'run_id': run_id, 'epoch': epoch, 'real_scenes': 85813,
                     'updates': 1341, 'online_epoch_H': means['h'], 'online_epoch_mu': means['mu'],
                     'online_epoch_total_loss': means['loss'], 'definition': 'real-scene-weighted changing-parameter epoch mean'}
            online.append(entry)
            online_by_epoch[epoch] = entry
        source_refs[run_id] = {'training_log_bounded_snapshot': logmeta,
                               'training_target_identity': ref(run / 'TARGET_IDENTITY.json'),
                               'effective_config': ref(run / 'effective_config.yaml'),
                               'completed_Test': {}, 'completed_Test_H': {}}
        for epoch in [8, 16, 24, 32]:
            folder = run / 'test' / f'epoch_{epoch:03d}'
            if not (folder / 'COMPLETE.json').is_file():
                continue
            test, refs = verified_summary(folder)
            assert test['behavior']['scenes'] == 12146
            assert test['full_population'] is True
            protocol = read(folder / 'PROTOCOL.json')
            # Behavior-only evaluation has no representation target; training
            # target identity above and representation-only protocol bind gain1.
            assert protocol['target_version'] is None
            hs = run / 'diagnostics/test_H' / f'epoch_{epoch:03d}'
            h = None
            if (hs / 'COMPLETE.json').is_file():
                rep, h_refs = verified_summary(hs)
                assert rep['checkpoint_sha256'] == test['checkpoint_sha256']
                assert rep['representation']['all']['scenes'] == 12146
                assert read(hs / 'PROTOCOL.json')['target_version'] == 'gain1'
                h = rep['representation']['all']['h_normalized']
                source_refs[run_id]['completed_Test_H'][str(epoch)] = h_refs
            source_refs[run_id]['completed_Test'][str(epoch)] = refs
            completed.append(epoch)
            behavior.append({'group': group, 'run_id': run_id, 'epoch': epoch, 'target_version': 'gain1',
                'status': 'COMPLETE_FULL_TEST_12146_S0_S4', 'Test_PDMS': test['behavior']['pdms_points'],
                'paired_Base_PDMS': test['behavior']['base_pdms_points'],
                'student_minus_Base_points': test['behavior']['delta_points'],
                'online_epoch_H': online_by_epoch[epoch]['online_epoch_H'],
                'online_epoch_total_loss': online_by_epoch[epoch]['online_epoch_total_loss'],
                'fixed_checkpoint_Test_H': h,
                'checkpoint_sha256': test['checkpoint_sha256']})
        run_status.append({'group': group, 'run_id': run_id, 'status': 'STOPPED_BY_USER',
            'latest_complete_training_epoch': latest, 'completed_Test_epochs': completed,
            'planned_40_epochs_completed': latest == 40})
    assert not (R / 'H_gain1_seed20261027/test/epoch_032/COMPLETE.json').exists()
    dump(HERE / 'BCD_snapshot/RESULTS.json', {
        'status': 'STOPPED_BY_USER', 'runs': run_status, 'completed_Test_endpoints': behavior,
        'online_epoch_loss': online, 'B_epoch32_Test': {'status': 'STOPPED_PARTIAL', 'full_Test_PDMS': None},
        'warnings': ['No B/C/D full-Train fixed-checkpoint H+behavior joint study.',
                     'All B/C/D representation targets are gain1; A uses full.',
                     'Online training loss is not fixed-checkpoint evaluation H.',
                     'C/D total loss includes scheduled functional mu; Test H excludes mu.',
                     'Test was used for model selection, not an independent holdout.']})
    dump(HERE / 'BCD_snapshot/SOURCE_PROVENANCE.json', {'sources': source_refs, 'canonical_log_reader': ref(helper)})
    write_csv(HERE / 'BCD_snapshot/TEST_SCORES_AND_LOSS.csv', behavior)
    write_csv(HERE / 'BCD_snapshot/ONLINE_EPOCH_LOSS.csv', online)
    bcd = ['# B/C/D 停止后完整 Test 与 loss 快照', '',
           '只报具有原生 COMPLETE 的完整 12,146 场景 × S0–S4。全部来自 seed20261027；没有把不完整 B32 评测外推成全量分数。', '',
           '| 组 | epoch | Test PDMS | 学生−Base | 在线 epoch H | 在线总 loss | 固定 checkpoint Test H |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for x in behavior:
        fmt = lambda v: '未完成' if v is None else f'{v:.6f}'
        bcd.append(f"| {x['group']} | {x['epoch']} | {fmt(x['Test_PDMS'])} | {x['student_minus_Base_points']:+.6f} | {fmt(x['online_epoch_H'])} | {fmt(x['online_epoch_total_loss'])} | {fmt(x['fixed_checkpoint_Test_H'])} |")
    bcd += ['', '同条件 Base=90.6792243763。B32 Test 在用户停止时为 partial，没有 COMPLETE，因此不报完整均分。B/C/D 最后完整训练 epoch 分别为 32/23/21；原 40 epoch 预算已被用户取消。', '',
            'B=H_gain1：只有表征损失；C=H_mu_teacher：使用冻结教师链状态计算功能损失；D=H_mu_student：每次由当前学生接口 detach 后刷新状态。三组表征目标均为 gain1，A 的目标为 full，不能直接按 H 绝对值跨目标排名。C/D 在所列 epoch 已使用 H+0.5μ；Test H 仅是表征项，不是 Test 总 loss。', '',
            '在线 epoch H/总 loss 是参数在训练中变化时的 85,813 个真实场景加权均值；固定 checkpoint Test H 是冻结权重后的全 Test 表征误差，二者不能混称。缺少的 Test H 未新算，B/C/D 未执行本次 A 组那样的完整 Train 固定权重行为/H 联合分析。', '',
            '已完成的 B/C/D Test 均低于同条件 Base。共同 epoch8 时 C/D 略高于 B，epoch16 时 B 高于 C/D，尚无稳定功能损失优势。B8→16 的 Test H 上升而 PDMS 也上升，因此 A 组的五点经验相关不能当作所有方法的一般定律。', '',
            '精确值见 TEST_SCORES_AND_LOSS.csv；各已完成训练 epoch 的在线值见 ONLINE_EPOCH_LOSS.csv；checkpoint、协议、结果文件及原训练日志 SHA 见 SOURCE_PROVENANCE.json。']
    (HERE / 'BCD_snapshot/RESULTS.md').write_text('\n'.join(bcd) + '\n')

    for filename in ['STOP_COMPLETE.json', 'STOP_STATE.json', 'CLEANUP_COMPLETE.json', 'A_PROTECTED_AFTER.json',
                     'inventory/SUMMARY.json', 'inventory/KEEP.json']:
        copy(STOP / filename, HERE / 'stop_cleanup' / filename)
    for filename in ['COMPLETE.json', 'INDEPENDENT_FULL_NUMERIC_REVIEW.json', 'MEASURED_GPU_COSTS.json']:
        copy(F / 'completion' / filename, HERE / 'provenance' / filename)
    refs = [F / 'INPUT_MANIFEST.json', F / 'ANALYSIS_FREEZE.json', F / 'analysis/COMPLETE.json',
            F / 'analysis/SUMMARY.json', F / 'analysis/SOURCE_PROVENANCE.json',
            F / 'completion/INDEPENDENT_FULL_NUMERIC_REVIEW.json', STOP / 'STOP_COMPLETE.json',
            STOP / 'CLEANUP_COMPLETE.json', STOP / 'A_PROTECTED_AFTER.json']
    dump(HERE / 'provenance/ORIGINAL_SOURCE_IDENTITIES.json', {
        'publication_is_existing_results_serialization_only': True,
        'source_references': [ref(p) for p in refs],
        'no_model_inference_scoring_training_or_new_scientific_acceptance': True,
        'raw_trajectories_large_arrays_weights_remain_local': True})
    readme = '''# 本轮蒸馏已停止：完整 Train/Test 结果与清理归档

2026-10-28 15:06:51 UTC，用户要求停止的训练、评测和控制器已全部实际退出，五台服务器本项目在跑进程为零，自动恢复与后续调度已禁止。15:12:10 UTC 完成清理。A 组训练已完整到 40 epoch；B/C/D 最后完整 epoch 为 32/23/21，未完成的原 40 epoch 预算已取消。本归档不将它们标为完成。

A 组全部 1,106 个文件均保留且元信息核验未变，其中 97 份 checkpoint 经 SHA256 核验未变。B/C/D 保留全部已完成 Test 关联权重及各 run 最后完整 epoch，共 12 份；删除其余 177 个中间 `.pt`，释放逻辑字节 4,409,115,022（4.106308 GiB）。原数据、教师、Base、缓存、日志、metadata、逐场景结果和报告全部保留。`last.pt` 重复副本删除后对应 canonical epoch 权重仍保留，替代路径记录在 KEEP 中。

## A：完整 Train 85,813 与完整 Test 12,146

固定 A/H_full seed20261027 的 epoch4/8/16/24/32，每点均为完整人群及同场景 S0–S4；H 每场景计算一次。Train 每点复用旧 8,192 条完整记录，真实新算余下 77,621 条，逐 token 合并为 85,813 条。Test 全部复用已完成完整结果，没有新增 Test 推理/评分。

- [完整报告](A_full_train/FULL_TRAIN_H_PDMS_GENERALIZATION_REPORT.md)
- [每个 checkpoint 的精确成绩与 H](A_full_train/CHECKPOINT_CURVES.csv)
- [Pearson/Spearman 相关性表](A_full_train/CORRELATIONS.csv)
- [相邻同场景差分及 log 聚类区间](A_full_train/ADJACENT_DIFFERENCES.csv)
- [保持组/修复组和严重退化](A_full_train/KEEP_REPAIR_CURVES.csv)
- [端点 95% log 聚类区间](A_full_train/ENDPOINT_EVIDENCE.json)
- [全量与旧 8,192 面板的实际差异](A_full_train/PANEL_REPRESENTATIVENESS.csv)
- [完整聚合统计 JSON](A_full_train/SUMMARY.json)

Train H 从 0.301131 降至 0.059741，PDMS 从 94.9994 升至 96.6635；epoch32 相对自己的 Base 提升 1.5915 点，保留教师收益 64.75%。五个 checkpoint 均值的 H–PDMS Pearson=-0.999245、Spearman=-1。

Test H 从 0.397157 升至 0.551774，PDMS 从 88.8716 降至 87.7419；epoch32 相对自己的 Base 为 -2.9373 点。Test 均值 H–PDMS Pearson=-0.977338、Spearman=-1。训练数据上的 H 下降确实伴随行为均分改善，但收益没有转移到 Test；不能笼统说 loss 与 PDMS 无关，也不能由此宣称逐场景必然相关。

![完整 Train/Test 的固定权重 H 与行为曲线](A_full_train/FULL_TRAIN_TEST_H_PDMS_CURVES.png)

这些是一个训练 seed 的五个相关 checkpoint，不是五个独立训练。Test 已参与模型选择，不是独立 holdout。比较迁移使用各自配对 Base 的收益与保持组退化，不使用原始 Train/Test 分数差直接命名泛化差距。全量相对旧面板的 PDMS 最大绝对差仅为这五点上的 0.0977 点，不据此声称面板对任意模型无偏。

## B/C/D：已完成结果与停止边界

[完整 Test 与在线/固定 loss 简表](BCD_snapshot/RESULTS.md) · [精确 CSV](BCD_snapshot/TEST_SCORES_AND_LOSS.csv) · [停止状态与在线 loss JSON](BCD_snapshot/RESULTS.json)。B32 Test 为停止时的 partial，不报告完整均分；所有负结果保留。B/C/D 目标为 gain1，A 为 full；B/C/D 未做本次完整 Train 联合行为/H 分析。

## 来源、成本与保留范围

[完整计算完成收据](provenance/COMPLETE.json)、[独立原始逐行数值核查](provenance/INDEPENDENT_FULL_NUMERIC_REVIEW.json)、[实测成本](provenance/MEASURED_GPU_COSTS.json)、[原文件 SHA](provenance/ORIGINAL_SOURCE_IDENTITIES.json)、[全 Train/Test 来源](A_full_train/SOURCE_PROVENANCE.json)。本次补全 Train 共使用 40 个物理 GPU，新增推理/评分执行跨度约 71.61 分钟；并发进程寿命和各 GPU 预留 wall union 分开报告，不把它们当精确 GPU kernel 时间。

[实际停止](stop_cleanup/STOP_COMPLETE.json) · [实际清理](stop_cleanup/CLEANUP_COMPLETE.json) · [A 全量保护核验](stop_cleanup/A_PROTECTED_AFTER.json) · [保留权重路径/SHA/last 替代](stop_cleanup/inventory/KEEP.json)。这些路径是本机溯源，不意味着大权重已经上传。

本提交只含聚合表格、JSON、报告、PNG 和 compact 收据，没有上传权重、大数组、原始轨迹或数据集。原逐场景 CSV、完整分片和逐行 protocol/SHA 仍在本机原目录，详见来源记录。原始完整报告中列出的逐场景文件名指本机产物；本目录中的文件清单以 PUBLICATION_MANIFEST.json 为准。
'''
    (HERE / 'README.md').write_text(readme)
    files = sorted(p for p in HERE.rglob('*') if p.is_file() and p.name != 'PUBLICATION_MANIFEST.json')
    dump(HERE / 'PUBLICATION_MANIFEST.json', {'status': 'STOPPED_AND_CLEANUP_COMPLETE_RESULTS_ARCHIVE',
        'files': [{'path': str(p.relative_to(HERE)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files],
        'total_bytes': sum(p.stat().st_size for p in files)})
    print(json.dumps({'files': len(files)+1, 'bytes': sum(p.stat().st_size for p in files), 'out': str(HERE)}))


if __name__ == '__main__':
    main()
