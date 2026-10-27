"""Supplement already verified A-arm correlation with keep/repair components.

This CPU reader never rescans GPU jobs or changes the frozen main analyzer.
Run only after analysis/CORRELATION_ANALYSIS.json is actually COMPLETE.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import correlation_analysis as main_analysis

METRICS = {
    'online_zero_target_h': ('整轮在线保持组H', 'lower'),
    'online_repair_target_h': ('整轮在线修复组H', 'lower'),
    'fixed_keep_h_normalized': ('固定面板保持组H', 'lower'),
    'fixed_repair_h_normalized': ('固定面板修复组H', 'lower'),
    'fixed_keep_predicted_rms': ('固定面板保持组预测修复RMS', 'lower'),
    'fixed_repair_direction_cosine': ('固定面板修复组方向cosine', 'higher'),
    'fixed_repair_amplitude_ratio': ('固定面板修复组平均幅度比', 'near_one'),
    'fixed_repair_mean_amplitude_deviation_from_one': ('平均幅度比偏离1的绝对值', 'lower'),
}


def optional_number(value, label):
    return None if value is None else main_analysis.finite(value, label)


def analyze(manifest_path, analysis_path, out):
    manifest_path, analysis_path, out = map(lambda p: Path(p).resolve(),
                                           (manifest_path, analysis_path, out))
    manifest = main_analysis.read_json(manifest_path)
    verified = main_analysis.read_json(analysis_path)
    if verified.get('status') != 'COMPLETE':
        raise ValueError('The complete primary correlation analysis is required first')
    reference = verified.get('manifest', {})
    if (Path(reference.get('path', '')).resolve() != manifest_path or
            reference.get('sha256') != main_analysis.sha256(manifest_path)):
        raise ValueError('Primary analysis belongs to a different frozen manifest')
    if manifest.get('run_id') != main_analysis.RUN_ID or manifest.get('epochs') != [1,4,8,12,16]:
        raise ValueError('Only the registered five A checkpoints are accepted')
    main_analysis.checked_reference(manifest['request'], 'user request')
    statistics_path = main_analysis.checked_reference(manifest['epoch_statistics'], 'epoch statistics')
    statistics = {r['epoch']: r for r in main_analysis.read_json(statistics_path)['epochs']}
    validated_rows = {row['epoch']: row for row in verified['rows']}
    if (set(validated_rows) != set(manifest['epochs']) or
            len(validated_rows) != len(verified['rows']) or verified.get('n_checkpoints') != 5):
        raise ValueError('Primary analysis must contain exactly all five verified checkpoint points')
    entries = {entry['epoch']: entry for entry in manifest['points']}
    rows, cohort_counts = [], None
    for epoch in manifest['epochs']:
        row = validated_rows[epoch]
        online = statistics[epoch]
        if (online.get('status') != 'COMPLETE' or
                online.get('checkpoint', {}).get('checkpoint_sha256') != row['checkpoint_sha256']):
            raise ValueError('Online loss components do not match the evaluated checkpoint')
        panel = Path(entries[epoch]['train_panel_representation']).resolve()
        if panel != Path(row['train_panel_representation']).resolve():
            raise ValueError('Panel path differs from the primary verified point')
        for name, field in [('SUMMARY.json', 'panel_summary_sha256'),
                            ('COMPLETE.json', 'panel_complete_sha256')]:
            if main_analysis.sha256(panel/name) != row[field]:
                raise ValueError('Panel evidence changed after the primary analysis: ' + name)
        for name, field in [('SUMMARY.json', 'test_summary_sha256'),
                            ('COMPLETE.json', 'test_complete_sha256')]:
            if main_analysis.sha256(Path(row['test_evaluation'])/name) != row[field]:
                raise ValueError('Test evidence changed after the primary analysis: ' + name)
        representation = main_analysis.read_json(panel/'SUMMARY.json')['representation']
        keep, repair = representation['keep'], representation['repair']
        counts = {'online_keep_scenes': int(online['zero_target_h']['weight']),
                  'online_repair_scenes': int(online['repair_target_h']['weight']),
                  'panel_keep_scenes': keep['scenes'], 'panel_repair_scenes': repair['scenes']}
        if (counts['online_keep_scenes']+counts['online_repair_scenes'] != 85813 or
                counts['panel_keep_scenes']+counts['panel_repair_scenes'] != 8192 or
                representation['all']['scenes'] != 8192):
            raise ValueError('Keep/repair populations do not cover the registered true scene counts')
        if cohort_counts is not None and counts != cohort_counts:
            raise ValueError('Keep/repair groups changed across checkpoints')
        cohort_counts = counts
        ratio = optional_number(repair.get('amplitude_ratio'), 'repair amplitude ratio')
        rows.append({'epoch': epoch, 'checkpoint_sha256': row['checkpoint_sha256'],
            'test_pdms_points': main_analysis.finite(row['test_pdms_points'], 'verified Test PDMS'),
            **counts,
            'online_zero_target_h': optional_number(online['zero_target_h']['mean'], 'online zero H'),
            'online_repair_target_h': optional_number(online['repair_target_h']['mean'], 'online repair H'),
            'fixed_keep_h_normalized': optional_number(keep.get('h_normalized'), 'panel keep H'),
            'fixed_repair_h_normalized': optional_number(repair.get('h_normalized'), 'panel repair H'),
            'fixed_keep_predicted_rms': optional_number(keep.get('predicted_rms'), 'panel keep predicted RMS'),
            'fixed_repair_direction_cosine': optional_number(repair.get('direction_cosine'), 'repair cosine'),
            'fixed_repair_amplitude_ratio': ratio,
            'fixed_repair_mean_amplitude_deviation_from_one': abs(ratio-1.) if ratio is not None else None})
    correlations = {}
    for key, (label, direction) in METRICS.items():
        paired = [row for row in rows if row[key] is not None]
        correlations[key] = {'label': label, 'nominal_metric_direction': direction,
            'n_checkpoint_pairs': len(paired), 'epochs_used': [r['epoch'] for r in paired],
            **{method: main_analysis.correlation([r[key] for r in paired],
                  [r['test_pdms_points'] for r in paired], method) for method in ('pearson','spearman')}}
    result = {'schema':'nrkd_loss_component_correlations_v1','status':'COMPLETE',
        'created_at':datetime.now(timezone.utc).isoformat(),'run_id':main_analysis.RUN_ID,
        'primary_analysis':{'path':str(analysis_path),'sha256':main_analysis.sha256(analysis_path)},
        'manifest':{'path':str(manifest_path),'sha256':main_analysis.sha256(manifest_path)},
        'rows':rows,'correlations':correlations,'test_used_for_model_selection':True,
        'independent_held_out_test':False,'five_checkpoints_are_not_independent_training_seeds':True,
        'note':'Descriptive same-trajectory association; no causal or independent-training-seed claim. '
               'Base is not a sixth point. Native group means are used as saved; missing cosine/ratio stays undefined.'}
    out.mkdir(parents=True,exist_ok=True)
    main_analysis.atomic_json(out/'LOSS_COMPONENT_CORRELATIONS.json',result)
    with (out/'LOSS_COMPONENTS.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report = ['# A组loss分解与完整Test PDMS的补充关联','',
        '使用主分析已经完整核验的同五个checkpoint和Test成绩；不重新运行GPU、不读取正在变化的日志。',
        '线上保持/修复组H各按真实组内场景数归一化；固定面板的保持/修复组使用相同8192人群。', '',
        '| 指标 | Pearson r | Spearman ρ | checkpoint数 |', '|---|---:|---:|---:|']
    def fmt(value):return '未定义' if value is None else f'{value:+.6f}'
    for key,value in correlations.items():
        report.append(f"| {value['label']} | {fmt(value['pearson']['value'])} | {fmt(value['spearman']['value'])} | {value['n_checkpoint_pairs']} |")
    report += ['',
        '这里x为该列指标，y为完整Test PDMS。对H、保持组修复RMS等越低越好的指标，负相关才对应“指标降低、Test升高”；正相关意味着两者同向变化。方向cosine通常越高越好，解释符号相反。', '',
        '保持组H或修复RMS下降并不推出PDMS必然提升；正常场景损害、失败场景收益和动作链非线性都可能影响完整行为成绩。这里仅列描述性相关，不判断因果。', '',
        '幅度比理想值在1附近，不能简单当作越高或越低越好。额外的偏离量是“平均幅度比与1的绝对差”，并非“逐场景绝对幅度误差的均值”；平均值会掩盖过修复与欠修复相互抵消。', '',
        '方向cosine与幅度比沿用原生SUMMARY对有效非空值求均值的定义，缺值不补0。分组结果是固定Train8192面板描述，不冒充完整Train误差或完整Train行为分数。', '',
        '五个点来自同一训练seed的同一训练轨迹，并非五个独立实验。S0–S4是动作噪声重复，Base只作为配对参考，不进入相关系数充当第六点。Test已用于选择和比较，不再称为独立held-out最终测试。']
    (out/'LOSS_COMPONENTS_INTERPRETATION.md').write_text('\n'.join(report)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.manifest,args.analysis,args.out)
    print(json.dumps({'status':result['status'],'checkpoint_count':len(result['rows']),'out':str(args.out)}))


if __name__=='__main__':main()
