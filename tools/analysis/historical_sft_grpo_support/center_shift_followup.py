"""Read-only, same-sampler center comparisons from the frozen 64-draw banks."""
from common_support import *
from analyze import PARENTS, bootstrap_difference
from datetime import datetime, timezone

DEST = OUT / 'center_shift_20260919'
REPORT = ROOT / 'reports/PSI_CENTER_SHIFT_COMPARISON_20260919.md'
LABELS = {
    'official_il': 'Official IL', 'psi_sft': 'PSI SFT epoch11',
    'a5_sft': 'A5 SFT', 'v6_sft': 'V6 SFT',
    'original_grpo_11970': 'Original GRPO step11970',
    'a5_grpo_300': 'A5 GRPO step300', 'a5_grpo_4842': 'A5 GRPO step4842',
    'v6_grpo_300': 'V6 GRPO step300', 'v6_grpo_3300': 'V6 GRPO step3300',
}


def center_ade(a, b):
    # Norm after averaging trajectories; not average pairwise/sample ADE.
    assert a.shape == b.shape == (8, 2)
    return float(np.linalg.norm(a - b, axis=-1).mean())


def markdown(frame):
    rows = ['| ' + ' | '.join(frame.columns) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(map(str, row)) + ' |'
             for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def main():
    protocol_hash = identity()
    all_scenes = scenes()
    assert len(all_scenes) == len({s['token'] for s in all_scenes}) == 1000
    hashes = {r['path']: r['sha256'] for r in read(OUT/'manifests/cache_hashes.json')}
    old = pd.read_parquet(OUT/'metrics/scene_metrics.parquet').set_index(['token', 'model', 'protocol'])
    membership = pd.read_csv(OUT/'metrics/training_membership.csv').set_index('token')
    rows = []
    max_previous_error = 0.
    for i, scene in enumerate(all_scenes):
        token = scene['token']
        centers = {}
        for model in LABELS:
            path = OUT/'cache/rollouts'/model/f'{token}.npz'
            assert sha(path) == hashes[str(path)], path
            with np.load(path) as bank:
                for protocol in CFG['protocols']:
                    t = bank[protocol]
                    assert t.shape == (4, 16, 8, 3) and np.isfinite(t).all()
                    centers[model, protocol] = t.reshape(64, 8, 3).mean(0)[:, :2]
        for model in LABELS:
            for protocol in CFG['protocols']:
                c = centers[model, protocol]
                reference_old = center_ade(c, centers['official_il', 'eval'])
                error = abs(reference_old - old.loc[(token, model, protocol), 'center_shift_IL_eval'])
                max_previous_error = max(max_previous_error, error)
                refs = [('official_IL_same_sampler', 'official_il')]
                parent = 'official_il' if model == 'psi_sft' else PARENTS.get(model)
                if parent:
                    refs.append(('actual_previous_stage_same_sampler', parent))
                for kind, ref in refs:
                    shift = center_ade(c, centers[ref, protocol])
                    if kind.startswith('actual_') and model in PARENTS:
                        error = abs(shift - old.loc[(token, model, protocol), 'center_shift_SFT_parent'])
                        max_previous_error = max(max_previous_error, error)
                    rows.append(dict(token=token, log=scene['log'], common_train=bool(membership.loc[token, 'common_train']),
                                     model=model, protocol=protocol, reference_kind=kind, reference_model=ref,
                                     stage='SFT' if model.endswith('_sft') else 'IL' if model == 'official_il' else 'GRPO',
                                     center_shift_m=shift))
        if (i + 1) % 200 == 0:
            print('CENTERS', i + 1, flush=True)
    assert max_previous_error < 1e-5, max_previous_error
    df = pd.DataFrame(rows)
    summaries, comparisons = [], []
    for scope, data in [('FULL1000', df), ('COMMON_TRAIN835', df[df.common_train])]:
        for keys, group in data.groupby(['model', 'protocol', 'reference_kind', 'reference_model'], sort=False):
            stats = bootstrap_difference(group.center_shift_m, group.log, 'center_followup:' + scope + ':'.join(keys))
            summaries.append(dict(scope=scope, **dict(zip(['model','protocol','reference_kind','reference_model'],keys)),
                                  scenes=stats['n'], mean_m=stats['mean_difference'], median_m=stats['median_difference'],
                                  p90_m=group.center_shift_m.quantile(.9), ci_low_m=stats['ci_low'], ci_high_m=stats['ci_high'],
                                  log_cluster_ci_low_m=stats['log_cluster_ci_low'], log_cluster_ci_high_m=stats['log_cluster_ci_high']))
        for protocol in CFG['protocols']:
            common = data[(data.protocol == protocol) & (data.reference_kind == 'official_IL_same_sampler')]
            psi = common[common.model == 'psi_sft'].set_index('token')
            for other in [m for m in LABELS if m not in {'official_il', 'psi_sft'}]:
                comparator = common[common.model == other].set_index('token').loc[psi.index]
                stats = bootstrap_difference(psi.center_shift_m-comparator.center_shift_m, psi.log,
                                             f'center_paired:{scope}:{protocol}:{other}')
                stats['fraction_PSI_displacement_larger'] = stats.pop('win_fraction')
                comparisons.append(dict(scope=scope, protocol=protocol, comparator=other, **stats))
    DEST.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEST/'scene_center_shifts.parquet', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(DEST/'center_shift_summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(DEST/'psi_paired_comparisons.csv', index=False)

    index_path = ROOT/'outputs/five_checkpoint_training_distribution/psi_training_chain_audit/stage3_historical_checkpoints.csv'
    index = pd.read_csv(index_path)
    paths = sorted(set(index.historical_source_path.dropna()) | set(index.historical_object_path.dropna()))
    readable = [p for p in paths if Path(p).is_file()]
    audit = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=protocol_hash,
                 input_cache_hash_manifest=str(OUT/'manifests/cache_hashes.json'),
                 verified_rollout_files=9000, scenes=1000, samples_per_scene=64, optimizer_updates=0,
                 max_existing_metric_reconstruction_error=max_previous_error,
                 PSI_GRPO_index_rows=len(index), PSI_GRPO_indexed_paths_checked=len(paths),
                 PSI_GRPO_readable_indexed_paths=readable,
                 PSI_GRPO_status='MISSING' if not readable else 'REQUIRES_IDENTITY_VALIDATION',
                 confidence_interval='3000 scene bootstrap; log-cluster sensitivity; conditional on existing 64 CRN draws',
                 interpretation='Empirical center displacement; not width, mean per-sample displacement, or a causal stage comparison.')
    save(DEST/'audit.json', audit)
    tables = []
    for kind in ['official_IL_same_sampler', 'actual_previous_stage_same_sampler']:
        s = summary[(summary.scope == 'FULL1000') & (summary.reference_kind == kind)]
        formatted = []
        for model in LABELS:
            g = s[s.model == model].set_index('protocol')
            if g.empty: continue
            r = {'权重': LABELS[model]}
            if kind.startswith('actual_'): r['参照权重'] = LABELS[g.iloc[0].reference_model]
            for protocol in CFG['protocols']:
                v = g.loc[protocol]
                r[protocol+' 均值米 [95%CI]'] = f'{v.mean_m:.4f} [{v.ci_low_m:.4f}, {v.ci_high_m:.4f}]'
                r[protocol+' 中位数米'] = f'{v.median_m:.4f}'
            formatted.append(r)
        tables.append(markdown(pd.DataFrame(formatted)))
    REPORT.write_text('''# PSI分布中心位移补充统计（2026-09-19）

复用已冻结的1000个NAVTRAIN场景与每模型每场景64条轨迹（4×16），无新推理、评分或权重更新。采样器分别统计，不跨采样器解释训练位移。835共同训练场景的敏感性与逐场景结果另存表格。

中心定义：每场景、每未来时刻，先对64条轨迹的XY取均值，得到8点“均值轨迹”；然后计算两模型均值轨迹的8点平均欧氏距离，最后1000场景等权汇总。单位米，不包含heading。它不是轨迹分布宽度，也不是两次随机采样轨迹距离的均值。CI为3000次场景bootstrap，log-cluster结果另列；有限64次采样仍有中心估计误差，CI没有声称穷尽Monte Carlo误差。

## 统一官方IL参照（双方使用同一采样器）

''' + tables[0] + '''

## 相对真实前一训练阶段（双方使用同一采样器）

''' + tables[1] + '''

PSI行测得的是IL→候选SFT的位移，其他GRPO行测得的是各自SFT→GRPO的位移，两者不是相同训练阶段的因果对照。不能据此推断PSI后续GRPO移动较小。A5/V6随机初始化action head，官方IL仅是共同几何参照，不是它们的实际初始化。

此前报告的PSI `center_shift_IL_eval=0.542425m` 使用PSI native采样中心对照官方IL eval中心，混合了采样协议。本补充的native主对比使用official native中心，故数值不同；旧数据没有改写。相对IL位移也不能与相对SFT位移直接相加，因为取了向量差的长度。

PSI后续SR-PGRPO的88条历史checkpoint索引仍保留。本次重查176个历史raw/store路径均无可读实体，原run的checkpoint/backup/eval目录也没有权重或轨迹数组。后续GRPO中心位移标为MISSING，不能从PDMS增益、KL或旧评分表推算，也未用APR或其他权重替代。此次不是对所有服务器所有磁盘重新穷尽搜索。

输出目录：`outputs/historical_sft_grpo_support/center_shift_20260919/`；`center_shift_summary.csv`为绝对位移，`psi_paired_comparisons.csv`为PSI相对其他模型的同场景位移差，`scene_center_shifts.parquet`保留逐场景值，`audit.json`记录输入哈希核验和权重检查。既有结果不变。
''')
    print(summary[summary.scope == 'FULL1000'].to_string(index=False))
    print('REPORT', REPORT)


if __name__ == '__main__':
    main()
