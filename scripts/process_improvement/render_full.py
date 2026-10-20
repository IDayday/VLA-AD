"""Figures from complete three-training-seed development evaluations only.

No checkpoint selection or missing-result interpolation is performed here.
Parent-record bootstrap results are produced separately by analyze.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = REPORTS/'figures'
    destination.mkdir(parents=True, exist_ok=True)
    groups = [('OPD 1 epoch', 'OPD', 1), ('OPD 2 epochs', 'OPD', 2),
              ('Anchor 1 epoch', 'ANCHOR', 1), ('Anchor 2 epochs', 'ANCHOR', 2)]
    il, rows = predictions('B0_IL', 'dev')
    teacher, _ = predictions('B0_RL', 'dev')
    sources = {}
    policies = {}
    analysis_names = []
    for label, method, epoch in groups:
        name = f'B{1 if method == "OPD" else 2}_{method}_epoch{epoch}_vs_IL'
        path = ROOT/'analysis'/f'{name}.json'
        record = read(path)
        expected = [f'full_{method}_{seed}_epoch{epoch}' for seed in [42, 43, 44]]
        if record['methods'] != expected or record['scenes'] != len(rows):
            raise ValueError('Group does not match completed locked development population')
        if record['cluster_unit'] != 'parent' or record['training_seeds'] != 3:
            raise ValueError('Figure requires primary three-training-seed analysis')
        sources[label] = record
        policies[label] = np.stack([predictions(n, 'dev')[0] for n in expected])
        analysis_names.append(str(path))

    labels = list(sources)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, field, title in zip(axes, ['G', 'M_plus', 'M_minus'],
                              ['Gain over IL', 'Positive scene-mean changes', 'Negative scene-mean magnitude']):
        point = np.array([sources[n]['point'][field] for n in labels])
        interval = np.array([sources[n]['CI95'][field] for n in labels])
        ax.scatter(x, point, color=['#376a93', '#376a93', '#a62c45', '#a62c45'], zorder=3)
        ax.vlines(x, interval[:, 0], interval[:, 1], color='gray')
        ax.axhline(0, color='black', lw=.6)
        ax.set(xticks=x, xticklabels=labels, ylabel='Normalized PDMS', title=title)
        ax.tick_params(axis='x', rotation=35)
    fig.suptitle('Complete dev: 1696 scenes, 15 parents, 3 training x 5 inference seeds; 95% intervals')
    fig.tight_layout()
    fig.savefig(destination/'Fig5_full_transfer_and_degradation.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, field, title in zip(axes, ['G', 'M_minus'], ['Mean gain: Anchor - OPD', 'Degradation: Anchor - OPD']):
        for e in [1, 2]:
            path = ROOT/'analysis'/f'Anchor_vs_OPD_epoch{e}.json'
            record = read(path)
            analysis_names.append(str(path))
            value = record['point'][field]
            lo, hi = record['CI95'][field]
            ax.plot(e, value, 'o', color='#a62c45')
            ax.vlines(e, lo, hi, color='#a62c45')
        ax.axhline(0, color='black', lw=.7)
        ax.set(xticks=[1, 2], xlabel='Complete training epochs', ylabel='New - control, normalized PDMS', title=title)
    fig.suptitle('Paired policies; M_minus for both refers to the same fixed IL baseline')
    fig.tight_layout()
    fig.savefig(destination/'Fig6_paired_anchor_effect.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, label in zip(axes, ['OPD 1 epoch', 'Anchor 1 epoch']):
        for j, seed in enumerate([42, 43, 44]):
            # Each point belongs to one trained policy. No policy-ensemble score.
            ax.scatter(il[:, :, -1].mean(1), policies[label][j, :, :, -1].mean(1),
                       s=6, alpha=.23, label=f'train seed {seed}')
        ax.plot([0, 1], [0, 1], color='black', lw=.7)
        ax.set(xlabel='IL scene mean PDMS', ylabel='Student scene mean PDMS', title=label)
        ax.legend(fontsize=7)
    fig.suptitle('Paired development outcomes; five inference draws per scene and policy')
    fig.tight_layout()
    fig.savefig(destination/'Fig7_full_paired_scene_scatter.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    progress_sources = {}
    for method, color in [('OPD', '#376a93'), ('ANCHOR', '#a62c45')]:
        for seed in [42, 43, 44]:
            path = ROOT/'runs'/f'full_{method}_{seed}'/'progress.json'
            progress = read(path)
            progress_sources[str(path)] = sha(path)
            step = [p['step'] for p in progress]
            axes[0, 0].plot(step, [p['loss'] for p in progress], color=color, alpha=.5, lw=.7,
                            label=method if seed == 42 else None)
            axes[0, 1].plot(step, [p['gradient_norm'] for p in progress], color=color, alpha=.5, lw=.7)
        data = read(ROOT/'runs'/f'full_{method}_42'/'progress.json')
        for k in range(5):
            axes[1, 0 if method == 'OPD' else 1].plot(
                [p['step'] for p in data], [p['per_step'][k] for p in data], lw=.7, label=f'index {k}')
    axes[0, 0].set(ylabel='Total normalized latent objective', yscale='log')
    axes[0, 0].legend()
    axes[0, 1].set(ylabel='Pre-clip gradient L2 norm')
    for ax, label in zip(axes[1], ['OPD seed42 per-step discrepancy', 'Anchor seed42 per-step discrepancy']):
        ax.set(ylabel=label, yscale='log')
        ax.legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel('Actual optimizer updates')
    fig.suptitle('Actual complete two-epoch lineages; sparse progress logging, no synthetic curves')
    fig.tight_layout()
    fig.savefig(destination/'Fig8_full_training_dynamics.png', dpi=170)
    plt.close(fig)
    write(REPORTS/'tables/full_development_figures_provenance.json', dict(
        script_sha256=sha(__file__), protocol_sha256=sha(ROOT/'protocol.yaml'),
        analysis_sources={p: sha(p) for p in set(analysis_names)}, progress_sources=progress_sources,
        population_sha256=sha(ROOT/'manifests/dev.json'),
        IL_PDMS=float(il[:, :, -1].mean()), T0_PDMS=float(teacher[:, :, -1].mean()),
        interpretation='All observed three-seed full-training endpoints retained; no final-test outcomes used'))
    print('COMPLETE_REAL_THREE_SEED_DEV_FIGURES', len(groups), flush=True)


if __name__ == '__main__':
    main()
