"""Plot actual predeclared coverage/exposure controls, training seed42 only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series = [
        ('4115 scenes', '#788998', [
            ('scaling_legacy_4k_OPD_42_epoch2', 8256),
            ('scaling_legacy_4k_OPD_42_epoch8', 32928),
            ('scaling_legacy_4k_OPD_42_epoch64k', 64000)]),
        ('25958 scenes', '#376a93', [
            ('scaling_train_25k_OPD_42_epoch1', 25984),
            ('scaling_train_25k_OPD_42_epoch2', 51936),
            ('scaling_train_25k_OPD_42_epoch64k', 64000),
            ('scaling_train_25k_OPD_42_epoch4', 103840)]),
        ('74885 scenes', '#a62c45', [('full_OPD_42_exposure64000', 64000)])]
    sources = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for label, color, points in series:
        records = []
        for name, exposure in points:
            path=ROOT/'analysis'/f'scaling_{name}_vs_B0_IL.json'
            record=read(path)
            assert record['scenes']==1696 and record['training_seeds']==1
            sources[str(path)]=sha(path)
            records.append(record)
        axes[0].plot([e for _,e in points], [r['point']['PDMS'] for r in records], 'o-',
                     color=color, label=label)
        # Single trained policy, paired parent/noise uncertainty only.
        final=next(r for (name,e),r in zip(points,records) if e==64000)
        axes[1].scatter(final['point']['M_plus'], final['point']['M_minus'], color=color, label=label)
    axes[0].axhline(.9151871137007114, color='gray', ls=':', label='Original IL')
    axes[0].axhline(.9496275362442816, color='black', ls='--', label='RL / COPY')
    axes[0].axvline(64000, color='gray', ls=':', lw=.7)
    axes[0].set(xlabel='Actual scene exposures', ylabel='Complete dev PDMS',
                title='Fixed recipe, one training seed; all registered points')
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel='M_plus relative to IL', ylabel='M_minus relative to IL',
                title='Matched 64000 exposures / 2000 updates')
    axes[1].legend(fontsize=7)
    fig.suptitle('Coverage and exposure are distinct; descriptive seed42 outcomes, not three-seed confirmation')
    fig.tight_layout()
    target=REPORTS/'figures/Fig9_coverage_and_exposure.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=170)
    plt.close(fig)
    write(REPORTS/'tables/scaling_figures_provenance.json',dict(
        script_sha256=sha(__file__),sources=sources,protocol_sha256=sha(ROOT/'protocol.yaml'),
        training_seeds=1,inference_seeds=5,scope='Seed42 scaling diagnostics; not an ensemble or a replacement for full three-seed inference'))
    print('REAL_SCALING_FIGURE_COMPLETE',len(sources),flush=True)


if __name__=='__main__':main()
