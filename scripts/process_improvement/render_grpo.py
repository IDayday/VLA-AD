"""Plot only actually completed three-seed native GRPO budget endpoints."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    observed=[];missing=[]
    for label,family,color in [('R1','IL','#376a93'),('R2','WARM','#a62c45'),('R3','RL','#638b40')]:
        for step in [977,3907,15625]:
            path=ROOT/'analysis'/f'{label}_{family}_step{step}_vs_IL.json'
            if not path.exists():
                missing.append(dict(family=family,step=step,status='NO_COMPLETE_THREE_SEED_ANALYSIS'))
                continue
            record=read(path);methods=[f'GRPO_{family}_{seed}_step{step}' for seed in [42,43,44]]
            if (record['methods']!=methods or record['training_seeds']!=3 or
                    record['baseline']!='B0_IL' or record['split']!='dev' or
                    record['cluster_unit']!='parent' or record['scenes']!=len(membership('dev')) or
                    record['analysis_source']['population_sha256']!=sha(ROOT/'manifests/dev.json')):
                raise ValueError('Figure source is not a complete locked three-seed group')
            costs=[]
            for seed,method in zip([42,43,44],methods):
                run=ROOT/'runs'/f'GRPO_{family}_{seed}'
                receipt=run/'endpoint_costs'/str(step)/'receipt.json'
                value=read(receipt)
                checkpoint=run/f'head_{step}.pt'
                spec=read(ROOT/'evaluation/dev'/method/'config_0.json')['spec']
                if not value['checkpoint_sha256']==spec['checkpoint_sha256']==sha(checkpoint):
                    raise ValueError('Cost and completed policy identities differ')
                if value['ledger']['budget']['physical_scores']!=step*256:
                    raise ValueError('Actual native scoring budget mismatch')
                costs.append(dict(training_seed=seed,receipt_sha256=sha(receipt),
                                  native_phase_GPU_hours=value['ledger']['GPU_seconds']/3600))
            observed.append(dict(label=label,family=family,color=color,step=step,scores=step*256,
                                 record=record,analysis_path=str(path),analysis_sha256=sha(path),costs=costs))
    if not observed:raise ValueError('No real complete three-seed native endpoints; do not draw placeholders')
    il,_=predictions('B0_IL','dev');teacher,_=predictions('B0_RL','dev')
    il_mean=float(il[:,:,-1].mean());teacher_mean=float(teacher[:,:,-1].mean())
    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    fields=['PDMS','G','M_plus','M_minus']
    titles=['Mean PDMS','Paired gain over IL','Positive scene-mean changes','Negative scene-mean magnitude']
    for ax,field,title in zip(axes.flat,fields,titles):
        for row in observed:
            value=row['record']['point'][field];lo,hi=row['record']['CI95'][field]
            x=row['scores']/1e6
            ax.scatter(x,value,color=row['color'],s=25,label=row['label']+' '+row['family'])
            ax.vlines(x,lo,hi,color=row['color'],lw=1.1)
        if field=='PDMS':
            ax.axhline(il_mean,color='gray',ls=':',lw=.8,label='Fixed IL mean')
            ax.axhline(teacher_mean,color='black',ls='--',lw=.8,label='Fixed T0 mean')
        else:ax.axhline(0,color='gray',lw=.6)
        ax.set(xlabel='Actual new native candidate scores (millions)',ylabel='Normalized PDMS',title=title,
               xticks=[.250112,1.000192,4.],xticklabels=['0.250112','1.000192','4.000000'],xlim=(.05,4.2))
    handles,labels=axes[0,0].get_legend_handles_labels()
    unique=dict(zip(labels,handles));axes[0,0].legend(unique.values(),unique.keys(),fontsize=7)
    fig.suptitle('Complete dev groups: 3 training x 5 paired inference seeds; parent-block 95% intervals\n'
                 'Fixed checkpoints only; missing groups omitted; no interpolated learning curves',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94))
    destination=REPORTS/'figures/Fig12_native_registered_budget_endpoints.png'
    fig.savefig(destination,dpi=170);plt.close(fig)
    write(REPORTS/'tables/native_GRPO_figures_provenance.json',dict(
        generator_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.yaml'),
        observed_groups=[{k:v for k,v in row.items() if k not in ['record','color']} for row in observed],
        missing_groups=missing,IL_PDMS=il_mean,T0_PDMS=teacher_mean,
        intervals='Parent record x training-policy x globally paired inference IDs; absolute PDMS intervals differ from paired differences',
        costs='Native-phase training ledgers only; not total compute. Warm initialization, tuning, cache and evaluation costs are separate and must not be omitted in total-budget comparison.',
        interpretation='Three models remain separate policies; no policy ensemble, missing-score fill or checkpoint selection'))
    print('REAL_THREE_SEED_NATIVE_BUDGET_FIGURE',len(observed),flush=True)


if __name__=='__main__':main()
