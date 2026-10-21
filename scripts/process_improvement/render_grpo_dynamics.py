"""Render observed native training logs at the common completed first endpoint.

These are training observations, not development/test scores or extra policy
replicates. Sparse logging is preserved; no smoothing, endpoint extrapolation,
unlogged clipping-rate estimate or missing-seed filling is performed.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    endpoint=977;records=[]
    fig,axes=plt.subplots(2,3,figsize=(12.2,7.2))
    panels=[('reward','Training group mean PDMS'),('grad_norm','Student gradient norm before clipping'),
            ('signal_fraction','Fraction of groups with reward variation'),
            ('policy_loss','Native signed policy objective'),('BC_loss','Native signed IL reference term'),
            ('loss','Native total objective')]
    for family,color in [('IL','#376a93'),('WARM','#a62c45'),('RL','#638b40')]:
        for seed,style in [(42,'-'),(43,'--'),(44,':')]:
            name=f'GRPO_{family}_{seed}';run=ROOT/'runs'/name
            cost=read(run/'endpoint_costs'/str(endpoint)/'receipt.json')
            checkpoint=run/f'head_{endpoint}.pt'
            if cost['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Endpoint policy changed')
            cfg=read(run/'config.json')
            if cfg['progress_weight']!=5. or cfg['global_batch']!=32 or cfg['group_size']!=8:
                raise ValueError('Noncanonical native training recipe')
            entries=[row for row in read(run/'progress.json') if row['step']<=endpoint]
            if not entries or entries[-1]['step']!=endpoint:
                raise ValueError('Incomplete first endpoint log; never draw a substitute')
            steps=[row['step'] for row in entries]
            if steps!=sorted(set(steps)):raise ValueError('Duplicate or unordered native observations')
            selected=[{key:row[key] for key in ['step','physical_scores','gradient_clipped',*[p[0] for p in panels]]}
                      for row in entries]
            for field,_ in panels:
                if not np.isfinite([row[field] for row in selected]).all():raise ValueError('Nonfinite native log')
            for ax,(field,title) in zip(axes.flat,panels):
                ax.plot(steps,[row[field] for row in selected],color=color,ls=style,lw=.7,alpha=.65,
                        label=family+' seed'+str(seed))
                ax.set(xlabel='Actual optimizer updates (logged observations)',title=title,xlim=(0,endpoint))
            records.append(dict(name=name,checkpoint_sha256=sha(checkpoint),
                endpoint_cost_receipt_sha256=sha(run/'endpoint_costs'/str(endpoint)/'receipt.json'),
                observations=selected,observations_sha256=digest(selected),
                logged_points=len(selected),logged_clipped_points=sum(row['gradient_clipped'] for row in selected),
                mean_logged_signal_fraction=float(np.mean([row['signal_fraction'] for row in selected]))))
    axes[0,0].legend(fontsize=6,ncol=3)
    axes[0,1].axhline(1.,color='black',ls=':',lw=.8,label='Configured gradient clipping threshold')
    fig.suptitle('Native GRPO: all 9 real lineages through the common 250,112-score endpoint\n'
                 'Training logs only; sparse observations retained without smoothing',fontsize=10)
    fig.text(.5,.012,'Native policy/BC terms are signed objectives. Logged clipping counts are not full-update rates. No development or test scores are shown.',
             ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.03,1,.94))
    fig.savefig(REPORTS/'figures/Fig13_native_training_dynamics_first_endpoint.png',dpi=170);plt.close(fig)
    value=dict(status='REAL_NINE_LINEAGE_LOG_FIGURE',generator_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'protocol.yaml'),endpoint=endpoint,actual_physical_scores_per_seed=endpoint*256,
        records=records,interpretation='Observed D_fit training logs; each policy remains separate; no smoothed or fabricated observations; no causal attribution of development-score changes to individual loss terms')
    write(ROOT/'analysis/native_training_dynamics_first_endpoint.json',value)
    write(REPORTS/'tables/native_training_dynamics_first_endpoint.json',value)
    print('REAL_NATIVE_LOG_FIGURE',len(records),'policies',flush=True)


if __name__=='__main__':main()
