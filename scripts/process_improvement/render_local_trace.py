"""Illustrate attenuation using an already saved, hash-selected real trace.

No rollout, simulation, new scoring, target construction or selection of a
scientific hyperparameter occurs here. The displayed largest attenuation is a
case explanation within the pre-existing trace subset, not a population test.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    name=diagnostic_plan()['confirm_name']
    paths=sorted((ROOT/'diagnostics'/name/'traces').glob('*_OLD_OPD_*.pt'))
    if not paths:raise ValueError('No real saved confirmation traces')
    candidates=[]
    for path in paths:
        value=torch.load(path,map_location='cpu',weights_only=False);r=value['record']
        candidates.append((r['selection_gain']-r['projection_gain'],path,value))
    _,path,value=sorted(candidates,key=lambda x:(-x[0],x[1].name))[0]
    r=value['record'];weights=value['weights'].numpy()
    independent=value['independent_scores']
    reconstructed=(weights[:,None,None]*independent).sum(0)
    if not np.allclose(reconstructed,np.asarray(r['P1']),rtol=0,atol=1e-12):
        raise ValueError('Stored weighted candidate labels do not reconstruct P1')
    gains=[r['selection_gain'],r['projection_gain'],r['guarded_gain']]
    fig,axes=plt.subplots(1,3,figsize=(12.1,4.3))
    ids=np.arange(len(weights));fits=value['fit_scores'][:,:,-1].mean(1);tests=independent[:,:,-1].mean(1)
    axes[0].scatter(ids,fits,label='Fit suffix Q',s=13,color='#376a93')
    axes[0].scatter(ids,tests,label='Held-out suffix Q',s=13,color='#a62c45',marker='x')
    axes[0].set(xlabel='Original candidate ID',ylabel='Absolute normalized PDMS',title='Fixed finite candidate bank')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
    axes[0].legend(fontsize=7)
    twin=axes[0].twinx();twin.bar(ids,weights,color='gray',alpha=.15)
    twin.set_ylabel('Locked fit-derived mixture weight',fontsize=8)
    shift=((value['proposed_mean']-value['reference_mean'])/value['std']).numpy()[0]
    for index,label in enumerate(['x','y','heading']):
        axes[1].plot(np.arange(1,9)*.5,shift[:,index],marker='.',lw=1,label=label)
    axes[1].axhline(0,color='gray',lw=.5)
    axes[1].set(xlabel='Joint action trajectory point time [s]',ylabel='Normalized mean shift / actual sigma',
                title='Projection in normalized diffusion state')
    axes[1].legend(fontsize=7)
    axes[2].bar(['P1 selection','Raw mean','Guarded P2'],gains,color=['#e88a28','#376a93','#638b40'])
    axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    axes[2].axhline(0,color='gray',lw=.5)
    axes[2].set(ylabel='Paired held-out PDMS gain over P0',title='Different local programs, same state')
    for ax in axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.tick_params(labelsize=8)
    token=value['row']['token']
    fig.suptitle(f'Actual saved OLD_OPD trace: {token}, denoising index {r["index"]}, timestep {r["timestep"]}',fontsize=10)
    fig.text(.5,.018,'One fixed bank/state: not a population estimate or learned student. P1 is a mixture, not an averaged trajectory. No new scoring.',
             ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.055,1,.93))
    fig.savefig(REPORTS/'figures/Fig14_real_local_projection_trace.png',dpi=170);plt.close(fig)
    accepted=sum(bool(x[2]['record']['accepted']) for x in candidates)
    record=dict(status='REAL_CACHED_TRACE_ILLUSTRATION',generator_sha256=sha(__file__),
        diagnostic=name,trace_path=str(path),trace_sha256=sha(path),record=r,
        trace_subset=len(paths),accepted_in_trace_subset=accepted,
        case_rule='Largest P1-minus-raw attenuation among the pre-existing OLD_OPD bank0 hash traces; filename tie-break; explanatory case selection, not a new confirmatory test',
        unavailable_case='No accepted OLD_OPD case exists in the saved trace subset; no success case fabricated' if not accepted else None,
        physical_scores_added=0,logical_scores_added=0,
        interpretation='Numerical conditional consequence only; no behavioral intent, environment interaction, causal factor attribution or student learning claim. Trajectory points are one joint action; mixture P1 has no single mean-trajectory score.')
    write(ROOT/'analysis/local_projection_trace_case.json',record)
    write(REPORTS/'tables/local_projection_trace_case.json',record)
    print('REAL_CACHED_TRACE_ONLY_FIGURE',token,gains,flush=True)


if __name__=='__main__':main()
