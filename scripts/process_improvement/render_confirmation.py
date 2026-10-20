"""Real independent mechanism confirmation, including its negative admission gate."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=ROOT/'analysis/confirmation_decomposition.json';d=read(path)
    gate=read(ROOT/'selection/process_gate.json');primary=d['results']['OLD_OPD']
    assert d['status']=='COMPLETE' and primary['scenes']==512
    figure=REPORTS/'figures';figure.mkdir(parents=True,exist_ok=True)
    labels=['Finite-bank P1 - P0','Raw native mean - P0','Guarded P2 - P0'];x=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    values=np.array(primary['mean'][:3]);bounds=np.array(primary['CI95'])[:,:3]
    axes[0].plot(x,values,'o',color='#a62c45');axes[0].vlines(x,bounds[0],bounds[1],color='#a62c45')
    axes[0].axhline(gate['material_gain'],color='black',ls='--',lw=.8,label='Registered material gain 0.002')
    axes[0].set(ylim=(0,.0022),title='Primary OLD_OPD states: full admission scale')
    axes[0].legend(fontsize=8)
    top=0
    for j,(name,color) in enumerate([('IL','#376a93'),('OLD_OPD','#a62c45'),('RL','#e88a28')]):
        value=d['results'][name];means=np.array(value['mean'][:3]);ci=np.array(value['CI95'])[:,:3]
        position=x+(j-1)*.10
        axes[1].plot(position,means,'o',color=color,label=name,ms=4)
        axes[1].vlines(position,ci[0],ci[1],color=color,lw=.8);top=max(top,float(ci[1].max()))
    axes[1].set(ylim=(0,top*1.12),title='Zoom: small positive local effects')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set(xticks=x,xticklabels=labels,ylabel='Normalized PDMS difference')
        ax.tick_params(axis='x',rotation=20);ax.axhline(0,color='gray',lw=.5)
    fig.suptitle('Held-out 512 scenes / 7 parents; paired banks and test noise; local programs, not student scores')
    fig.tight_layout();fig.savefig(figure/'Fig10_independent_process_confirmation.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,4.5));means=np.array(primary['mean'][3:6]);ci=np.array(primary['CI95'])[:,3:6]
    ax.plot(x,means,'o',color='#376a93');ax.vlines(x,ci[0],ci[1],color='#376a93');ax.axhline(0,color='black',lw=.7)
    ax.set(xticks=x,xticklabels=['P1 - raw mean','Raw mean - guarded mean','P1 - guarded mean'],
           ylabel='Paired normalized PDMS difference',title='Primary held-out procedure differences; not KL projection error')
    fig.tight_layout();fig.savefig(figure/'Fig11_paired_process_attenuation.png',dpi=170);plt.close(fig)
    write(REPORTS/'tables/confirmation_figures_provenance.json',dict(script_sha256=sha(__file__),
        decomposition_sha256=sha(path),protocol_sha256=sha(ROOT/'protocol.yaml'),
        scene_manifest_sha256=sha(ROOT/'manifests'/f"{diagnostic_plan()['confirm_split']}.json"),
        scope='Actual held-out local programs; no virtual trained-process-student point'))
    print('REAL_INDEPENDENT_CONFIRMATION_FIGURES',gate['status'],flush=True)


if __name__=='__main__':main()
