"""Small publishable figures/tables from actual complete results, no dummy data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import numpy as np
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');a=p.parse_args()
    figures=REPORTS/'figures';figures.mkdir(parents=True,exist_ok=True)
    tables=REPORTS/'tables';tables.mkdir(parents=True,exist_ok=True)
    sources={};metrics=[]
    for folder in sorted((ROOT/'evaluation'/a.split).glob('*')):
        if folder.name.startswith('qualification_') or folder.name.endswith('_screen'):continue
        try:values,rows=predictions(folder.name,a.split)
        except ValueError:continue # Missing results are recorded, never synthesized.
        sources[folder.name]=values
        metrics.append(dict(method=folder.name,training_seeds=1,scenes=len(rows),parents=len({r['driver_record'] for r in rows}),
                      inference_seeds=values.shape[1],PDMS=float(values[:,:,-1].mean()),
                      NC=float(values[:,:,0].mean()),DAC=float(values[:,:,1].mean()),EP=float(values[:,:,2].mean()),
                      TTC=float(values[:,:,3].mean()),Comfort=float(values[:,:,4].mean()),DDC=float(values[:,:,5].mean()),
                      zero_outputs=int((values[:,:,-1]<=1e-8).sum()),
                      source=str(folder)))
    if metrics:
        with (tables/f'actual_{a.split}_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]),lineterminator='\n');writer.writeheader();writer.writerows(metrics)
        # These are per-training-policy descriptive means. Aggregated three-seed
        # inferential comparisons are separate, never pseudo-ensembles here.
        fig,ax=plt.subplots(figsize=(max(7,len(metrics)*.42),4.2))
        ax.bar([r['method'] for r in metrics],[r['PDMS'] for r in metrics],color='#376a93')
        ax.set(ylabel='Canonical PDMS [0,1]',ylim=(0,1),title=f'Actual complete {a.split} evaluations')
        ax.tick_params(axis='x',rotation=75);fig.tight_layout();fig.savefig(figures/f'Fig1_actual_{a.split}_PDMS.png',dpi=160);plt.close(fig)
    if 'B0_IL' in sources:
        baseline=sources['B0_IL'].mean(1)[:,-1]
        names=[n for n in sources if n not in ['B0_IL','B0_COPY']]
        if names:
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for name in names:
                mean=sources[name].mean(1)[:,-1];delta=mean-baseline
                axes[0].scatter(baseline,mean,s=5,alpha=.25,label=name)
                axes[1].scatter(np.maximum(delta,0).mean(),np.maximum(-delta,0).mean(),label=name)
            axes[0].plot([0,1],[0,1],color='black',lw=.7);axes[0].set(xlabel='IL scene mean PDMS',ylabel='New scene mean PDMS')
            axes[1].set(xlabel='M_plus',ylabel='M_minus');axes[1].legend(fontsize=6)
            fig.tight_layout();fig.savefig(figures/f'Fig3_paired_and_tradeoff_{a.split}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7));plotted=False
    for path in sorted((ROOT/'runs').glob('*/progress.json')):
        if path.parent.name.startswith(('qual_','qualification_')):continue
        data=read(path)
        if not data:continue
        steps=[r['step'] for r in data];loss=[r['loss'] for r in data]
        grad=[r.get('gradient_norm',r.get('grad_norm')) for r in data]
        family=1 if 'GRPO' in path.parent.name else 0
        axes[family,0].plot(steps,loss,label=path.parent.name,lw=.9);axes[family,1].plot(steps,grad,lw=.9);plotted=True
    if plotted:
        for row,label in enumerate(['OPD / Anchor latent mean loss','Native GRPO objective']):
            axes[row,0].set(xlabel='Actual optimizer updates',ylabel=label)
            if axes[row,0].lines:axes[row,0].legend(fontsize=5)
            axes[row,1].set(xlabel='Actual optimizer updates',ylabel='Gradient L2 norm')
        fig.tight_layout();fig.savefig(figures/'Fig2_actual_training.png',dpi=160)
    plt.close(fig)
    plan=diagnostic_plan()
    for phase in dict.fromkeys(['local_train_grid','local_cal_grid','local_confirm','local_confirm_expand32',
                               plan['train_name'],plan['cal_name'],plan['confirm_name'],plan['expansion_name']]):
        path=ROOT/'analysis'/f'{phase}.json'
        if not path.exists():continue
        result=read(path)
        if not result['complete']:continue
        values=result['results'];labels=[f"{r['source']} t={r['temperature']} r={r['rho']}" for r in values]
        mean=np.asarray([r['mean'][:3] for r in values]);lo=np.asarray([r['CI95'][0][:3] for r in values]);hi=np.asarray([r['CI95'][1][:3] for r in values])
        fig,ax=plt.subplots(figsize=(max(7,len(values)*.5),4.8));positions=np.arange(len(values))
        for j,label in enumerate(['Value selection','Native mean projection','Guarded native mean']):
            x=positions+(j-1)*.2
            artist=ax.plot(x,mean[:,j],'o',ms=3,label=label)[0]
            ax.vlines(x,lo[:,j],hi[:,j],color=artist.get_color(),lw=.8)
        ax.axhline(.002,color='red',ls='--',lw=.7,label='Registered material gain');ax.axhline(0,color='black',lw=.5)
        role=ROOT/'diagnostics'/phase/'sampling_role.json'
        title=('Exploratory token-hash / ' if role.exists() else '')+phase
        ax.set(xticks=positions,xticklabels=labels,ylabel='Held-out suffix PDMS difference',title=title)
        ax.tick_params(axis='x',rotation=75);ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(figures/f'Fig4_{phase}.png',dpi=160);plt.close(fig)
        write(tables/f'{phase}.json',result)
        if role.exists():write(tables/f'{phase}_sampling_role.json',read(role))
    inventory={}
    for path in (ROOT/'runs').glob('*/config.json'):
        name=path.parent.name
        inventory[name]=dict(config=read(path),completion=read(path.parent/'complete.json') if (path.parent/'complete.json').exists() else None,
                             paused=read(path.parent/'paused.json') if (path.parent/'paused.json').exists() else None)
    write(tables/'actual_run_inventory.json',inventory)
    write(tables/f'render_provenance_{a.split}.json',dict(study=STUDY,script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.yaml'),
          actual_complete_evaluations=list(sources),tables='Descriptive per-policy means; confidence intervals require analyze.py',
          external_artifact_root=str(ROOT)))
    print('REAL_RESULT_FIGURES',a.split,len(metrics),flush=True)


if __name__=='__main__':main()
