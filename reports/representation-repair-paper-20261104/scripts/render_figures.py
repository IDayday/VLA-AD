"""Regenerate all paper figures from small aggregate tables; no research mount needed."""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'representation-repair-paper-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def main(root):
    root=Path(root)
    data=json.loads((root/'tables/figure_data.json').read_text())
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.18,'axes.axisbelow':True,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','savefig.dpi':180})
    blue='#28658C';orange='#CC773E';green='#378470';gray='#7B8791'
    def save(fig,name):
        for ext in ['svg','pdf','png']:
            opts={'metadata':{'Date':None}} if ext=='svg' else ({'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {})
            fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',**opts)
            if ext=='svg':
                svg=out/f'{name}.{ext}'
                svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5.7));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color=blue):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=color+'14',edgecolor=color,linewidth=1.2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.2,linespacing=1.4)
    def arrow(x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color=gray))
    ax.text(.1,5.75,'Offline: privileged construction on TRAIN',fontweight='bold',fontsize=12)
    upper=[('Frozen driver\nfailed scene',.1),('GT-guided latent\noptimization',2.5),('Behavior-validated\nsame-scene pairs',4.9),('Normalized D48\nfull-384 clustering',7.3),('8 shared experts\n6 shifts + 2 MLPs',9.7)]
    for label,x in upper:box(x,4.35,2.12,1.0,label)
    for (_,x),(_,xx) in zip(upper,upper[1:]):arrow(x+2.15,4.85,xx-.1,4.85)
    ax.text(.1,3.65,'Online: current front view + navigation / vehicle state',fontweight='bold',fontsize=12)
    lower=[('Frozen driving\nprefix: h',.1),('Base + 8 repaired\nlatent conditions',2.5),('A: three-step\nbehavior previews',4.9),('F/H: front-view\njoint scoring',7.3),('Selected original G\nfive full steps',9.7)]
    for j,(label,x) in enumerate(lower):box(x,2.1,2.12,1.05,label,orange if j in [2,3] else blue)
    for (_,x),(_,xx) in zip(lower,lower[1:]):arrow(x+2.15,2.62,xx-.1,2.62)
    ax.text(10.75,3.99,'fixed expert bank',ha='center',fontsize=8,color=gray)
    ax.text(.1,1.35,'Train A from original G behavior; train F/H from actual candidate feedback.',fontsize=10)
    ax.text(.1,.87,'Original driving prefix / projector / G stay frozen.  Only the chosen original G branch runs in full.',fontsize=10)
    ax.text(.1,.4,'Previews are scoring inputs, not executable prefixes.  A fixed S0 probe determines the scene choice for S0-S4.',fontsize=9,color='#425564')
    save(fig,'fig01_method_pipeline')

    main_rows=data['main_results']
    def rl(dose,pop):return next(r for r in main_rows if r['branch']=='RL_v1' and r['population']==pop and r['configuration'].endswith('_'+dose))
    fig,axs=plt.subplots(1,2,figsize=(9,3.7))
    for ax,pop in zip(axs,['TUNE','TEST']):
        rows=[rl(d,pop) for d in ['L1','L15','L2']]
        ax.errorbar([1,1.5,2],[r['mean'] for r in rows],yerr=[r['training_seed_SD'] for r in rows],fmt='o-',color=blue,capsize=4)
        ax.scatter([1.5],[rows[1]['mean']],marker='*',s=140,color=orange,zorder=3)
        ax.set(xlabel='Global repair strength',ylabel='PDMS',title=pop,xticks=[1,1.5,2],xlim=(.85,2.15));ax.margins(y=.25)
        for x,r in zip([1,1.5,2],rows):ax.annotate(f"{r['mean']:.4f}",(x,r['mean']),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
    fig.suptitle('Matched workflow: strength chosen by complete TUNE',fontsize=12)
    fig.text(.5,-.02,'Error bars: sample SD across three independently trained models. No ensemble.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig02_strength_selection')

    labels={'E2_RANDOM_GROUPS':'Direction grouping vs. random grouping (E1 - E2)',
        'E3_SHUFFLED_PAIRS':'Same-scene vs. shuffled pairs (E1 - E3)',
        'E4_RANDOM_D48':'Structured vs. random directions in D48 (R6 - E4)',
        'R6_FIXED_NATIVE_CLUSTERS':'Mixed vs. all-fixed operators (Main - R6)'}
    rows=data['ablations'];fig,ax=plt.subplots(figsize=(9,4.1))
    for i,r in enumerate(rows):
        value=r['ordinary_PDMS_difference'];lo,hi=r['descriptive_session_bootstrap95CI']
        ax.errorbar(value,i,xerr=np.array([[value-lo],[hi-value]]),fmt='o',capsize=4,color=blue if lo>0 else gray)
        ax.text(.62,i,f'{value:+.4f}',va='center',fontsize=9)
    ax.axvline(0,color='black',linewidth=.8);ax.set(yticks=range(len(rows)),yticklabels=[labels[r['other']] for r in rows],xlabel='Ordinary TEST PDMS difference',xlim=(-.16,.73),title='Core frontend controls after retraining each library');ax.invert_yaxis()
    ax.grid(axis='y',visible=False)
    fig.text(.5,-.04,'44-session descriptive bootstrap 95% intervals; not training-seed CIs; no multiplicity correction.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig03_core_ablations')

    main=next(r for r in main_rows if r['role']=='formal_main' and r['population']=='TEST')
    v2=next(r for r in main_rows if r['role']=='formal_transfer' and r['population']=='TEST')
    hard=next(r for r in main_rows if r['branch']=='IL_v2_navhard')
    fig,axs=plt.subplots(1,3,figsize=(10,4))
    for ax,row,title in zip(axs,[main,v2,hard],['RL / v1 PDMS','IL / v2 ePDMS','IL / v2 navhard ePDMS']):
        ax.bar([0,1],[row['Base'],row['mean']],color=[gray,blue],width=.6)
        if row['training_seed_SD']!='':ax.errorbar([1],[row['mean']],yerr=[row['training_seed_SD']],fmt='none',color='black',capsize=4)
        for x,val in enumerate([row['Base'],row['mean']]):ax.text(x,val+1.2,f'{val:.4f}',ha='center',fontsize=9)
        ax.set(xticks=[0,1],xticklabels=['Base','Repair'],ylim=(0,104),title=title+f"\nGain {row['delta']:+.4f}")
    axs[0].set_ylabel('Score (0-100)')
    fig.text(.5,-.01,'Left / middle: three-model means +/- seed SD. Right: one TUNE-selected model, five noise seeds.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig04_main_results')

    bands=data['main_bands'];values=[r['contribution_to_total_delta'] for r in bands]
    fig,ax=plt.subplots(figsize=(7.4,4));start=0
    for i,v in enumerate(values):
        ax.bar(i,abs(v),bottom=min(start,start+v),color=green if v>=0 else orange,width=.58)
        ax.text(i,max(start,start+v)+.08,f'{v:+.4f}',ha='center',fontsize=10)
        if i<2:ax.plot([i+.3,i+.7],[start+v,start+v],color=gray,linestyle='--',linewidth=.8)
        start+=v
    ax.bar(3,start,color=blue,width=.58);ax.text(3,start+.08,f'{start:+.4f}',ha='center',fontsize=10)
    ax.axhline(0,color='black',linewidth=.7);ax.set(xticks=range(4),xticklabels=['Base < 70\n845 scenes','Base 70-90\n1,742 scenes','Base > 90\n9,559 scenes','Net gain\n12,146 scenes'],ylabel='Contribution to overall TEST PDMS gain',ylim=(-.1,2.7),title='Repair gains and preservation losses coexist')
    fig.text(.5,-.02,'Bands use Base five-noise means. Contributions averaged over three separately evaluated models.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig05_repair_and_preservation')

    rows=data['cost'];fig,axs=plt.subplots(1,2,figsize=(8.2,3.7))
    for ax,key,label in [(axs[0],'P50_ms','P50 end-to-end latency (ms)'),(axs[1],'peak_allocated_GiB','Peak allocated memory (GiB)')]:
        vals=[r[key] for r in rows];ax.bar([0,1],vals,color=[gray,blue],width=.6)
        for x,v in enumerate(vals):ax.text(x,v*1.02,f'{v:.2f}',ha='center')
        ax.set(xticks=[0,1],xticklabels=['Base','Core K8 / 1.5'],ylabel=label,ylim=(0,max(vals)*1.18))
    fig.suptitle('Raw current-front input: 128 scenes x 5 repeats on PPU-ZW810E',fontsize=11)
    fig.text(.5,-.02,'Same device type; model resident; JPEG decode included; storage reads reported separately.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig06_deployment_cost')

    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    colors=[blue,orange,green,'#8D6799','#8F874A']
    names=['PURE_SYN','REAL25_LONG','REAL50_LONG','PURE_SYN_CALIBRATED','PURE_SYN_STATE_CALIBRATED']
    for name,col in zip(names,colors):
        rows=sorted([r for r in data['synthetic_curve'] if r['name']==name],key=lambda r:r['updates'])
        axs[0].plot([r['updates'] for r in rows],[r['TUNE'] for r in rows],'.-',color=col,label=name,linewidth=1,markersize=3)
    rows=sorted([r for r in data['synthetic_curve'] if r['name']=='CRITICAL_POSTTRAIN'],key=lambda r:r['updates'])
    axs[1].plot([r['updates'] for r in rows],[r['TUNE'] for r in rows],'o-',color=blue,markersize=3)
    best=max(rows,key=lambda r:r['TUNE']);axs[1].scatter([best['updates']],[best['TUNE']],marker='*',s=120,color=orange,zorder=3)
    for ax in axs:
        ax.axhline(data['original_RL_TUNE'],color='black',linestyle='--',linewidth=.9)
        ax.set(xlabel='Recorded training updates / data iterations',ylim=(93.0,94.9))
    axs[0].set(ylabel='Complete real TUNE PDMS',title='Five 3,072-update courses')
    axs[0].legend(fontsize=6.5,loc='lower left')
    axs[1].set_title('Critical-scene post-training')
    axs[1].annotate('Selected: 3,398\nTUNE 94.6198',(best['updates'],best['TUNE']),xytext=(1900,94.77),fontsize=8,arrowprops={'arrowstyle':'-','color':gray})
    fig.text(.5,-.025,'Dashed: original RL Base. Different courses have different data / trainability / rewards; no pure-real budget control.',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'fig07_synthetic_rl_curves')
    print('Rendered 7 figures in SVG, PDF and PNG.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);main(p.parse_args().root)
