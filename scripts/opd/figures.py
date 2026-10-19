"""All curves and cases come from completed, identity-checked real results."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv
from scripts.opd.common import *
from scripts.opd.analyze import bank,EXPERIMENTS


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'figure.dpi':150})
    (REPORTS/'figures').mkdir(parents=True,exist_ok=True)
    return plt


def figures(split):
    plt=setup();data=read(ROOT/'analysis'/f'formal_{split}.json');base=read(ROOT/'analysis'/f'baseline_{split}.json')
    methods=list(data['methods']);b=bank(split,'E0','IL');t=bank(split,'E0','RL');bj=b[...,-1].mean(-1);tj=t[...,-1].mean(-1)
    vals={e:np.stack([bank(split,f'train_{e}_{s}') for s in config()['train_seeds']]) for e in methods}
    fig,ax=plt.subplots(figsize=(10,4));names=['E0-IL','E0-RL','E0-COPY']+methods
    scores=[base['components'][n]['PDMS'] for n in names[:3]]+[data['methods'][e]['metrics']['PDMS']['estimate'] for e in methods]
    ax.bar(range(len(names)),scores,color=['#8798a5']*3+['#457b9d']*len(methods))
    for i,e in enumerate(methods,3):
        v=data['methods'][e]['metrics']['PDMS'];ax.errorbar(i,v['estimate'],yerr=[[max(0,v['estimate']-v['lower95'])],[max(0,v['upper95']-v['estimate'])]],fmt='none',color='black',capsize=3)
    ax.set_xticks(range(len(names)),names,rotation=35);ax.set(ylabel='Normalized PDMS',ylim=(max(0,min(scores)-.1),1),title=f'{split}: complete population, paired5 draws /3 training seeds')
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig1_PDMS_{split}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5));contrasts=data['contrasts'];labels=list(contrasts)
    for i,key in enumerate(labels):
        v=contrasts[key]['PDMS'];axes[0].errorbar(v['estimate'],i,xerr=[[max(0,v['estimate']-v['lower95'])],[max(0,v['upper95']-v['estimate'])]],fmt='o',capsize=3)
        v=contrasts[key]['M_minus'];estimate=-v['estimate'];lo=-v['upper95'];hi=-v['lower95'];axes[1].errorbar(estimate,i,xerr=[[max(0,estimate-lo)],[max(0,hi-estimate)]],fmt='o',capsize=3)
    for ax,threshold,title in [(axes[0],.002,'Paired mean PDMS difference'),(axes[1],.0005,'Paired M_minus reduction')]:
        ax.set_yticks(range(len(labels)),labels);ax.axvline(0,color='gray',lw=.7);ax.axvline(threshold,color='#d05c50',ls='--',lw=.9,label='Material threshold');ax.set(xlabel=title,title='Positive favors the first method');ax.invert_yaxis();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig1_paired_contrasts_{split}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for e in methods:
        h=read(ROOT/'runs'/f'train_{e}_42'/'progress.json');x=[r['step'] for r in h]
        axes[0].plot(x,[max(r['loss'],1e-12) for r in h],label=e);axes[1].plot(x,[r['grad_norm'] for r in h],label=e)
    axes[0].set(yscale='log',xlabel='Optimizer updates',ylabel='Own training objective',title='Different objectives have different scales')
    axes[1].set(yscale='log',xlabel='Optimizer updates',ylabel='Unclipped gradient norm',title='Seed42; clipping threshold1.0');axes[1].axhline(1,color='grey',ls='--');axes[0].legend(ncol=2)
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig2_learning_{split}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for e in ['E2','E3']:
        if e not in methods:continue
        values=[]
        for s in config()['train_seeds']:
            h=read(ROOT/'runs'/f'train_{e}_{s}'/'progress.json');values.append(h[-1]['per_step'])
        mean=np.mean(values,0);axes[0].plot([80,60,40,20,0],mean,'o-',label=e)
    chain=ROOT/'diagnostics/chain_distributions.json'
    if chain.exists():
        c=read(chain);y=[r['records'][-1]['state_RMS'] for r in c['training_seeds']]
        axes[1].plot(range(6),np.mean(y,0),'o-',label='State RMS')
        response=[r['records'][-1]['teacher_response_RMS'] for r in c['training_seeds']]
        axes[1].plot(range(5),np.mean(response,0),'s--',label='Teacher response RMS');axes[1].legend()
        axes[1].set(xlabel='Chain state index',ylabel='Normalized RMS difference',title='Rollout entering update1000; response sensitivity, not reliability')
    else:axes[1].set_axis_off()
    axes[0].set(xlabel='DDIM timestep',ylabel='Teacher/student mean MSE',yscale='log');axes[0].legend()
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig3_chain_discrepancy_{split}.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ev=[e for e in methods if e in ['E3','E4-A','E4-B','E4-C','E5']]
    for offset,key,label in [(-.18,'zero_to_positive_fraction','Zero -> positive'),(.18,'positive_to_zero_fraction','Positive -> zero')]:
        ax.bar(np.arange(len(ev))+offset,[data['methods'][e]['metrics'][key]['estimate']*len(b) for e in ev],width=.35,label=label)
    ax.set_xticks(range(len(ev)),ev);ax.set(ylabel='Mean scene count across3 trained policies',title='Scene-mean PDMS; zero tolerance1e-12');ax.legend()
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig4_forgetting_{split}.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));sj=vals['E3'][...,-1].mean((0,2))
    for ax,x,label in [(axes[0],bj,'IL'),(axes[1],tj,'RL teacher')]:
        ax.scatter(x,sj,s=5,alpha=.3);ax.plot([0,1],[0,1],color='black',lw=.7);ax.set(xlabel=f'{label} scene mean PDMS',ylabel='E3 mean over3 policies and5 draws')
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig5_paired_{split}.png');plt.close(fig)
    if 'E5' in methods:
        fig,ax=plt.subplots(figsize=(9,4));categories=['A','B','C','D'];groups=data['methods']['E5']['categories']
        for off,k,label in [(-.25,'IL_PDMS','IL'),(0,'RL_PDMS','RL'),(.25,'PDMS','E5')]:ax.bar(np.arange(4)+off,[groups[c][k] or 0 for c in categories],.25,label=label)
        ax.set_xticks(range(4),[f'{c} (n={groups[c]["scenes"]})' for c in categories]);ax.set(ylabel='PDMS',title='Held-out quality strata; training gate never fit on these scenes');ax.legend()
        fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig6_quality_strata_{split}.png');plt.close(fig)
    costs={};fig,ax=plt.subplots(figsize=(8,4))
    for e in methods:
        runs=[read(ROOT/'runs'/f'train_{e}_{s}'/'complete.json') for s in config()['train_seeds']]
        cost=np.mean([r['seconds'] for r in runs])/3600;gain=data['methods'][e]['metrics']['G']['estimate'];costs[e]=dict(mean_training_GPU_hours=float(cost),mean_peak_GPU_GiB=float(np.mean([r['peak_gpu_bytes'] for r in runs])/2**30),gain=gain,
            teacher_queries=[r['teacher_forward_counts'] for r in runs])
        ax.scatter(cost,gain);ax.annotate(e,(cost,gain),xytext=(4,3),textcoords='offset points')
    ax.axhline(0,color='grey',lw=.7);ax.set(xlabel='Single-GPU training hours per policy',ylabel='PDMS gain vs IL',title='Shared encoding/gate/cache and tuning costs reported separately')
    fig.tight_layout();fig.savefig(REPORTS/'figures'/f'Fig8_cost_{split}.png');plt.close(fig)
    write(ROOT/'analysis'/f'costs_{split}.json',costs)


def cases():
    plt=setup();split='dev';records=rows(split);b=bank(split,'E0','IL');t=bank(split,'E0','RL');s=bank(split,'train_E3_42');e=bank(split,'train_E5_42')
    bj=b[...,-1].mean(-1);tj=t[...,-1].mean(-1);sj=s[...,-1].mean(-1);ej=e[...,-1].mean(-1)
    rules={'repair':sj-bj>.05,'regression':sj-bj<-.05,'protected':(sj-bj<-.05)&(ej-bj>-.02),'teacher_transfer_failure':(tj-sj>.1)&(tj>.8)}
    from scripts.opd.scoring import Scoring,load_cache
    details=[]
    for kind,mask in rules.items():
        eligible=np.flatnonzero(mask)
        if not len(eligible):continue
        i=min(eligible,key=lambda i:digest([records[i]['token'],'OPD_case_v1']));row=records[i];token=row['token']
        z=int(np.argmin(abs((s[i,:,-1]-b[i,:,-1])-(sj[i]-bj[i]))));cache=load_cache(row['metric_cache_path'])
        all_arrays=[]
        for folder,key in [('E0','IL'),('E0','RL'),('train_E3_42','train_E3_42'),('train_E5_42','train_E5_42')]:all_arrays.append(np.load(ROOT/'evaluation/dev'/folder/'predictions'/f'{token}.npz')[key])
        all_arrays=np.array(all_arrays);arrays=all_arrays[:,z]
        full=Scoring().batch(cache,all_arrays.reshape(-1,8,3));expected=np.array([b[i],t[i],s[i],e[i]])
        error=float(abs(full['scores'].reshape(4,5,7)-expected).max());assert error<=1e-6,error
        scored={k:(v.reshape(4,5,*v.shape[1:])[:,z] if isinstance(v,np.ndarray) and len(v)==20 else v) for k,v in full.items()}
        ego=cache.ego_state.rear_axle;angle=ego.heading;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        local=lambda xy:(np.asarray(xy)-[ego.x,ego.y])@rotation
        fig,axes=plt.subplots(1,3,figsize=(16,4));ax=axes[0]
        for polygon in cache.drivable_area_map._geometries:
            if polygon.geom_type=='Polygon':
                xy=local(polygon.exterior.coords);ax.fill(xy[:,0],xy[:,1],color='#e9ecef',alpha=.5)
        event_keys=['first_collision_s','first_dac_s','first_ttc_check_failure_s'];times={k:[float(v) if np.isfinite(v) else None for v in scored[k]] for k in event_keys}
        finite=[float(v) for key in event_keys for v in full[key] if np.isfinite(v)];event=min(finite) if finite else 0.
        for tidx,style in [(0,'-'),(int(round(event/.1)),'--')]:
            for obj in cache.observation[tidx].tokens:
                polygon=cache.observation[tidx][obj]
                if polygon.geom_type=='Polygon':
                    xy=local(polygon.exterior.coords);ax.plot(xy[:,0],xy[:,1],style,color='grey',lw=.5,alpha=.6)
        for j,(name,color) in enumerate([('IL','#457b9d'),('RL','#777777'),('E3','#e76f51'),('E5','#2a9d8f')]):
            for draw in range(5):
                ax.plot(all_arrays[j,draw,:,0],all_arrays[j,draw,:,1],color=color,alpha=.16,lw=.7)
            trajectory=arrays[j];ax.plot(trajectory[:,0],trajectory[:,1],'o-',color=color,ms=3,label=name)
            tracked=local(scored['simulated_states'][j,:,:2]);ax.plot(tracked[:,0],tracked[:,1],'--',color=color,lw=.8)
            axes[1].plot(config()['inference_seeds'],[b,t,s,e][j][i,:,-1],'o-',color=color,label=name)
            for k,event_key in enumerate(event_keys):
                event_groups={}
                for draw,moment_all in enumerate(full[event_key].reshape(4,5)[j]):
                    if np.isfinite(moment_all):event_groups.setdefault(float(moment_all),[]).append(config()['inference_seeds'][draw])
                for moment_all,draws in event_groups.items():
                    axes[2].plot(moment_all,k+(j-1.5)*.12,'.',color=color,alpha=.4,ms=5)
                    label='all5' if len(draws)==5 else ','.join(map(str,draws))
                    axes[2].annotate(label,(moment_all,k+(j-1.5)*.12),xytext=(2,5),textcoords='offset points',fontsize=6,color=color)
                moment=times[event_key][j]
                if moment is not None:axes[2].plot(moment,k+(j-1.5)*.12,'o',color=color,ms=5,label=name if k==0 else None)
                else:axes[2].text(4.05,k+(j-1.5)*.12,name+': none',color=color,fontsize=6,va='center')
        xy=all_arrays[...,:2].reshape(-1,2);ax.set_xlim(xy[:,0].min()-8,xy[:,0].max()+8);ax.set_ylim(xy[:,1].min()-8,xy[:,1].max()+8)
        ax.set_aspect('equal');ax.set(xlabel='Rear-axle x (m)',ylabel='Rear-axle y (m)',title=kind+' / '+token);ax.legend()
        axes[1].set(ylim=(-.05,1.05),xlabel='Paired inference seed',ylabel='PDMS',title=f'IL={bj[i]:.3f}, RL={tj[i]:.3f}, E3={sj[i]:.3f}, E5={ej[i]:.3f}');axes[1].legend()
        axes[1].set_xticks(config()['inference_seeds'],list(map(str,config()['inference_seeds'])))
        axes[2].set(xlim=(-.1,5.2),ylim=(-.5,2.5),xlabel='Simulated elapsed time (s)',title='First infractions within4s: all5 / selected highlight')
        axes[2].set_xticks([0,1,2,3,4]);axes[2].axvspan(4,5.2,color='#eeeeee',alpha=.3)
        axes[2].set_yticks([0,1,2],['At-fault collision','Offroad','TTC check failure']);axes[2].grid(axis='x',alpha=.2)
        fig.text(.01,.01,f'Faint: all5 draws; thick: registered representative; dashed: tracked. Objects: current / first event {event:.1f}s, diagnostic only. Shading beyond4s is unscored. TTC check failure is not time to impact.',fontsize=7)
        fig.tight_layout(rect=(0,.05,1,1));fig.savefig(REPORTS/'figures'/f'Fig7_{kind}.png');plt.close(fig)
        path=ROOT/'diagnostics/case_states'/f'{kind}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,simulated_states=full['simulated_states'].reshape(4,5,*full['simulated_states'].shape[1:]),requested_trajectories=all_arrays,scores=full['scores'].reshape(4,5,7))
        details.append(dict(kind=kind,token=token,log=row['log_name'],seed=config()['inference_seeds'][z],scene_means=dict(IL=float(bj[i]),RL=float(tj[i]),E3=float(sj[i]),E5=float(ej[i])),
            event_seconds=times,all_draw_event_seconds={k:[[float(v) if np.isfinite(v) else None for v in row] for row in full[k].reshape(4,5)] for k in event_keys},
            raw_progress_m=scored['progress_raw_m'].tolist(),reference_progress_m=scored['reference_progress_raw_m'],
            extra_case_state_log=str(path),state_log_sha256=sha(path),canonical_score_error=error,rule='First stable token hash among threshold-eligible cases; displayed draw closest to mean E3-minus-IL change'))
    write(ROOT/'analysis/cases.json',details)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev');p.add_argument('--cases',action='store_true');a=p.parse_args()
    figures(a.split)
    if a.cases:cases()
