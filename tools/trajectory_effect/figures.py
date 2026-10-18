"""Publication-exportable plots drawn only from completed numeric artifacts."""
from __future__ import annotations
import numpy as np
from .common import *

FIGURES=REPORT_DIR/'figures'


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    FIGURES.mkdir(parents=True,exist_ok=True)
    return plt


def h1():
    plt=setup();value=read(ROOT/'analysis/h1.json');rows=[r for r in value['families'] if r['split']=='audit']
    changes=read(ROOT/'analysis/h1_scene_changes.json');audit={r['family'] for r in rows}
    data=[r['delta_J'] for r in changes if r['family'] in audit and r['step']==4]
    from .bank import baseline_score_path
    base=np.stack([np.load(baseline_score_path(r['token']))['scores'][:,-1] for r in scene_rows('D_audit')])
    noise=base[:,5:10].mean(-1)-base[:,:5].mean(-1)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for d,label in [(data,'Actual update, step4'),(noise,'Same policy, independent5-draw means')]:
        x=np.sort(d);axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=label)
    axes[0].axvline(0,color='black',lw=.7);axes[0].set(xlabel='Scene-mean PDMS change',ylabel='Empirical cumulative probability');axes[0].legend(fontsize=8)
    xs=np.arange(len(rows))
    for offset,key,color in [(-.22,'M_plus','#2a9d8f'),(0,'M_minus','#d05c50'),(.22,'G','#457b9d')]:
        y=np.array([r['metrics'][key]['estimate'] for r in rows]);lo=np.array([r['metrics'][key]['lower95'] for r in rows]);hi=np.array([r['metrics'][key]['upper95'] for r in rows])
        axes[1].bar(xs+offset,y,width=.21,label=key,color=color)
        axes[1].errorbar(xs+offset,y,yerr=np.maximum(np.stack([y-lo,hi-y]),0),fmt='none',color='black',lw=.8,capsize=2)
    axes[1].set_xticks(xs,[r['family']+' / '+str(r['step']) for r in rows],rotation=35,ha='right');axes[1].set(ylabel='Normalized PDMS',title='Paired log bootstrap 95% intervals');axes[1].legend()
    fig.tight_layout();fig.savefig(FIGURES/'h1_update_noise.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for r in value['families']:
        x=r['metrics']['M_plus']['estimate'];y=r['metrics']['M_minus']['estimate']
        assert abs((x-y)-r['metrics']['G']['estimate'])<1e-10
        ax.scatter(x,y,marker='o' if r['step']==4 else 'x',label=None,color={'fit':'#adb5bd','cal':'#e9c46a','audit':'#e76f51'}[r['split']])
        if r['split']=='audit':ax.annotate(r['family']+'/'+str(r['step']),(x,y),fontsize=8)
    lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],'k--',lw=.8,label='G=0');ax.set(xlabel='M_plus',ylabel='M_minus',title='G = M_plus - M_minus');ax.legend();fig.tight_layout();fig.savefig(FIGURES/'gain_degradation.png');plt.close(fig)


def probes(prefix='formal'):
    plt=setup();value=read(ROOT/'analysis'/f'probes_{prefix}_D_audit.json');arms=['T','FQ','FE','JQ','JE','ZERO']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,arm in enumerate(arms):
        v=value['arms'][arm]['intervals']['RMSE'];axes[0].errorbar(i,v['estimate'],yerr=[[max(0,v['estimate']-v['lower95'])],[max(0,v['upper95']-v['estimate'])]],fmt='o',capsize=4)
    axes[0].set_xticks(range(len(arms)),arms);axes[0].set(ylabel='Actual-update delta_R RMSE',title='Unseen logs x unseen update families')
    pairs=['JQ-vs-FQ','JE-vs-FE','JE-vs-JQ']
    for i,key in enumerate(pairs):
        v=value['comparisons'][key];axes[1].errorbar(i,100*v['estimate'],yerr=[[100*max(0,v['estimate']-v['lower95'])],[100*max(0,v['upper95']-v['estimate'])]],fmt='o',capsize=4)
    axes[1].axhline(10,color='#d05c50',ls='--',label='Registered material threshold10%');axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set_xticks(range(3),pairs);axes[1].set(ylabel='Relative RMSE reduction (%)',title='Paired log/family/train/inference bootstrap');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIGURES/'consequence_prediction.png');plt.close(fig)


def gated_rl(reason):
    plt=setup();fig,axes=plt.subplots(1,2,figsize=(11,3))
    for ax,title in zip(axes,['Same policy-update budget: RL-Q vs RL-E','Matched positive PDMS gain: RL-Q vs RL-E']):
        ax.set_axis_off();ax.set_title(title);ax.text(.5,.55,'NOT RUN: stage gate not passed',ha='center',va='center',fontsize=13)
        ax.text(.5,.25,reason,ha='center',va='center',wrap=True,fontsize=10)
    fig.tight_layout();fig.savefig(FIGURES/'rl_comparison_gate.png');plt.close(fig)


def cases(prefix='formal'):
    plt=setup()
    from .scoring import load_cache,Scoring
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import get_collision_type
    from .updates import trajectory_path,score_path,families
    from .bank import baseline_path,baseline_score_path
    from .probe_runner import merge_eval
    audit={f['id'] for f in families() if f['split']=='audit'}
    changes=[r for r in read(ROOT/'analysis/h1_scene_changes.json') if r['family'] in audit]
    choices={}
    for kind,condition in [('improvement',lambda r:r['delta_J']>.02),('degradation',lambda r:r['delta_J']<-.02)]:
        eligible=[r for r in changes if condition(r)]
        if eligible:choices[kind]=min(eligible,key=lambda r:digest([r['token'],r['family'],r['step'],'case']))
    try:
        rows,pred,target=merge_eval(f'{prefix}_JE_42','D_audit');failures=[]
        lookup={(r['token'],r['family'],r['step']):r for r in changes}
        for i,r in enumerate(rows):
            for f,name in enumerate(r['families']):
                for s,step in enumerate((1,4)):
                    if abs(pred[i,f,s].mean()-target[i,f,s].mean())>=.02:
                        rec=dict(lookup[r['token'],name,step],predicted_delta_J=float(pred[i,f,s].mean()))
                        failures.append(rec)
        if failures:choices['prediction_failure']=min(failures,key=lambda r:digest([r['token'],r['family'],r['step'],'case']))
    except ValueError:pass
    lookup={r['token']:r for r in scene_rows('D_audit')};details=[]
    for kind,case in choices.items():
        token=case['token'];row=lookup[token];cache=load_cache(row['metric_cache_path'])
        old=np.load(baseline_path(token))['trajectories'][:5];new=np.load(trajectory_path(case['family'],case['step'],token))['trajectories']
        oldscore=np.load(baseline_score_path(token));newscore=np.load(score_path(case['family'],case['step'],token))
        z=int(np.argmin(abs(np.array(case['paired_deltas'])-case['delta_J'])))
        ego=cache.ego_state.rear_axle;angle=ego.heading;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        def local(xy):return (np.asarray(xy)-np.array([ego.x,ego.y]))@rotation
        fig,axes=plt.subplots(1,3,figsize=(15,4));ax=axes[0]
        for polygon in cache.drivable_area_map._geometries:
            if polygon.geom_type=='Polygon':
                xy=local(np.asarray(polygon.exterior.coords));ax.fill(xy[:,0],xy[:,1],color='#e9ecef',alpha=.35,lw=.3,edgecolor='#adb5bd')
        event_times=[]
        for key in ['first_collision_s','first_dac_s','first_ttc_check_failure_s']:
            t=float(newscore[key][z]);event_times.append(t)
        finite=[t for t in event_times if np.isfinite(t)];event=min(finite) if finite else 0.
        collision_objects=[]
        if np.isfinite(event_times[0]):
            diagnostic=Scoring();check=diagnostic.batch(cache,np.stack([old[z],new[z]]))
            assert np.allclose(check['scores'][1],newscore['scores'][z],rtol=0,atol=1e-6)
            tidx=int(round(event_times[0]/.1));sc=diagnostic.scorer;polygon=sc._ego_polygons[2,tidx]
            for obj_token in cache.observation[tidx].intersects(polygon):
                if obj_token not in cache.observation.unique_objects:continue
                obj=cache.observation.unique_objects[obj_token]
                kind_value=get_collision_type(sc._states[2,tidx],polygon,obj,cache.observation[tidx][obj_token])
                collision_objects.append(dict(token=obj_token,object_type=str(obj.tracked_object_type),collision_type=str(kind_value),time_s=event_times[0],
                    provenance='Derived intersection/classification at native first at-fault collision time; not a replay of prior non-fault exclusion history.'))
        for tidx,color,style in [(0,'#6c757d','-'),(int(round(event/.1)),'#e9a23b','--')]:
            occupancy=cache.observation[tidx]
            for tok in occupancy.tokens:
                poly=occupancy[tok]
                if poly.geom_type!='Polygon':continue
                xy=local(np.asarray(poly.exterior.coords))
                ax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=.6,alpha=.7)
        for tr,sc,label,color in [(old[z],oldscore,'base','#457b9d'),(new[z],newscore,'updated','#e76f51')]:
            ax.plot(tr[:,0],tr[:,1],'o-',color=color,label=label+' requested',ms=3)
            sim=local(sc['simulated_states'][z,:,:2]);ax.plot(sim[:,0],sim[:,1],'--',color=color,label=label+' tracked')
        xy=np.concatenate([old[z,:,:2],new[z,:,:2],[[0,0]]]);ax.set_xlim(xy[:,0].min()-8,xy[:,0].max()+8);ax.set_ylim(xy[:,1].min()-8,xy[:,1].max()+8)
        ax.set_aspect('equal');ax.set(xlabel='Ego x (m)',ylabel='Ego y (m)',title=kind+' / '+token);ax.legend(fontsize=7)
        labels=['collision','offroad','TTC check failure'];axes[1].set_xlim(-.1,4.1);axes[1].set_ylim(-.7,2.7)
        old_event_times=[float(oldscore[key][z]) for key in ['first_collision_s','first_dac_s','first_ttc_check_failure_s']]
        for j,(label,t) in enumerate(zip(labels,event_times)):
            axes[1].hlines(j,0,4,color='#dee2e6')
            for value,offset,marker,color in [(old_event_times[j],-.15,'o','#457b9d'),(t,.15,'x','#e76f51')]:
                if np.isfinite(value):
                    axes[1].plot(value,j+offset,marker,color=color,ms=7)
                    axes[1].annotate(f'{value:.1f}s',(value,j+offset),xytext=(4,2 if offset>0 else -9),textcoords='offset points',fontsize=8)
                else:axes[1].text(3.9,j+offset,'none within4s',ha='right',va='center',fontsize=8,color=color)
        title=f'Scene mean delta_J={case["delta_J"]:+.4f}; seed{1101+z}'
        if 'predicted_delta_J' in case:title+=f'\nPredicted delta_J={case["predicted_delta_J"]:+.5f}'
        axes[1].set_yticks(range(3),labels);axes[1].set(xlabel='Simulated elapsed time (s)',title=title)
        axes[1].text(.02,-.3,f'Dashed objects: logged occupancy at{event:.1f}s, diagnostic only.\nTTC check failure is not time remaining to impact.',transform=axes[1].transAxes,fontsize=8)
        indices=np.arange(5)
        axes[2].plot(indices,oldscore['scores'][:5,-1],'o-',color='#457b9d',label='base')
        axes[2].plot(indices,newscore['scores'][:5,-1],'x--',color='#e76f51',label='updated')
        axes[2].axvline(z,color='#adb5bd',lw=.7,label='shown draw')
        axes[2].set_xticks(indices,[str(1101+i) for i in indices],rotation=30)
        axes[2].set(ylim=(-.05,1.05),xlabel='Paired inference seed',ylabel='Normalized PDMS',title=f'Shown draw delta_R={case["paired_deltas"][z]:+.4f}')
        axes[2].legend(fontsize=8)
        fig.tight_layout();fig.savefig(FIGURES/f'case_{kind}.png',bbox_inches='tight');plt.close(fig)
        details.append(dict(kind=kind,**case,shown_seed=1101+z,shown_delta_R=float(case['paired_deltas'][z]),first_events_s={k:(t if np.isfinite(t) else None) for k,t in zip(labels,event_times)},
            base_first_events_s={k:(t if np.isfinite(t) else None) for k,t in zip(labels,old_event_times)},
            old_components=oldscore['scores'][z].tolist(),new_components=newscore['scores'][z].tolist(),
            collision_intersections=collision_objects,
            selection_rule='Threshold0.02 then token/family/step hash; displayed draw closest to scene mean delta',
            object_overlay='Current and event-time occupancy; future is diagnostic, never predictor input'))
    atomic_json(ROOT/'analysis/cases.json',details)
