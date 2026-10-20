"""Predefined numerical case rules; real native trajectories and event times."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import numpy as np
from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.opd.scoring import Scoring,load_cache
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--split',default='dev');a=p.parse_args()
    il,rows=predictions('B0_IL',a.split);rl,_=predictions('B0_RL',a.split);new,_=predictions(a.method,a.split)
    im=il[:,:,-1].mean(1);rm=rl[:,:,-1].mean(1);nm=new[:,:,-1].mean(1);delta=nm-im
    # Defined independently of method results: largest gain/drop, then token;
    # conditional counterexamples are chosen only if the numerical class exists.
    rules=[('largest_repair',np.ones(len(rows),bool),delta,True),('largest_degradation',np.ones(len(rows),bool),delta,False),
           ('teacher_failure_student_success',(rm<=.01)&(nm>=.8),nm-rm,True),
           ('teacher_success_student_failure',(rm>=.8)&(nm<=.01),nm-rm,False),
           ('joint_failure_improvement',(im<=.6)&(rm<=.6)&(nm>np.maximum(im,rm)),nm-np.maximum(im,rm),True)]
    directory=REPORTS/'figures/cases'/a.method;directory.mkdir(parents=True,exist_ok=True)
    records=[];used=set();scorer=Scoring()
    for rule,mask,value,descending in rules:
        possible=[i for i in range(len(rows)) if mask[i] and i not in used]
        if not possible:continue
        i=sorted(possible,key=lambda j:((-1 if descending else 1)*value[j],rows[j]['token']))[0]
        used.add(i);row=rows[i];cache=load_cache(row['metric_cache_path']);ego=cache.ego_state.rear_axle
        rotation=np.array([[np.cos(ego.heading),-np.sin(ego.heading)],[np.sin(ego.heading),np.cos(ego.heading)]])
        def local(points):return (np.asarray(points)-np.array([ego.x,ego.y]))@rotation
        fig,ax=plt.subplots(figsize=(7,5.4))
        def polygon(geometry,**kwargs):
            if hasattr(geometry,'geoms'):
                for part in geometry.geoms:polygon(part,**kwargs)
            elif hasattr(geometry,'exterior'):
                xy=local(geometry.exterior.coords)
                if np.min(np.linalg.norm(xy,axis=1))<90:ax.fill(xy[:,0],xy[:,1],**kwargs)
        for geometry in cache.drivable_area_map._geometries:polygon(geometry,color='lightgray',alpha=.12,lw=.3,edgecolor='gray')
        for time_index,color,alpha in [(0,'#555555',.5),(20,'#aaaaaa',.2)]:
            for geometry in cache.observation._occupancy_maps[time_index]._geometries:polygon(geometry,color=color,alpha=alpha,lw=.4)
        cases={};all_xy=[]
        for name,color in [('B0_IL','#376a93'),('B0_RL','#e88a28'),(a.method,'#a62c45')]:
            if name in cases:continue
            with np.load(ROOT/'evaluation'/a.split/name/'scenes'/f"{row['token']}.npz") as data:
                trajectory=data['trajectories'][0];seed=int(data['seeds'][0]);mean=float(data['scores'][:,-1].mean())
            details=scorer.batch(cache,trajectory[None]);simulated=details['simulated_states'][0]
            xy=local(simulated[:,:2]);all_xy.append(xy)
            ax.plot(trajectory[:,0],trajectory[:,1],'o--',ms=3,lw=.8,color=color,alpha=.7)
            ax.plot(xy[:,0],xy[:,1],color=color,label=f'{name}: mean PDMS {mean:.3f}')
            events={}
            for field in ['first_collision_s','first_dac_s','first_ttc_check_failure_s']:
                event=float(details[field][0]);events[field]=event if np.isfinite(event) else None
                if np.isfinite(event):
                    index=min(int(round(event*10)),len(xy)-1)
                    ax.scatter(*xy[index],marker='x',s=35,color=color)
            cases[name]=dict(inference_seed=seed,scene_mean_PDMS=mean,first_seed_PDMS=float(details['scores'][0,-1]),events=events,
                             prediction_sha256=sha(ROOT/'evaluation'/a.split/name/'scenes'/f"{row['token']}.npz"))
        combined=np.concatenate(all_xy);lo=combined.min(0)-8;hi=combined.max(0)+8
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Ego x forward [m]',ylabel='Ego y left [m]',title=f'{rule}: {row["token"]}')
        ax.scatter(0,0,color='black',s=20);ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=7);fig.tight_layout()
        fig.savefig(directory/f'{rule}.png',dpi=150);plt.close(fig)
        records.append(dict(rule=rule,token=row['token'],log=row['log_name'],cases=cases,
                            interpretation='Dashed planned trajectory; solid tracked simulation; x marks first violation. TTC check failure is not remaining collision time. Occupancy at t=0 and t=2s is privileged illustration only.'))
    write(REPORTS/'tables'/f'case_provenance_{a.split}_{a.method}.json',records)
    print('REAL_CASE_FIGURES',a.method,len(records),flush=True)


if __name__=='__main__':main()
