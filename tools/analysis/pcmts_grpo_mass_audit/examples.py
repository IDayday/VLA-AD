"""Supplementary, prespecified token-hash examples; unavailable types stay empty."""
from common_mass import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures import COLORS,LABELS,export
def main():
    raw=pd.read_parquet(OUT/'metrics/candidate_metrics.parquet');pools=pd.read_csv(OUT/'metrics/pool_scene_metrics.csv');chosen=pd.read_parquet(OUT/'metrics/selected_pools_4x1000x16.parquet')
    external=raw.all_source_tags.map(lambda x:bool(set(json.loads(x))&{'ddv2','drivor'}));eligible=raw.hard_safe&raw.quality_floor_pass
    masks={'covered_external_gain':external&eligible&(raw.p_B>=.03)&(raw.hit_count_B>=10)&(raw.candidate_local_gain>=.5),'covered_external_redundant':external&eligible&(raw.p_B>=.03)&(raw.hit_count_B>=10)&(raw.candidate_local_gain<.5),'external_quality_low_coverage':external&eligible&(raw.p_B<.03)}
    sets={k:set(raw.loc[v,'token']) for k,v in masks.items()};op=pools[pools.pool_type=='operational'];pc=op[op.method=='grpo_mass_pc'].set_index('token');score=op[op.method=='score'].set_index('token')
    sets['strict_shortage_or_fallback']=set(pc.index[pc.strict_count<16]);sets['PC_nonwin']=set(pc.index[(pc.mean_candidate_p_B<=score.mean_candidate_p_B+1e-12)&(pc.candidate_PDMS<=score.candidate_PDMS+1e-12)])
    manifest=[];points=[];lookup={s['token']:s for s in scenes()};fig,axes=plt.subplots(1,5,figsize=(20,4.5))
    for ax,(category,tokens) in zip(axes,sets.items()):
        if not tokens:
            manifest.append(dict(category=category,status='MISSING_NO_QUALIFYING_EXAMPLE',qualifying_scene_count=0));ax.text(.5,.5,'No qualifying scene',ha='center',transform=ax.transAxes);ax.set_title(category);continue
        t=min(tokens,key=lambda t:digest(['supplementary_example',t]));manifest.append(dict(category=category,status='AVAILABLE',token=t,qualifying_scene_count=len(tokens),selection='minimum fixed token hash within prespecified category'))
        a=np.load(OUT/'cache/raw'/f'{t}.npz')['trajectories'];gt=np.asarray(lookup[t]['gt']);ax.plot(gt[:,0],gt[:,1],color='black',lw=2,label='GT')
        for method in METHODS:
            rows=chosen[(chosen.token==t)&(chosen.method==method)&chosen.valid_mask.fillna(False)]
            for j,r in enumerate(rows.itertuples()):
                traj=a[int(r.raw_index)];ax.plot(traj[:,0],traj[:,1],color=COLORS[method],alpha=.25,lw=.8,label=LABELS[method] if j==0 else None)
                for k,(x,y,h) in enumerate(traj):points.append(dict(category=category,token=t,method=method,candidate_id=r.candidate_id,point=k+1,x=x,y=y,heading=h))
        ax.set(title=category+'\n'+t,xlabel='forward x (m)',ylabel='left y (m)');ax.axis('equal')
    axes[-1].legend(fontsize=7);export(fig,'Supplementary_fixed_example_types',pd.DataFrame(points));save(OUT/'manifests/supplementary_examples.json',manifest)
if __name__=='__main__':main()
