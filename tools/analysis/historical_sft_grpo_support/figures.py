"""Figures from measured tables; each plot saves its source table."""
from common_support import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'official_il':'Official GT-IL','psi_sft':'IL + PSI SFT','a5_sft':'Random multi-SFT A5','v6_sft':'Random multi-SFT V6'}
COLORS={'official_il':'#57636e','psi_sft':'#247caa','a5_sft':'#e58b32','v6_sft':'#9a61b2'}
FIG=OUT/'figures'
def savefig(fig,name,data):
    FIG.mkdir(parents=True,exist_ok=True);fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    svg=FIG/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    data.to_csv(FIG/f'{name}.csv',index=False);plt.close(fig)

def main():
    sf=pd.read_csv(OUT/'metrics/scene_metrics.csv');tf=pd.read_csv(OUT/'metrics/teacher_scene_metrics.csv')
    ts=pd.read_csv(OUT/'metrics/teacher_summary.csv');primary=CFG['primary_models']
    own=ts[(ts.scope=='COMMON_TRAIN835')&(ts.policy==ts.origin)&(ts.kind=='NON_GT')]
    fig,axs=plt.subplots(1,3,figsize=(13,3.8))
    for ax,metric,title in zip(axs,['weighted_hit16_0p5','weighted_hit64_0p5','weighted_mass64_0p5'],['Teacher hit in a G16 group','Teacher hit in 64 draws','Probability mass near teacher']):
        for j,pr in enumerate(CFG['protocols']):
            x=np.arange(3)+(j-.5)*.34;h=own[own.protocol==pr].set_index('policy').reindex(primary[1:])
            ax.bar(x,h[metric]*100,width=.32,label=pr)
        ax.set_xticks(range(3),['PSI','A5','V6']);ax.set_ylabel('%');ax.set_title(title);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fig.suptitle('Own non-GT supervision: scene-weighted, common actual training scenes; ADE <= 0.5 m',y=1.04)
    savefig(fig,'Fig-M1_teacher_coverage',own)
    matrix=ts[(ts.scope=='COMMON_TRAIN835')&(ts.kind=='NON_GT')&(ts.protocol=='eval')]
    fig,ax=plt.subplots(figsize=(7,4));p=matrix.pivot(index='policy',columns='origin',values='weighted_hit64_0p5').reindex(index=primary,columns=primary[1:])*100
    im=ax.imshow(p,vmin=0,vmax=100,cmap='Blues');ax.set_xticks(range(3),['PSI teachers','A5 teachers','V6 teachers']);ax.set_yticks(range(4),[LABEL[x] for x in primary])
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{p.iloc[i,j]:.1f}%',ha='center',va='center',color='white' if p.iloc[i,j]>60 else 'black')
    fig.colorbar(im,ax=ax,label='Weighted teacher coverage in 64 evaluation draws (%)');ax.set_title('Same teacher bank evaluated under each policy')
    savefig(fig,'Fig-M1b_cross_teacher_coverage',matrix)
    subset=sf[sf.model.isin(primary)&sf.protocol.isin(CFG['protocols'])];summary=subset.groupby(['model','protocol']).mean(numeric_only=True)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for j,pr in enumerate(CFG['protocols']):
        g=summary.xs(pr,level='protocol').reindex(primary);x=np.arange(4)+(j-.5)*.25
        axs[0].errorbar(x,g.mean_PDMS,yerr=np.stack([g.mean_PDMS-g.min_PDMS,g.max_PDMS-g.mean_PDMS]),fmt='o',capsize=4,label=pr)
        axs[1].bar(x,g.pairwise_ADE,width=.23,label=pr);axs[2].bar(x,g.feasible_rate*100,width=.23,label=pr)
    for ax in axs:ax.set_xticks(range(4),['GT-IL','PSI','A5','V6']);ax.legend(fontsize=8)
    axs[0].set(title='Average group mean / minimum / maximum',ylabel='PDMS points',ylim=(0,101));axs[1].set(title='Within-group trajectory spread',ylabel='Pairwise ADE (m)',ylim=(0,None));axs[2].set(title='Conservative feasible members',ylabel='%',ylim=(0,100))
    fig.suptitle('1000 scenes, 4 groups per scene, 16 members per group',y=1.03)
    savefig(fig,'Fig-M2_native_group16',summary.reset_index())
    factorial=set(read(OUT/'manifests/scenes.json')['factorial_tokens']);sub=sf[sf.model.isin(primary)&sf.token.isin(factorial)]
    agg=sub.groupby(['model','protocol']).mean(numeric_only=True);prs=['eval','clip_only','floor_only','floor_and_clip','native_grpo']
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for m in primary:
        g=agg.loc[m].reindex(prs)
        for ax,metric in zip(axs,['pairwise_ADE','mean_PDMS','feasible_rate']):ax.plot(range(5),g[metric]*(100 if metric=='feasible_rate' else 1),'o-',label=LABEL[m],color=COLORS[m])
    for ax in axs:ax.set_xticks(range(5),['eval','clip only','floor only','both','native'],rotation=25);ax.grid(alpha=.2)
    axs[0].set_ylabel('Pairwise ADE (m)');axs[1].set_ylabel('PDMS points');axs[2].set_ylabel('Feasible rate (%)');axs[0].legend(fontsize=8)
    fig.suptitle('Same checkpoint + same random numbers: sampler interventions on fixed 256 scenes',y=1.04)
    savefig(fig,'Fig-M3_sampler_factorial',agg.reset_index())
    chains=[['official_il','original_grpo_11970'],['a5_sft','a5_grpo_300','a5_grpo_4842'],['v6_sft','v6_grpo_300','v6_grpo_3300']]
    fig,axs=plt.subplots(3,3,figsize=(13,10));source=[]
    for j,chain in enumerate(chains):
        step=[0]+[int(n.rsplit('_',1)[1]) for n in chain[1:]]
        for pr in CFG['protocols']:
            g=sf[(sf.model.isin(chain))&(sf.protocol==pr)].groupby('model').mean(numeric_only=True).reindex(chain)
            for i,metric in enumerate(['mean_PDMS','feasible_rate','pairwise_ADE']):axs[i,j].plot(step,g[metric]*(100 if metric=='feasible_rate' else 1),'o-',label=pr)
            h=g.reset_index();h['step']=step;h['protocol']=pr;source.append(h)
        axs[0,j].set_title(LABEL[chain[0]]);axs[0,j].legend(fontsize=8)
        for i in range(3):axs[i,j].grid(alpha=.2);axs[i,j].set_xlabel('Actual historical optimizer step')
    axs[0,0].set_ylabel('Internal NAVTRAIN PDMS');axs[1,0].set_ylabel('Conservative feasible (%)');axs[2,0].set_ylabel('Within-G16 pairwise ADE (m)')
    fig.suptitle('Frozen historical checkpoint evolution; not a new GRPO training run',y=1.01)
    savefig(fig,'Fig-M4_historical_distribution_evolution',pd.concat(source,ignore_index=True))
    # One common scene chosen solely by token hash, not by observed success.
    member=pd.read_csv(OUT/'metrics/training_membership.csv');token=min(member[member.common_train].token,key=lambda t:digest(['illustrative_scene',t]))
    teachers=np.load(OUT/'cache/teachers'/f'{token}.npz');meta=json.loads(str(teachers['metadata']))['rows'];tr=teachers['trajectories']
    fig,axs=plt.subplots(2,4,figsize=(15,7));sources=[]
    for j,m in enumerate(primary):
        z=np.load(OUT/'cache/rollouts'/m/f'{token}.npz');indices=[i for i,r in enumerate(meta) if r['origin']==m]
        for i,pr in enumerate(CFG['protocols']):
            ax=axs[i,j]
            for a in z[pr][0]:ax.plot(a[:,0],a[:,1],color=COLORS[m],alpha=.28,lw=.8)
            for idx in indices:ax.plot(tr[idx,:,0],tr[idx,:,1],'--',color='black' if meta[idx]['is_gt'] else '#cd3f37',lw=1.3)
            ax.set_aspect('equal',adjustable='box');ax.set_title(LABEL[m]+' / '+pr,fontsize=9);ax.set_xlabel('ego X (m)');ax.set_ylabel('ego Y (m)')
            for member,a in enumerate(z[pr][0]):
                sources.extend(dict(token=token,policy=m,protocol=pr,kind='rollout',trajectory_id=str(member),future_time=(step+1)*.5,x=float(point[0]),y=float(point[1]),heading=float(point[2])) for step,point in enumerate(a))
            for idx in indices:
                sources.extend(dict(token=token,policy=m,protocol=pr,kind='GT' if meta[idx]['is_gt'] else 'teacher',trajectory_id=meta[idx]['teacher_id'],future_time=(step+1)*.5,x=float(point[0]),y=float(point[1]),heading=float(point[2])) for step,point in enumerate(tr[idx]))
    drawing=pd.DataFrame(sources)
    xmin,xmax=drawing.x.min(),drawing.x.max();ymin,ymax=drawing.y.min(),drawing.y.max()
    for ax in axs.flat:ax.set_xlim(xmin-1,xmax+1);ax.set_ylim(ymin-1,ymax+1)
    fig.suptitle('Fixed hash-selected scene '+token+'; solid: 16 draws; dashed: actual supervision',y=1.01)
    savefig(fig,'Fig-M5_fixed_scene_trajectories',drawing)

if __name__=='__main__':main()
