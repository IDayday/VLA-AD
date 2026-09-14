"""Standalone figures and a local, dependency-free same-scene trajectory viewer."""
from common_fd import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABEL={'official_il':'Official IL','mts_8692':'MTS86.92','mts_8751':'MTS87.51','grpo_9041':'GRPO90.41','apr_9145':'APR91.45'}
COLORS=dict(zip(MODELS,['#555555','#247ba0','#63a375','#d97927','#9955a8']))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def readcsv(name):return pd.read_csv(OUT/'metrics'/name)
def finish(fig,name,tables):
    fig.tight_layout();folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf','svg']:
        path=folder/f'{name}.{suffix}';fig.savefig(path,dpi=170,bbox_inches='tight')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    for key,frame in tables.items():frame.to_csv(folder/f'{name}_{key}.csv',index=False)
    plt.close(fig)

def main():
    p=readcsv('policy_scene.csv');s=readcsv('policy_summary.csv').set_index('model').loc[MODELS]
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for m in MODELS[1:]:
        v=np.sort(p[p.model==m].spread_ratio);ax[0].plot(v,np.arange(1,len(v)+1)/len(v),label=LABEL[m],color=COLORS[m])
    ax[0].set_xscale('log');ax[0].axvline(1,color='black',ls=':');ax[0].set(xlabel='Per-scene Spread-AUC / IL (log scale)',ylabel='Scene ECDF');ax[0].legend(fontsize=8)
    x=np.arange(5);ax[1].bar(x,s.PDMS,color=[COLORS[m] for m in MODELS]);ax[1].set(xticks=x,xticklabels=[LABEL[m] for m in MODELS],ylabel='Mean PDMS (points)',ylim=(0,100));ax[1].tick_params(axis='x',rotation=30)
    for i,m in enumerate(MODELS):
        ax[2].scatter(s.loc[m,'pairwise_ADE'],s.loc[m,'Hit8']*100,color=COLORS[m],s=150)
        ax[2].annotate(LABEL[m],(s.loc[m,'pairwise_ADE'],s.loc[m,'Hit8']*100),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax[2].set(xlabel='Mean pairwise ADE (m)',ylabel='Hit@8 (%) — common IL mean + 1 point',ylim=(0,100))
    finish(fig,'Fig1_width_quality',{'scene':p[['token','model','spread_ratio','pairwise_ADE','PDMS','Hit8']],'summary':s.reset_index()})

    t=readcsv('teacher_scene.csv');logs=readcsv('historical_training_curves.csv');last=readcsv('historical_training_at_checkpoint.csv')
    fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for m in ARCHIVES:
        z=t[t.model==m].support_count.value_counts(normalize=True).sort_index()
        ax[0].plot(z.index,z.values*100,'o-',label=LABEL[m],color=COLORS[m])
    ax[0].set(xlabel='Actual selected support trajectories / scene',ylabel='Scenes (%)');ax[0].legend()
    for m in ARCHIVES:
        z=logs[(logs.model==m)&(logs.metric=='train/dpsi_gt_weight_ratio_epoch')]
        ax[1].plot(z.epoch,z.value*100,label=LABEL[m]+' pre-budget',color=COLORS[m])
    z=logs[(logs.model=='mts_8692')&(logs.metric=='train/dpsi_non_gt_target_weight_ratio_post_budget_epoch')]
    ax[1].plot(z.epoch,(1-z.value)*100,label='MTS86 actual post-budget',color=COLORS['mts_8692'],ls='--')
    ax[1].set(xlabel='Training epoch',ylabel='GT fraction of loss target weight (%)',ylim=(0,100));ax[1].legend(fontsize=8)
    keys=['train/dpsi_non_gt_residual_mass_ratio_pre_budget_epoch','train/dpsi_non_gt_residual_mass_ratio_post_budget_epoch','train/dpsi_residual_budget_active_ratio_epoch']
    for k,label in zip(keys,['Non-GT residual share before','Non-GT residual share after','Budget active scene fraction']):
        z=logs[(logs.model=='mts_8692')&(logs.metric==k)];ax[2].plot(z.epoch,z.value*100,label=label)
    ax[2].set(xlabel='Training epoch (MTS86)',ylabel='Percent',ylim=(0,100));ax[2].legend(fontsize=8)
    finish(fig,'Fig2_actual_teachers_weights',{'teacher_scene':t,'training_log':logs[logs.metric.isin(keys+['train/dpsi_gt_weight_ratio_epoch','train/dpsi_non_gt_target_weight_ratio_post_budget_epoch'])]})

    tr=readcsv('teacher_output_scene.csv');fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for i,m in enumerate(ARCHIVES):
        g=tr[tr.model==m];eps=CFG['teacher_coverage_ADE_m']
        for model,style in [('official_il','--'),(m,'-')]:
            vals=[g[f'{model}_nonGT_mass_covered_{e}'].mean()*100 for e in eps]
            ax[i].plot(eps,vals,'o'+style,label=LABEL[model],color=COLORS[model])
        ax[i].set(xlabel='Finite64 rollout ADE coverage radius (m)',ylabel='Non-GT teacher weighted mass covered (%)',ylim=(0,100),title=LABEL[m]);ax[i].legend()
    for m in ARCHIVES:
        z=tr[tr.model==m].merge(p[p.model==m][['token','pairwise_ADE']],on='token')
        ax[2].scatter(z.weighted_pair_ADE,z.pairwise_ADE,s=8,alpha=.25,color=COLORS[m],label=LABEL[m],rasterized=True)
    ax[2].set(xlabel='Weighted teacher pairwise ADE (m)',ylabel='Output pairwise ADE (m)');ax[2].set_xscale('symlog',linthresh=.01);ax[2].set_yscale('symlog',linthresh=.01);ax[2].legend()
    finish(fig,'Fig3_teacher_to_output',{'teacher_output_scene':tr})

    cf=readcsv('counterfactual_summary.csv');raw=readcsv('counterfactual_scene.csv');fig,ax=plt.subplots(1,2,figsize=(14,5))
    contrasts=[a+'__'+b for a,b in CFG['counterfactual']['contrasts']]
    for i,e in enumerate(['center_contribution','residual_shape_contribution']):
        z=cf[(cf.metric=='PDMS')&(cf.effect==e)].set_index('contrast').loc[contrasts]
        ax[0].errorbar(np.arange(len(z))+(i-.5)*.16,z['mean'],yerr=np.vstack([z['mean']-z.ci_low,z.ci_high-z['mean']]),fmt='o',capsize=3,label=['Center','Residual shape'][i])
    ax[0].axhline(0,color='gray',ls=':');ax[0].set(xticks=np.arange(len(contrasts)),xticklabels=[LABEL[a]+' → '+LABEL[b] for a,b in CFG['counterfactual']['contrasts']],ylabel='Geometric Shapley contribution (PDMS points)');ax[0].tick_params(axis='x',rotation=25);ax[0].legend()
    g=raw[(raw.contrast=='official_il__grpo_9041')&(raw.metric=='PDMS')]
    vals=g[['old','new_center_old_residual','old_center_new_residual','new']].mean()
    control=raw[raw.contrast=='official_il__width_only']['new'].mean()
    ax[1].bar(np.arange(5),list(vals)+[control],color=['#555555','#d97927','#adadad','#d97927','#888888'])
    for i,v in enumerate(list(vals)+[control]):ax[1].text(i,v+.4,f'{v:.3f}',ha='center',fontsize=9)
    ax[1].set(xticks=np.arange(5),xticklabels=['IL','GRPO center\nIL residual','IL center\nGRPO residual','GRPO','IL width-only\nscaled to GRPO'],ylabel='Mean true NAVSIM PDMS',ylim=(0,103));ax[1].tick_params(axis='x',rotation=18)
    finish(fig,'Fig4_center_residual_counterfactual',{'attribution':cf,'GRPO_scene':g,'width_only':raw[raw.contrast=='official_il__width_only']})

    ch=readcsv('historical_grpo_summary.csv');fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(ch.step,ch.PDMS,'o-');ax[0,0].set(ylabel='Mean PDMS',xlabel='Original GRPO optimizer step')
    ax[0,1].plot(ch.step,ch.pairwise_ADE,'o-',label='Pairwise ADE');ax[0,1].plot(ch.step,ch.Spread_AUC,'o-',label='Spread-AUC');ax[0,1].set(xlabel='Original GRPO optimizer step',ylabel='Metres');ax[0,1].legend()
    for k in ['feasible','DAC','TTC','DDC']:ax[1,0].plot(ch.step,ch[k]*100,'o-',label=k)
    ax[1,0].set(xlabel='Original GRPO optimizer step',ylabel='Safety (%)');ax[1,0].legend()
    ax[1,1].plot(ch.step,ch.Hit8*100,'o-',label='Hit@8');ax[1,1].plot(ch.step,ch.HQ_mass*100,'o-',label='High-quality sample mass');ax[1,1].set(xlabel='Original GRPO optimizer step',ylabel='Percent');ax[1,1].legend()
    for a in ax.flat:a.axvline(11970,color='gray',ls=':',lw=.8);a.grid(alpha=.2)
    finish(fig,'Fig5_historical_GRPO_evolution',{'summary':ch,'scene':readcsv('historical_grpo_scene.csv')})

    strata=readcsv('policy_stratification.csv');fig,ax=plt.subplots(1,3,figsize=(15,4.5))
    for i,(key,order) in enumerate([('IL_spread_quartile',['Q1','Q2','Q3','Q4']),('command',['straight','left','right']),('heading_tertile',['low','middle','high'])]):
        for m in MODELS[1:]:
            z=strata[(strata.stratification==key)&(strata.model==m)].set_index('stratum').loc[order]
            ax[i].plot(order,z.median_spread_ratio,'o-',label=LABEL[m],color=COLORS[m])
        ax[i].axhline(1,color='gray',ls=':');ax[i].set(xlabel=key.replace('_',' '),ylabel='Median per-scene Spread-AUC ratio / IL')
    ax[0].legend(fontsize=8)
    finish(fig,'Fig6_scene_stratification',{'strata':strata})
    viewer()

def viewer():
    data=read(OUT/'cache/scene_examples.json')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>同场景：五模型64条轨迹与真实teacher</title>
<style>body{font-family:system-ui,sans-serif;max-width:1250px;margin:25px auto;color:#253044}header{line-height:1.7}label{display:inline-block;margin:6px}canvas{border:1px solid #ddd;width:100%;height:640px}select{padding:7px}small{color:#536070}</style>
<header><h2>同一场景的五模型64条采样，与实际训练teacher</h2><p>细线：64条真实模型输出；粗线：平均轨迹；虚线：实际训练teacher。每种驾驶指令按固定token hash选2个示例，共6个；主统计仍覆盖全部1000场景。坐标为ego局部米，黑点为当前时刻原点，轨迹从未来0.5秒开始。</p>
<select id="scene"></select><div id="controls"></div><p id="stats"></p></header><canvas width="1200" height="640" id="plot"></canvas>
<small>Teacher透明度按重建的期望监督权重展示（残差预算之前）；不代表生成概率。64条采样使用统一evaluation DDIM协议，并非GRPO训练探索sampler。</small>
<script>const DATA=__DATA__,LABEL=__LABEL__,COLORS=__COLORS__;const select=document.querySelector('#scene'),controls=document.querySelector('#controls'),canvas=document.querySelector('#plot'),ctx=canvas.getContext('2d');
for(const [token,d] of Object.entries(DATA)){const op=document.createElement('option');op.value=token;op.textContent=token+' | '+d.command;select.appendChild(op)}
for(const [m,label] of Object.entries(LABEL)){controls.innerHTML+=`<label style="color:${COLORS[m]}"><input type="checkbox" value="${m}" checked>${label}</label>`}
controls.innerHTML+='<label><input type="checkbox" value="teachers" checked>实际teacher</label><label><input type="checkbox" value="gt" checked>GT</label>';
function draw(){let d=DATA[select.value];let enabled=new Set([...controls.querySelectorAll('input:checked')].map(x=>x.value));ctx.clearRect(0,0,1200,640);
let all=[[0,0],...d.gt];for(let m of Object.keys(LABEL))all.push(...d.trajectories[m].flat());for(let t of Object.values(d.teachers))all.push(...t.trajectory.flat());
let xs=all.map(p=>p[0]),ys=all.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);let scale=Math.min(1080/Math.max(xmax-xmin,1),520/Math.max(ymax-ymin,1));let X=x=>60+(x-xmin)*scale,Y=y=>580-(y-ymin)*scale;
ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.setLineDash([]);ctx.font='12px sans-serif';ctx.fillStyle='#666';let tick=Math.max(1,Math.round((xmax-xmin)/12));for(let x=Math.ceil(xmin/tick)*tick;x<=xmax;x+=tick){ctx.beginPath();ctx.moveTo(X(x),30);ctx.lineTo(X(x),590);ctx.stroke();ctx.fillText(x+' m',X(x)-10,610)}
function line(t,c,a,w,dash=[]){ctx.strokeStyle=c;ctx.globalAlpha=a;ctx.lineWidth=w;ctx.setLineDash(dash);ctx.beginPath();t.forEach((p,i)=>{if(i)ctx.lineTo(X(p[0]),Y(p[1]));else ctx.moveTo(X(p[0]),Y(p[1]))});ctx.stroke();ctx.globalAlpha=1}
if(enabled.has('teachers'))for(let [m,t] of Object.entries(d.teachers))t.trajectory.forEach((a,i)=>line(a,COLORS[m],Math.max(.18,t.weights[i]),2,[6,5]));
for(let m of Object.keys(LABEL)){if(!enabled.has(m))continue;let t=d.trajectories[m];t.forEach(a=>line(a,COLORS[m],.18,1));let avg=t[0].map((p,i)=>[0,1].map(j=>t.reduce((s,r)=>s+r[i][j],0)/t.length));line(avg,COLORS[m],1,3)}
if(enabled.has('gt'))line(d.gt,'#101010',1,2,[2,3]);ctx.setLineDash([]);ctx.fillStyle='#000';ctx.beginPath();ctx.arc(X(0),Y(0),4,0,2*Math.PI);ctx.fill();document.querySelector('#stats').textContent=Object.keys(LABEL).map(m=>LABEL[m]+': '+d.PDMS[m].toFixed(2)+' PDMS').join('　');}
select.onchange=draw;controls.onchange=draw;draw();</script></html>'''
    html=template.replace('__DATA__',json.dumps(data)).replace('__LABEL__',json.dumps(LABEL)).replace('__COLORS__',json.dumps(COLORS))
    (OUT/'figures/same_scene_viewer.html').write_text(html)

if __name__=='__main__':main()
