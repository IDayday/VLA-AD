from __future__ import annotations
import argparse, concurrent.futures, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

LABELS={'official_stage2_il':'Official IL','local_original_grpo_epoch8':'GRPO 90.41','final_apr_9145':'APR 91.45','candidate_sft_v6_epoch165':'SFT V6 86.92','candidate_sft_a5_epoch155':'SFT A5 87.51'}
ORDER=list(LABELS)

def metrics(traj, gt):
    xy=np.asarray(traj,dtype=np.float64)[...,:2];k=len(xy)
    dist=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    ade=dist.mean(-1);fde=dist[:,:,-1];tri=np.triu_indices(k,1)
    medoid=int(ade.mean(1).argmin());r=ade[medoid]
    errors=np.linalg.norm(xy-np.asarray(gt)[None,:,:2],axis=-1)
    z=linkage(squareform(ade,checks=False),method='complete')
    result=dict(pair_ade_m=float(ade[tri].mean()),pair_fde_m=float(fde[tri].mean()),pair_ade_p90_m=float(np.quantile(ade[tri],.9)),endpoint_long_std_m=float(xy[:,-1,0].std(ddof=1)),endpoint_lat_std_m=float(xy[:,-1,1].std(ddof=1)),medoid_radius90_ade_m=float(np.quantile(r,.9)),mass_within_05m=float((r<=.5).mean()),mass_within_1m=float((r<=1.).mean()),kernel_concentration_05=float(np.exp(-ade[tri]**2/(2*.5**2)).mean()),mean_gt_ade_m=float(errors.mean()),best_gt_ade_m=float(errors.mean(1).min()),mean_gt_fde_m=float(errors[:,-1].mean()),best_gt_fde_m=float(errors[:,-1].min()))
    # Fixed physical thresholds; geometric clusters are not semantic behavior labels.
    for threshold,tag in [(.5,'05'),(1.,'1'),(2.,'2')]:
        ids=fcluster(z,t=threshold,criterion='distance');counts=np.bincount(ids)[1:];p=counts/counts.sum()
        result[f'modes_{tag}m']=len(counts);result[f'effective_modes_{tag}m']=float(np.exp(-(p*np.log(p)).sum()));result[f'largest_mode_{tag}m']=float(p.max())
    result['endpoint_std_area_m2']=float(np.sqrt(max(0,np.linalg.det(np.cov(xy[:,-1].T)))))
    return result

def one(task):
    path,scene,model=task
    data=np.load(path);rows=[]
    assert set(data.files)=={'initial_noise_only','stochastic_ddim'}
    for protocol in data.files:
        a=data[protocol]
        if a.shape!=(64,8,3) or not np.isfinite(a).all():raise ValueError(path)
        for k in [16,32,64]:rows.append(dict(model=model,token=scene['token'],log=scene['log'],protocol=protocol,k=k,**metrics(a[:k],scene['gt'])))
    return rows

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(args):
    tokens=json.loads((args.out/'tokens.json').read_text());models=json.loads((args.out/'models.json').read_text())
    scenes={r['token']:r for f in args.out.glob('features_shard*.json') for r in json.loads(f.read_text())}
    assert len(tokens)==1000 and set(scenes)==set(tokens)
    tasks=[(args.out/'predictions'/m['id']/f'{t}.npz',scenes[t],m['id']) for m in models for t in tokens]
    assert all(p.is_file() for p,_,_ in tasks)
    parity=[]
    for m in models:
        directory=args.out/'predictions'/m['id']
        assert {p.stem for p in directory.glob('*.npz')}==set(tokens)
        checks=json.loads((args.out/f"production_parity_{m['id']}.json").read_text())
        assert len(checks)==4 and max(r['max_abs_error'] for r in checks)<=1e-6
        parity.extend(checks)
        for shard in range(8):
            meta=json.loads((directory/f'metadata_shard{shard}.json').read_text())
            assert meta['checkpoint_sha256']==m['sha256'] and meta['scenes']==125
    args.report.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows=[r for rs in ex.map(one,tasks,chunksize=10) for r in rs]
    write_csv(args.report/'per_scene_metrics.csv',rows)
    fields=[k for k in rows[0] if k not in ['model','token','log','protocol','k']]
    logs=sorted({scenes[t]['log'] for t in tokens});li={v:i for i,v in enumerate(logs)}
    index=np.asarray([li[scenes[t]['log']] for t in tokens]);sizes=np.bincount(index,minlength=len(logs))
    rng=np.random.default_rng(20260914)
    multiplicities=rng.multinomial(len(logs),np.ones(len(logs))/len(logs),size=3000)
    den=multiplicities@sizes
    summary=[];arrays={}
    for m in ORDER:
        for protocol in ['initial_noise_only','stochastic_ddim']:
            for k in [16,32,64]:
                found={r['token']:r for r in rows if r['model']==m and r['protocol']==protocol and r['k']==k}
                assert set(found)==set(tokens)
                matrix=np.asarray([[found[t][f] for f in fields] for t in tokens]);arrays[m,protocol,k]=matrix
                sums=np.zeros((len(logs),len(fields)));np.add.at(sums,index,matrix)
                boots=(multiplicities@sums)/den[:,None]
                for j,f in enumerate(fields):summary.append(dict(model=m,protocol=protocol,k=k,metric=f,mean=float(matrix[:,j].mean()),median=float(np.median(matrix[:,j])),ci_low=float(np.quantile(boots[:,j],.025)),ci_high=float(np.quantile(boots[:,j],.975))))
    write_csv(args.report/'summary.csv',summary)
    comparisons=[]
    for protocol in ['initial_noise_only','stochastic_ddim']:
        for i,a in enumerate(ORDER):
            for b in ORDER[:i]:
                delta=arrays[a,protocol,64]-arrays[b,protocol,64]
                sums=np.zeros((len(logs),len(fields)));np.add.at(sums,index,delta)
                boots=(multiplicities@sums)/den[:,None]
                for j,f in enumerate(fields):comparisons.append(dict(a=a,b=b,protocol=protocol,metric=f,delta_mean=float(delta[:,j].mean()),ci_low=float(np.quantile(boots[:,j],.025)),ci_high=float(np.quantile(boots[:,j],.975)),fraction_a_gt_b=float((delta[:,j]>1e-8).mean()),fraction_a_lt_b=float((delta[:,j]<-1e-8).mean())))
    write_csv(args.report/'paired_comparisons.csv',comparisons)
    (args.report/'validation.json').write_text(json.dumps(dict(scenes=1000,logs=len(logs),models=5,protocols=2,samples_per_scene=64,trajectories=640000,finite=True,invalid_trajectories=0,missing_scenes=0,checkpoint_hashes_verified=5,production_agent_parity_scenes=20,production_agent_parity_max_abs=max(r['max_abs_error'] for r in parity),bootstrap='3000 resamples of logs, scene-weighted mean',gt_used_in_inference=False),indent=2))
    plot(args,scenes,tokens,summary,arrays,fields)
    report(args,summary,comparisons,len(logs))

def plot(args,scenes,tokens,summary,arrays,fields):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#2878b5','#cc4b37','#31a354','#8e63b6']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for row,protocol in enumerate(['initial_noise_only','stochastic_ddim']):
        for col,metric in enumerate(['pair_ade_m','endpoint_lat_std_m','largest_mode_1m']):
            ax=axes[row,col]
            for i,m in enumerate(ORDER):
                r=next(r for r in summary if r['model']==m and r['protocol']==protocol and r['k']==64 and r['metric']==metric)
                ax.errorbar(i,r['mean'],yerr=[[r['mean']-r['ci_low']],[r['ci_high']-r['mean']]],fmt='o',color=colors[i],capsize=4)
            ax.set_xticks(range(5),[LABELS[m] for m in ORDER],rotation=25,ha='right');ax.set_title(protocol+'\n'+metric);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.report/'distribution_summary.png',dpi=170);fig.savefig(args.report/'distribution_summary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,protocol in zip(axes,['initial_noise_only','stochastic_ddim']):
        for m,c in zip(ORDER,colors):
            v=np.sort(arrays[m,protocol,64][:,fields.index('pair_ade_m')]);ax.plot(v,np.arange(1,1001)/1000,label=LABELS[m],color=c)
        ax.set_xlabel('Scene pairwise ADE (m)');ax.set_ylabel('Fraction of scenes');ax.set_title(protocol);ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.report/'width_ecdf.png',dpi=170);plt.close(fig)
    # First six randomly ordered scenes, selected before looking at model outputs.
    for page in range(2):
        fig,axes=plt.subplots(3,5,figsize=(17,10),layout='constrained')
        for row,t in enumerate(tokens[page*3:page*3+3]):
            all_xy=[np.load(args.out/'predictions'/m/f'{t}.npz')['initial_noise_only'][...,:2] for m in ORDER]
            g=np.asarray(scenes[t]['gt']);combined=np.concatenate([a.reshape(-1,2) for a in all_xy]+[g[:,:2]])
            for col,(m,xy) in enumerate(zip(ORDER,all_xy)):
                ax=axes[row,col]
                for tr in xy:ax.plot(tr[:,1],tr[:,0],color=colors[col],alpha=.15,lw=.6)
                ax.plot(g[:,1],g[:,0],'k--',lw=1.5,label='GT');ax.scatter([0],[0],c='k',s=12)
                ax.set_xlim(combined[:,1].min()-1,combined[:,1].max()+1);ax.set_ylim(min(-1,combined[:,0].min()-1),combined[:,0].max()+1)
                ax.set_title(LABELS[m]+'\n'+t);ax.set_xlabel('lateral y (m)');ax.set_ylabel('forward x (m)');ax.grid(alpha=.2)
        fig.savefig(args.report/f'same_scene_trajectories_{page+1}.png',dpi=170);fig.savefig(args.report/f'same_scene_trajectories_{page+1}.pdf');plt.close(fig)

def report(args,summary,comparisons,nlogs):
    lines=['# 五个 checkpoint 的同场景轨迹分布实验','',f'1000 个固定随机 Navtest 场景，覆盖 {nlogs} 个日志；每模型每场景每协议 64 条轨迹，总计 640,000 条。以下为实际采样结果，不是重新计算的 benchmark PDMS。','', '统一前视图 VLM 条件缓存和 FP32 计算；DDIM 5 步；两个协议共享初始高斯样本：initial_noise_only 关闭中途加噪，stochastic_ddim 开启 eta=1 的中途加噪。关闭 dropout，固定场景级随机种子。SFT 模型保留训练所需 FS-Norm 与 Planning Token Adapter，其他模型使用对应原生代码。','']
    for protocol in ['initial_noise_only','stochastic_ddim']:
        lines+=['## '+protocol,'','|模型|两两 ADE m（95% CI）|两两 FDE m|终点横向 std m|1m 最大模式占比|平均 GT ADE m|','|---|---:|---:|---:|---:|---:|']
        for m in ORDER:
            get=lambda f:next(r for r in summary if r['model']==m and r['protocol']==protocol and r['k']==64 and r['metric']==f)
            a=get('pair_ade_m')
            lines.append(f"|{LABELS[m]}|{a['mean']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}]|{get('pair_fde_m')['mean']:.4f}|{get('endpoint_lat_std_m')['mean']:.4f}|{get('largest_mode_1m')['mean']:.3f}|{get('mean_gt_ade_m')['mean']:.4f}|")
    lines+=['','## 解读边界','','- 两两 ADE/FDE 和方差越大表示越宽；局部集中比例、最大几何模式占比越大表示越集中。','- 模式是用物理距离阈值做 complete-linkage 聚类得到的几何簇，不等于左转/右转等语义行为。0.5/1/2m 阈值敏感性和 16/32/64 样本敏感性见 CSV。','- GT 平均误差辅助识别发散，best-of-K GT 误差仅是离线覆盖指标；本实验未计算候选 PDMS，不能把宽分布直接解释为更好的驾驶或更好的安全覆盖。','- 每个场景先计算统计量，再跨场景平均；CI 采用按日志重采样的 3000 次 bootstrap。配对差值使用相同场景，不能把 64 条轨迹当作独立场景扩大样本量。','- 五模型并非严格同初始化、同结构、同训练数据的随机对照；A5/V6 的归一化和结构也不同。这是 checkpoint 分布对比，不能将全部差异归因于 SFT/GRPO。APR 曾使用 Navtest 进行模型选择。','- 历史 86.92/87.51/90.41/91.45 分数仅用于识别模型，不是本次 1000 场景上的 PDMS。','','## 文件','','- [交互式同场景轨迹查看器](trajectory_viewer.html)','- [分布汇总图](distribution_summary.png)','- [宽度分布 ECDF](width_ecdf.png)','- [同场景轨迹图 1](same_scene_trajectories_1.png)','- [同场景轨迹图 2](same_scene_trajectories_2.png)','- [均值、中位数与置信区间](summary.csv)','- [全部模型配对差值](paired_comparisons.csv)','- [逐场景指标](per_scene_metrics.csv)','- [完整性检查](validation.json)']
    (args.report/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=16);main(p.parse_args())
