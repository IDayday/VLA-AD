"""Report all four rules on the same parent-eligible scenes, retaining coverage."""
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from scene_statistics import summary_and_tests

def markdown(frame):
    def cell(x):
        if pd.isna(x):return '—'
        return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '\n'.join(['|'+'|'.join(frame.columns)+'|','|'+'|'.join(['---']*len(frame.columns))+'|']+['|'+'|'.join(map(cell,row))+'|' for row in frame.itertuples(index=False,name=None)])

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir']
    pending=pending_pc_tokens(out,cfg,sc);dest=out/'metrics/conditional_pc';cohort=read(dest/'cohort.json');n=cohort['paired_scene_count']
    assert n==len(sc['scenes'])-len(pending)
    pool=pd.read_csv(dest/'candidate_pool_scene.csv');local=pd.read_csv(dest/'local_robustness_scene.csv')
    candidate=pd.read_csv(dest/'candidate_records.csv');figdir=out/'figures/conditional_pc';figdir.mkdir(parents=True,exist_ok=True)
    for frame in [pool,local]:
        assert frame.groupby(['method','subset']).size().eq(n).all()
        assert not frame.duplicated(['token','method','subset']).any()
    validation=[]
    for r in sc['scenes']:
        token=r['token']
        if token in pending:continue
        path=out/'candidate_pools/pc_mts'/f'{token}.npz';p=np.load(path);heldpath=out/'heldout/pc_mts'/f'{token}.npz';held=np.load(heldpath)
        scores=np.load(out/'evaluator/heldout/pc_mts'/f'{token}.npz')
        assert json.loads(str(held['metadata']))['parent_sha256']==sha(path)
        assert json.loads(str(scores['metadata']))['input_sha256']==sha(heldpath)
        assert held['trajectories'].shape==(cfg['pool_size'],12,8,3)
        assert scores['trajectories'].shape==(cfg['pool_size'],12,7) and np.isfinite(scores['trajectories']).all()
        ids=list(p['candidate_id']);maxdis=0.
        for i in np.flatnonzero(p['is_fill']):
            j=ids.index(p['fill_parent_id'][i]);assert not p['is_fill'][j]
            displacement=float(np.linalg.norm(p['trajectories'][i,:,:2]-p['trajectories'][j,:,:2],axis=1).max())
            assert displacement<=.010000001
            assert p['fallback_level'][i] in [4,5]
            assert feasible(p['scores'][i][None])[0]
            maxdis=max(maxdis,displacement)
        validation.append(dict(token=token,fill_count=int(p['is_fill'].sum()),parent_count=int((~p['is_fill']).sum()),maximum_filler_displacement_m=maxdis))
    save(out/'manifests/pc_partial_heldout_validation.json',dict(identity=identity(cfg,sc),full_scene_count=len(sc['scenes']),completed_scene_count=n,pending_scene_count=len(pending),heldout_trajectories=n*cfg['pool_size']*12,all_cache_parent_hashes_match=True,all_fillers_trace_to_accepted_non_fill_parent=True,rows=validation))
    texts=['# PC-MTS：有合格父轨迹场景的条件性诊断','',f'完整冻结队列仍为 {len(sc["scenes"])} 场景；{n} 个场景可构建 PC-MTS，{len(pending)} 个（{len(pending)/len(sc["scenes"]):.1%}）经过全部已规定放宽后仍没有合格父轨迹。后者保留为 pending，没有新增回退规则，也没有生成空池或伪造候选。','',f'以下四种方法都在相同的 {n} 个场景上重新聚合、配对 bootstrap（3000 次）。这是按父轨迹可得性筛选后的条件性比较，不能外推成全 {len(sc["scenes"])} 场景的 PC-MTS 效果。选择可得性本身依赖模型与质量；这不是随机子集。','']
    tables={}
    for subset in ['all16','unique_non_fill']:
        for prefix,frame in [('candidate',pool),('local',local)]:
            part=frame[frame.subset==subset]
            summary=summary_and_tests(part,'method',dest,prefix+'_'+subset,cfg['bootstrap_replicates'])
            tables[(prefix,subset)]=summary
            means=part.groupby('method').mean(numeric_only=True).reindex(METHODS).reset_index()
            columns=['method','count','candidate_pdms','d_GT','q_policy','core_fraction','boundary_fraction','ood_fraction','pool_diversity','filler_fraction','unique_fraction'] if prefix=='candidate' else ['method','count','original_pdms','local_mean','local_p10','local_cvar20','local_feasible_rate','local_drop','robustness_slope']
            if prefix=='candidate':
                means['pool_diversity_valid_scenes']=means.method.map(part.groupby('method').pool_diversity.count());columns.append('pool_diversity_valid_scenes')
            means[columns].to_csv(dest/(prefix+'_'+subset+'_table.csv'),index=False)
            texts += [f'## {prefix} / {subset}','',markdown(means[columns]),'']
    stats=[('candidate','pool_diversity','Pool pairwise ADE (m)'),('candidate','boundary_fraction','Boundary fraction'),('candidate','filler_fraction','Filler fraction'),('local','local_mean','Held-out mean PDMS'),('local','local_cvar20','Held-out CVaR20'),('local','local_feasible_rate','Held-out feasible fraction')]
    for subset in ['all16','unique_non_fill']:
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,(prefix,metric,label) in zip(axes.flat,stats):
            table=tables[(prefix,subset)];x=table[table.metric==metric].set_index('group').reindex(METHODS)
            ax.errorbar(np.arange(4),x['mean'],yerr=np.maximum(0,np.stack([x['mean']-x.ci_low,x.ci_high-x['mean']])),fmt='o',capsize=4)
            if metric=='pool_diversity':
                for i,(_,row) in enumerate(x.iterrows()):ax.annotate(f'n={int(row.n_scenes)}',(i,row['mean']),xytext=(7,7),textcoords='offset points',fontsize=8)
            ax.set_xticks(np.arange(4),['Score','Pareto','GT-distance','PC-MTS'],rotation=15);ax.set_title(label);ax.grid(alpha=.2)
        fig.suptitle(f'Conditional on accepted PC-MTS parent: {n}/{len(sc["scenes"])} scenes | {subset}\nCommon cohort; diversity valid counts shown; remaining {len(pending)} scenes pending')
        fig.tight_layout(rect=(0,0,1,.92))
        for ext in ['png','pdf']:fig.savefig(figdir/f'pc_conditional_{subset}.{ext}',dpi=180)
        plt.close(fig)
    selected_tests=[]
    for subset in ['all16','unique_non_fill']:
        for prefix in ['candidate','local']:
            df=pd.read_csv(dest/f'{prefix}_{subset}_paired_tests.csv');df=df[(df.a=='pc_mts')&df.metric.isin(['pool_diversity','ood_fraction','boundary_fraction','candidate_pdms','local_mean','local_cvar20','local_feasible_rate','local_drop'])]
            df.insert(0,'subset',subset);selected_tests.append(df)
    tests=pd.concat(selected_tests,ignore_index=True);tests.to_csv(dest/'pc_vs_others_paired_tests.csv',index=False)
    texts += ['## 配对差值','', '差值为 PC-MTS 减去 comparator；区间为 scene bootstrap 95% CI，未作多重比较校正。','',markdown(tests[['subset','b','metric','mean_difference','ci_low','ci_high','n_scenes']]),'']
    pc=candidate[candidate.method=='pc_mts'];fallback=pc.groupby(['is_fill','fallback_level']).size().rename('candidate_count').reset_index();fallback.to_csv(dest/'pc_fallback_counts.csv',index=False)
    texts += ['## 填充与回退','',markdown(fallback),'',f'已完成 {n*cfg["pool_size"]} 条 PC-MTS 候选与 {n*cfg["pool_size"]*12} 条独立 held-out 扰动评分。所有 filler 都可追溯到非 filler 合格父轨迹，最大位移不超过 0.01m，并已重新评分。UNIQUE-NON-FILL 中不足两条的 diversity 为缺失；对应统计文件保留有效场景数。','', '即使条件性局部得分较高，也不足以证明 PC-MTS 在完整队列有效：零父轨迹覆盖率是必须同时报告的主要限制。']
    report=out/'report/PC_MTS_CONDITIONAL_DIAGNOSTICS.md';report.parent.mkdir(parents=True,exist_ok=True);report.write_text('\n'.join(texts)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
