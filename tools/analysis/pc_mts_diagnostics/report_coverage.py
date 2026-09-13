"""Full-cohort inference with transparent coverage fallback and stratified results."""
import argparse,concurrent.futures
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from scene_statistics import summary_and_tests
from report_partial_pc import markdown
from select_candidate_pools import COVERAGE_POLICY,pareto_ranks

def validate_scene(task):
    r,cfg,out=task;token=r['token'];raw=np.load(out/'raw_candidates'/f'{token}.npz');raw_score=np.load(out/'evaluator/raw_candidates'/f'{token}.npz')['trajectories']
    loc=feasible(np.load(out/'evaluator/selection_local/common'/f'{token}.npz')['trajectories']).mean(1);rank=pareto_ranks(raw_score)
    ref=float(np.load(out/'evaluator/rollouts/official_il'/f'{token}.npz')['reference'][0,6]);records=[]
    for method in METHODS:
        path=out/'candidate_pools'/method/f'{token}.npz';pool=np.load(path);a=pool['trajectories'];s=pool['scores'];meta=json.loads(str(pool['metadata']))
        assert a.shape==(cfg['pool_size'],8,3) and s.shape==(cfg['pool_size'],7)
        assert np.isfinite(a).all() and np.isfinite(s).all()
        hpath=out/'heldout'/method/f'{token}.npz';held=np.load(hpath);evaluated=np.load(out/'evaluator/heldout'/method/f'{token}.npz')
        assert held['trajectories'].shape==(cfg['pool_size'],12,8,3)
        assert evaluated['trajectories'].shape==(cfg['pool_size'],12,7) and np.isfinite(evaluated['trajectories']).all()
        assert json.loads(str(held['metadata']))['parent_sha256']==sha(path)
        assert json.loads(str(evaluated['metadata']))['input_sha256']==sha(hpath)
        ids=list(pool['candidate_id'])
        for i in range(cfg['pool_size']):
            level=int(pool['fallback_level'][i]);idx=int(pool['raw_index'][i])
            if pool['is_fill'][i]:
                parent=ids.index(pool['fill_parent_id'][i]);assert not pool['is_fill'][parent]
                assert np.linalg.norm(a[i,:,:2]-a[parent,:,:2],axis=1).max()<=.010000001
            else:
                assert idx>=0 and np.array_equal(a[i],raw['trajectories'][idx]) and np.array_equal(s[i],raw_score[idx])
            if method=='pc_mts' and level>=6:
                assert meta['coverage_policy_hash']==sha(COVERAGE_POLICY) and meta['raw_initial_selected_count']==0
                if level==6:assert rank[idx]<=cfg['pareto_qualified_max_front'] and feasible(s[i][None])[0] and s[i,6]>=ref-1e-10 and loc[idx]>=.5
                elif level==7:assert feasible(s[i][None])[0] and loc[idx]>=.5
                elif level==8:assert feasible(s[i][None])[0]
                elif level==9:assert not hard_failure(s[i][None])[0]
                elif level==10:assert pool['is_fill'][i]
                else:raise AssertionError(level)
        records.append(dict(token=token,method=method,candidates=len(a),filler_count=int(pool['is_fill'].sum()),coverage_count=int((pool['fallback_level']>=6).sum()) if method=='pc_mts' else 0,original_feasible_count=int(feasible(s).sum()),heldout_scored=cfg['pool_size']*12))
    return records

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];md=out/'metrics';dest=md/'coverage';dest.mkdir(exist_ok=True)
    coverage=read(out/'manifests/coverage_completion.json');assert coverage['identity']==identity(cfg,sc)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(32,len(sc['scenes']))) as ex:validation=[x for rows in ex.map(validate_scene,[(r,cfg,out) for r in sc['scenes']],chunksize=1) for x in rows]
    save(out/'manifests/coverage_validation.json',dict(identity=identity(cfg,sc),scene_count=len(sc['scenes']),pool_size=cfg['pool_size'],methods=METHODS,full_completion=True,candidate_count=sum(r['candidates'] for r in validation),heldout_scored_count=sum(r['heldout_scored'] for r in validation),all_parent_hashes_match=True,all_raw_candidates_traceable=True,all_coverage_tier_predicates_verified=True,rows=validation))
    pools=pd.read_csv(md/'candidate_pool_scene.csv');local=pd.read_csv(md/'local_robustness_scene.csv');candidates=pd.read_parquet(md/'candidate_records.parquet')
    assert len(candidates)==len(sc['scenes'])*4*cfg['pool_size']
    for frame in [pools,local]:
        assert len(frame)==len(sc['scenes'])*4*3
        assert frame.groupby(['method','subset']).size().eq(len(sc['scenes'])).all()
    lines=['# 全场景四方法候选与覆盖回退分析','',f'固定 {len(sc["scenes"])} 场景，每方法每场景 {cfg["pool_size"]} 条，共 {len(candidates):,} 条候选；{len(candidates)*12:,} 条 held-out 扰动全部评分完成。没有删除场景。','',f'PC-MTS 原规则在 {coverage["original_eligible_scene_count"]} 个场景有父轨迹，另外 {coverage["coverage_scene_count"]} 个场景使用用户授权的 coverage 扩展。主结果中的 PC-MTS + coverage 是扩展方法，不冒称全部满足原 Boundary 条件。','', '扩展只在原规则零父轨迹时启用：level 6 取消 q 限制但保留质量、Front 1、reference 和局部可行性；level 7 保留完整可行性与局部可行性；level 8 保留完整可行性；level 9 保留 NC=DAC=1。层内按原 PDMS 和去冗余规则选取共同库中的轨迹。该规则在新增 held-out 评分之前固定，未使用 held-out 结果排序。','', 'UNIQUE-NON-FILL 包含真实 raw coverage 候选；QUALIFIED-NON-FILL 进一步排除 coverage 候选，仅描述原规则支持的父轨迹，缺失记为缺失。','']
    summary_tables={}
    for subset in ['all16','unique_non_fill','qualified_non_fill']:
        for prefix,frame in [('candidate',pools),('local',local)]:
            part=frame[frame.subset==subset];summary_and_tests(part,'method',dest,prefix+'_'+subset,cfg['bootstrap_replicates'])
            mean=part.groupby('method').mean(numeric_only=True).reindex(METHODS).reset_index()
            columns=['method','count','candidate_pdms','d_GT','q_policy','core_fraction','boundary_fraction','ood_fraction','pool_diversity','filler_fraction','unique_fraction','coverage_fraction'] if prefix=='candidate' else ['method','count','original_pdms','local_mean','local_p10','local_cvar20','local_feasible_rate','local_drop','robustness_slope']
            mean['metric_valid_scenes']=mean.method.map(part.groupby('method')['pool_diversity' if prefix=='candidate' else 'local_mean'].count());columns.append('metric_valid_scenes')
            table=mean[columns];summary_tables[prefix+'_'+subset]=table;table.to_csv(dest/f'{prefix}_{subset}_table.csv',index=False)
            lines += [f'## {prefix} / {subset}','',markdown(table),'']
    original_tokens=set(candidates[(candidates.method=='pc_mts')&(~candidates.is_coverage_fallback)].token)
    for label,tokens in [('original_parent_available',original_tokens),('coverage_required',set(r['token'] for r in sc['scenes'])-original_tokens)]:
        for prefix,frame in [('candidate',pools),('local',local)]:
            part=frame[(frame.subset=='all16')&frame.token.isin(tokens)]
            if not len(part):continue
            summary_and_tests(part,'method',dest,f'{prefix}_{label}',cfg['bootstrap_replicates'])
            part.groupby('method').mean(numeric_only=True).to_csv(dest/f'{prefix}_{label}_means.csv')
    pc=candidates[candidates.method=='pc_mts'];levels=pc.groupby(['fallback_level','is_fill','is_coverage_fallback']).size().rename('count').reset_index();levels.to_csv(dest/'fallback_counts.csv',index=False)
    lines += ['## 回退构成','',markdown(levels),'']
    tests=[]
    for prefix,metrics in [('candidate',['candidate_pdms','boundary_fraction','ood_fraction','pool_diversity']),('local',['local_mean','local_cvar20','local_feasible_rate','local_drop'])]:
        p=pd.read_csv(dest/f'{prefix}_all16_paired_tests.csv');tests.append(p[(p.a=='pc_mts')&p.metric.isin(metrics)])
    tests=pd.concat(tests,ignore_index=True);tests.to_csv(dest/'pc_full_cohort_paired_tests.csv',index=False)
    lines += ['## PC-MTS + coverage 的全队列配对差值','', '差值为 PC-MTS + coverage 减去另一方法；95% CI 使用场景 bootstrap，未作多重比较校正。','',markdown(tests[['b','metric','mean_difference','ci_low','ci_high','n_scenes']]),'']
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(levels.fallback_level.astype(str),levels['count']);ax.set_xlabel('Fallback level (0–5 original; 6–10 coverage)');ax.set_ylabel('PC-MTS candidate count');ax.set_title(f'All {len(sc["scenes"])} scenes; {coverage["coverage_scene_count"]} require coverage')
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'Fig-coverage-levels.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    report=out/'report/PC_MTS_FULL_COVERAGE.md';report.parent.mkdir(exist_ok=True);report.write_text('\n'.join(lines)+'\n');print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
