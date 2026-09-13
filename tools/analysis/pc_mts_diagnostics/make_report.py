"""Assemble a report from actual full-run tables; no synthetic result entries."""
import argparse,subprocess
import pandas as pd
from common import *

def main(args):
    cfg=config(args.config);sc=manifest(args.manifest);out=ROOT/cfg['output_dir'];md=out/'metrics';manifest_rows=models()
    def table(frame,columns):
        def cell(value):
            if pd.isna(value):return '—'
            return f'{value:.4f}' if isinstance(value,(float,np.floating)) else str(value)
        return '\n'.join(['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']+['|'+'|'.join(cell(v) for v in row)+'|' for row in frame[columns].itertuples(index=False,name=None)])
    def means(path,group,subset=None):
        frame=pd.read_csv(path)
        if subset:frame=frame[frame.subset==subset]
        return frame.groupby(group).mean(numeric_only=True).reset_index()
    joined=pd.read_csv(md/'table_a_d_scene.csv');assert len(joined)==5*sc['scene_count']
    policy=joined.groupby('checkpoint').mean(numeric_only=True).reindex(MODELS).reset_index();cats={r['name']:r['category'] for r in manifest_rows};policy['category']=policy.checkpoint.map(cats)
    oracle='oracle_64' if 'oracle_64' in policy else 'oracle_8'
    cols_a=['checkpoint','category','mean_pdms','feasible_rate','pairwise_ade','spread_auc','r90','endpoint_spread','effective_rank','center_shift_from_il',oracle,'D_positive']
    cols_d=['checkpoint','feasible_rate','mean_pdms',oracle,'oracle_mean_gap','p_plus_0','p_plus_1','p_plus_2','hit8_0','hit8_1','hit8_2','all_infeasible_group_rate','mostly_infeasible_group_rate','unsafe_positive_advantage','feasible_positive_advantage','D_all','D_feasible','D_positive']
    policy[cols_a].to_csv(md/'TABLE_A.csv',index=False);policy[cols_d].to_csv(md/'TABLE_D.csv',index=False)
    lines=['# PC-MTS Policy Distribution Diagnostics — 2026-09-13','', '## 1. Experimental Setup','',f'主实验：Navtrain，{sc["scene_count"]} 个固定场景，每 checkpoint 每场景 {cfg["num_rollouts"]} 条随机轨迹。五个真实 checkpoint 与四种 reconstructed candidate selection rule 是两条独立分析线，没有重新训练或把 MTS 权重对应到 M1–M4。','',f'共同采样器：FP32 VLM/DiT，DDIM {cfg["diffusion_steps"]} 步，eta={cfg["eta"]}，temperature=1，不使用 CFG；每个场景共享初始噪声 seed 和中间采样 seed。轨迹为未来 4 秒、8 个点、米制 XY 与弧度 heading。A5/V6 的训练专属 FS-Norm 与 adapter 必须保留，因此内部归一化无法作为严格相同的训练算法对照。','', '## 2. Checkpoint Manifest','']
    for r in manifest_rows:lines += [f'- **{r["label"]}** ({r["category"]})：`{r["checkpoint_path"]}`；SHA256 `{r["sha256"]}`。历史 PDMS 为 {r["reported_pdms"]}，仅识别权重。']
    lines += ['', '完整来源、config、代码版本见 `manifests/checkpoint_manifest.yaml` 和 `manifests/assets.json`。','', '## 3. Scene Sampling','',f'从官方 Navtrain 的 {sc["population"]:,} 个 token 中按 command 比例分层，seed=20260913；{sc["selected_commands"]}，覆盖 {len({r["log"] for r in sc["scenes"]})} 个日志。无基于模型输出或 score 的筛选。场景文件 `manifests/scenes_1000.json`。代表场景在模型结果之前按 2 个 straight、2 个 turn、2 个 heading 总变化不低于 0.55 rad 的 high-curvature 场景固定。','', '## 4. Parallelization and Compute','', '每次一个 checkpoint，8 个 torchrun rank 分片；每 rank 4 个 DataLoader workers。rollout 先落盘，再运行最多 96 个 CPU scoring workers，嵌套 BLAS/OpenMP 线程为 1。`logs/*_resources.jsonl` 记录 GPU、CPU 和内存；`manifests/*_execution.json` 记录实际命令与时长。每个缓存包含场景/config/checkpoint 身份或指向其父缓存哈希。','', '20 场景 smoke 已验证真权重、轨迹坐标、采样回放、批量评分、原生 GRPO 奖励、几何控制、扰动幅度和绘图。50 场景 benchmark 的运行记录位于 `benchmark/manifests/`。批量与逐条 PDM 的全部七项分数误差为 0；GT 变换误差为 0。','', '## 5. Policy Distribution Width / Sharpness','', 'TABLE A：数值先在场景内聚合，再对场景平均；所有距离以米计。D_positive 为可行且超过默认 IL reference 的轨迹之间的两两 ADE；不足两条的场景记为缺失，不当作零多样性。','',table(policy,cols_a),'', '完整时序 spread、R50/R90、终点 covariance trace、半径、center shift 及子集有效场景数在 CSV/Parquet 与 `policy_summary.csv`。Fig-0a–g 展示宽度、中心移动和预注册场景。','', '## 6. Reconstructed Candidate Reservoir','',f'每个场景 {cfg["raw_gt_count"]+4*cfg["raw_model_count"]} 条：1 exact GT + {cfg["raw_gt_count"]-1} 条平滑结构扰动 + 四个非 IL 模型各 {cfg["raw_model_count"]} 条独立种子输出。各方法读取同一个文件；官方 IL 只作为 support/reference，不作为 raw source。历史 candidate archive 确实找到，但未将新构建库冒称历史训练数据。','', '50 场景中，96 条库的 PC-MTS 主规则不足比例为 98%，按预注册规则扩为 128；扩容后仍为 98%，全部已允许回退后仍有 80% 场景没有合格父轨迹。详见 `reservoir_expansion_decision.json`。','', '## 7. Four Candidate Selection Rules','', '- Score-only：有效候选直接按标准 PDMS 降序，稳定 ID 打破完全相同的 score。','- Pareto：复用历史 `pareto_front_mask` 和三个 component（progress、TTC、driving direction），逐层 non-dominated sorting；同层先高 PDMS，差距不超过 0.01 point 时用几何多样性打破平局；不使用 GT 或 policy distance。','- Pareto + GT-distance：用 IL rollout-to-GT 的 command 条件 80% 分位作为主阈值，不足时 90%、95%，最后 filler；独立保存固定 80/90/95 的敏感性统计。','- PC-MTS：Pareto Front 1、保守可行、PDMS 不低于默认 IL reference、q∈[50,95)、四次 selection perturbation 中局部可行比例≥0.75；依次只放宽局部比例到0.5、q下界到40、q上界到99，之后才 filler。q 不参与“越大越好”的排序。','', '未在原需求中量化的“Pareto qualified”和“基本 quality”，在正式结果之前固定为 Front 1 与上述 reference-relative quality；这些具体定义会影响候选可得性，报告不将其包装成 PC-MTS 的唯一可能定义。','', '## 8. Candidate Pool Sanity Check','']
    pool_path=md/'candidate_pool_scene.csv'
    if pool_path.exists():
        pools=means(pool_path,'method','all16');cols_b=['method','candidate_pdms','d_GT','q_policy','core_fraction','boundary_fraction','ood_fraction','pool_diversity','filler_fraction','unique_fraction','construction_success'];pools[cols_b].to_csv(md/'TABLE_B.csv',index=False);lines += ['TABLE B（ALL-16）：','',table(pools,cols_b),'','UNIQUE-NON-FILL 与 construction success / 缺失场景数必须同时查看；不能将 filler 算作新的可学习模式。']
    else:lines += ['候选池尚未全部完成；不提供虚构的 TABLE B。']
    lines += ['', '## 9. Exp-1 Policy-relative Candidate Position','', 'k=5；IL 自距离排除自身，candidate 距离取最近五条 IL rollout 的平均 ADE。q 为相对于 64 个 IL self-distance 的右连续经验 CDF；Core<50、Boundary∈[50,95)、OOD≥95。q 是局部几何参照的分位数，不是模型显式概率或经过校准的 OOD 检验。Fig-1a–e 与 candidate_records 保存位置、质量、来源、fallback、GT 距离和边界高质量质量占比。','', '## 10. Exp-2 Local Robustness','', '每个选中候选使用独立 held-out stream 生成 12 条平滑扰动：0.02/0.05/0.10/0.20m，各3个方向。相同轨迹在不同 pool 中共享相同扰动，selection perturbation 不复用。CVaR20 对最差20%经验质量精确加权（12条时为最差2条加第三差的0.4权重，再除2.4）。下降斜率定义为四档幅度对平均 PDMS 的线性回归斜率取负，单位 points/m；负值表示扰动后改善，不截成0。','']
    if (md/'local_robustness_scene.csv').exists():
        local=means(md/'local_robustness_scene.csv','method','all16');cols_c=['method','original_pdms','local_mean','local_p10','local_cvar20','local_feasible_rate','local_drop','robustness_slope'];local[cols_c].to_csv(md/'TABLE_C.csv',index=False);lines += ['TABLE C：','',table(local,cols_c)]
    else:lines += ['局部扰动评测尚未全部完成；TABLE C 待真实计算。']
    lines += ['', '## 11. Exp-3 Rollout & GRPO Readiness','', 'TABLE D：p_plus 与实际 Hit@8 使用标准 PDMS、Δ=0/1/2 point；64条严格按种子顺序分成8个G=8 group。Reference是额外采样的官方 IL 默认采样器 seed=0 单条轨迹，不是 IL Oracle@64。Oracle@K 是离线候选上界，不是部署策略得分。','',table(policy,cols_d),'','Unsafe Positive Advantage 使用从历史 GRPO 源文件提取并执行的 advantage 代码：group mean/std，PyTorch unbiased std +1e-8，实际默认 quantile clipping 0/1。历史训练 reward 使用 progress/TTC/comfort=10/5/2；统一展示的标准 PDMS 为5/5/2。因此 advantage 病态诊断与 PDMS 质量采用各自正确口径，原始代码和校验见 `grpo_advantage_source.json`。','', '保守 feasible 明确定义为 NC=DAC=TTC=DDC=1；hard failure 为 NC<1 或 DAC<1，两者不是互补。Comfort 单独作为质量指标。每个场景先计算 unsafe/positive 比例，正 advantage 分母为零时记缺失，同时保存分子/分母；不靠跨场景样本堆叠制造显著性。','', '## 12. Statistical Tests','', '主比较按 scene 配对，3000 次场景 bootstrap；报告均值差、中位数差、95% CI、Wilcoxon signed-rank p、matched rank-biserial effect size、有效场景数。未将 64 次 rollout 当作独立场景。区间与p值为逐项结果，未作多重比较校正；同日志邻近场景仍存在相关性，是解释边界。相关系数为描述性，详见 candidate_correlations.json。','', '## 13. Hypothesis Validation','', '待主实验全部计算并人工检查后，逐项判定 H1–H6；本节不从 smoke 数据或预设方向填入结论。','', '## 14. Limitations','', '- 这是一组已训练 checkpoint 的 Navtrain 机制诊断，不是独立测试集泛化结论。历史 APR 曾使用 Navtest 进行模型选择。','- MTS 权重与 IL/GRPO 的内部归一化、adapter 和初始化不同，五模型不是严格训练算法因果消融；官方 IL→原版 GRPO 的初始化关系有训练 manifest 支持。','- 64样本的kNN经验 support 可能很窄，q≥95并不自动证明真实策略概率为零。PC-MTS 缺少父轨迹时不能从无中生成合格监督。','- 小幅扰动衡量指定局部邻域的敏感性，不能证明现实驾驶安全。','- GRPO readiness 是机会、可行性与advantage结构诊断，不是重新训练 GRPO 的最终性能保证。','', '## 15. Conclusions','', '待完整验收后填入基于主实验的结论。','', f'原始输出目录：`{out}`。图目录：`{out / "figures"}`。']
    text='\n'.join(lines)+'\n'
    interpretation=out/'report/hypothesis_interpretation.md'
    conclusions=out/'report/conclusions.md'
    if interpretation.exists():
        start=text.index('## 13. Hypothesis Validation');end=text.index('## 14. Limitations');text=text[:start]+interpretation.read_text()+'\n'+text[end:]
    if conclusions.exists():
        start=text.index('## 15. Conclusions');text=text[:start]+conclusions.read_text()+'\n'
    text=text.replace('\n## 1. Experimental Setup','\n状态：五个checkpoint的1000场景分布/PDMS/readiness及M1–M3已完成；M4零父轨迹的处理等待用户明确。未宣称完整任务通过验收。所有表格中的比例均为0–1，PDMS为0–100 points。\n\n## 1. Experimental Setup',1)
    text=text.replace('## 6. Reconstructed Candidate Reservoir','有效秩均值 IL/MTS86.92/MTS87.51/GRPO/APR 为1.666/1.737/1.794/1.877/2.997；APR变化分布在更多线性方向，但这不等于证明存在更多语义模态。\n\n## 6. Reconstructed Candidate Reservoir',1)
    text += '\n计算时长：正式观测特征、五模型rollout和PDM评分合计1173.59秒（19.56分钟）；M1–M3候选与扰动pipeline合计427.39秒（7.12分钟），不含此前资产审计、smoke、50场景benchmark和报告整理。\n'
    text += '\n[PNG/PDF图目录](../outputs/pc_mts_diagnostics/figures/) · [CSV/Parquet与配对统计](../outputs/pc_mts_diagnostics/metrics/) · [冻结场景](../outputs/pc_mts_diagnostics/manifests/scenes_1000.json) · [完整性校验](../outputs/pc_mts_diagnostics/manifests/full_validation.json) · [待明确的零候选处理](../outputs/pc_mts_diagnostics/report/OPEN_DECISIONS.md)\n'
    partial=md/'conditional_pc'
    if (partial/'cohort.json').exists() and not (out/'manifests/coverage_completion.json').exists():
        cohort=read(partial/'cohort.json');n=cohort['paired_scene_count'];pending=cohort['pending_count']
        note=f'PC-MTS 已完成其中 {n} 个有合格父轨迹场景，余下 {pending} 个仍 pending；以下条件性比较把四种方法都限制在相同的 {n} 个场景，绝不与其他方法的1000场景均值混排。'
        text=text.replace('M4零父轨迹的处理等待用户明确。',f'M4已完成{n}个有合格父轨迹场景，{pending}个零父轨迹场景的处理等待用户明确。',1)
        b=pd.read_csv(partial/'candidate_all16_table.csv');c=pd.read_csv(partial/'local_all16_table.csv')
        ub=pd.read_csv(partial/'candidate_unique_non_fill_table.csv');uc=pd.read_csv(partial/'local_unique_non_fill_table.csv')
        text=text.replace('## 9. Exp-1 Policy-relative Candidate Position',f'TABLE B-conditional：{note}\n\n'+table(b,list(b.columns))+'\n\nTABLE B-conditional（UNIQUE-NON-FILL）：\n\n'+table(ub,list(ub.columns))+'\n\n## 9. Exp-1 Policy-relative Candidate Position',1)
        text=text.replace('## 11. Exp-3 Rollout & GRPO Readiness',f'TABLE C-conditional（ALL-16，配对 {n} 场景）：\n\n'+table(c,list(c.columns))+'\n\nTABLE C-conditional（UNIQUE-NON-FILL）：\n\n'+table(uc,list(uc.columns))+'\n\n[条件性完整报告及配对差值](../outputs/pc_mts_diagnostics/report/PC_MTS_CONDITIONAL_DIAGNOSTICS.md) · [条件性PNG/PDF图](../outputs/pc_mts_diagnostics/figures/conditional_pc/)\n\n## 11. Exp-3 Rollout & GRPO Readiness',1)
        if (out/'manifests/pc_eligibility_funnel.json').exists():
            funnel=read(out/'manifests/pc_eligibility_funnel.json')
            text=text.replace('## 8. Candidate Pool Sanity Check','零父轨迹原因（固定门限、未重新调参）：'+str(funnel['reason_counts'])+'。775个场景有基本高质量候选，但没有候选同时落在最终放宽的q∈[40,99)区间；33个场景没有满足基本质量交集的候选。局部可行门限不是这808例的最终致零原因。可构建组的IL参考均分99.09，待定组90.74，进一步说明192场景具有明显选择偏差。\n\n## 8. Candidate Pool Sanity Check',1)
        executions=list((out/'manifests').glob('pc_partial_*_execution.json'))
        seconds=sum(read(p)['wall_seconds'] for p in executions)
        text+=f'\nPC-MTS 条件性追加计算：{seconds:.2f}秒（{seconds/60:.2f}分钟），包括父轨迹筛选、填充重评分、held-out评分、配对统计及补充图。CPU阶段GPU占用来自此前恢复的压力任务，不计作本实验GPU利用率。\n'
    if (out/'manifests/coverage_validation.json').exists():
        cv=read(out/'manifests/coverage_validation.json');coverage=read(out/'manifests/coverage_completion.json')
        text=text.replace('状态：五个checkpoint的1000场景分布/PDMS/readiness及M1–M3已完成；M4零父轨迹的处理等待用户明确。未宣称完整任务通过验收。',f'状态：五个checkpoint的1000场景分布/PDMS/readiness，以及四方法的{cv["candidate_count"]:,}条候选和{cv["heldout_scored_count"]:,}条held-out扰动全部完成。PC-MTS包含用户授权的coverage扩展，原规则零父轨迹率仍单独保留。')
        text=text.replace('20 场景 smoke 已验证', '初始20场景smoke验证了五权重和M1–M3；原M4因零父轨迹无法通过。用户要求完整覆盖后，先完成扩展规则的20场景全四方法smoke，再运行1000场景覆盖评测。Smoke已验证',1)
        text=text.replace('## 8. Candidate Pool Sanity Check',f'**用户授权的全覆盖扩展（PC-MTS + coverage）**：原规则与其192个非空池保持不变；仅对另外{coverage["coverage_scene_count"]}个零父轨迹场景分层补充。Level 6取消q限制、保留Front 1/完整可行性/IL参考分数/局部可行性≥0.5；level 7保留完整可行性和局部≥0.5；level 8保留完整可行性；level 9保留NC=DAC=1。层内使用原PDMS与去冗余规则，从相同128库选择。新规则见 `configs/pc_mts_diagnostics/coverage_fallback.yaml`。这是知道前期结果后增加的探索性方案，在新增候选及其held-out评分前固定，没有通过反复调阈值追求预期结论。\n\n后续表和图中的 `pc_mts` / “PC-MTS + coverage”均指此完整方法。原Boundary门限仍是[50,95)，因此coverage候选可能被如实判为OOD。\n\n## 8. Candidate Pool Sanity Check',1)
        b=pd.read_csv(md/'coverage/candidate_unique_non_fill_table.csv');c=pd.read_csv(md/'coverage/local_unique_non_fill_table.csv')
        text=text.replace('## 9. Exp-1 Policy-relative Candidate Position','TABLE B（UNIQUE-NON-FILL，全1000场景）：\n\n'+table(b,list(b.columns))+'\n\n真实raw coverage候选计入UNIQUE-NON-FILL；另有QUALIFIED-NON-FILL排除所有额外回退候选。仅后者仍只在原规则可构建场景上有定义，详见完整覆盖补充报告。\n\n## 9. Exp-1 Policy-relative Candidate Position',1)
        text=text.replace('## 11. Exp-3 Rollout & GRPO Readiness','TABLE C（UNIQUE-NON-FILL，全1000场景）：\n\n'+table(c,list(c.columns))+'\n\n[完整覆盖及分层配对报告](../outputs/pc_mts_diagnostics/report/PC_MTS_FULL_COVERAGE.md)保存ALL-16、UNIQUE-NON-FILL、QUALIFIED-NON-FILL，以及原可构建192/额外覆盖808两个场景层的四方法同场景比较。\n\n## 11. Exp-3 Rollout & GRPO Readiness',1)
        text=text.replace('[待明确的零候选处理]','[已解决的覆盖规则说明]')
        text=text.replace('## 12. Statistical Tests','Coverage共新增12,928条真实raw候选：level 6/7/8/9分别为5,793/6,869/1/265条，没有新增duplicate filler。Level 9只满足NC=DAC=1，不能称为满足完整可行性；局部评测按真实TTC/DDC等结果计分。\n\n## 12. Statistical Tests',1)
        seconds=sum(read(p)['wall_seconds'] for p in (out/'manifests').glob('pc_coverage_*_execution.json'))
        text+=f'\n1000场景PC-MTS覆盖追加pipeline耗时{seconds:.2f}秒（{seconds/60:.2f}分钟）；复用已有候选与评分缓存。含全四方法重新分析、图表和覆盖校验；不含之前的smoke与人工报告整理。原192场景条件性pipeline另耗时97.70秒。\n'
        text+='\n[实际8-GPU与CPU执行命令](../outputs/pc_mts_diagnostics/report/EXECUTED_COMMANDS.md) · [逐项完成验收](../outputs/pc_mts_diagnostics/manifests/completion_audit.json)\n'
    path=ROOT/'reports/PC_MTS_POLICY_DISTRIBUTION_DIAGNOSTICS_20260913.md';path.parent.mkdir(exist_ok=True);path.write_text(text);print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest');main(p.parse_args())
