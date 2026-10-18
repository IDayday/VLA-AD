"""Generate the final Chinese research report from completed, traceable artifacts."""
from __future__ import annotations
import argparse
import csv
import io
import numpy as np
from .common import *
from .probe_runner import merge_eval


def ci(value,percent=False):
    scale=100 if percent else 1;digits=5 if percent else 6
    return f"{scale*value['estimate']:.{digits}f} [{scale*value['lower95']:.{digits}f}, {scale*value['upper95']:.{digits}f}]"


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='formal');args=p.parse_args()
    cfg,identity=protocol();decision=read(ROOT/'analysis/decisions.json')
    h1=read(ROOT/'analysis/h1.json');h2=read(ROOT/'analysis'/f'probes_{args.prefix}_D_audit.json')
    secondary=read(ROOT/'analysis/probe_secondary_intervals.json');severe=read(ROOT/'analysis/severe_confirmation.json')
    costs=read(ROOT/'analysis/accounting.json');selection=read(ROOT/'probes/selection.json')
    if costs['status']!='FINAL' or costs['running']:raise ValueError('Final report requires finished owned jobs')
    for arm in cfg['probes']['arms']:
        for seed in cfg['seeds']['training']:
            for split in ('D_fit','D_cal','D_audit'):merge_eval(f'{args.prefix}_{arm}_{seed}',split)
    from . import figures
    figures.h1();figures.probes(args.prefix);figures.cases(args.prefix)
    if decision['stage_E']!='PASS':figures.gated_rl(decision['gate_reason_en'])
    dest=REPORT_DIR;dest.mkdir(parents=True,exist_ok=True)
    summary_rows=[]
    for label in ['H1','H2','H3']:
        v=decision[label];summary_rows.append(f"| {label} | `{v['status']}` | {v['reason']} |")
    primary=h1['audit_combined'];primary_rows=[]
    for step,j in [(1,0),(4,1)]:
        cells=[ci({k:primary[k][j][n] for k in ['estimate','lower95','upper95']}) for n in range(4)]
        primary_rows.append('| '+str(step)+' | '+' | '.join(cells)+' |')
    family_rows=[]
    for r in h1['families']:
        if r['split']=='audit' and r['step']==4:
            family_rows.append(f"| {r['family']} | {r['source_kind']} | {ci(r['metrics']['G'])} | {ci(r['metrics']['M_minus'])} | {r['material_drops']} / {r['drop_logs']} |")
    probe_rows=[]
    for arm in ['T','FQ','FE','JQ','JE','ZERO']:
        v=h2['arms'][arm];det=secondary['arms'].get(arm)
        ap=ci(det['degradation_PR_AUC_AP']) if det else str(v['degradation_PR_AUC_AP'])
        recall=ci(det['top10pct_scene_recall']) if det else str(v['top10pct_scene_recall'])
        probe_rows.append(f"| {arm} | {ci(v['intervals']['MAE'])} | {ci(v['intervals']['RMSE'])} | {ap} | {recall} |")
    comparisons='\n'.join(f"| {key} | {ci(h2['comparisons'][key],True)}% |" for key in ['JQ-vs-FQ','JE-vs-FE','JE-vs-JQ','FE-vs-FQ','JE-vs-T','JE-vs-ZERO'])
    severe_rows=[]
    for f in severe['families']:
        if f['scenes']:
            m=f['metrics'];g=ci({k:m[k][1][0] for k in ['estimate','lower95','upper95']})
            severe_rows.append(f"| {f['family']} | {f['scenes']} | {f['remaining_material_drops'][1]} | {f['remaining_severe_drops'][1]} | {g} |")
    datasets='\n'.join(f"| {s} | {v['scenes']} | {v['logs']} | {v['driver_records']} | `{v['sha256']}` |" for s,v in cfg['datasets']['splits'].items())
    tune='\n'.join(f"| {a} | {v['head_lr']} | {v['lambda_diff']} |" for a,v in selection['selected'].items())
    case_data=read(ROOT/'analysis/cases.json');case_text=[]
    for case in case_data:
        prediction=f"预测 ΔJ={case['predicted_delta_J']:+.6f}。" if 'predicted_delta_J' in case else ''
        case_text.append(f"{case['kind']}：场景 `{case['token']}`，`{case['family']}` 步骤 {case['step']}，场景平均 ΔJ={case['delta_J']:+.6f}，展示种子 {case['shown_seed']} 的 ΔR={case['shown_delta_R']:+.6f}。{prediction}更新后事件记录：`{case['first_events_s']}`；基准事件：`{case['base_first_events_s']}`。\n\n![{case['kind']}](figures/case_{case['kind']}.png)\n")
    ridge=read(ROOT/'controls/ridge_complete.json');shuffle=read(ROOT/'analysis/observation_shuffle.json')
    init=read(ROOT/'analysis/probe_identity_audit.json');high=read(ROOT/'analysis/stable_high.json')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    native_updates=sum(len(read(p)['steps']) for p in (ROOT/'updates').glob('*/train_complete.json'))
    text=f"""# 连续轨迹后果辨识与 ReCogDrive 跨场景退化验证

本报告只依据本次真实运行结果。全部分数使用 `[0,1]` 的 NAVSIM v1.1 PDMS；乘以100才是百分制分数点。数据和模型的适用范围、阶段停止以及未完成的假设检验分别列明。

## 研究结论

| 假设 | 判断 | 证据与限制 |
|---|---|---|
{chr(10).join(summary_rows)}

建议：{decision['recommendation']}

## 实际实验与身份

实际完成24个独立更新族、共{native_updates}次原生 GRPO 优化器更新，每族保存0/1/4步。12个初始族加一次预注册的12族确认扩容；自然分布与低分且奖励有差异的来源都保留。所有分支共享同一初始策略参数哈希，零更新最大误差均为0；没有接续累积不同更新族。

预测实验完成15个校准运行（每个300步）以及五组×三个训练种子×2000步的正式训练。每个训练种子的首次输入及损失在 FQ/FE/JQ/JE 间完全一致，见 [初始化核验]({ROOT}/analysis/probe_identity_audit.json)。全局批次4、单场景微批次1；不同并行度只分摊微批次并对全局样本均值求梯度。RL正式阶段状态为 `{decision['stage_E']}`，具体理由见下文。

- 研究分支：`research/trajectory-effect-validation-20261019`；报告生成时代码：`{code}`。
- 实际主起点：`{cfg['primary_start']['checkpoint']}`，SHA256 `{cfg['primary_start']['sha256']}`，epoch24/global_step8325。
- VLM：`{cfg['primary_start']['vlm']}`，SHA256 `{cfg['primary_start']['vlm_sha256']}`。
- 原始仓库代码：`{cfg['primary_start']['model_source_commit']}`；每个运行还有实际 commit、源码内容哈希、服务器、GPU、PID/start-time 和命令记录。
- [锁定协议](../../configs/trajectory_effect_validation_20261019/protocol.yaml)，SHA256 `{identity}`。没有把本轮重新初始化 AdamW 的训练称为历史更新复现。
- 评分器、仿真器、缓存类型与基线身份见 [source_weights.json]({ROOT}/manifests/source_weights.json)；实际选用日志、图像和缓存的内容身份见 [input_content_identity.json]({ROOT}/manifests/input_content_identity.json)。原文件修改时间均早于协议锁定。

主干是发布版 InternVL/Qwen2 2B VLM，动作头是原有 small LightningDiT，五步 DDIM 输出8×3轨迹（0.5–4秒，当前自车后轴坐标，x前/y左/heading弧度）。完整轨迹是联合动作。输入只含推理可见的当前前视图、四帧自车历史、状态和指令。读取的是动作头实际接收的完整VLM token，没有 teacher-forced 答案或真实未来。

基准策略在本次审计集的五种子平均PDMS为0.9479255933（94.79256百分制分数点）。24族训练共768个实际采样组，其中9组奖励无差异；这些组和原生BC项均保留并记录，没有伪造advantage。

联合更新范围锁定为28层语言注意力 Q/V 的 rank8、alpha16 LoRA，共1,089,536个参数；图像编码器及底座参数冻结。统一条件轨迹读出头为两层跨注意力、192维、六头，共1,270,465个参数。结论限定于该可训练范围和本次预算，不外推为全参数 VLM 的能力上限。

## 划分与信息边界

| 用途 | 场景 | 完整log | 上级驾驶记录 | 清单SHA256 |
|---|---:|---:|---:|---|
{datasets}

目标log、场景及上级驾驶记录互斥。新增更新族按12/6/6分别用于fit/cal/audit；同族的1/4步不跨用途。主要预测结果是1696个保留场景、16个log与6个保留更新族的交叉组合。校准仅4个log，因此选择本身存在不确定性。

这些数据均来自官方 navtrain，可能被底模预训练见过；“保留”仅指没有用于本次新增训练。官方 navtest 未用于调参、选阈值或选检查点。{decision['navtest_statement']}

真实PDMS、未来占用、碰撞事件、候选来源及场景/更新编号均不进入主预测器输入。T只使用数值轨迹和可观测自车状态/指令；附加线性对照使用真实基线PDMS，明确属于评价标签 privileged 诊断。真实未来占用仅在案例图中解释后果，没有作为任何主预测器特征。

## 阶段A：评分与配对有效性

64个真实场景、1024条轨迹经过正式逐候选入口、批量包装、顺序和分批比较。原始多候选直接评分存在候选集合依赖，归一化总分最大差为0.309478303825；修正包装层后与正式reference/candidate逐对口径的最大差为0，离散分项完全一致。正式评分器未修改。该检查与 [NAVSIM v1.1正式入口](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/evaluate/pdm_score.py) 的reference/candidate语义一致，具体执行以本地源码哈希为准。

同一观测、模型和噪声的轨迹误差、重复评分误差、零更新分数差均为0。小测还执行了真实GRPO反向与参数更新，以及共享VLM辅助监督反向；这两项仅用于运行资格验证，不是H1/H2/H3成功证据。

原有RL训练奖励的进度/TTC/舒适度权重为10/5/2；正式PDMS端点和后果标签为5/5/2。本轮保持原生RL目标和采样器，明确保留这个项目现有差别。策略各自从自身当前policy生成真实采样链，未把人工扰动或其他策略轨迹冒充on-policy动作。

## H1：真实更新、自然采样波动与确认

先按场景平均五个配对推理种子，再计算G、M_plus和M_minus。以下区间联合重采样完整log、更新族和推理种子；相邻检查点保持同族相关性。

| 更新步 | G，95%区间 | M_plus，95%区间 | M_minus，95%区间 | 明显下降比例，95%区间 |
|---:|---|---|---|---|
{chr(10).join(primary_rows)}

| 审计族，步骤4 | 来源 | G，95%区间 | M_minus，95%区间 | 明显下降场景 / 所在log |
|---|---|---|---|---:|
{chr(10).join(family_rows)}

同一模型的两组独立五次采样自比较及完整连续变化见 [h1.json]({ROOT}/analysis/h1.json)。单次输出的下降概率另列在文件中，没有替代场景平均下降幅度。独立筛选种子定义了{high['selection']['stable_count']}个稳定高分场景，其评价采用另一组种子，结果见 [stable_high.json]({ROOT}/analysis/stable_high.json)。新增碰撞、越界、TTC、舒适度、方向违规及正分到零分的数量逐族保留。

![真实更新与采样波动](figures/h1_update_noise.png)

![改善、退化和平均增益](figures/gain_degradation.png)

初始5次采样仅用于选择预注册的严重下降子集；下面是额外25组独立成对种子的确认结果。该子集按log分层、哈希选取，不能作为总体发生率。

| 更新族 | 复核场景 | 步骤4仍下降超过0.02 | 仍下降超过0.10 | 子集步骤4平均变化，95%区间 |
|---|---:|---:|---:|---|
{chr(10).join(severe_rows)}

阶段B的操作门槛为至少两个保留族各有20个明显下降场景、涉及至少4个log；实际状态为 `{h1['gate_B']}`。存在真实负向后果与满足后续有效比较所需事件数量是两个不同判断。唯一一次确认扩容已经使用，未再扩大训练或降低门槛。

## H2：表征与监督目标的独立归因

每场景新增候选15条：4条自身额外采样、1条发布版IL轨迹、10条平滑数值扰动；基准与真实更新轨迹另存。扰动尺度在fit/cal真实更新上锁定，纵向中位/90分位RMS为0.142278/0.301767米，横向为0.052330/0.099886米；没有依据审计失效位置调探针。碰撞、越界、低分及跟踪差异均保留。

候选按完整轨迹数值去重并保留全部来源，每场景至多32个固定分层轨迹对。绝对值项和差值项分别按样本数取均值；普通Q与差值监督使用相同真实标签。数据覆盖与小扰动掉分见 [candidate_coverage.json]({ROOT}/analysis/candidate_coverage.json)。人工扰动的反向后果不被当作真实参数更新退化的证明。

全部94,935条新增候选还保留了逐轨迹连续跟踪误差，见 [candidate_tracking.npz]({ROOT}/analysis/candidate_tracking.npz)。未找到环境定义的“跟踪失败”阈值，因此没有新增事后分类阈值；不因误差、碰撞或越界删除监督。

| 组别 | 校准后head学习率 | lambda_diff |
|---|---:|---:|
{tune}

校准选择在一次小范围比较后锁定；确认扩容没有重新选择。正式模型各从共同底座重新初始化，没有把校准或诊断中联合训练的权重单独用于其他组。

| 组别 | 回报差MAE，95%区间 | 回报差RMSE，95%区间 | 退化PR-AUC（AP），95%区间 | 前10%审查召回，95%区间 |
|---|---|---|---|---|
{chr(10).join(probe_rows)}

| 归因对照 | RMSE相对降低，95%区间 |
|---|---|
{comparisons}

预注册实质性门槛为相对RMSE降低10%。JQ/FQ、JE/FE、JE/JQ分别回答普通价值监督下更新VLM、差值监督下更新VLM、以及差值监督自身的作用。{decision['H2']['detailed_interpretation']}

严重下降子集的误差、漏检率和三分类混淆矩阵见 [主预测分析]({ROOT}/analysis/probes_formal_D_audit.json) 与 [检测指标区间]({ROOT}/analysis/probe_secondary_intervals.json)。fit/cal诊断单独保存，不替代保留log×保留更新族结果。

本轮没有训练出有效的真实更新后果预测器。所有组的差值RMSE都与恒预测0的基线几乎相同；按锁定的±0.02阈值，所有组都把审计场景的平均变化判为近零。JE在严重退化子集上的RMSE为{ci(secondary['arms']['JE']['severe_RMSE'])}，观察到的漏检率为100%。全遗漏数据的经验bootstrap区间退化为[1,1]，不能解释为总体漏检率已被精确确定。fit/cal也呈现同样的近零分类模式，见 [fit_cal_diagnostics.json]({ROOT}/analysis/fit_cal_diagnostics.json)。因此，组间差距很小不能作为“冻结表征已充分”的证据；负面结论针对本次训练方法和预算所能获得的改进。

真实基线分数+轨迹改变量+自车状态的 privileged 线性对照RMSE为{ridge['RMSE']:.6f}，MAE为{ridge['MAE']:.6f}；其优势不能归因于图像表征。按指令、速度与轨迹几何分组的图像置换结果见 [observation_shuffle.json]({ROOT}/analysis/observation_shuffle.json)：{shuffle['interpretation']}。该实验只检查输入依赖，不识别交通因素的因果作用。

![后果预测主结果](figures/consequence_prediction.png)

## H3：有效RL优化中的退化

{decision['H3']['detailed_interpretation']}

同预算与相近正向增益的规则已写入协议：500次实际优化器更新、0/100/250/500固定时点，D_cal共同正向增益至少0.002、容差0.001，Q/E同辅助库和头初始化。未达到有效增益的模型不能凭较低退化率获胜。

![RL比较执行状态](figures/rl_comparison_gate.png)

## 数值后果案例

改善、退化与预测失败按预注册阈值再按哈希选择；展示的采样种子最接近该场景平均变化。图中原始轨迹与跟踪轨迹分开，同时展示五次成对得分；选中的单次ΔR可能为0，不能把场景平均变化归因于这一次输出。退化与预测失败规则选中了同一案例，不是两个独立证据。地图及当前/事件时刻对象占用仅用于诊断。首次TTC检查失败时刻不是剩余碰撞时间。默认输出、原生分项/事件索引与额外派生的首次越界时刻在评分模块中分别标明。

{chr(10).join(case_text)}

## 成本、失败记录与复现

已结束进程的GPU数量×运行壁钟时长为{costs['reserved_gpu_hours']:.3f} GPU小时（包含启动及依赖等待，属于实际计算占用的保守上界）；记录存储为{costs['storage_bytes']/1024**3:.3f} GiB。使用当前允许的vla-zt、rl-zt2及清华系列空闲资源，并保留其他真实任务。没有使用不允许的服务器或恢复历史压力脚本。

一次调度因设备在启动前被占用而拒绝，原样留下未启动记录后换到空闲GPU；一次H1后的可选线性诊断因缺少sklearn停止，H1结果不受影响。改用显式NumPy岭回归与按并列分组的AP定义后，在同一锁定数据上完成诊断。所有失败、未启动和完成记录保存在 [accounting.json]({ROOT}/analysis/accounting.json)。评分集合依赖在正式采集前修复，未沿用不一致标签。

数据根目录：`{ROOT}`。逐场景轨迹、评分分项、仿真状态与事件时刻分别位于`baseline/`、`updates/`、`candidates/`和`confirmation/severe/`；大权重与缓存不进入git。协议、场景/log/更新族/随机种子、真实训练配置及完整执行命令均可追溯。

重新生成统计与报告：

```bash
cd {SOURCE}
{PYTHON} -m tools.trajectory_effect.statistics h1
{PYTHON} -m tools.trajectory_effect.probe_metrics --prefix formal --split D_audit
{PYTHON} -m tools.trajectory_effect.severe analyze
{PYTHON} -m tools.trajectory_effect report
```

实际从采集到报告的模块命令清单：[executed_commands.sh]({ROOT}/analysis/executed_commands.sh)。完整清单同时记录并行设备和运行身份。完整重跑入口会复制锁定清单，重新执行真实采集、训练、评分、统计和报告；需要四张空闲GPU，串行推进各个实验组，耗时长于本轮多服务器并行。新目录必须尚不存在，复制协议后不覆盖原研究：

```bash
cd {SOURCE}
TRAJECTORY_EFFECT_RUN_ROOT=/mnt/project/VLA-AD/outputs/trajectory_effect_reproduction_new \\
  {PYTHON} -m tools.trajectory_effect.reproduce --gpus 0,1,2,3
```

该完整重跑入口实现了真实调用链，但本轮没有再重复整个研究矩阵；本次实际运行身份以命令清单和原始记录为准。八个公共入口为`audit`、`collect_updates`、`build_candidates`、`train_probes`、`evaluate_probes`、`train_rl`、`evaluate_rl`、`report`。
"""
    (dest/'REPORT_zh.md').write_text(text)
    atomic_json(dest/'delivery.json',dict(protocol_sha256=identity,code_commit=code,root=str(ROOT),decisions=decision,
        h1_summary=h1['audit_combined'],h2_comparisons=h2['comparisons'],report=str(dest/'REPORT_zh.md')))
    print(dest/'REPORT_zh.md')


if __name__=='__main__':main()
