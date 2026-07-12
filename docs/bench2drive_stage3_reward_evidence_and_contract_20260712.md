# Bench2Drive Stage3 reward：公开证据、口径边界与候选合同

日期：2026-07-12

分支：`feature/bench2drive-recogdrive`

状态：v2 report-aligned reward state、聚合器与 CPU 组合器已固定；尚未接入正式 Stage3 cache/GRPO，且不影响正在运行的 Stage1 → Stage2 基线流水线。

## 1. 先固定口径

用户的判断正确：Bench2Drive 不能把 NAVSIM PDMS 当作官方评价指标。

Bench2Drive V0.0.3 的正式闭环汇报包括：

1. Driving Score（DS）；
2. Success Rate（SR）；
3. Driving Efficiency；
4. Driving Smoothness/Comfortness；
5. Merging、Overtaking、Emergency Brake、Give Way、Traffic Signs 五类 Multi-Ability。

[Bench2Drive 论文](https://arxiv.org/html/2406.03877)明确给出 220 条短路线、DS/SR、Efficiency 和 Smoothness 的定义；[官方仓库](https://github.com/Thinklab-SJTU/Bench2Drive)也要求分别运行 merge、ability、efficiency/smoothness 工具。官方仓库在 2024-08-19 后还将 minimum-speed penalty 从 DS 中移除，改为独立汇报 Efficiency。

本地核查固定到 Bench2Drive commit
`2645714eb1f3a100217928dd113093cae0779f36`：

- [DS 统计代码](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/leaderboard/leaderboard/utils/statistics_manager.py)：`DS = route_completion × multiplicative_infraction_penalty`；
- [Efficiency/Smoothness 代码](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/tools/efficiency_smoothness_benchmark.py)：从闭环日志独立计算；
- [Multi-Ability 代码](https://github.com/Thinklab-SJTU/Bench2Drive/blob/2645714eb1f3a100217928dd113093cae0779f36/tools/ability_benchmark.py)：按成功路线聚合五种能力。

因此本文把后续自建训练信号称为 **B2D candidate reward vector**，绝不称为
“Bench2Drive PDMS”或“官方 ReCogDrive B2D reward”。PDMS 只可作为分项建模和安全门控思想的来源。

## 2. ReCogDrive 公开资料仍留下什么缺口

[ReCogDrive ICLR 2026 论文](https://openreview.net/pdf?id=JoXwhGbuMi)写明第三阶段使用完整 NAVSIM 与 B2D 数据，并以 EP、TTC、Comfort 权重描述 PDMS reward；但论文没有公开以下 B2D 合同：

- 如何在 B2D 六点、3 秒轨迹上构造 NAVSIM metric cache；
- 是否把 B2D 轨迹转换成 NAVSIM 8 点、4 秒轨迹；
- B2D 的 actor rollout、地图查询、交通规则和 controller；
- Table 2 的 B2D 模型究竟是否经过该 Stage3；
- B2D reward 的精确聚合式。

[ReCogDrive 公开仓库](https://github.com/xiaomi-research/recogdrive)的 DiffGRPO 路径只提供 NAVSIM PDM simulator/metric-cache 实现。因此它不能证明 B2D 使用了官方不存在的 PDMS，也不能据此把 NAVSIM scorer 直接接到当前 B2D 基线上。

结论：paper-exact B2D Stage3 仍不可复原；只能建立单独标记的
`custom-aligned research extension`。

## 3. 公开工作实际怎样定义 B2D/RL reward

### 3.1 直接在 B2D 数据或环境上使用的定义

| 工作 | 场景与优化 | 公开 reward | 可采用部分 | 不足 |
|---|---|---|---|---|
| [Think2Drive](https://arxiv.org/html/2402.16720) | 生成 B2D 专家数据的 CARLA-v2 世界模型 RL | safe-speed reward、沿目标路线的 travel reward、lane deviation penalty、steering-change cost | 安全、进度、平顺四类分项；密集 reward | 控制级 reward，不是 ReCogDrive 候选轨迹 reward；固定 lane-center penalty 会误伤合法绕行/超车 |
| [MindDrive](https://github.com/xiaomi-mlab/MindDrive) | B2D 在线 RL | 到达终点 `+1`；碰撞、红灯、停车牌、偏航/驶出路线等 `-1`；其余 `0` | 违规集合与在线终止语义 | 极度稀疏，不能表达安全—效率—舒适冲突 |
| [WoTE](https://arxiv.org/html/2504.01941) | B2D 轨迹评价/选择 | imitation reward + simulation NC + DAC；B2D 使用 256 anchors | 模仿与模拟信号互补、NC/DAC cache | 未包含 B2D Efficiency、Smoothness 和交通规则；不是 GRPO |
| [iPad](https://arxiv.org/html/2505.15111) | B2D 3 秒 log-replay scorer | 2 Hz perfect-controller rollout；NC、DAC、TTC、Comfort、Ego Progress | 与当前六点/3 秒合同最接近；明确给出 B2D 子项 | 作者发现 NC/TTC/Comfort 与真实闭环成绩可能负相关，说明非反应式 scalar scorer 不能直接充当闭环真值 |
| [TakeVLA](https://arxiv.org/html/2603.14972) | B2D takeover 场景 GRPO | 2 秒、5 Hz、自行车模型、部署同款 PID；discounted expert-trajectory L2 + collision indicator | 同 controller pseudo-sim、连续模仿项和碰撞项 | 只覆盖模仿与碰撞，缺少效率、舒适、路线/交通规则 |
| [MindDriver](https://arxiv.org/html/2602.21952) | B2D progressive GRPO | trajectory ADE/L2 reward + 六点输出 format reward | 六点格式与连续几何 reward | 不评估碰撞、规则、效率与舒适，低 L2 也不保证闭环正确 |

MindDrive 的精确在线实现已核查到 commit
`1a4085dab1c20895a0c8d2b67b4f8e65712fa8de`：
[环境 reward](https://github.com/xiaomi-mlab/MindDrive/blob/1a4085dab1c20895a0c8d2b67b4f8e65712fa8de/team_code/carla_env/carla_env_scenario.py#L365-L473)与
[违规配置](https://github.com/xiaomi-mlab/MindDrive/blob/1a4085dab1c20895a0c8d2b67b4f8e65712fa8de/adzoo/minddrive/configs/minddrive_qwen25_3B_lora_rollout.py#L38-L49)相互一致。

### 3.2 只能作为设计参考，不能冒充 B2D 官方 reward

| 工作 | reward | 本项目如何使用 |
|---|---|---|
| [DriveReward](https://arxiv.org/html/2606.08525) | NC、DAC、TTC、EP、Comfort、Command Following、Legality 七维 reward model；在 B2D 上又比较其自称的 rule-based/model-predicted “PDMS” | 七维输出证明 reward 应保留分项；其 PDMS 是跨基准代理，不是 B2D 官方指标。其 B2D 表中 DS 上升时 Comfort 从 42.9 降至 11.4，是安全/任务收益与舒适冲突的直接证据 |
| [DriveStack-VLA](https://arxiv.org/html/2606.24051) | NAVSIM 上用 NC、TTC、DAC 均值与格式 reward 做 GRPO，B2D 仅闭环测试 | 支持安全分项与格式项；不能作为 B2D reward 证据 |
| [Generative Scenario Rollouts](https://arxiv.org/html/2601.11475) | collision rate、inverse TTC、VLA semantic alignment 的 surrogate reward | 支持连续安全余量和语义/任务一致性，但仍是 learned surrogate |
| [VDRive](https://arxiv.org/html/2510.15446) | off-road、lane centering 与大 VLM 风险评分的混合 reward | off-road 可借鉴；无条件 lane centering 不适合 B2D 的 overtaking、detour、merge |

截至 2026-07-12，没有找到同时满足以下条件的公开 reward：

1. 明确针对 B2D 六点、3 秒轨迹；
2. 完整覆盖 B2D 官方 DS/SR、Efficiency、Smoothness、Multi-Ability 语义；
3. 有可执行公开代码；
4. 已证明离线 reward 与 220-route 闭环结果同向；
5. 能保留多目标冲突而非提前压成单一分数。

因此必须自行构造，但每个分项都要注明“官方对齐”还是“训练代理”。

## 4. 为什么 v1 需要被替换

v1 的 `[Safety, Compliance, Efficiency, Comfort]` 能表达冲突，但仍有三处不够严格：

1. continuous TTC、command following 和 comfort margin 不是 B2D 最终汇报列；
2. capped local speed ratio 不能还原官方可超过 100% 的 Efficiency；
3. 从四个 v1 分量不能精确重建 DS、SR 和 Multi-Ability。

因此 v1 配置只保留作审计历史，正式候选合同升级为：

- 配置：`configs/bench2drive_recogdrive_stage3_report_reward_v2.json`；
- 实现：`navsim/agents/recogdrive/bench2drive_report_reward.py`；
- 单测：`tests/test_bench2drive_report_reward.py`。

v2 不再先“猜”局部 reward，而是先保存能还原官方报告的 route-level sufficient statistics，再让每条 3 秒候选轨迹预测一次 after-state。

## 5. v2 route state 与官方结果的精确关系

每条 route 保存：

- `RC`：官方 Route Completion，范围 `[0,1]`；
- `P`：collision、red light、stop、yield、outside-route 等官方 penalty 的累乘；
- `infraction_free`：除 minimum-speed 外没有任何正式 infraction，且没有 deviation、blocked、route timeout；
- Efficiency 的有效 checkpoint percentage `sum/count`；
- Smoothness 的有效 segment `pass_count/segment_count`；
- `route_finished`、`target_reached`；
- ability labels 与 Traffic-Signs 的特殊 junction outcome。

这些量可以直接重建：

\[
DS_i=100\,RC_iP_i,
\qquad
DS=\frac{1}{220}\sum_i DS_i.
\]

\[
Success_i=I(route\ finished\land target\ reached\land RC_i=1
\land infraction\ free),
\]

\[
SR=\frac{100}{220}\sum_i Success_i.
\]

官方 Efficiency 仍是每条有效 route 的 checkpoint speed-percentage 均值，再在有效 routes 间取均值；官方 Smoothness 是有效 route 的 smooth-segment pass ratio 均值。五类 Multi-Ability 是同一个 `Success_i` 按场景组聚合，Traffic Signs 保留官方脚本额外计算 junction outcome 的行为。

这意味着 v2 route aggregator 可以和官方 220-route JSON/metric logs 逐项对账，不再只有“同方向”关系。

## 6. Pareto 目标：官方量的非冗余分解

对每个 candidate after-state，定义：

\[
\mathbf f=[P,\ RC,\ E_{linear},\ Smoothness].
\]

四个分量分别对应：

1. `Infraction Compliance = P`：DS 的官方安全/守法乘法项；
2. `Route Completion = RC`：DS 的官方任务进度项；
3. `Driving Efficiency = E_linear`：官方 Efficiency 的正线性归一化；
4. `Driving Smoothness`：官方 segment pass ratio，不再混入 dense margin。

### 6.1 Efficiency 必须线性归一化

B2D 的 Efficiency 可以超过 100%，官方只过滤单个大于 1000% 的异常值。直接 cap 到 100% 会改变最终排名；对数变换虽然保留单条 route 的大小关系，却会改变跨 routes 取算术平均后的模型排名。因此优化量只做正线性归一化：

\[
E_{linear}=\frac{E_{official}}{1000}.
\]

它把官方有效范围映射到 `[0,1]`，并且在同一组有效 routes 上满足 `mean(E_linear)=mean(E_official)/1000`，所以算术聚合和模型排序都不变。若模型的有效 route coverage 不同，则由下一节的显式缺失值规则处理，不能宣称官方省略口径也天然不变。最终报告仍输出 raw percentage。Pareto 支配对正线性缩放不敏感，训练时再做组内按维标准化，不能修改原始目标定义。

### 6.2 缺失值不能成为刷 reward 的路径

- 未到第一个 5% checkpoint 的 route 在官方 Efficiency 中被省略；训练目标将其记为 `E_linear=0`，同时记录 Efficiency route coverage；
- 没有完整 smooth segment 时优化值为 0，同时记录 Smoothness coverage；
- 正式报告仍照官方省略规则计算，但必须同时给出 coverage，防止“不开动所以没有低 Efficiency”“过早失败所以没有不舒适段”造成误判。

### 6.3 SR 作为约束支配，而不是稀疏第五维

- SR 是 `RC=1 + infraction_free + terminal` 的二元派生量，直接加入 vector 会形成稀疏且冗余的第五维，但完全忽略它又会偏离最终汇报；
- 因此定义 `sr_feasible = infraction_free AND (尚未终止 OR terminal_success)`，采用 constrained dominance：仍有 SR 成功可能的候选无条件优先于已经不可成功的候选；两者可行性相同时，再比较四维 Pareto vector；
- blocked、route deviation 和 route timeout 虽然没有单独的 DS 乘法因子，也会使 `infraction_free=false`；minimum-speed 按官方规则不触发该约束；
- Multi-Ability 是同一个 route success 按 scenario group 聚合，不是新的 per-candidate 物理量；
- 训练时采用五种 ability/44 scenarios 平衡采样，验证时直接报告五维 ability success vector。

TTC、clearance、command following 和 dense comfort margin 仍可作为 diagnostics/tie-breaker，但不进入 report-aligned 主 Pareto vector。

## 7. B2D-RAS：普通 GRPO 的标量链路基线

类比 NAVSIM 的“penalty × weighted quality”，先定义同类可行状态内的基础效用：

\[
U=P\frac{2RC+E_{linear}+Smoothness}{4}.
\]

再把 SR 的零违章条件编码为不可被效率补偿的 barrier：

\[
B2D\text{-}RAS=
\begin{cases}
U,& sr\_feasible,\\
-1+0.25U,& otherwise.
\end{cases}
\]

- `P` 使用 B2D 官方 penalty，不使用 NAVSIM NC/DAC；
- `RC` 权重 2，因为 DS/SR 是 B2D 主结果；
- Efficiency 和 Smoothness 各权重 1，因为二者独立汇报；
- 任意 SR-feasible 状态位于 `[0,1]`，任意 infeasible 状态位于 `[-1,-0.75]`，因此速度或平顺性不能补偿导致 SR 归零的违章；
- `0.25U` 只在一组候选已经全部不可行时保留相对学习信号；
- 该分数可验证普通 scalar GRPO 数据链路，但不是官方指标，也不是 Pareto 方法；
- 真正 Pareto 训练直接消费上一节的四维向量，避免权重掩盖冲突。

ADE 不进入主 reward。Stage2 reference policy 继续通过 KL/BC 保持，避免把专家单一路径变成与官方报告无关的第五目标。

## 8. 3 秒 candidate 如何与完整 route 对齐

每个 GRPO group 共享同一个 before-state。候选轨迹经同 controller rollout 后得到各自 after-state：

\[
s_t\xrightarrow{\tau_k}s_{t+3s}^{(k)}.
\]

组内先应用 SR constraint dominance，再比较 after-state potential：

\[
[P_{t+3s}^{(k)},RC_{t+3s}^{(k)},E_{t+3s}^{(k)},S_{t+3s}^{(k)}].
\]

因为 before-state 对组内全部候选相同，相对优势不会受到共同历史项影响。若以后采用连续 route rollout，则使用 potential difference：

\[
r_t=\Phi(s_{t+3s})-\Phi(s_t),
\]

DS 与 B2D-RAS 的差分在整条 route 上 telescoping，累计值等于终局 potential 减去初值。

候选分支必须继承：

- 当前 RC 与 penalty；
- 已有 Efficiency checkpoint `sum/count` 以及下一个 5% checkpoint；
- Smoothness 的 segment boundary 和运动学历史；
- 已经发生的 infraction 状态；
- route/scenario/ability metadata。

如果独立 frame cache 没有这些 route context，就不能声称 report-aligned，应先补全 route-state cache。

## 9. 本定义保留的真实冲突

1. **Infraction Compliance vs Route Completion**：merge/overtake/detour 中过度保守可以保持 `P=1`，但降低 RC，甚至 blocked/timeout。
2. **Infraction Compliance vs Efficiency**：更大安全余量通常需要更早制动，而 B2D Efficiency 奖励更高的周车速度比。
3. **Efficiency vs Smoothness**：快速起步、抢 gap、急减速提高进度/速度，却增加 acceleration 与 jerk。
4. **Infraction Compliance vs Smoothness**：Emergency Brake 中必要急刹可能保住无碰撞，却损失 Smoothness。
5. **Route Completion vs Smoothness**：静止或 blocked route 可能得到完美 Smoothness，但 RC/DS 很低。
6. **非反应式 proxy vs reactive closed-loop**：回放 actor 不会响应 ego，因此 candidate after-state 必须和 held-out CARLA reactive rollout 校准。

[NAVSIM 原论文](https://proceedings.neurips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Paper-Datasets_and_Benchmarks_Track.pdf)本身就采用 penalty 与 progress/comfort 的分项聚合；[近期跨基准研究](https://arxiv.org/html/2605.00066)进一步观察到安全过度优化会因低进度和 timeout 导致 B2D 排名反转。这正是保留 Pareto 分项而不是只优化 scalar 的研究依据。

## 10. Cache、训练与验证门

### 10.1 两档 rollout

1. **参考真值**：CARLA reactive rollout，复用正式 agent 的 PID、20 Hz control 和官方 event logic；
2. **低成本 proxy**：B2D future actor box log replay；必须明确标注 non-reactive，并在 held-out routes 上对照 reactive rollout。

无论哪一档，都必须保存原始 route state、四维 vector 和 B2D-RAS，不能只保存 scalar。

### 10.2 正式 Stage3 前的门

- v2 aggregator 在现有官方 evaluation JSON/metric logs 上逐项复现 DS、SR、Efficiency、Smoothness；
- 在 route XML 与官方 CARLA map 上单独复现五类 ability，包含 Traffic-Signs 的额外 junction contribution；
- 注入 collision、red-light、stop、outside-route、blocked 等反例，确保它们失去 SR feasibility，且 B2D-RAS 跨越负 barrier；
- 检查候选转移的 RC 单调、penalty 不可恢复、accumulator count 不可减少；
- 检查四个目标在 held-out training routes 的组内方差；
- 报告 proxy 与 reactive rollout 的 component-wise Spearman/Kendall 相关性和 ranking inversion；
- 在 reward/cache/controller 固定前，不用完整 220-route 结果调权重。

已用现有 Stage2 完整 220-route 工件执行第一项数值门：

| 指标 | v2 聚合器 | 官方后处理 | 差值 |
|---|---:|---:|---:|
| DS | 45.10557816 | 45.10557675 | `+1.41e-6` |
| SR | 21.36363636 | 21.36363636 | `0` |
| Efficiency | 137.51002511 | 137.51002511 | `+8.53e-14` |
| Smoothness | 37.71364351 | 37.71364351 | `-7.11e-15` |

DS 的微小差值来自已保存 route score 的六位小数累加；`1e-5` 精度门通过。该次评估的 Efficiency route coverage 为 `97.27%`，Smoothness coverage 为 `100%`，进一步证明 coverage 不能省略。真实五类 ability 的 per-route 特殊处理仍待 route XML/CARLA 对账。

### 10.3 Stage3 完成后的最终汇报

- 官方 DS、SR、raw Efficiency、Smoothness；
- 五类 Multi-Ability 与 mean；
- per-infraction counts、RC、Efficiency/Smoothness coverage；
- 四个 Pareto objectives 的分布、相关矩阵和 Pareto front；
- B2D-RAS 只作为 scalar baseline ablation；
- offline proxy 到 reactive closed-loop 的相关性审计。

## 11. 当前可执行边界

已经完成：

- v1 的局限审计并标记 superseded；
- v2 route sufficient-state 数据结构；
- 官方 DS/SR/Efficiency/Smoothness/Multi-Ability 聚合器；
- report-aligned 四维 Pareto vector；
- SR constraint-dominance 与不可补偿的 scalar barrier；
- B2D-RAS scalar baseline；
- candidate potential/delta 及状态不变量；
- machine-readable 配置与 CPU 单元测试。
- 在既有 220-route artifacts 上复现官方 DS、SR、Efficiency、Smoothness，`1e-5` 数值门通过；
- 独立验证脚本 `scripts/bench2drive/validate_b2d_report_reward.py`。

尚未完成、不能假装完成：

- 用 route XML/CARLA 复现真实五类 Multi-Ability 的 per-route 特殊处理；
- 3 秒 actor/map/controller candidate rollout cache builder；
- non-reactive proxy 与 reactive CARLA 的相关性门；
- 将 vector/scalar adapter 接入 ReCogDrive DiffGRPO trainer；
- Pareto-GRPO 算法迁移；
- 正式 Stage3 训练。

以上设计仍遵守既定顺序：先完成并验收纠正后的 Stage2；Stage3 baseline 与后续 Pareto 方法作为明确标注、可相互比较的独立阶段。
