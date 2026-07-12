# Bench2Drive Stage3 reward：公开证据、口径边界与候选合同

日期：2026-07-12

分支：`feature/bench2drive-recogdrive`

状态：奖励语义与 CPU 组合器已固定；尚未接入正式 Stage3 cache/GRPO，且不影响正在运行的 Stage1 → Stage2 基线流水线。

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

## 4. B2D reward vector v1

配置：`configs/bench2drive_recogdrive_stage3_reward_contract_v1.json`

CPU 组合器：`navsim/agents/recogdrive/bench2drive_reward_contract.py`

对每个 6 点、3 秒候选轨迹，pseudo-sim 输出四维向量：

\[
\mathbf r(\tau) = [S(\tau), L(\tau), E(\tau), C(\tau)]\in[0,1]^4.
\]

这里只定义 reward vector，不实现 Pareto-GRPO 算法。

### 4.1 Safety：碰撞门控 + 连续 TTC 余量

\[
S = I_{\mathrm{no\ collision}}
\operatorname{clip}\left(\frac{TTC_{min}-1\mathrm{s}}{3\mathrm{s}-1\mathrm{s}},0,1\right).
\]

- 碰撞来自候选 ego swept box 与未来 actor box 的时序相交；
- TTC `<=1s` 是 iPad 的 B2D scorer 阈值；3 秒为当前候选 horizon；
- 没有 closing actor 时 `TTC=+inf`，TTC score 为 1；
- TTC 是训练代理与安全诊断，不得写进 B2D 官方结果列。

二元 collision-free 保证底线，连续 TTC 保留“更早制动更安全，但可能降低效率”的 Pareto 冲突。

### 4.2 Compliance：可行驶区、路线、规则、指令

\[
L=(D_{area}D_{route}D_{rule}D_{cmd}D_{ability})^{1/5}.
\]

- `D_area`：ego footprint 位于可行驶区域的比例；
- `D_route`：ego 位于**场景允许路线走廊**的比例；
- `D_rule`：红灯、停车牌、让行紧急车辆等规则符合度；
- `D_cmd`：到达正确路线分支并满足 high-level command 的连续分数。
- `D_ability`：按场景类型构造的局部能力信号，例如 merge/overtake 是否进入正确空隙、emergency brake 是否避开风险、give-way 是否让行；正式 Multi-Ability 仍只由闭环 route success 计算。

禁止把“到 lane center 的距离”直接当通用惩罚。B2D 的超车、事故绕行、施工绕行、merge、双向道路避障会合法离开当前车道中心；route corridor 必须按场景和导航分支构造。

### 4.3 Efficiency：路线进度 × B2D 周车速度比

\[
P=\operatorname{clip}(\Delta s_{candidate}/\Delta s_{reference},0,1),
\]

\[
V=\operatorname{clip}(\bar v_{ego}/\bar v_{background},0,1),\qquad
E=\sqrt{PV}.
\]

- `V` 直接对应 B2D Efficiency 的局部形式；正式评价仍用 20 个 route checkpoints 的官方实现；
- 对训练 reward 将速度比 cap 到 1，避免通过超速无限刷分；原始百分比仍写入 diagnostics；
- 没有 background vehicles 时按官方逻辑令 `V=1`；
- reference 处于必要停车状态时，候选也在 0.5 m 容差内停车则 `P=1`，避免在红灯/紧急制动时错误奖励前进；
- 几何均值要求“有进度且不比周车过慢”，单纯原地停车不能在整个 route 上获得高效率。

### 4.4 Comfort：官方通过率 + 稠密阈值余量

B2D 官方 2 秒分段阈值：

| 运动量 | 阈值 |
|---|---:|
| longitudinal acceleration | `[-4.05, 2.40] m/s²` |
| absolute lateral acceleration | `< 4.89 m/s²` |
| absolute yaw rate | `< 0.95 rad/s` |
| absolute yaw acceleration | `< 1.93 rad/s²` |
| absolute longitudinal jerk | `< 4.13 m/s³` |
| absolute magnitude jerk | `< 8.37 m/s³` |

训练分量定义为：

\[
C=0.70C_{official-pass}+0.30C_{margin}.
\]

- `C_official-pass` 是所有变量均过阈值的 segment 比例，可与官方 Smoothness 做同方向审计；
- `C_margin` 是到各阈值的归一化余量，只提供组内稠密差异；
- 最终论文汇报只能把 `C_official-pass`/官方闭环脚本结果称为 B2D Smoothness，不能把混合后的 `C` 冒充官方指标。

### 4.5 Hard feasibility gate

候选只有同时满足以下条件才是 feasible：

- 轨迹有限且格式有效；
- 无碰撞；
- 无 route-deviation terminal event；
- 没有会使 B2D route success 失效的正式 infraction；
- drivable-area compliance `>=0.99`；
- traffic-rule compliance `=1`。

TTC 与 command following 不设硬门，保留连续学习信号。后续 Pareto 方法应先处理可行性，再比较四维向量；不要让高速但碰撞/闯灯的候选依靠效率补偿安全失败。

## 5. 普通 GRPO 的兼容标量基线

为了在 Pareto 方法实现前验证整个 Stage3 数据链路，定义一个明确标注的普通 GRPO baseline：

\[
U= P_{B2D-infraction}\frac{S+L+E+C}{4}.
\]

\[
R_{scalar}=\begin{cases}
U,& feasible,\\
-1+0.25U,& infeasible.
\end{cases}
\]

性质：

- 任意 feasible candidate 都优于任意 infeasible candidate；
- 当一组候选全部失败时，残差项仍提供相对排序，避免全组 constant reward；
- `P_B2D-infraction` 使用官方 collision/red-light/stop/yield/outside-route 乘法因子；
- 该标量只是数据链路与普通 GRPO ablation，**不是** Pareto 方法、DS、SR 或 PDMS。

模仿 ADE 不进入四维 reward。Stage2 reference policy 通过既有 KL/BC 项保留，避免把专家单一路径变成目标维度后压制合法的多模态 merge/overtake/detour 解。

## 6. 本定义显式保留的天然冲突

1. **Safety vs Efficiency**：更大 TTC 余量通常需要更早减速；merge/overtake 又需要主动加速进入空隙。
2. **Efficiency vs Comfort**：快速起步/制动提高进度，但增加 acceleration/jerk。DriveReward 的 B2D 实验也出现 DS/SR 上升而 Comfort 大幅下降。
3. **Safety vs Comfort**：Emergency Brake、pedestrian crossing 等场景中，必要急刹可能违反平顺阈值。iPad 观察到 Comfort 与闭环 B2D 得分负相关，给出了同类证据。
4. **Route compliance vs scenario completion**：过强 lane-center 约束会阻止超车、施工绕行或双向路避障。
5. **Imitation vs multimodality**：低 ADE 偏好专家单一路径，但多个安全可行策略可能具有不同横纵向动作。
6. **离线非反应式安全 vs 闭环交互**：回放 actor 不会响应 ego；iPad 已报告其 NC/TTC 与闭环表现可能反向，因此必须审计相关性，不能假定 proxy 就是真值。

这正是后续 Pareto 方法应该发挥作用的位置：保留冲突分项和可行性关系，而不是先用一组未经验证的权重把它们永久压成一个数。

## 7. Cache/pseudo-sim 合同

当前 B2D Base 原始标注足以提供：ego pose/state、未来车辆/行人 oriented boxes、交通灯/标志、road/lane IDs、HD map 和导航命令。v1 cache builder 应固定：

1. 输入只来自 training split，不读取 220-route 结果作为 label；
2. 当前六点轨迹间隔 0.5 秒、horizon 3 秒；
3. 使用闭环部署相同的 PID，并以 20 Hz 自行车模型 rollout ego；
4. 把 10 Hz future actor boxes 插值到 rollout ticks；
5. 保存每个候选的原始事件/运动学量和四维 reward，不能只保存 scalar；
6. 保存 `data commit + cache builder commit + controller config + map commit + reward contract id`；
7. cache manifest 必须标注非反应式 actor 假设。

必须保存的最小字段已列在 JSON 配置的 `required_cache_fields` 中。

## 8. 训练前验证门与最终评价

### 8.1 不使用 220-route 调 reward

- 用 training clips 内部划分 held-out validation；
- 注入碰撞、越界、闯灯、停止、激进加速、原地不动等可控 counterexamples；
- 检查每个 GRPO group 的四维方差、方向性和 reward-hacking；
- 报告 offline component 与 held-out outcome 的 rank correlation；
- 在阈值、controller、cache 和权重冻结前，不用完整 220 routes 做选择。

### 8.2 Stage3 结束后的同口径评价

必须继续运行 B2D 官方 220-route evaluator，汇报：

- DS、SR；
- Efficiency、Smoothness；
- 五类 Multi-Ability 与 mean；
- route completion 与各类 infraction count；
- 四维训练 reward 的分布与 pairwise conflict；
- offline reward 对 closed-loop 结果的相关性审计。

训练 reward 与评价“一致”不等于把二者强行写成同一个标量，而是：同一交通事件、同一 controller、同一运动学阈值、同一路线语义可追溯；同时承认局部非反应式 proxy 与完整闭环 route metric 的层级差异。

## 9. 当前可执行边界

已经完成：

- 公开 reward 证据分类与固定来源；
- B2D 官方 event penalty 组合器；
- B2D Smoothness 阈值通过率及稠密 margin 组合器；
- 四维 reward vector；
- 普通 GRPO scalar baseline；
- 单元测试与 machine-readable JSON 合同。

尚未完成、不能假装完成：

- 3 秒 B2D actor/map pseudo-simulation cache builder；
- reward cache 的 held-out correlation audit；
- 将 vector/scalar adapter 接入 ReCogDrive DiffGRPO trainer；
- Pareto-GRPO 算法迁移；
- 正式 Stage3 训练。

以上缺口不改变当前复现结论：先完成并验收纠正后的 Stage2；Stage3 作为单独的、明确标注的研究扩展启动。
