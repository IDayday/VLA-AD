# LFP-GRPO：面向学习前沿的 Pareto 组相对策略优化

> **文档状态：原始设计基线，非当前实验结论。** 其中“Stage2 已形成宽策略分布”等前提已被
> 后续 SNSAD 与 NAVTEST 实验修正。当前可证实结论、被否定假设和默认配置以
> `docs/LFP_GRPO_STAGE3_RESEARCH_SUMMARY.md` 及
> `reports/recogdrive_stage3/lfp_grpo_r4_evidence_review_20260712.md` 为准。

> **Learning-Frontier Pareto Group Relative Policy Optimization**
> 面向 VLA-AD / ReCogDrive 的简洁、高效 Stage3 策略优化方案

---

## 0. 文档定位

本文给出 VLA-AD 在 Stage2 已完成 Pareto 支持集学习、FS-Norm、Planning Token Adapter 和轨迹辅助约束之后的 Stage3 完整设计。

方案目标是：

1. 保留此前 Core-Pareto GRPO v2 已验证有效的安全—进展优化方向；
2. 删除后续版本中相互叠加、难以归因的冗余机制；
3. 用一套简洁算法同时支持 NAVSIM v1 和 NAVSIM v2；
4. 把昂贵的 rollout 与 evaluator 预算集中到当前真正具有学习价值的场景；
5. 保持 on-policy、critic-free、单次更新，不引入复杂 PPO replay 或额外价值网络。

最终方法命名为：

> **LFP-GRPO：Learning-Frontier Pareto GRPO**

论文层面建议只强调两个核心算法创新：

1. **Reference-Relative Pareto Credit Assignment**：参考相对的 Pareto 正信用分配；
2. **Learning-Frontier Scene Curriculum**：由双向 Pareto 优势能量驱动的学习前沿课程。

此外，提供一个统一的 NAVSIM v1/v2 指标因子化方式，作为跨 benchmark 的方法泛化设计。

---

# 1. Stage2 与 Stage3 的职责划分

新的 Stage2 已经完成以下工作：

- 用 evaluator 验证的 Pareto support 扩展监督分布；
- 用 FS-Norm 把绝对 waypoint 空间改造成逐步运动增量空间；
- 用 Planning Token Adapter 提取规划相关的 VLM 语义；
- 用 trajectory-space auxiliary 和轻量相对几何约束提高可行性；
- 使 diffusion policy 能够从一个较宽、较可行的策略分布中采样。

因此，Stage3 不应该重新承担“把轨迹变平滑、变可行”的全部责任。Stage3 应集中解决两个问题：

> **问题 A：哪些 rollout 应当被增加概率？**
> **问题 B：哪些 scene 当前值得消耗 rollout 和 evaluator 预算？**

LFP-GRPO 分别用 Pareto 信用分配和学习前沿课程回答这两个问题。

---

# 2. 为什么不继续沿用当前复杂 Stage3 堆叠方案

当前仓库中的 Stage3 已逐渐包含：

- Core-Pareto；
- Feasible-Pareto；
- PDAS；
- phenotype bucket；
- asymmetric advantage；
- GSPO ratio；
- PPO replay；
- dynamic group weight；
- diversity reward；
- buffer distillation；
- diffusion DPO；
- self-imitation；
- BC 与 reference KL。

这些机制单独看都有动机，但同时启用时会产生三个问题。

## 2.1 信用分配被重复修改

一个 rollout 的 advantage 可能依次经历：

```text
安全门控
→ EP floor
→ 几何门控
→ Pareto front
→ bucket 内归一化
→ bucket 间修正
→ positive cap
→ scene weight
→ PDAS
→ ratio clipping
```

最终很难判断性能变化来自哪里，也容易让正优势样本过少。

## 2.2 Stage3 重复惩罚 Stage2 已解决的问题

新的 Stage2 已经通过 support 质量门控和轨迹辅助约束改善几何质量。如果 Stage3 再用 curvature、reverse、tail-reverse、jerk proxy 等启发式代价参与主 reward 和 valid mask，容易压低 EP，并使策略过度保守。

这些几何指标可以继续用于：

- 日志；
- support archive 审计；
- 失败案例分析；

但不进入 LFP-GRPO 的主策略梯度。

## 2.3 多个稳定项方向不一致

同时加入 BC、support distill、DPO、self-imitation 和 KL，会同时要求策略：

- 靠近旧策略；
- 靠近离线支持集；
- 靠近本轮高分样本；
- 提高 evaluator reward。

Stage2 已负责 imitation。Stage3 应尽量只保留 evaluator-driven policy gradient，以及一个轻量 KL 防止策略漂移。

---

# 3. 论文动机与核心创新

## 3.1 核心问题一：标量奖励会错误强化被支配轨迹

驾驶天然是多目标优化：

- Ego Progress 更高，可能伴随 TTC 下降；
- 更高安全余量可能牺牲进展；
- 更舒适、更靠近车道中心可能限制快速操作。

直接把多个指标加权成一个 scalar，再进行 GRPO，可能给“在所有关键目标上都比另一条轨迹差”的 rollout 正优势。

GD²PO 从 reward-wise advantage 冲突角度处理多奖励相互抵消问题，但驾驶中的 EP–TTC 冲突经常是合法权衡，不能简单要求所有 reward 方向一致。

LFP-GRPO 的核心观点是：

> **标量 benchmark score 决定更新强度；Pareto dominance 决定一个 rollout 是否有资格获得正信用。**

因此，我们不照搬 GD²PO 的符号一致性过滤，而使用更贴合驾驶多目标结构的 Pareto 正信用门控。

## 3.2 核心问题二：均匀采样浪费 evaluator 预算

对当前策略而言，场景可分为：

```text
太简单：
全部 rollout 都很好且相近，advantage 接近 0。

太困难：
全部 rollout 都不满足约束，没有可靠改进方向。

学习前沿：
同一 scene 内既有值得增强的 rollout，也有值得抑制的 rollout。
```

DAPO 的动态采样说明全对或全错 group 往往缺少有效 advantage。Self-Paced RL 和 Prioritized Level Replay 则表明，应根据当前策略的学习潜力自适应选择任务。

但本方法不使用简单 pass-rate 或人工难度标签，而直接从最终 Pareto advantage 构造场景学习价值：

> **一个 scene 如果同时提供正优势和负优势，并且优势幅度明显，就处于当前策略的学习前沿。**

由此提出 **Bidirectional Pareto Advantage Energy，双向 Pareto 优势能量**。

---

# 4. 方法概览

完整训练流：

```text
最终 Stage2 checkpoint
        ↓
冻结副本作为 reference policy
        ↓
每个 scene 采样 G 条 diffusion trajectories
        ↓
NAVSIM evaluator 计算官方 scalar 和分量指标
        ↓
Benchmark-specific Metric Adapter
        ↓
硬约束 + coherent reference + progress floor
        ↓
Pareto front
        ↓
scene 内中心化 + DDP global normalization
        ↓
Pareto-gated advantage
        ↓
Bidirectional Pareto Advantage Energy
        ↓
更新下一 epoch 的 scene sampling priority
        ↓
trajectory-level REINFORCE + exact transition KL
```

算法保持：

- critic-free；
- on-policy；
- 每批 rollout 只更新一次；
- 不复用陈旧 rollout；
- 不需要 PPO ratio；
- 不需要额外 value network。

---

# 5. 统一指标因子化

LFP-GRPO 将 benchmark 指标分为三类：

1. **Hard constraints**：不能被标量收益补偿的约束；
2. **Canonical Pareto objectives**：表示合法权衡的多目标向量；
3. **Official scalar reward**：决定策略梯度强度，并与最终榜单对齐。

## 5.1 NAVSIM v1 Adapter

### Hard constraints

```text
NC = 1
DAC = 1
DDC_candidate >= DDC_GT - δ_DDC
```

推荐：

```text
δ_DDC = 0.01
```

DDC 在 v1 中可作为辅助语义约束。门槛不使用固定 0.95，而以 GT DDC 为参考：

- GT DDC 高时，候选必须保持高 DDC；
- GT DDC 本身较低时，不会把所有候选错误过滤。

### Canonical Pareto vector

\[
\mathbf z^{v1} = (EP,\ TTC,\ Comfort)
\]

### Official scalar

\[
R^{v1} = PDMS
\]

## 5.2 NAVSIM v2 Adapter

NAVSIM v2 的 EPDMS 增加：

- multiplier：DDC、TLC；
- weighted metrics：LK、HC、EC；
- human-aware false-positive filtering；
- pseudo-closed-loop aggregation。

### Hard constraints

```text
NC = 1
DAC = 1
DDC_candidate >= DDC_GT - δ_DDC
filtered TLC = 1
```

DDC guard 优先使用 raw/unfiltered DDC。如果 evaluator 只提供 filtered DDC，则使用 filtered candidate/GT DDC 做同样的相对约束，并在日志中记录 fallback。

### Canonical Pareto vector

为了避免在 G=8 的小 group 上出现高维 Pareto 退化，使用三维统一向量：

\[
\mathbf z^{v2} = (EP,\ TTC,\ Q)
\]

其中：

\[
Q = \frac{LK + HC + EC}{3}
\]

LK、HC、EC 在官方 EPDMS 中权重相同，因此平均值是自然的综合质量维度。

### Official scalar

\[
R^{v2} = EPDMS
\]

## 5.3 是否共用一个 Stage3 checkpoint

建议：

```text
Common Stage2 checkpoint
        ├── LFP-GRPO-v1 checkpoint
        └── LFP-GRPO-v2 checkpoint
```

算法与主要超参数共用，但 metric adapter、reference cache 和训练 split 分开。

为了冲击各自榜单，不建议强制一个 Stage3 checkpoint 同时优化 v1 与 v2。两者评价函数和测试分布不同，分开 fine-tune 更合理。

---

# 6. Coherent Reference：同一条参考轨迹提供全部参考指标

每个 scene 准备：

1. GT trajectory；
2. 最终 Stage2 checkpoint 的 deterministic trajectory。

分别用对应 benchmark evaluator 计算完整指标向量。

## 6.1 Reference 选择

先判断两条参考是否满足该 benchmark 的 hard constraints。在满足约束的参考中选择官方 scalar 更高者：

\[
\tau_s^{ref}
=
\arg\max_{\tau \in \{\tau^{GT},\tau^{S2}\}}
R_b(\tau)
\]

如果两者都不满足约束，则选择 scalar 更高者作为 fallback，并记录 `reference_fallback=1`。

所有参考指标必须来自同一条轨迹：

```text
ref_scalar
ref_EP
ref_TTC
ref_quality
ref_NC
ref_DAC
ref_DDC
ref_TLC（v2）
```

禁止分别取 `max(GT_EP, Stage2_EP)`、`max(GT_TTC, Stage2_TTC)` 等，因为这种拼接 reference 可能在现实中不可同时达到。

## 6.2 DDC guard 的参考来源

虽然 scalar/core reference 是 coherent reference，但 DDC guard 单独使用 GT DDC：

\[
DDC_{candidate} \ge DDC_{GT} - \delta_{DDC}
\]

原因：DDC 更接近路线语义与方向合规，应以数据场景的真实行为为语义基准，而不是以当前 Stage2 策略能力为基准。

## 6.3 离线缓存

Reference 应在 Stage3 开始前离线构建，避免每个训练 step 重复：

- 运行 Stage2 deterministic sampling；
- evaluator scoring；
- GT scoring。

建议缓存 schema：

```yaml
version: 1
benchmark: navsim_v1 | navsim_v2
stage2_checkpoint_sha256: ...
scene_token: ...
gt:
  trajectory: [8, 3]
  scalar: ...
  components: {...}
stage2:
  trajectory: [8, 3]
  scalar: ...
  components: {...}
selected_source: gt | stage2
selected_scalar: ...
selected_components: {...}
gt_ddc_guard_value: ...
reference_fallback: false
```

v1 和 v2 使用独立 reference cache。

---

# 7. Reference-Relative Pareto Credit Assignment

设每个 scene \(s\) 采样 \(G\) 条轨迹 \(\tau_{s,i}\)。

## 7.1 Feasible mask

由 metric adapter 返回：

\[
F_{s,i} \in \{0,1\}
\]

v1：

\[
F = NC \land DAC \land DDC_{GT-relative}
\]

v2：

\[
F = NC \land DAC \land DDC_{GT-relative} \land TLC
\]

TTC、EP、Comfort/Quality 不作为 hard gate，因为这些指标之间存在合法权衡。

## 7.2 Progress floor

定义：

\[
P_{s,i}
=
\mathbf 1
\left[
EP_{s,i} \ge EP_s^{ref}-\delta_{EP}
\right]
\]

推荐：

```text
δ_EP = 0.02
```

它的含义不是“低 EP 轨迹绝对非法”，而是：

> 明显慢于 reference 的轨迹不能获得正信用。

## 7.3 Pareto front

只在：

\[
F_{s,i}=1 \land P_{s,i}=1
\]

的 rollout 中，根据 canonical vector 计算 non-dominated set：

v1：

\[
(EP,TTC,Comfort)
\]

v2：

\[
(EP,TTC,Q)
\]

若存在另一条可行轨迹在所有目标上不差，并至少一个目标更好，则当前 rollout 被支配。

## 7.4 Scalar score

使用官方 scalar，并加入轻量 reference-relative margin：

\[
S_{s,i}
=
R_b(\tau_{s,i})
+
\lambda_{ref}
\operatorname{clip}
\left(
\frac{R_b(\tau_{s,i})-R_b(\tau_s^{ref})}{\tau_{ref}},
-1,1
\right)
\]

默认：

```text
λ_ref = 0.20
τ_ref = 0.05
```

官方 scalar 保证优化目标与排行榜一致；reference margin 则提供绝对改进信号，避免只做 group 内相对排序。

## 7.5 Scene baseline

每个 scene 先确定 baseline mask：

1. feasible rollout 数量不少于 2：用 feasible mean；
2. 否则：用全部 G 条 rollout mean。

\[
\bar S_s = \operatorname{mean}_{i\in B_s} S_{s,i}
\]

中心化：

\[
D_{s,i}=S_{s,i}-\bar S_s
\]

## 7.6 DDP global advantage normalization

不除以 scene 内标准差。跨所有 GPU 汇总 normalization mask 上的：

```text
count
sum(D)
sum(D²)
```

得到：

\[
\mu_g,
\quad
\sigma_g
\]

最终：

\[
Z_{s,i}
=
\frac{D_{s,i}-\mu_g}{\max(\sigma_g,0.05)}
\]

统计量：

- 使用 float64 累加；
- 通过 `torch.distributed.all_reduce` 汇总；
- 对梯度 detach；
- 单进程时自动退化为本地统计。

这借鉴 REINFORCE++ 的 global normalization 原则，但在论文中定位为稳定训练基础，而非核心创新。

## 7.7 最终 advantage

定义 Pareto front mask 为 \(N_{s,i}\)。

\[
A_{s,i}
=
\begin{cases}
Z_{s,i},
& F=1,\ P=1,\ N=1
\\[4pt]
\min(Z_{s,i},0),
& F=1,\ (P=0\ \lor\ N=0)
\\[4pt]
-1,
& F=0\ \text{且该 scene 存在 feasible rollout}
\\[4pt]
0,
& \text{该 scene 所有 rollout 均 infeasible}
\end{cases}
\]

最后：

\[
A_{s,i}\leftarrow \operatorname{clip}(A_{s,i},-3,3)
\]

关键性质：

1. 只有约束可行、进展不退化、Pareto non-dominated 的 rollout 才可能得到正优势；
2. 被支配轨迹仍可得到负优势，但不能错误获得正强化；
3. 低 EP 轨迹不能通过 TTC 或 Comfort 较高获得正优势；
4. mixed group 中不安全轨迹明确得到负优势；
5. all-infeasible group 不更新，避免在没有可靠方向时盲目强化“相对不坏”的危险轨迹。

---

# 8. Learning-Frontier Scene Curriculum

## 8.1 双向 Pareto 优势能量

对每个 scene：

\[
p_s^+
=
\frac1G\sum_i \mathbf1[A_{s,i}>0]
\]

\[
p_s^-
=
\frac1G\sum_i \mathbf1[A_{s,i}<0]
\]

\[
M_s
=
\frac1G\sum_i |A_{s,i}|
\]

定义 **Bidirectional Pareto Advantage Energy（BPAE）**：

\[
E_s
=
4p_s^+p_s^-M_s
\]

## 8.2 解释

### 太简单

```text
所有 rollout 分数相近
→ |A| 很小
→ M_s 很小
→ E_s 接近 0
```

### 太困难

```text
所有 rollout 都 infeasible
→ A 全部为 0
→ E_s = 0
```

### 学习前沿

```text
存在 Pareto-good 正优势 rollout
同时存在 dominated/unsafe 负优势 rollout
→ p+ 和 p- 同时非零
→ M_s 有幅度
→ E_s 较高
```

BPAE 直接回答：

> 当前 scene 是否同时提供“应该增强什么”和“应该抑制什么”的策略梯度证据？

这比 binary pass rate 更适合连续、多目标驾驶。

## 8.3 学习进展

为每个 scene 维护两个 EMA：

\[
E_s^{fast}
\leftarrow
\alpha_f E_s^{fast}
+
(1-\alpha_f)E_s
\]

\[
E_s^{slow}
\leftarrow
\alpha_s E_s^{slow}
+
(1-\alpha_s)E_s
\]

默认：

```text
α_fast = 0.80
α_slow = 0.98
```

学习进展：

\[
LP_s = |E_s^{fast}-E_s^{slow}|
\]

优先级：

\[
W_s
=
\epsilon
+
E_s^{fast}
+
\lambda_{LP}LP_s
\]

默认：

```text
λ_LP = 0.50
ε = 1e-3
```

## 8.4 采样分布

采用 epoch-lagged online curriculum：第 \(e\) 个 epoch 采样权重来自此前 rollout 统计。

\[
P(s)
=
(1-\rho)
\frac{W_s^{\eta}}{\sum_j W_j^{\eta}}
+
\rho\frac1N
\]

默认：

```text
η = 0.50
ρ = 0.20
```

即：

```text
80% frontier-prioritized sampling
20% uniform exploration
```

平方根指数避免少数高能量场景垄断训练。

## 8.5 训练调度

```text
Epoch 0：完全 uniform，初始化 scene state。
Epoch 1+：启用 frontier sampler。
未见过 scene：使用全局中位 priority 初始化。
priority：做 95% quantile cap。
每个 epoch 结束：跨 rank 合并本 epoch scene energy，更新 EMA。
下一 epoch：使用新权重。
```

这种 epoch-wise 更新比每 step 全局同步简单、稳定且高效。

## 8.6 为什么不是 hard filtering

不永久删除难场景或简单场景：

- 20% uniform 避免遗忘；
- 困难场景随着策略进步可能进入 frontier；
- 简单场景仍会被周期性复习；
- curriculum 改变的是采样概率，不改变目标分布定义。

---

# 9. 策略优化目标

## 9.1 Trajectory-level REINFORCE

对 diffusion denoising chain 的各 reverse transition log-probability 做 discounted mean，得到：

\[
\log \pi_\theta(\tau_{s,i}|s)
\]

策略损失：

\[
L_{PG}
=
-
\frac1{BG}
\sum_{s,i}
A_{s,i}^{detach}
\log \pi_\theta(\tau_{s,i}|s)
\]

不再增加额外 scene weight。课程 sampler 已经在数据层控制场景曝光频率，重复在 loss 中加 scene weight 会产生双重加权。

## 9.2 Exact transition KL

冻结最终 Stage2 policy 作为 reference，使用已有 Gaussian reverse-transition 分布计算：

\[
L_{KL}
=
\mathbb E_t
D_{KL}
\left(
 p_\theta(x_{t-1}|x_t,s)
\|
 p_{S2}(x_{t-1}|x_t,s)
\right)
\]

总损失：

\[
L = L_{PG}+\beta_{KL}L_{KL}
\]

默认：

```text
β_KL = 0.005
```

推荐小范围 sweep：

```text
0.002 / 0.005 / 0.010
```

## 9.3 明确关闭的机制

主方案关闭：

```text
BC loss
GSPO ratio
PPO replay
PDAS
Feasible-Pareto geometry reward
phenotype bucket GRPO
adaptive dual
buffer distillation
buffer DPO
self-imitation
diversity reward
Stage3 trajectory geometry loss
```

---

# 10. NAVSIM v2 的 pseudo-closed-loop 处理

NAVSIM v2 的最终分数包含：

1. first-stage EPDMS；
2. follow-up scenes 的加权 EPDMS；
3. 根据 first-stage endpoint 与 follow-up start point 的 Gaussian 权重；
4. 两阶段聚合。

第一版 LFP-GRPO 建议：

- 把 first-stage 和 follow-up token 都作为训练 scene；
- 每个 token 用 one-stage official EPDMS 计算 rollout reward；
- curriculum 对每个 token 独立维护 frontier state；
- 最终验证使用官方 pseudo-closed-loop aggregation。

不在每个 Stage3 rollout 内展开完整两阶段闭环，因为这会显著放大 evaluator 成本，并破坏“简洁高效”的设计目标。

必须额外报告：

- one-stage EPDMS；
- official aggregated EPDMS；
- first-stage endpoint distribution；
- EC/HC/LK 子指标。

若后续发现 one-stage 与 aggregated EPDMS 明显错位，再考虑 root-scene-aware reward backend，但不改变 LFP-GRPO 主算法。

---

# 11. 参数训练范围

冻结：

```text
VLM backbone
frozen Stage2 reference policy
NAVSIM evaluator / scorer
```

训练：

```text
LightningDiT
action encoder / decoder
fusion projector
history/status encoder
Planning Token Adapter 与 planning gates
VLM feature projector
```

不重新启用 VLM LoRA。

---

# 12. 默认训练配置

```yaml
stage3_algorithm: lfp_grpo
stage3_objective: grpo
benchmark_adapter: navsim_v1  # 或 navsim_v2

# Rollout
group_size: 8
sampling_method: ddim
num_inference_steps: 5
on_policy_single_update: true
trajectory_logprob_reduce: discounted_mean
min_sampling_denoising_std: 0.04
min_logprob_denoising_std: 0.10
gamma_denoising: 0.60

# Constraints
require_nc: true
require_dac: true
ddc_guard_mode: gt_relative
ddc_gt_tolerance: 0.01
v2_require_tlc: true

# Pareto credit
ep_reference_tolerance: 0.02
reference_margin_weight: 0.20
reference_margin_scale: 0.05
global_std_floor: 0.05
advantage_clip: 3.0
all_infeasible_update: skip

# Curriculum
curriculum_enabled: true
curriculum_warmup_epochs: 1
frontier_fast_ema: 0.80
frontier_slow_ema: 0.98
frontier_progress_weight: 0.50
frontier_priority_exponent: 0.50
frontier_uniform_ratio: 0.20
frontier_priority_eps: 1.0e-3
frontier_priority_quantile_cap: 0.95

# Stability
reference_kl_coeff: 0.005
learning_rate: 1.0e-5
max_epochs: 10
max_grad_norm: 1.0
scheduler_warmup_epochs: 1
scheduler_min_lr: 1.0e-6

# Disable redundant paths
use_gspo_ratio: false
use_ppo_replay: false
use_feasible_pareto_grpo: false
use_pdas: false
use_phenotype_buckets: false
use_adaptive_dual: false
use_diversity_reward: false
use_buffer_guidance: false
use_buffer_distillation: false
use_buffer_dpo: false
use_self_imitation: false
bc_weight: 0.0
```

---

# 13. 算法伪代码

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)

    for scenes in loader:
        # 1. On-policy rollouts
        chains, trajectories = policy.sample_group(scenes, G=8)

        # 2. Official evaluator
        scalar_reward, components = metric_adapter.evaluate(
            trajectories,
            scenes,
        )

        # 3. Coherent reference + GT-relative DDC
        reference = reference_cache.lookup(scenes.tokens)
        feasible = metric_adapter.feasible_mask(
            components,
            gt_ddc=reference.gt_ddc,
        )
        progress_ok = components.ep >= reference.ep - ep_tolerance

        # 4. Pareto front
        objectives = metric_adapter.canonical_objectives(components)
        pareto = pareto_front(
            objectives,
            mask=feasible & progress_ok,
        )

        # 5. Scalar score
        ref_margin = clip(
            (scalar_reward - reference.scalar) / reference_margin_scale,
            -1,
            1,
        )
        score = scalar_reward + reference_margin_weight * ref_margin

        # 6. Scene baseline + global normalization
        centered = group_center(score, preferred_mask=feasible)
        global_mean, global_std = distributed_masked_moments(
            centered,
            mask=feasible,
            std_floor=0.05,
        )
        z = (centered - global_mean) / global_std

        # 7. Pareto-gated advantage
        advantage = zeros_like(z)
        mixed = feasible.any(-1) & (~feasible.all(-1))
        all_infeasible = ~feasible.any(-1)

        advantage[feasible & progress_ok & pareto] = z[
            feasible & progress_ok & pareto
        ]
        advantage[feasible & (~progress_ok | ~pareto)] = minimum(
            z[feasible & (~progress_ok | ~pareto)],
            0,
        )
        advantage[(~feasible) & mixed[:, None]] = -1.0
        advantage[all_infeasible[:, None]] = 0.0
        advantage = advantage.clamp(-3, 3).detach()

        # 8. Learning-frontier signal
        energy = bidirectional_pareto_advantage_energy(advantage)
        curriculum_accumulator.add(scenes.tokens, energy)

        # 9. On-policy trajectory REINFORCE + exact KL
        trajectory_logp = policy.chain_logprob(chains)
        policy_loss = -(advantage.flatten() * trajectory_logp).mean()
        kl_loss = exact_transition_kl(policy, stage2_reference, chains)
        loss = policy_loss + kl_coeff * kl_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

    # 10. Epoch-wise curriculum update
    energy_map = distributed_merge(curriculum_accumulator)
    frontier_state.update_fast_slow_ema(energy_map)
    sampler.update_weights(frontier_state.priorities())
```

---

# 14. 论文贡献写法

## Contribution 1

> We formulate driving policy improvement as constrained Pareto optimization and introduce Pareto-gated credit assignment, where official benchmark scores determine update magnitude while Pareto dominance determines positive-credit eligibility.

## Contribution 2

> We introduce a learning-frontier curriculum driven by Bidirectional Pareto Advantage Energy, which directly measures whether a scene simultaneously provides behaviors to reinforce and suppress.

## Contribution 3

> We provide a metric-factorized instantiation that applies the same optimizer to NAVSIM v1 and v2 through benchmark-specific constraint, objective, and scalar adapters.

不建议把以下内容写成主要创新：

- global advantage normalization；
- exact KL；
- trajectory-level log-probability；
- DDIM chain likelihood。

这些应描述为稳定且必要的实现基础。

---

# 15. 与相关方法的区别

## REINFORCE++

采用其 global advantage normalization 原则，但本方法的创新在于驾驶专用的 Pareto credit 与 frontier curriculum。

## DAPO

DAPO 过滤全对/全错 group，减少零 advantage 样本。LFP-GRPO 在 rollout 之前通过历史 frontier state 分配 scene 采样概率，并用连续、多目标 advantage energy 替代 binary correctness。

## GD²PO

GD²PO 根据不同 reward advantage 的一致性过滤冲突 rollout。LFP-GRPO 不要求目标方向一致，而使用 Pareto dominance 保留合法 safety–progress trade-off。

## Self-Paced RL / Prioritized Level Replay

这些工作根据任务难度或学习潜力构建 curriculum。LFP-GRPO 的 scene priority 直接来自最终 Pareto-gated policy advantage，而不是独立的 value error、pass rate 或人工 difficulty。

---

# 16. 消融实验

## 16.1 算法消融

| 实验 | Global Norm | Pareto Positive Gate | BPAE Curriculum | Learning Progress |
|---|---:|---:|---:|---:|
| S0 | 否 | 否 | 否 | 否 |
| S1 | 是 | 否 | 否 | 否 |
| S2 | 是 | 是 | 否 | 否 |
| S3 | 是 | 是 | 是 | 否 |
| S4 | 是 | 是 | 是 | 是 |

S0 应使用曾达到 PDMS 91 的旧 Core-Pareto v2 配置，并在新的 Stage2 checkpoint 上重新复现。

## 16.2 Curriculum 对照

```text
Uniform sampling
DAPO-style zero-variance filtering
Pass-rate Gaussian curriculum
BPAE only
BPAE + learning progress
```

## 16.3 KL sweep

```text
0.002 / 0.005 / 0.010
```

## 16.4 必须报告的效率指标

除最终 PDMS/EPDMS 外，还应报告：

```text
score gain per million evaluated trajectories
evaluator calls to reach a fixed score
zero-signal scene ratio
all-infeasible scene ratio
frontier scene hit ratio
positive-and-negative advantage coexistence ratio
curriculum effective sample entropy
```

课程贡献不能只依靠最终分数证明，还需证明 evaluator 预算利用率提高。

---

# 17. 诊断指标

## Benchmark metrics

v1：

```text
PDMS, EP, TTC, Comfort, NC, DAC, DDC
```

v2：

```text
EPDMS, EP, TTC, LK, HC, EC, NC, DAC, DDC, TLC
Official pseudo-closed-loop aggregated EPDMS
```

## Pareto credit

```text
feasible_ratio
progress_ok_ratio
pareto_front_ratio
dominated_ratio
positive_advantage_ratio
negative_advantage_ratio
zero_advantage_ratio
all_infeasible_group_ratio
advantage_clip_ratio
```

## Global normalization

```text
global_centered_mean
global_centered_std
global_normalization_count
per-rank rollout count
```

## Curriculum

```text
frontier_energy_mean / p50 / p90
frontier_fast_ema_mean
frontier_slow_ema_mean
learning_progress_mean
frontier_sampled_scene_ratio
uniform_sampled_scene_ratio
sampler_entropy
priority_max_to_median_ratio
unseen_scene_ratio
```

## Optimization stability

```text
policy_loss
exact_transition_kl
policy_gradient_norm
planning_adapter_gradient_norm
trajectory_logprob
FS output-bound hit ratio
```

---

# 18. 风险与控制

## 18.1 Pareto front 过大

风险：v2 使用过多目标会导致多数 rollout non-dominated。

控制：只使用 `(EP, TTC, Q)` 三维 canonical vector，不直接使用五维指标。

## 18.2 Curriculum 过度集中

控制：

- 20% uniform mixture；
- priority 平方根；
- 95% quantile cap；
- sampler entropy 日志；
- unseen scene 使用中位 priority。

## 18.3 All-infeasible scene 无梯度

这是有意设计：没有可行 rollout 时，组相对 RL 无法提供可靠正方向。通过 20% uniform 继续探索；随着 Stage2/策略提升，一旦出现可行 rollout，该 scene 会自动进入 learning frontier。

## 18.4 DDC 指标可用性

优先使用 raw DDC 与 GT raw DDC。若缓存只提供 filtered DDC，允许 fallback，但必须记录：

```text
ddc_raw_available_ratio
ddc_filtered_fallback_ratio
```

## 18.5 v2 one-stage 与最终 aggregation 不一致

首版保持简单，先优化 one-stage EPDMS 并用 official aggregated EPDMS 做验证。如果两者相关性不足，再单独研究 root-aware reward，不修改 LFP-GRPO 主体。

---

# 19. 推荐实验顺序

```text
R0：最终 Stage2 checkpoint，不做 Stage3
R1：旧 91 分 Core-Pareto v2，在新 Stage2 上复现
R2：R1 + coherent reference + GT-relative DDC
R3：R2 + DDP global normalization
R4：R3 + Pareto positive-credit gate
R5：R4 + BPAE curriculum
R6：R5 + learning progress，完整 LFP-GRPO
R7：R6 做 KL 0.002/0.005/0.010 小 sweep
```

先在 NAVSIM v1 快速验证，再迁移同一算法到 NAVSIM v2 adapter。

---

# 20. 参考工作

1. **REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization**, arXiv:2501.03262.
2. **DAPO: An Open-Source LLM Reinforcement Learning System at Scale**, arXiv:2503.14476.
3. **GD²PO: Mitigating Multi-Reward Conflicts via Group-Dynamic Reward-Decoupled Policy Optimization**, arXiv:2606.16771.
4. **Self-Paced Deep Reinforcement Learning**, arXiv:2004.11812.
5. **Prioritized Level Replay**, arXiv:2010.03934.
6. NAVSIM official metric documentation: PDMS / EPDMS and pseudo-closed-loop aggregation.

---

# 21. 最终一句话

> **LFP-GRPO only positively reinforces constraint-feasible, progress-preserving, Pareto-nondominated trajectories, while continuously reallocating rollout budget toward scenes that provide balanced positive and negative Pareto policy-gradient evidence.**

中文：

> **LFP-GRPO 只正向强化满足约束、保持进展且位于 Pareto 前沿的轨迹，并持续把 rollout 预算转移到同时提供正负策略梯度证据的学习前沿场景。**
