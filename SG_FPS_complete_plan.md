# SG-FPS：Scorer-Guided Feasible Pareto Support for VLA Diffusion Driving

> 目标：在 ReCogDrive / VLA + DiT diffusion policy 框架中，用**强外部轨迹生成器 + 高效结构化扩展 + evaluator-verified Pareto support + vector scorer + B-PDAS + Feasible Pareto-GRPO**，解决单 GT IL 的窄策略、GRPO 后期坍缩、DDC 退化、轨迹折线/倒退投机、candidate selection regret 等问题。

本文档是面向实现和论文写作的完整方案。核心原则是：**不暴力生成大量候选，而是构建小而强的可行 Pareto 前沿支撑集；不把 scorer 当最终 reward oracle，而是用于 candidate mining 和 selection-regret reduction；不让 GRPO 单纯追逐 PDMS，而是加入 DDC、轨迹可行性、regression guard 与多样性保持机制。**

---

## 0. 现象与动机

当前 V2 Pareto-GRPO 相比 Stage2 IL 已有明显收益：

- PDMS、EP、DAC、TTC、SafetyMean 均有提升。
- 大量场景满足“安全项不降且 EP 提升”，说明模型确实学到了安全约束下的效率提升。
- 主要 PDMS 收益来自安全项修复，符合 PDMS 乘法安全 gate 的机制。

但还存在问题：

1. 仍有若干场景从 `PDMS > 0` 退化为 `PDMS = 0`。
2. DDC 在 V2 中存在轻微均值退化和较多 `>0 -> 0` 退化。
3. 部分轨迹出现“前两个点局部折线”和“最后几个点轻微倒退”，疑似 reward hacking。
4. 输出和 GT 差距大；这本身不一定错误，但需要判断其是否仍在 evaluator-verified support manifold 内。
5. GRPO 后期可能因为 group reward variance 变小、采样轨迹趋同而失去探索动力。
6. 如果候选池里存在好轨迹但选择器 top-1 选错，scorer / selection regret 会成为主要瓶颈；如果候选池里根本没有好轨迹，scorer 也无能为力。

因此，最终方案不是单纯调 reward，而是构建一个闭环：

```text
高质量候选生成
  -> evaluator-verified feasible Pareto support
  -> support imitation
  -> vector scorer 降低 selection regret
  -> B-PDAS 选择 learnable policy-boundary scenes
  -> bucketed feasible Pareto-GRPO 做后训练
```

---

## 1. 方法总览

方法名建议：

## SG-FPS: Scorer-Guided Feasible Pareto Support

核心模块：

1. **Efficient Candidate Funnel**  
   用 GT、Stage2 IL、当前 V2 policy、DiffusionDriveV2、DriveOR/DrivoR 等强 generator 产生少量高质量 seed anchors；经过 cheap filter 和 Pareto-NMS 后，只对 Pareto-potential anchors 做 failure-conditioned structured expansion 和 trust-region control expansion。

2. **Feasible Pareto Support Archive**  
   对候选轨迹计算 evaluator submetrics 和 trajectory feasibility metrics，按固定 quota 选择 `GT / IL / best-PDMS / safe EP improver / safety repair / DDC repair / smooth feasible / vector-Pareto / diversity-max valid`，形成每个 scene 的 8–12 条可行 Pareto 支撑集。

3. **FS-Norm + Trajectory-centric Auxiliary Loss**  
   用简化的 step-wise delta normalization 降低数值学习难度；保持 diffusion noise prediction 主干不变，同时从预测噪声推导 clean trajectory estimate \(\hat{x}_0\)，在轨迹空间加入 curvature、early-kink、tail-reverse、jerk 等辅助约束。

4. **Diverse Pareto-Support Imitation, DPSI**  
   Stage2 不再只模仿单条 GT，而是对 evaluator-verified support set 做加权 diffusion imitation，并加入 pairwise ranking / DPO-style preference loss。

5. **Pareto-Vector Scorer**  
   不预测单一 confidence，而是预测 `NC/DAC/TTC/EP/Comfort/DDC/TLC/FeasCost/PDMS/ParetoFrontProb`。Scorer 只用于 candidate pre-filter、support refinement 和 B-PDAS uncertainty signal，不直接替代 true evaluator。

6. **B-PDAS: Boundary-aware Pareto-Diversity Active Sampling**  
   在 GRPO 阶段动态关注 learnable policy-boundary scenes：success rate 适中、group reward variance 高、Best-of-N gap 高、Pareto bucket 多样、存在 regression risk 的样本。

7. **Bucketed Feasible Pareto-GRPO**  
   在 GRPO 中优化 `EP/TTC/DDC/FeasCost` 的 safety-constrained utility，并加入 DDC guard、geometry cap、regression cap、dominated cap、off-support cap；用 bucketed intra/inter advantage 防止多样化 support 在 GRPO 后期坍缩。

---

## 2. Efficient Candidate Funnel

### 2.1 候选来源

每个 scene 先构造 seed anchors：

```text
GT:                         1
Stage2 IL deterministic:     1
Current V2 deterministic:    1
DiffusionDriveV2 top-k:      3~5
DriveOR / DrivoR top-k:      3~5
--------------------------------
Total seed anchors:          9~13
```

外部 generator 只作为 **proposal source**，不能直接作为 teacher target 蒸馏。所有外部轨迹必须经过 cheap filter、true evaluator 和 feasibility filter。

### 2.2 Cheap Filter

在跑完整 PDM / PDMS 前，先计算轻量几何与路线风险指标，过滤明显无效轨迹。

Cheap filter 指标：

```text
early_kink_violation
tail_reverse_violation
large_heading_jump
curvature_proxy_bad
final_progress_too_low
route_heading_inconsistent
distance_to_anchor_too_large
```

建议阈值初始设置：

```text
max_heading_jump_rad = 0.45
max_early_heading_jump_rad = 0.35
tail_reverse_eps_m = 0.05
max_tail_reverse_steps = 1
max_anchor_distance_m = 6.0
min_final_progress_ratio = 0.3
```

### 2.3 Pareto-NMS

对通过 cheap filter 的 seed anchors，估计：

\[
\hat{F}=[\widehat{EP},\widehat{TTC},\widehat{DDC},-\widehat{FeasCost}]
\]

保留 predicted Pareto front，然后按 endpoint distance / trajectory distance 做 NMS：

```text
同一 source 最多保留 2 条
endpoint distance < 0.5m 且 final heading 接近时，只保留 utility 更高的
progress bucket / lateral bucket 至少覆盖 2 类
最终保留 4~6 个 anchors
```

### 2.4 Failure-conditioned Structured Expansion

不要对所有 anchor 做全组合扰动，而是先判断短板再扩展。

#### Case A：安全好但 EP 低

条件：

\[
NC=1,\ DAC=1,\ TTC \ge TTC_{ref}-\epsilon,\ DDC \ge DDC_{ref}-\epsilon,\ EP < EP_{target}
\]

生成：

```text
mild progress extension: +0.5m
moderate progress extension: +1.0m
time-gamma earlier progress
```

#### Case B：TTC / collision risk

条件：

\[
TTC < TTC_{ref}-\epsilon_{ttc}
\quad \text{or}\quad NC \text{ risk}
\]

生成：

```text
slow-first
delay-then-recover
creep-then-stop
```

#### Case C：DDC / route consistency risk

条件：

\[
DDC < DDC_{ref}-\epsilon_{ddc}
\]

生成：

```text
route-heading smoothing
endpoint return-to-route-center
lateral offset decay-to-zero
```

#### Case D：geometry bad

不继续扩展，先 repair：

```text
spline smoothing
monotonic progress projection
heading step clamp
```

如果 repair distance 太大，则作为 hard negative，不进入 support。

### 2.5 Trust-region Control Expansion

吸收 TOAD 的关键思想：不要在 pose space 做大扰动，而是在 control-like space 小范围搜索。

流程：

1. 选 top-2 Pareto-potential anchors。
2. 近似反解控制序列：

\[
u_t=[a_t,\omega_t]
\]

3. 小范围采样：

\[
u'_t=u_t+\delta u_t
\]

4. 用 kinematic bicycle / unicycle rollout 得到轨迹。
5. cheap filter + true evaluator 验证。

每个 top anchor 只采 8 条即可。最终每个 scene 的 candidate 数控制在：

```text
seed anchors:             9~13
Pareto-NMS anchors:       4~6
structured variants:      8~18
control variants:         16
--------------------------------
pre-evaluator candidates: 20~40
verified support:         8~12
```

---

## 3. Feasible Pareto Support Archive

### 3.1 Candidate Labels

对每条候选轨迹计算：

\[
y(\tau)=[PDMS,NC,DAC,TTC,EP,Comfort,DDC,TLC,FeasCost]
\]

其中：

\[
FeasCost=
\alpha_\kappa V_\kappa
+\alpha_{rev}V_{reverse}
+\alpha_{kink}V_{early-kink}
+\alpha_{jerk}V_{jerk}
\]

推荐初始权重：

```text
alpha_curvature = 1.0
alpha_reverse   = 1.0
alpha_kink      = 1.0
alpha_jerk      = 0.2
```

### 3.2 Valid Gate

\[
Valid(\tau)=
\mathbb{1}[NC=1]
\cdot
\mathbb{1}[DAC=1]
\cdot
\mathbb{1}[DDC\ge d_{min}]
\cdot
\mathbb{1}[FeasCost\le f_{max}]
\]

建议：

```text
d_min = max(0.95, DDC_ref - 0.01)
f_max = percentile-based threshold, e.g. P90 of GT/IL FeasCost
```

### 3.3 Pareto Objectives

Comfort 基本饱和，不作为 Pareto 主轴；只作为 veto。

Pareto vector：

\[
F(\tau)=[EP,TTC,DDC,-FeasCost]
\]

Comfort veto：

\[
Comfort < c_{min} \Rightarrow A\le0
\]

推荐：

```text
c_min = 0.95
```

### 3.4 Support Quota

每个 scene 最终保留 8–12 条 support candidates：

```text
1 GT anchor
1 Stage2 IL anchor
1 best-PDMS valid
2 safe EP improvers
1 safety repair
1 DDC repair
1 smooth feasible candidate
2 vector-Pareto candidates
1 diversity-max valid candidate
```

如果某类不存在，用 vector-Pareto candidate 或 best-valid candidate 补齐。

### 3.5 Support 类型定义

#### Safe EP Improver

\[
EP_i>EP_{ref}+\delta_{ep}
\]

且：

\[
NC_i\ge NC_{ref},\quad DAC_i\ge DAC_{ref},\quad TTC_i\ge TTC_{ref}-\epsilon_{ttc}
\]

\[
DDC_i\ge DDC_{ref}-\epsilon_{ddc},\quad FeasCost_i\le FeasCost_{ref}+\epsilon_f
\]

#### Safety Repair

\[
EP_i\ge EP_{ref}-\epsilon_{ep}
\]

且至少一个安全项提升：

```text
NC / DAC / TTC / DDC improves
```

#### DDC Repair

\[
DDC_i>DDC_{ref}+\delta_{ddc}
\]

且：

\[
PDMS_i\ge PDMS_{ref}-\epsilon
\]

#### Smooth Feasible Candidate

\[
FeasCost_i=\min_j FeasCost_j
\]

并满足基本 valid gate。

#### Vector-Pareto Candidate

在：

\[
[EP,TTC,DDC,-FeasCost]
\]

上非支配。

---

## 4. FS-Norm：Feasibility-aware Step-wise Normalization

PARS-Norm 原方案包含 anchor residual、Frenet、source-bucket 等设计，归因困难。最终使用更简洁的 FS-Norm。

### 4.1 主表示：Step-wise Delta

主 diffusion target 从 absolute trajectory：

\[
[x_t,y_t,\theta_t]
\]

改为 ego-local step delta：

\[
u_t=[\Delta x_t,\Delta y_t,\Delta\theta_t]
\]

其中：

\[
\Delta x_t=x_t-x_{t-1}
\]

\[
\Delta y_t=y_t-y_{t-1}
\]

\[
\Delta\theta_t=wrap(\theta_t-\theta_{t-1})
\]

轨迹恢复：

\[
x_t=\sum_{k=1}^{t}\Delta x_k,
\quad
y_t=\sum_{k=1}^{t}\Delta y_k,
\quad
\theta_t=\sum_{k=1}^{t}\Delta\theta_k
\]

### 4.2 Step-wise Robust Normalization

对每个 horizon \(t\)、维度 \(d\) 统计：

\[
\mu_{t,d},\quad \sigma_{t,d}
\]

统计来源：

```text
GT + Stage2 IL + evaluator-verified Pareto support
```

归一化：

\[
\bar{u}_{t,d}=clip\left(\frac{u_{t,d}-\mu_{t,d}}{\sigma_{t,d}+\epsilon},-c,c\right)
\]

推荐：

```text
clip c = 5
epsilon = 1e-6
```

若候选异常值较多，使用 median / MAD：

\[
\bar{u}_{t,d}=\frac{u_{t,d}-median_{t,d}}{1.4826\cdot MAD_{t,d}+\epsilon}
\]

### 4.3 Heading Loss

不要直接对 heading 做普通 MSE。辅助 loss 使用：

\[
\mathcal{L}_{heading}=|\sin\hat{\theta}-\sin\theta|+|\cos\hat{\theta}-\cos\theta|
\]

### 4.4 保留 Noise Prediction，新增 \(x_0\) Auxiliary

主 diffusion 仍预测 noise：

\[
\mathcal{L}_{\epsilon}=\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2
\]

从预测噪声推出：

\[
\hat{x}_0=
\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}
\]

将 \(\hat{x}_0\) 解码回 raw trajectory 后加：

\[
\mathcal{L}_{aux}=\lambda_{x0}\mathcal{L}_{x0}+\lambda_{geo}\mathcal{L}_{geo}
\]

其中：

\[
\mathcal{L}_{x0}=\|\hat{\tau}-\tau\|_1+\lambda_h\mathcal{L}_{heading}
\]

\[
\mathcal{L}_{geo}=\lambda_\kappa\mathcal{L}_{curv}
+\lambda_{rev}\mathcal{L}_{reverse}
+\lambda_{kink}\mathcal{L}_{early-kink}
+\lambda_{jerk}\mathcal{L}_{jerk}
\]

只在低噪声或中低噪声 timestep 上启用：

```text
timestep in lowest-noise 30%~50%
```

---

## 5. DPSI：Diverse Pareto-Support Imitation

Stage2 不再只模仿 GT，而是模仿 support set。

\[
\mathcal{L}_{DPSI}=
\sum_{\tau_i\in\mathcal{S}(o)}w_i\mathcal{L}_{diff}(\tau_i)
+\lambda_{x0}\mathcal{L}_{x0}
+\lambda_{geo}\mathcal{L}_{geo}
+\lambda_{rank}\mathcal{L}_{rank}
\]

推荐权重：

```text
safe EP improver:        1.0
vector-Pareto candidate: 0.9
safety repair:           0.8
DDC repair:              0.8
best-PDMS valid:         0.7
smooth feasible:         0.6
IL anchor:               0.5
GT anchor:               0.4
invalid / repair far:    0.0
```

Pairwise ranking：

\[
\mathcal{L}_{rank}= -\log \sigma\left(\beta[\log p_\theta(\tau^+|o)-\log p_\theta(\tau^-|o)]\right)
\]

正样本：safe EP improver、vector-Pareto、safety repair、DDC repair。  
负样本：dominated valid、unsafe、DDC-regressed、geometry-bad、off-support low-score。

---

## 6. Pareto-Vector Scorer

### 6.1 作用边界

Scorer 不直接替代 evaluator，不直接作为 GRPO 主 reward。只做三件事：

```text
1. candidate pre-filter
2. Pareto target proposal
3. B-PDAS uncertainty / selection-regret signal
```

所有 scorer-selected targets 必须经过 true evaluator 验证后才能进入 support archive。

### 6.2 输入输出

\[
S_\phi(o,\tau)\rightarrow[
\hat{NC},\hat{DAC},\hat{TTC},\hat{EP},\hat{Comfort},\hat{DDC},\hat{TLC},\widehat{FeasCost},\widehat{PDMS},\hat{p}_{front}]
\]

输入特征：

```text
VLA / DiT scene tokens
ego status
route command
trajectory delta features
speed / acceleration / yaw-rate / curvature proxy
source embedding: GT / IL / DDv2 / DriveOR / structured / control
```

### 6.3 Scorer Loss

Metric loss：

\[
\mathcal{L}_{metric}=\sum_m\lambda_m\ell(\hat{y}_m,y_m)
\]

Pareto-front classification：

\[
\mathcal{L}_{front}=BCE(\hat{p}_{front},\mathbb{1}[\tau\in\mathcal{P}])
\]

Pairwise ranking：

\[
\mathcal{L}_{rank}= -\log\sigma\left(\frac{\hat{U}_i-\hat{U}_j}{\tau}\right)
\]

Formula consistency：

\[
\hat{PDMS}_{formula}=\hat{NC}\cdot\hat{DAC}\cdot\frac{5\hat{EP}+5\hat{TTC}+2\hat{Comfort}}{12}
\]

\[
\mathcal{L}_{formula}=|\hat{PDMS}-\hat{PDMS}_{formula}|
\]

Total：

\[
\mathcal{L}_{scorer}=\mathcal{L}_{metric}+\lambda_f\mathcal{L}_{formula}+\lambda_p\mathcal{L}_{front}+\lambda_r\mathcal{L}_{rank}
\]

---

## 7. B-PDAS：Boundary-aware Pareto-Diversity Active Sampling

B-PDAS 的目标是在 GRPO 训练阶段优先选择当前策略边界附近、learnable、reward 有差异、Best-of-N 有潜力、且存在 regression risk 的场景。

### 7.1 在线统计

对 scene \(o_i\)，当前 policy 采样 \(G\) 条轨迹：

\[
\{\tau_{i,1},\ldots,\tau_{i,G}\}
\]

#### A. Success-rate Learnability

\[
p_i=\frac{1}{G}\sum_{g=1}^{G}\mathbb{1}[Valid(\tau_{i,g})\land U(\tau_{i,g})>U_{ref}(i)+\delta]
\]

\[
L_i^{succ}=4p_i(1-p_i)
\]

当 \(p_i\approx0.3\sim0.7\) 时，样本最有学习价值。

#### B. Group Reward Variance

\[
L_i^{var}=clip\left(\frac{std(R_{i,1:G})}{\sigma_0},0,1\right)
\]

#### C. Best-of-N Gap / Selection Regret

\[
L_i^{gap}=clip\left(\frac{\max_g U_{i,g}-mean_g U_{i,g}}{\Delta_0},0,1\right)
\]

#### D. Pareto Bucket Diversity

轨迹按 phenotype 分桶：

```text
progress: slow / normal / fast
lateral endpoint: left / center / right
DDC: route-consistent / route-risky
feasibility: smooth / kink-risk / reverse-risk
```

\[
L_i^{bucket}=\frac{\#occupied\ buckets}{\#max\ buckets}
\]

#### E. Regression Risk

\[
L_i^{reg}=\mathbb{1}[PDMS_{ref}>0\land\exists g:PDMS(\tau_{i,g})=0]
+\mathbb{1}[\exists g:DDC(\tau_{i,g})<DDC_{ref}-\epsilon]
+\mathbb{1}[\exists g:FeasBad(\tau_{i,g})]
\]

### 7.2 Sampling / Weighting Score

\[
W_i=(\epsilon+L_i^{succ})^{\alpha}(\epsilon+L_i^{var})^{\beta}(\epsilon+L_i^{gap})^{\gamma}(1+\lambda_bL_i^{bucket})(1+\lambda_rL_i^{reg})C_i
\]

推荐初始参数：

```text
alpha = 1.0
beta = 1.0
gamma = 0.5
lambda_bucket = 0.2
lambda_regression = 0.5
epsilon = 0.05
```

`C_i` 是 category / cluster balancing，用于防止训练分布偏移。

### 7.3 Schedule

#### Early GRPO

重点：

```text
>0 -> 0 regression risk
DDC regression risk
DAC/TTC boundary
geometry bad
```

#### Middle GRPO

重点：

```text
success rate 0.3~0.7
group reward std 高
Best-of-N gap 高
Pareto bucket 多样
```

#### Late GRPO

重点：

```text
仍有非零 reward variance
support bucket 仍有多样性
scorer uncertainty 高但 evaluator 可验证
```

同时降低 LR，提高 KL/reference weight，加强 positive advantage cap。

---

## 8. Feasible Pareto-GRPO

### 8.1 Utility

\[
U(\tau)=0.60EP+0.25TTC+0.10DDC-0.05FeasCost
\]

### 8.2 Valid Gate

\[
G(\tau)=\mathbb{1}[NC=1]\cdot\mathbb{1}[DAC=1]\cdot\mathbb{1}[DDC\ge d_{min}]\cdot\mathbb{1}[FeasCost\le f_{max}]
\]

### 8.3 Reward

\[
R(\tau)=G(\tau)[U(\tau)+\lambda_p\mathbb{1}[\tau\in ParetoFront]-\lambda_{trade}P_{bad-trade}]+(1-G(\tau))R_{unsafe}
\]

其中：

\[
P_{bad-trade}=[-(\Delta EP+\rho\Delta TTC)-\epsilon]_+
\]

这表示不允许用明显 TTC 退化换 EP。

### 8.4 Positive Advantage Caps

必须加入以下 cap：

```text
if NC=0 or DAC=0:
    A <= negative

if DDC < max(0.95, DDC_ref - 0.01):
    A <= 0

if FeasCost > f_max:
    A <= 0

if dominated by vector-Pareto front:
    A <= 0

if off-support and not clearly better than ref:
    A <= 0

if PDMS_ref > 0 and candidate PDMS == 0:
    A <= negative

if Comfort < c_min:
    A <= 0
```

### 8.5 Bucketed Advantage

按照 phenotype bucket 做 intra-bucket advantage：

\[
A^{intra}=zscore(U\mid bucket)
\]

跨 bucket 只做弱比较：

\[
A=A^{intra}+\eta\cdot clip(A^{inter},-c,c)
\]

推荐：

```text
inter_bucket_weight = 0.2~0.3
inter_bucket_clip = 0.5
```

### 8.6 KL / BC

KL 参考应靠近 DPSI policy，而不是单 GT：

\[
D_{KL}(\pi_\theta||\pi_{DPSI})
\]

BC 也不建议强拉 GT，而是弱拉 support：

\[
\mathcal{L}_{support}=\sum_{\tau_i\in\mathcal{S}}w_i\mathcal{L}_{diff}(\tau_i)
\]

最终：

\[
\mathcal{L}_{Stage3}=\mathcal{L}_{GRPO}+\lambda_{KL}D_{KL}(\pi_\theta||\pi_{DPSI})+\lambda_{sup}\mathcal{L}_{support}
\]

推荐：

```text
KL to DPSI:       中等
support BC:       弱到中等
GT BC:            很弱，或并入 support
```

---

## 9. 完整训练流程

```text
Step 0: Base Stage2 IL
  训练原始 ReCogDrive / VLA-DiT diffusion planner。

Step 1: Efficient Candidate Funnel
  GT + IL + V2 + DiffusionDriveV2 + DriveOR seed anchors。
  cheap filter + Pareto-NMS。

Step 2: Failure-conditioned Expansion
  progress expansion / yield expansion / DDC repair / geometry repair / trust-region control expansion。

Step 3: Evaluator Labeling
  对所有候选计算 PDMS, NC, DAC, TTC, EP, Comfort, DDC, TLC, FeasCost。

Step 4: Feasible Pareto Support Archive
  每个 scene 保留 8~12 条 support。

Step 5: FS-Norm + DPSI
  用 support archive 训练 diffusion policy。

Step 6: Train Vector Scorer
  用 evaluator labels 训练 metric-vector scorer。

Step 7: Scorer-guided Support Refinement
  Scorer 预筛更大候选池，true evaluator 验证 top-k / vector-Pareto，更新 support archive。

Step 8: B-PDAS + Feasible Pareto-GRPO
  用 B-PDAS 选择 learnable boundary scenes，用 bucketed Pareto-GRPO 后训练。
```

---

## 10. 实验与消融

### 10.1 主对比

```text
Stage2 IL
Original ReCogDrive Stage3 RL
V2 Pareto-GRPO
V2 + DDC/geometry/regression cap
DPSI
DPSI + B-PDAS Pareto-GRPO
DPSI + scorer-guided support + B-PDAS Pareto-GRPO
```

### 10.2 模块消融

```text
Candidate source ablation:
  GT/IL only
  + DiffusionDriveV2
  + DriveOR/DrivoR
  + structured expansion
  + control expansion

Support selection ablation:
  scalar top-k
  PDMS top-k
  vector Pareto
  vector Pareto + quota
  vector Pareto + scorer-guided refinement

Normalization ablation:
  original normalization
  step-wise absolute normalization
  FS-Norm: step-wise delta + x0 geometry auxiliary

GRPO ablation:
  scalar GRPO
  core-pareto GRPO
  feasible pareto GRPO
  feasible pareto + bucketed advantage
  feasible pareto + B-PDAS
```

### 10.3 必报指标

```text
PDMS / EPDMS
EP / NC / DAC / TTC / Comfort / DDC / TLC
SafetyMean
0->>0 / >0->0
safe EP improver count
DDC regression count
early_kink_rate
tail_reverse_rate
curvature_violation_rate
support_distance_to_GT/IL
group reward std
Best-of-N gap
Pareto-front hit rate
scorer selection regret
scorer metric MAE / AUC / calibration error
oracle top-k gap
```

---

## 11. 风险与防护

### 风险 1：Scorer 过拟合候选分布

防护：

```text
scorer 不直接进 GRPO reward
scorer-selected targets 必须 true evaluator 验证
训练集加入 hard negatives / geometry-bad / off-support samples
```

### 风险 2：外部 generator 风格不兼容

防护：

```text
只作为 proposal source
不无条件蒸馏 teacher outputs
必须进入 evaluator-verified Pareto support
```

### 风险 3：B-PDAS 导致训练分布偏移

防护：

```text
category / cluster balancing
保留少量 uniform sampling
每个 epoch 固定 regression holdout scenes
```

### 风险 4：FeasCost 太强导致 EP 下降

防护：

```text
FeasCost 主要做 positive advantage cap
不要过度加入 scalar penalty
只惩罚明确异常：early kink / tail reverse / curvature violation
```

### 风险 5：DDC guard 误伤特殊场景

防护：

```text
使用 relative guard:
DDC >= max(0.95, DDC_ref - 0.01)
不要强制所有样本 DDC=1
```

---

## 12. 代码实现建议路径

建议新增文件：

```text
navsim/agents/recogdrive/trajectory_feasibility.py
navsim/agents/recogdrive/fs_norm.py
navsim/agents/recogdrive/candidate_funnel.py
navsim/agents/recogdrive/pareto_support.py
navsim/agents/recogdrive/pareto_vector_scorer.py
navsim/agents/recogdrive/pdas.py
scripts/tools/build_sg_fps_support_archive.py
scripts/tools/build_fs_norm_stats.py
scripts/training/run_train_pareto_vector_scorer.py
scripts/tools/refine_support_with_scorer.py
```

需要修改：

```text
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
navsim/agents/recogdrive/offline_rl_buffer.py
navsim/agents/recogdrive/offline_action_explorer.py
navsim/agents/recogdrive/recogdrive_agent.py
navsim/planning/training/agent_lightning_module.py
navsim/planning/script/run_training_recogdrive_rl.py
```

配置新增：

```text
use_fs_norm
fs_norm_stats_path
x0_aux_weight
geo_aux_weight
use_feasible_pareto_grpo
fp_ep_weight
fp_ttc_weight
fp_ddc_weight
fp_feas_weight
fp_ddc_min_absolute
fp_ddc_relative_tolerance
fp_feas_max
fp_use_bucketed_advantage
fp_use_pdas
pdas_alpha
pdas_beta
pdas_gamma
pdas_regression_weight
support_archive_path
scorer_checkpoint_path
```

---

## 13. 论文叙事模板

### 标题

**SG-FPS: Scorer-Guided Feasible Pareto Support for Vision-Language-Action Diffusion Driving**

### 摘要核心

Existing VLA diffusion planners are trained from a single logged trajectory, which narrows policy support and limits subsequent GRPO exploration. We propose SG-FPS, a compact and efficient framework that constructs evaluator-verified feasible Pareto support from strong teacher generators and trust-region structured expansion. A vector scorer predicts planning sub-metrics to reduce selection regret, while FS-Norm and trajectory-centric auxiliary losses improve numerical conditioning and feasibility. Finally, B-PDAS and bucketed Pareto-GRPO focus training on learnable policy-boundary scenes, preserving diversity while optimizing safety, progress, DDC and trajectory feasibility.

### 贡献点

```text
1. Feasible Pareto Support Archive:
   将 IL 从 single-GT cloning 改成 evaluator-verified Pareto support learning。

2. Efficient Candidate Funnel:
   用强 generator + cheap filter + failure-conditioned expansion 构造小而强的候选池。

3. Pareto-Vector Scorer:
   不预测单一 confidence，而预测 PDMS 子指标和 Pareto-front probability，用于 support mining 和 selection-regret reduction。

4. B-PDAS + Bucketed Pareto-GRPO:
   动态关注 learnable policy-boundary scenes，并用 bucketed advantage 防止 GRPO 后期 mode collapse。

5. FS-Norm + x0 feasibility auxiliary:
   简化轨迹数值表示，并显式抑制局部折线、末端倒退、曲率异常。
```

---

## 14. 推荐实验推进顺序

虽然这是完整方案，但实验推进可以并行化：

```text
Track A: GRPO 修复线
  实现 DDC guard + geometry cap + regression cap + bucketed advantage + B-PDAS weighting。

Track B: Support 数据线
  接外部 generator，构建 candidate funnel 和 feasible Pareto support archive。

Track C: 模型训练线
  实现 FS-Norm + x0 geometry auxiliary，训练 DPSI。

Track D: Scorer 线
  训练 vector scorer，用于 support refinement。
```

最终组合：

```text
SG-FPS = Candidate Funnel + Pareto Support Archive + FS-Norm/DPSI + Vector Scorer + B-PDAS Feasible Pareto-GRPO
```
