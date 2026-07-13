# LFP-GRPO Stage3 探究总结

更新时间：2026-07-13 06:17 UTC

## 1. 术语与研究目标

- `NAVSIM v1`：PDMS 评估口径。
- `NAVSIM v2`：EPDMS 评估口径。
- `Pareto GRPO v2`：历史 Stage3 算法，不等同于 NAVSIM v2。

研究主线是：学习多样但可行的策略分布，以 Pareto 方式处理安全、进度和质量之间的冲突，
再用 learning-frontier curriculum 主动选择尚未学好的场景。KL、FS 表示、credit gate 和其他
稳定化机制只服务这三项核心创新，不能替代它们。

当前 Stage2 起点是 A5 epoch155：

```text
outputs/stage2_pta_fs_dit_a5_full103k_standardv2_clean_20260710T091344Z/epoch_155.ckpt
```

## 2. 已确认的事实

### 2.1 Stage2 没有明显过拟合，但策略分布偏窄

- A5 相比官方 ReCogDrive Stage2，固定协议下 v1 PDMS `+0.01361`、v2 EPDMS `+0.00770`。
- epoch100--200 的 navtest 没有持续下降，L1 仍缓慢改善；epoch155 位于稳定平台期。
- SNSAD v2 的 1,024-scene、K=32 评估显示 A5 支持邻域精度更好，但支持 recall、effective-mode
  coverage 和相对 dispersion 没有变宽。
- “分布窄”表示 Stage3 的有效探索半径和信用分辨率更重要，不等同于 train-set overfit。

### 2.2 Stage3 退化是安全尾部与物理轨迹漂移，不是整体 reward 不学习

- R4 step300 的多数样本 PDMS 上升或持平，但约 8.25% 样本出现 NC/DAC/TTC 回退；少数负尾部
  抵消了多数 progress 收益。
- 策略终点纵向明显前移，train rollout scalar 与 EP 上升，但 held-out NC/TTC 和 v2 extended
  comfort 下降。
- exact reverse-transition KL 约束 diffusion transition，不等价于 decoded trajectory trust。
  Stage2 L1 为 `0.28334`，多个 300-step LFP 对照升至 `0.44--0.62`。

### 2.3 safety gate 必须保留，但在当前 on-policy 分布中接近饱和

- LFP 仍先要求 NC、DAC、GT-relative DDC，NAVSIM v2 额外要求 TLC；只有 feasible rollout
  可进入安全域内 Pareto front。
- 首批 train feasible 约 `99%`，NC 基本饱和，主要失败来自 DAC/DDC。
- mixed group 的 unsafe rollout 有明确负 advantage；all-infeasible rescue 也不授予正 credit。
- 因而当前问题不是“unsafe 没被罚”，而是 sampled train safety 信号太稀疏，不能约束共享 DiT
  更新后的 held-out deterministic tail。

### 2.4 v3 是可行候选库，但未校准为多模态 teacher 分布

- 全量 103,288 scene 的 selected support 均通过当前 evaluator/几何门控，`58.51%` 场景的
  cache-best 是非 GT；这证明轨迹池有可用模式，但不能证明它们适合作为直接监督 teacher。
- reward 标签过度饱和：`75.81%` 场景 best support reward 等于 1，5,000-scene
  抽样中 `35.16%` 场景所有 support reward 都等于 1。
- support 有平均 `4.91` 个几何模式，但近邻 ADE 均值仅 `0.436m`，存在明显模式内冗余。
- 现有 adaptive beta 使用 source entropy，而 source entropy 与 trajectory mode capacity 的
  Spearman 相关为 `-0.263`。A5 名义采样 `3.821` 个 target，effective target count 仅
  `1.842`，最终 diffusion loss 的 GT mass 为 `87.29%`。
- cache-best 非 GT 相对 GT 的 EP 增益均值 `+0.16063`，而 TTC 已饱和；其 GT-relative ADE
  均值 `2.443m`、FDE 均值 `4.867m`，主要来自 external/failure-expand/progress generator。
- 因此 Stage2 的基础性能是正向的，但“学到宽、support-aligned distribution”的监督目标没有
  真正成立。同一 v3 archive 上的 mode/density-balanced 对照已经完成：高 mode mass 能显著
  提高 recall 和 width，证明 archive 中存在可学习多样性；但 `beta>=0.45` 明确损害安全、
  PDMS 和 L1。当前可接受边界为 `beta=0.25`，其相对 legacy D0 的 SNSAD `+0.00589`、
  width ratio `+0.00925`，PDMS `-0.00143` 且 95% CI 跨 0。

### 2.5 Stage2 数据问题的准确定位

- v3 是有效候选库，不是自然校准的目标概率分布；archive quota/source frequency 不能直接作为 mode mass。
- 旧 v3 的 `anchor_distance` 误把 candidate 0 当 GT，而 candidate 0 只有 `66.23%` 是 GT；新构建
  和 runtime 读取均已修为真实 GT-relative distance。
- legacy sampler 使 diffusion loss 的实际 GT mass 达到约 87.3%，解释了 A5 的窄分布。
- 把 GT mass 一次降到约 56% 可以显著学宽，但会产生共享 DiT 的质量/安全干扰；延长到
  累计 1,000 step、降低 sampler variance 都不能消除该退化。
- `beta=0.15` 退回更高 GT mass，保住 PDMS 但没有新增多样性；`beta=0.25` 是首个同时具有
  显著 width/SNSAD 增益且 PDMS 差异未显著的点。
- 纯 GT continuation 在相同 FS-Norm/PTA 下，相对 legacy multi-support 得到 NC `+0.00416`、
  TTC `+0.00952`、L1 `-0.04467`，但 EP `-0.00636`。这排除了 FS-Norm/PTA 是主因。
- 每 scene 只训练唯一 cache-best 后，相对纯 GT 得到 EP `+0.01981`，但 PDMS `-0.01622`、
  NC `-0.02220`、TTC `-0.04972`、L1 `+0.22422`。这排除了同一步多目标平均是唯一原因。
- 因此下一步不是删除 FS-Norm/PTA，也不是把 cache-best 固化为 teacher，而是把模式去重、条件
  可学习性和实际 residual/gradient budget 接入渐进式 support exposure。

### 2.6 Stage2 的训练方式直接决定 Stage3 难度

Stage3 需要的不是最大宽度，而是局部可优化的 policy prior：高 feasible rollout ratio、足够的
组内 reward spread、可解释模式和连续的 decoded-trajectory 响应。当前 Stage2 的静态 beta 只
控制 target probability，不控制难 target 的实际残差影响；A5 epoch155 已达到当前分布的 loss
floor，因此继续同配方训练不能自动获得更合适的 RL prior。

`M=1 mode-balanced beta=0.25` 双 seed 对照已完成。相对同 seed M=4，NAVSIM v1 PDMS
`-0.00090` 且区间跨 0，L1 `-0.00719`；SNSAD `-0.00240`、recall `-0.00298`、width
`-0.00496`，区间均低于 0。跨 step 单模式学习略微降低轨迹误差，但同时收窄分布，证明并行
多目标平均不是主要退化原因，目标分布的期望偏差更重要。

进一步静态审计发现，既有 mode-balanced 路径仍继承 legacy reward shortcut：约 `51.4%` scene
因 cache-best reward margin 被提高 beta，且声明为 true 的 scene normalization 没有生效，scene
loss mass 均值被放大到 `1.111`。这与 cache-best 的 progress 偏差叠加。代码已仅对新
mode-balanced 路径改为 capacity-only、scene-uniform，legacy 路径保持不变，并完成独立双 seed
短训验证。

修正路径的 2-seed 训练已显示：non-GT target mass 约 `14.8%`，但 residual-mass proxy 约
`37.7%`，non-GT/GT diffusion loss ratio 约 `8.7`。这给出了 Stage2 训练失衡的直接机制：
概率质量远小于实际优化影响。后续 support curriculum 必须同时调度 mode probability 和
residual/gradient budget，不能只退火 beta。

其 NAVSIM v1 2-seed 配对结果相对旧 mode-balanced 为 PDMS `+0.00073`、NC `+0.00130`、
TTC `+0.00288`、L1 `-0.00914`，置信区间均支持改善；EP `-0.00121`。这验证了 legacy
progress shortcut 确实是可复现的退化来源，而不是纯代码审美问题。与此同时，SNSAD
`-0.01081`、support recall `-0.01832`、width ratio `-0.01527`、pairwise ADE `-0.01603m`，
置信区间均低于 0，precision `+0.00076`。旧 shortcut 的确扩宽了策略，但方式是过度优化
progress-biased hard targets；capacity-only 则回到更精确、更安全但更窄的分布。下一版需要显式
约束安全下的模式覆盖，而不是在两个端点之间继续手调静态 beta。

基于上述机制证据，已实现默认关闭的 paired residual-budget prototype：同 scene 的候选共享
noise/timestep，以 `weight * sqrt(epsilon_mse)` 标定 non-GT residual mass，并在保持 scene loss
mass 不变的前提下将其逐 scene 限制到 `0.38`。A5 epoch155 的 10-step smoke 中，non-GT
exposure 为 `0.1906`，residual mass 从 `0.4811` 降至 `0.2897`，budget 在 `65.63%` scene
生效，unanchored ratio 为 0，所有 loss finite。该结果仅证明机制按定义工作，尚未证明最终
PDMS/EPDMS/SNSAD 改善；完整结果见 `docs/Stage2_V3_Target_Distribution_Study.md`。

完整实验表、置信区间和路径见 `docs/Stage2_V3_Target_Distribution_Study.md`。

## 3. 已完成的关键因果对照

| 对照 | 结果 | 决定 |
|---|---|---|
| 旧 Pareto GRPO v2 完整配方迁移 | 当前 PTA+FS Stage2 上 10 step 即明显下降 | 不原样恢复 LR=1e-4/BC/Core 组合 |
| KL 0.005 vs 0.02 | 0.02 显著减轻 NC/TTC/EC 和 L1 退化，但仍低于 A5 | 0.02 作为最低 trust 基线 |
| FS-aware rollout covariance | 排除“完全采不到可比较候选”，但单独加噪不能改善 navtest | 保留标定工具，默认关闭 |
| scene-group std vs DDP-global std | group std 在 PDMS/EPDMS/NC/TTC/L1 上更差 | 保留 scene mean + global std |
| quality/reference Pareto gate | 首批 credit 几乎不变 | 不进入主配置 |
| safety-first frontier | sampler/EMA 正常，短 smoke 未损害 progress | 作为课程基线 |
| support-capacity frontier | v1 PDMS 相对 safety-only `+0.004758`，CI 不跨 0 | 有效缓解退化，仍默认关闭等待复验 |
| KL 0.02/0.05/0.10 | L1 `0.444/0.386/0.360`，但 PDMS `0.8716/0.8710/0.8693` | KL 约束漂移但不解决 credit trade-off |
| Stage2 target D0 vs mode balance | 0.25 提高 SNSAD/width 且 PDMS CI 跨 0；0.45 以上单调退化 | 保留 legacy 默认，mode 实验上限改为 0.25 |
| Stage2 GT-only vs multi-support | GT-only 显著改善 NC/TTC/L1，但降低 EP | support 带来 progress，同时引入安全和可学习性代价 |
| Stage2 single cache-best vs GT-only | EP 提高，但 PDMS/NC/TTC/L1 大幅退化 | cache-best 不能直接作为 teacher；问题不只来自 M>1 |
| Stage2 M=1 vs M=4 | M=1 改善 L1，但 SNSAD/recall/width 显著下降，PDMS 无显著收益 | 多目标同步平均不是主因；不能靠减小 M 修复 |
| Stage2 capacity-only vs reward shortcut | 安全/PDMS/L1 改善，同时 SNSAD/recall/width 显著下降 | 将安全锚定与多样性释放拆成动态约束课程 |

优势归一化的准确表述是：每个 scene 内减 feasible mean，再用 DDP-global std 缩放。它不是
跨 scene 的 batch mean 排序。纯 scene-group std 会放大低 reward-spread 场景中的噪声。

## 4. 支持对齐多样性与容量课程

### 4.1 SNSAD v2

`docs/Support_Aligned_Policy_Diversity_Evaluation.md` 定义模型无关的支持对齐评价：

- density-corrected support precision/recall；
- kernel effective-mode ratio；
- relative dispersion calibration；
- 对确定性单轨迹、GT-only diffusion 和显式多策略模型使用同一接口。

它避免用 pairwise ADE 奖励随机发散，也通过 scene normalization 避免天然多模态场景占便宜。

### 4.2 Capacity-normalized learning frontier

旧实验从 v3 selected positive support 缓存了一个观测到的参考宽度。它不是 scene 固有容量；
在固定 proposal、evaluator、初始化 policy 和阈值下没有合格替代模式时，`K_i=0` 是合法结果。
当前定义为：

```text
mode_capacity = 1 - 1 / reference_mode_count
coverage_ratio = clip(
    (policy_group_pairwise_ADE + 0.05)
    / (support_pairwise_ADE + 0.05),
    0, 1,
)
coverage_gap = mode_capacity * (1 - coverage_ratio)
frontier_energy = base_BPAE * (
    1 + 0.05 * coverage_gap * all_feasible
)
```

coverage gap 只调制下一 epoch sampler，不进入 advantage 或 policy loss，并且不能在
`base_BPAE=0` 时凭空生成课程能量。旧 v3 cache 有 103,288 scenes，support ADE 均值
`1.36082m`，mode count 均值 `4.9077`；它由于旧构建会填充候选，不能继续作为新版无配额数据的
容量事实，必须由 v4 archive 重建。

300-step 配对中，它使 frontier energy 与 mode capacity 的 Spearman 相关从 `0.0965` 提高到
`0.2327`。epoch1 实际采样的 multimodal ratio 提高 `0.49` 个百分点，没有 sampler collapse。
该历史实验使用的是加性 bonus，下面结果保留为方向性证据，不能证明加性公式正确。

固定 seed0 全量 v1：

| 配置 | PDMS | NC | TTC | EP | L1 |
|---|---:|---:|---:|---:|---:|
| A5 Stage2 | 0.873111 | 0.980804 | 0.942907 | 0.820586 | 0.283344 |
| safety-only | 0.866870 | 0.966345 | 0.908634 | 0.846880 | 0.464194 |
| diversity-capacity | 0.871628 | 0.969764 | 0.916955 | 0.847800 | 0.444137 |

capacity 相对 safety-only 的 PDMS delta 为 `+0.004758`，scene-cluster bootstrap 95% CI
`[+0.002472,+0.007083]`；NC、TTC、DDC 和 L1 也有统计支持的改善。相对 A5 的 PDMS 差
`-0.001483`，CI 跨 0，但 NC/TTC/L1 仍明确更差。因此主动学习方向得到支持，Stage3 的
physical trust 与 progress-biased credit 仍未解决。

后续在无配额 v4 的 128-scene 固定候选池上发现了加性公式的反例：122 个多模态 scene 中有 70 个
`base_BPAE=0`，加性 gap 会给这 70 个零策略梯度 scene 制造非零 priority。按当前 sampler 的
`sqrt(priority)` 和 20% uniform mixture 重放，credit-ready 采样占比从 68.0% 降至 44.5%，
零 BPAE scene 的采样占比从 31.0% 升至 54.0%。乘性调制分别保持为 68.1% 和 30.9%。因此当前
实现改为乘性公式：多样性只在已有双向 Pareto credit 的学习前沿之间调整顺序。

为保留因果消融，代码提供两个互斥且显式命名的 priority mode：主方案
`credit_multiplicative` 使用上式；`additive_preservation` 复现旧公式，只用于检验零优势 scene
上的 frozen-Stage2 KL 是否能保持已经学到的多样性。后者不能被解释为学习新策略的 frontier，
也不作为默认配置。

## 5. 当前原因排序

1. **首要：decoded trajectory trust 不足。** transition KL 不能阻止轨迹在物理空间显著漂移。
2. **首要：安全饱和后的 credit 偏向 progress。** Pareto gate 逻辑正确，但安全域内标量仍更容易
   奖励纵向激进轨迹；NAVSIM v2 comfort 不在 v1 reward 中。
3. **Stage2 多模态监督错位。** v3 轨迹库有可行模式，但 reward 饱和、progress-biased winner、
   远距离 teacher、模式内冗余和静态 beta 使它没有变成条件可学习的 policy prior。单变量实验
   已确认 mode mass 存在安全边界：0.25 有温和多样性收益，0.45 以上干扰明显。
4. **次要：课程目标曾与核心多样性目标错位。** 旧 BPAE 更偏低 reference reward；旧加性容量
   bonus 还会优先采样零梯度 scene。容量课程提供过正向方向性证据，但新版必须用乘性调制复验。
5. **不是主因：优势使用 global std。** 严格对照已否定恢复 scene-group std。
6. **不是主因：Stage2 明显过拟合、FS-Norm 或 PTA。** navtest 趋势、GT-only 对照和 A5 相对
   Official 的 v1/v2 改善均不支持。更准确的问题是 support teacher 分布缺少概率和可学习性校准。

## 6. 代码与复现入口

核心代码：

```text
navsim/agents/recogdrive/stage3_lfp_grpo.py
navsim/agents/recogdrive/stage3_policy_geometry.py
navsim/agents/recogdrive/stage3_diversity_curriculum.py
navsim/agents/recogdrive/support_aligned_diversity.py
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
```

实验入口：

```text
scripts/training/sg_fps/run_train_lfp_grpo_fs_probe.sh
scripts/training/sg_fps/run_lfp_fs_calibration_matrix.sh
scripts/stage3/build_lfp_diversity_capacity_cache.py
scripts/evaluation/run_recogdrive_support_aligned_diversity_sharded.sh
scripts/evaluation/analyze_lfp_frontier_alignment.py
```

完整证据、置信区间和输出路径见：

```text
reports/recogdrive_stage3/lfp_grpo_r4_evidence_review_20260712.md
```

## 7. 下一步最小实验

1. 按 GT-relative distance、source holdout 和非饱和 safety gain 分层支持集，测试哪些 support
   具有可迁移的条件可学习性；不再用 cache-best 作为 teacher oracle。
2. 设计 anchor、local-support、mode 三阶段课程，用 target loss ratio、residual budget 和 held-out
   safety 控制 beta；课程 priority 只控制 exposure，不乘入 loss。
3. 加入动态 non-GT residual-mass budget，在旧 shortcut 与 capacity-only 之间搜索 Pareto 点；
   晋级条件同时约束 NC/TTC、L1、SNSAD 和 support recall。
4. 用无配额 v4 重建 capacity cache，对乘性 capacity frontier 复验至少一个 seed，并补 NAVSIM
   v2 EPDMS；在通过前保持默认关闭。
5. 单变量测试 reference-relative decoded trajectory trust，不能同时改 reward、KL 和 curriculum。
6. trust 有效后，再测试 v1/v2 分离的 quality objective；不能把 NAVSIM v2 与历史 Pareto GRPO v2
   混为一谈。
7. 最终候选必须同时报告 PDMS、EPDMS、NC/TTC/EC、L1、SNSAD 和 Stage3 reward-spread probe，
   不以单一训练 reward 晋级。

本轮没有启动完整 Stage3 长训练，也没有继续占用 `training-rl-zt3`。容量课程的 v2 评估因该
资源已释放而未执行，不能从 v1 结果外推 v2 改善。
