# Stage2 v3 Target Distribution Study

更新时间：2026-07-13 06:17 UTC

## 1. 研究问题与当前结论

本轮把 Stage2 问题拆成四层，避免用一句“数据好或坏”掩盖不同含义：

1. **候选可行性**：轨迹本身是否通过 evaluator 与几何筛选；
2. **teacher 分布校准**：候选频率和分数能否解释为给定场景下的策略概率；
3. **条件可学习性**：仅凭当前场景条件，DiT 是否能稳定预测这些目标，而不是依赖不可见变量；
4. **Stage3 readiness**：采样策略是否同时具有较高可行率、有效模式覆盖和可分辨的组内 reward。

现有证据表明：v3 作为**候选可行轨迹库**是有价值的，但作为**直接监督 teacher 分布**并不合格。
问题同时存在于数据语义和训练方法，不能只归因于 FS-Norm、Planning Token Adapter，或“同场景
多目标平均”中的任何单一因素。

原版 ReCogDrive Stage2 的 GT-only 监督条件熵低、目标唯一，因此容易拟合。它的随机采样宽度
并不自动等价于可控的行为多模态。我们的目标也不是最大化无约束宽度，而是学习一个围绕可行
Pareto 支持集、可由场景解释、且能被 Stage3 稳定优化的分布。

## 2. v3 Archive 全量审计

全量 archive 有 103,288 个 scene、约 103 万条 selected support：

- 每个 scene 都有 evaluator-valid selected candidate；
- candidate 数均值 `23.284`，selected 数均值 `10.279`；
- `58.505%` scene 的 cache-best candidate 是非 GT；
- `91.125%` scene 至少选择了一条 external support；
- `96.078%` scene 的 selected set 来自至少两个 source；
- support 几何模式数均值约 `4.91`，但模式内有明显冗余；
- `75.81%` scene 的 best support reward 为 1；5,000-scene 抽样中，`35.16%` scene 的所有
  selected support reward 都为 1。

selected tag 主要由 `diversity_max`、`best_pdms` 和 `vector_pareto` 构成。标签表示构建时的
quota 和选择理由，不是行为出现概率。因此 archive record frequency 不能直接当成监督概率。

### 2.1 Cache-best 的系统偏差

在 60,429 个 cache-best 非 GT scene 中：

- 相对 GT 的 cached reward 增益均值为 `+0.09479`；
- EP 增益均值为 `+0.16063`，P50 `+0.1403`，P90 `+0.3120`；
- TTC 增益为 0，因为该项基本饱和；
- source 主要是 external `45,575` 和 progress `14,453`；
- 最大来源是 `ddv2:failure_expand_0`、`progress_endpoint`、`driveor:failure_expand_0`。

真实 GT-relative 几何距离并不小：cache-best 非 GT 的 trajectory ADE 均值 `2.443m`、P50
`2.073m`、P90 `4.555m`；FDE 均值 `4.867m`、P90 `9.535m`。这说明 cache-best 往往是
更激进、更远离 GT 的 progress winner，而不是局部、容易学习的替代模式。

### 2.2 Anchor distance 正确性问题

旧 v3 构建代码把 `candidates[0]` 当作 anchor，但全量数据中 candidate 0 只有 `66.227%` 是 GT。
因此约三分之一 scene 的 `anchor_distance` 语义错误。

修复后：

- 新 archive 始终相对真实 GT trajectory 计算距离，并写入 `anchor_distance_reference`；
- 旧 v3 在 runtime load 时，只要 selected candidates 中存在 GT，就重新计算 GT-relative distance；
- 没有 GT 时才兼容回退旧字段；
- 当前 mode-balanced q 不依赖该字段，所以修复不改变已有对照的 loss；它防止后续 trust curriculum
  和 archive 重选继续使用错误距离。

审计文件：

```text
outputs/stage2_mode_balance_multiseed_analysis_20260713T0500Z/
  support_v3_teacher_bias_audit.json
  support_v3_gt_distance_audit.json
```

## 3. Target-distribution 对照

所有短训都从 A5 epoch155 checkpoint 开始，固定全局 batch 64、LR `1e-5`、300 optimizer
steps。除明确列出的变量外，FS-Norm、PTA、auxiliary、archive 和 DDIM 设置相同。

Mode-balanced 的预期定义使用 SNSAD 轨迹距离的 kernel density，以 inverse-density 消除近重复
候选的额外票数：

```text
beta(scene) = beta_max * (1 - 1 / effective_mode_count(scene))
q = (1 - beta) * q_anchor + beta * q_pareto
```

`M=4` 版本使用 systematic resampling，保持 q 的无偏期望并降低 multinomial count variance。

复核实际代码后发现，以下表格中的既有 mode-balanced 训练仍继承了两个 legacy reward shortcut：

- 当 `best_selected - GT >= 0.05` 时，把 beta 强制提高到至少 0.4，再由 `beta_max` clamp；
- `dpsi_scene_normalize_weights=True` 未实际生效，improver scene 仍按 reward margin 放大总 loss mass。

在 `beta_max=0.25` 训练中约 `51.4%` scene 命中 improver，beta 均值为 `0.194`，而纯
`0.25 * mode_capacity` 约为 `0.158`；scene 总权重均值为 `1.111`。因此已有实验严格表示的是
“mode density + legacy progress shortcut”，不是纯几何 mode balance。代码现已仅对新
`mode_balanced` 路径修为 capacity-only、scene-uniform；`legacy` 保持历史行为。修正后的独立对照
尚未完成，下面结果不追溯改名或混用。

### 3.1 M=4 mode mass 剂量

| 配置 | sampled GT mass | target ESS | sampled unique ratio |
|---|---:|---:|---:|
| D0 legacy | 0.8735 | 1.856 | legacy sampler |
| beta 0.15 | 0.8851 | 1.245 | 0.353 |
| beta 0.25 | 0.8180 | 1.465 | 0.420 |
| beta 0.45 | about 0.700 | about 2.00 | about 0.54 |
| beta 0.75 | about 0.56 | about 3.20 | higher |

固定 seed0 NAVSIM v1：

| 配置 | PDMS | NC | DAC | TTC | EP | DDC | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 epoch155 | 0.873111 | 0.980804 | 0.953782 | 0.942907 | 0.820586 | 0.980269 | 0.283344 |
| D0 legacy | 0.870513 | 0.978909 | 0.952134 | 0.939117 | 0.820878 | 0.979527 | 0.292797 |
| beta 0.15 | 0.869893 | 0.980269 | 0.950816 | 0.942000 | 0.817506 | 0.979980 | 0.287590 |
| beta 0.25 | 0.869079 | 0.978827 | 0.951145 | 0.938540 | 0.819112 | 0.979692 | 0.305483 |
| beta 0.45 | 0.865694 | - | - | - | - | - | 0.338254 |
| beta 0.75 | 0.862283 | 0.973019 | 0.948509 | 0.926018 | 0.820043 | 0.976726 | 0.376257 |

`beta=0.25` 的 3-seed pooled scene bootstrap 相对 D0 的 PDMS 均值差约 `-0.00129`，置信区间
略跨 0；SNSAD 则稳定提高约 `+0.00550`。因此它是可复现的 diversity-quality Pareto 交换，
不是对 D0 的全面支配。`beta>=0.45` 随 mode mass 增加单调损害安全和 L1，累计训练 1,000
steps 仍未消除差距。systematic 与 multinomial 近似相同，排除了采样 count variance 是主因。

### 3.2 Support-aligned diversity

协议为固定 1,024 个 navtrain scene、每 scene 32 次 5-step DDIM、相同采样 seed 和 v3 support。

| 配置 | Precision | Recall | KEMR AUC | Width ratio | Pairwise ADE m | SNSAD v2 |
|---|---:|---:|---:|---:|---:|---:|
| Official Stage2 | 0.905771 | 0.617563 | 0.504625 | 0.302735 | 0.142481 | 0.485336 |
| A5 epoch155 | 0.984085 | 0.639157 | 0.498695 | 0.221943 | 0.087649 | 0.464131 |
| D0 legacy | 0.984083 | 0.652396 | 0.499506 | 0.227664 | 0.093886 | 0.469284 |
| beta 0.25 | 0.984095 | 0.658400 | 0.501027 | 0.236909 | 0.103358 | 0.475173 |
| beta 0.75 | 0.972458 | 0.765235 | 0.534785 | 0.373021 | 0.270971 | 0.559025 |

Official Stage2 更宽但 support precision 明显更低，说明“宽”里含有较多不对齐噪声。A5 精度
更高但分布偏窄。`beta=0.75` 证明 v3 中存在可学习多样性，但无约束释放会以单样本安全为代价。

## 4. 三个关键因果对照

### 4.1 纯 GT continuation

在同一个 A5 起点上关闭 support/DPSI，保留 FS-Norm、PTA 和新 auxiliary，运行 2 seeds：

| 指标 | GT-only 相对 D0 pooled delta | scene bootstrap 95% CI |
|---|---:|---:|
| PDMS | +0.001374 | [-0.000743,+0.003457] |
| NC | +0.004160 | [+0.002792,+0.005628] |
| TTC | +0.009516 | [+0.007355,+0.011709] |
| EP | -0.006361 | [-0.008572,-0.004099] |
| DDC | +0.001936 | [+0.001146,+0.002759] |
| L1 | -0.044673 | [-0.048887,-0.040630] |

同样的 FS-Norm/PTA 在 GT-only 下提高安全和轨迹精度，因此它们不是当前退化的主要原因。
multi-support 的实际作用是保留更高 EP，同时牺牲 NC/TTC/L1。

### 4.2 每场景唯一 cache-best teacher

该对照完全去掉同一步的多目标平均：每个 scene 只训练 archive 中 reward 最高的一条 selected
support，`58.7%` target 是非 GT。2-seed pooled 结果相对 GT-only：

| 指标 | single-best minus GT-only | scene bootstrap 95% CI |
|---|---:|---:|
| PDMS | -0.016216 | [-0.021452,-0.011199] |
| NC | -0.022203 | [-0.026395,-0.018207] |
| TTC | -0.049720 | [-0.055880,-0.043621] |
| EP | +0.019806 | [+0.014283,+0.025245] |
| DDC | -0.007291 | [-0.009057,-0.005594] |
| L1 | +0.224216 | [+0.210984,+0.237609] |

相对未继续训练的 A5 epoch155，single-best 仍为 PDMS `-0.020075`、NC `-0.022182`、TTC
`-0.046960`、EP `+0.011195`、L1 `+0.186187`，全部关键区间不跨 0。

该实验排除了“同场景 M 条轨迹平均是唯一问题”。即使目标唯一，cache-best teacher 仍系统性
推进更远，并且难以从场景条件稳定泛化。

### 4.3 条件可学习性和 loss 难度

300-step 两个 seed 的训练均值高度一致：

| 配置 | diffusion loss | trajectory aux | full-x0 L1 |
|---|---:|---:|---:|
| GT-only | about 0.0194 | about 0.0093 | 0.16 to 0.18 |
| D0 multi-support | about 0.0331 | about 0.0300 | 0.22 |
| single cache-best | about 0.0682 | about 0.0787 | 0.30 |

cache-best 的 diffusion loss 约为 GT-only 的 3.5 倍。A5 完整 200 epoch 中，diffusion loss 从
epoch0 的 `0.2037` 降到 epoch155 的 `0.0328`、epoch199 的 `0.0294`，selected GT ratio 和
beta 基本不变。这说明 A5 已在当前静态目标分布上接近 loss floor，不能用“训练时间不够”解释。

当前 `beta` 控制的是目标概率质量，不是实际优化影响。对于 epsilon MSE，较远、较难 target 的
残差更大，即使概率较小，也可能贡献更大的梯度。代码已新增 GT/non-GT target loss、loss ratio、
target weight ratio 和 residual-mass proxy 诊断，以直接监控该不等价性。

## 5. 训练方法为什么会影响 Stage3

Stage3 RL 并不只需要 Stage2 有较高 PDMS，还需要一个合适的 policy prior：

- rollout 的大多数样本在 NC/DAC/DDC hard guard 内；
- 同一 scene 中仍有可比较的 progress/quality 差异；
- 支持模式是场景可解释的，不是由不可见 generator source 决定；
- policy 对小梯度更新有连续响应，不会把少量 positive credit 扩散成安全尾部漂移；
- exact KL 附近存在足够的正负 advantage，而不是全相同或全不可行。

当前 Stage2 的问题是：GT anchor 很稳定，但非 GT support 中混入了远距离、progress-biased、条件
难预测的 teacher。Stage3 从这种 prior 出发，要么采样过窄、组内无差异，要么扩大探索后迅速
越过物理安全边界。仅增加 Stage3 KL 或调整 advantage normalization 不能修复 Stage2 prior 的
这类结构性缺口。

## 6. 训练机制对照

`M=1 mode-balanced beta=0.25` 已按 2 seeds 完成训练和评估。该对照与既有 M=4 使用相同的
旧 reward shortcut，只改变每个 scene 每个 step 反向传播的目标数，因此仍能严格区分两个解释：

1. 若 M=1 明显优于 M=4，则同一步共享条件下的多目标梯度平均是重要干扰源；
2. 若 M=1 仍退化，则主要问题是 q 中非 GT teacher 的条件可学习性和概率质量，而不是并行平均。

输出：

```text
outputs/stage2_mode_balance_beta025_m1_seed0_20260713T053334Z
outputs/stage2_mode_balance_beta025_m1_seed1_20260713T053334Z
```

NAVSIM v1 已完成。M=1 相对同 seed M=4 的 pooled scene bootstrap：

| 指标 | M=1 minus M=4 | 95% CI |
|---|---:|---:|
| PDMS | -0.000896 | [-0.002165,+0.000375] |
| NC | -0.000577 | [-0.001142,+0.000000] |
| TTC | +0.000165 | [-0.000773,+0.001097] |
| EP | -0.001209 | [-0.002470,+0.000016] |
| L1 | -0.007191 | [-0.008195,-0.006200] |

M=1 稳定改善 L1，但 PDMS、NC、TTC、EP 的区间均不能支持正向收益。SNSAD pooled 结果为：

| 指标 | M=1 minus M=4 | 95% CI |
|---|---:|---:|
| Precision | -0.000618 | [-0.001088,-0.000130] |
| Recall | -0.002979 | [-0.004675,-0.001290] |
| KEMR AUC | -0.000941 | [-0.001462,-0.000394] |
| Width ratio | -0.004961 | [-0.009869,-0.001440] |
| Pairwise ADE | -0.003347m | [-0.005693,-0.000753] |
| SNSAD | -0.002395 | [-0.003481,-0.001343] |

因此 M=1 的 L1 改善确实伴随轻微但显著的分布收窄。同步 M=4 平均增加了一点轨迹误差，但不是
策略质量退化的主要原因；带偏的期望 q 才是更重要的问题，不能靠继续调 M 解决。

紧随其后的对照使用修正后 capacity-only、scene-uniform 的 mode-balanced 路径。它检验之前的
退化有多少来自 progress-improver scene 被同时提高 beta 和 scene loss mass。

该对照的 2-seed 300-step 训练诊断已经完成：

| 诊断 | seed0 | seed1 |
|---|---:|---:|
| beta mean | 0.157989 | 0.158495 |
| scene weight mean | 1.000000 | 1.000000 |
| non-GT target weight ratio | 0.146953 | 0.149727 |
| non-GT residual-mass ratio | 0.373352 | 0.380623 |
| non-GT / GT diffusion loss | 8.577 | 8.854 |
| GT target loss | 0.019573 | 0.019604 |
| non-GT target loss | 0.134365 | 0.133841 |

这是目前最直接的训练机制证据：non-GT 只有约 `14.8%` 的 target mass，却贡献约 `37.7%` 的
residual-mass proxy，单 target loss 是 GT 的约 `8.7` 倍。因此 beta 不能被解释成实际梯度预算。
下一版 curriculum 除了控制 mode probability，还必须限制 hard support 的 residual/gradient mass，
并在其 loss ratio 随学习下降后逐步释放。

NAVSIM v1 相对旧 mode-balanced reward-shortcut 路径的 2-seed pooled 配对结果：

| 指标 | capacity-only minus old mode | 95% CI |
|---|---:|---:|
| PDMS | +0.000728 | [+0.000001,+0.001494] |
| NC | +0.001298 | [+0.000637,+0.002000] |
| TTC | +0.002884 | [+0.001947,+0.003867] |
| EP | -0.001212 | [-0.002030,-0.000411] |
| L1 | -0.009140 | [-0.010005,-0.008287] |

结果跨 seed 一致，并与删除的 progress shortcut 方向吻合：停止放大 reward improver scene 后，
安全、PDMS 和轨迹精度提高，代价是少量 EP。相同 1,024 scene、K=32 的 SNSAD 配对结果为：

| 指标 | capacity-only minus old mode | 95% CI |
|---|---:|---:|
| Precision | +0.000763 | [+0.000574,+0.000975] |
| Recall | -0.018322 | [-0.020941,-0.015918] |
| KEMR AUC | -0.002948 | [-0.003364,-0.002561] |
| Width ratio | -0.015269 | [-0.020048,-0.011659] |
| Pairwise ADE | -0.016025m | [-0.018248,-0.013871] |
| SNSAD | -0.010805 | [-0.012089,-0.009527] |

配对统计保存在
`outputs/stage2_mode_balance_multiseed_analysis_20260713T0500Z/capacity_only_minus_reward_shortcut_mode025_snsad.json`。

修正路径提高 support precision，却显著降低 recall、有效模式覆盖和宽度。这说明旧 shortcut 的
确扩宽了策略，但扩宽机制是放大 progress-biased hard targets，并以安全和精度为代价；简单删除
shortcut 也不是最终方案。训练目标必须把安全锚定与多样性释放拆开：先满足 held-out safety 和
residual budget，再按 coverage gap 逐步释放可学习模式。

### 6.3 Paired residual-budget prototype

针对“约 `14.8%` target mass 产生约 `37.7%` residual mass”的失衡，已实现默认关闭的
训练机制，而不是继续把 `beta` 当作梯度预算：

1. 同一 scene 的 M 个 target 共用 diffusion noise 和 timestep，降低 target 难度比较的采样方差；
2. 使用 detached `weight * sqrt(epsilon_mse)` 作为 residual-gradient proxy；
3. 每 scene 统一缩放 non-GT 权重，使其 proxy mass 不超过配置上限，再恢复原 scene loss mass；
4. 同一 effective weight 同时作用于 diffusion、trajectory 和 feasibility auxiliary；
5. 每个 active scene 必须含真实 GT anchor。v3 support filter 丢失 GT 时只在新路径追加原 archive
   GT，并记录注入比例；新功能关闭时不改变候选或权重。

入口为：

```text
scripts/training/sg_fps/run_train_pta_fs_dit_residual_frontier.sh
```

默认实验点为 mode mass `beta_max=0.35`、residual cap `0.38`、M=4。A5 epoch155 起点上的完整
navtrain cache、单卡 10 optimizer-step smoke 已通过：

| 诊断 | 10-step mean |
|---|---:|
| loss | 0.024461 |
| non-GT exposure mass | 0.190625 |
| non-GT residual mass before budget | 0.481078 |
| non-GT residual mass after budget | 0.289742 |
| budget active scene ratio | 0.656250 |
| non-GT scale | 0.502973 |
| unanchored scene ratio | 0.000000 |
| scene target weight sum | 1.000000 |
| planning adapter forward count | 1.000000 |

输出位于：

```text
outputs/stage2_residual_frontier_smoke_gtanchor_20260713T063730Z
```

该 smoke 只验证数值、权重守恒和调用链，不证明 NAVTEST/SNSAD 改善。抽到的 batch 中 archive-level
GT 注入比例为 0，因此不能用该短样本估计全 archive 的边缘条件频率。下一项因果实验仍应是
paired-only 与 paired+budget 的双 seed 300-step 对照，再按 v1/v2、L1、SNSAD 和 RL-readiness
联合晋级。

## 7. 下一版 Stage2 训练原则

本轮 M=1 和 capacity-only 对照均已闭环，residual-budget prototype 已通过短 smoke；在完成
双 seed NAVTEST/SNSAD 因果验证前不启动新的 200-epoch 全训。
当前最有依据的方向是渐进式、受 trust budget 约束的 support 学习，而不是固定 beta 从第一个
epoch 训练到底：

1. **Anchor phase**：先以 GT 或高置信局部 teacher 建立可行、低误差的 base policy；
2. **Local-support phase**：只释放 GT-relative 距离小、held-out evaluator 可靠、非饱和指标有真实
   改善的 support；
3. **Mode phase**：按几何模式聚类，每个模式选一个代表，跨 step 采样单 target；
4. **Learning frontier**：根据非 GT/GT loss ratio、residual budget、held-out safety 和 coverage gap
   动态增加或收回每场景的 beta；
5. **RL-readiness gate**：只有当 feasible rollout ratio、reward spread、正负 advantage energy 和
   SNSAD 同时达标，才把 checkpoint 交给 Stage3。

建议的约束不是把所有 support 拉回 GT，而是把“模式是否存在”“模式应该有多少概率”“模型当前
能否学它”分开决定：Pareto archive 提供可行模式，density correction 决定模式去重，learning
frontier 决定曝光顺序，residual/safety budget 决定当前可释放质量。

## 8. 当前判断边界

- 已否定：FS-Norm/PTA 是退化主因；训练不够久；sampler count variance；仅 M>1 才有问题；
  cache-best 可以直接作为最优 teacher。
- 已确认：v3 有可学习多样性；过高 mode mass 损害安全；GT-only 更容易且更安全；cache-best
  明显 progress-biased；旧 anchor distance 存在正确性 bug；M=1 会轻微收窄而不会修复质量；
  移除 legacy progress shortcut 会提高安全/精度但显著收窄分布；target probability 不等于实际
  residual/gradient budget。
- 尚待确认：auxiliary 是否放大 hard-target drift；按 GT 距离和 held-out safety 分层后，哪一类
  support 能在不损害 NC/TTC 的情况下提高有效模式覆盖；动态 residual budget 是否能位于旧
  shortcut 与 capacity-only 两个端点之间，形成更优 Pareto 点。

因此当前最准确的结论不是“数据本身不好”或“训练方法不好”二选一，而是：**候选库可用，但
teacher 分布缺少概率语义和 learnability calibration；静态多目标训练又没有按模型学习状态控制
这些难目标的实际梯度预算。**

## 9. SG-FPS v4 implementation update

The data-level cause has now been isolated and corrected. The v3 builder first reduced 33 internal
candidates to about 12 with a scalar AWAC score before combining them with external candidates. A
512-scene raw rebuild showed that 54.32% of the non-GT modes selected by the corrected contract had
been removed from the v3 candidate pool; 50.59% of scenes recovered at least one such mode.

The new `mode_pareto_v4` path removes that preselection, disables external mechanical expansion,
keeps one GT anchor, and selects at most three non-GT modes using hard safety, GT-relative trust
bounds, GT-inclusive epsilon-Pareto filtering, and SNSAD mode separation. Stage2 uses a fixed
per-scene mode mass, uniform probability per explicit mode, paired noise, and a residual-gradient
budget. Migrated v3 shadow archives are rejected by the production loader.

The complete contract, artifacts, promotion gates, and commands are documented in
`docs/SG_FPS_V4_Data_And_Training_Contract.md`.

## 10. Reachable Pareto data and gradient-budget conclusion

后续实验把“数据是否正确”“模型是否学得动”“采样分布是否变宽”拆成了三个独立问题。

### 10.1 数据侧修复

最终 v4 不再按 scalar reward 对 Pareto front 做最后一次 top-k，而是按场景内归一化的
`[EP, TTC, comfort]` 做最远点覆盖，再用轨迹 SNSAD 打破饱和指标的平局。每条非 GT teacher
还必须位于 A5 八次随机 rollout 的最近 SNSAD `0.50` 内。最终 128-scene 重构加入不扩增的
DDV2/DriveOR 原始 proposal 后，multi-mode ratio 为 95.31%，平均 2.047 条 non-GT mode，policy
reachability 为 mean 0.337 / p90 0.439 / max 0.484，严格 validator 为零错误。6 个 scene 合法地
保持 GT-only（`no_distinct_mode=1`、`no_pareto_candidate=5`），没有用低质量候选补数。

FS-Norm 实验同时确认：FS stats 是 checkpoint 的动作坐标系，不是随 teacher archive 更新的数据
统计。用新 v4 stats 直接微调 A5 会造成 representation reconstruction 明显漂移；保留 A5 原始
stats 后训练恢复正常。因此 fine-tune 默认允许 archive provenance 与 FS stats provenance 不同并
明确 warning，只有从头训练才要求两者一致。

### 10.2 可学习性不是问题，但原预算会饿死模式

同一 v4 数据上的 16-batch 固定随机性 replay 中，step100 相对 A5：total loss 平均下降
0.00356（15/16 batch 改善），non-GT denoising loss 下降 0.00617（14/16），trajectory loss
16/16 改善。teacher 与当前 policy 并未断开。

真正的问题是 residual budget 的反馈方向。名义 non-GT target mass 约 24%，cap=0.25 时实际
non-GT weight 只有约 8%。GT 先变容易后，relative residual ratio 上升，预算继续压低 mode
weight，形成“GT 越容易 -> mode 越少 -> 策略越窄”的正反馈。100-step rollout 的表现正好符合
该机制：precision 和 GT ADE 改善，但 dispersion ratio 下降 0.04795，pairwise ADE 下降
0.01141m，kernel mode coverage 也显著下降。

### 10.3 当前有效修复

在同一 128 scenes、同一初始化、同一 global batch 和 seed 上：

| 设置 | recall AUC vs A5 | F1 AUC | SNSAD | dispersion ratio | mean GT ADE |
|---|---:|---:|---:|---:|---:|
| cap 0.25 + trajectory/feasibility aux | +0.00151 | +0.00320 | -0.00304 | -0.04795 | -0.02790m |
| cap 0.45 + auxiliary | +0.00498 | +0.00454 | +0.00134 | -0.00487 | -0.01836m |
| cap 0.45 + no x0 auxiliary | +0.00722 | +0.00571 | +0.00321 | -0.00218 | -0.01386m |

最后一行的 recall、F1、SNSAD、GT ADE 的 paired bootstrap 95% CI 均不跨零，dispersion 变化
不显著。它说明提高可达 mode 的梯度预算能阻止收缩；trajectory/feasibility x0 auxiliary 在这个
多模态目标上是次要 mean-seeking 因素，移除后 support recall、kernel coverage、dispersion 和
SNSAD 均显著优于相同 cap 的 auxiliary 版本。

因此 v4 launcher 当前采用：`beta_max=0.25`、40-epoch exposure warmup、residual cap `0.45`、
trajectory/feasibility auxiliary 为 0，保留 epsilon prediction、FS-Norm、PTA 和 hard output bounds。
这不是以无约束宽度换安全，而是先通过数据的 safety/reachability gate 缩小 teacher hardness，再
给予通过 gate 的 Pareto mode 足够梯度。完整 103k rebuild 和长训仍必须经过 held-out NAVTEST、
support-aligned coverage 和 Stage3 BPAE 三重晋级，不能由该 128-scene train-subset probe 直接替代。

### 10.4 不使用逐 scene 候选配额

全量 token 覆盖不等于每个 scene 都要产生替代轨迹。新契约允许 `K_i=0`：这类 scene 只保留 GT，
不会用低质量、不可达或重复轨迹填满 top-k。archive 新增可重算的 `candidate_funnel`，逐级记录
proposal、evaluator-valid、trajectory-quality、GT trust region、1 个/要求数量的 policy witness、
teacher eligibility、与 GT 的模式分离、Pareto eligibility 和最终 pairwise-distinct mode 数。模式分离
必须先于 Pareto：否则近 GT 的高指标候选可能支配真正独立的候选，随后自己又被距离门槛丢弃，
导致增加 proposal source 反而损失 mode。GT-only scene 记录第一处空漏斗的
原因，但该原因只用于改进 proposal generator，不进入 target 权重，也不是单 scene promotion gate。

同一原则也用于 Stage2 curriculum：如果某个 scene 的全部 non-GT mode 都高于当前难度阈值，
该 scene 在这一 epoch 暂时只学习 GT；不会绕过阈值强行启用“最容易的一条”。阈值随后从
`0.45` 逐步升到 `1.0`，因此这是延迟暴露而不是永久丢弃。训练日志用
`dpsi_frontier_all_modes_deferred_scene_ratio` 单独记录这类暂时 GT-only scene。

这里的 capacity 是冻结的候选生成器、A5 policy、evaluator 和阈值下的**观测容量**，不能宣称是
scene 的固有容量。跨 scene 晋级只检查全局多模态比例、仅在正容量 scene 上计算的 capacity-normalized
coverage，以及 Stage3 credit-ready 比例。

新的 8-GPU、8-scene `train_val` 端到端 smoke 位于：

```text
outputs/sg_fps_v4_repro_a_smoke_20260713
outputs/sg_fps_v4_repro_b_smoke_20260713
```

它确认真实 dataset domain 为 103,288 scenes，8/8 shard 成功、contract error 为 0。平均每 scene
有 40 条 non-GT proposal；39.625 条 evaluator-valid、39.0 条 trajectory-quality、30.375 条进入 trust
region、29.375 条满足 2/8 policy-density witness，最终选择 2.25 条 non-GT mode。policy feasible ratio
为 0.921875，Stage3 credit-ready 比例为 2/8。两次独立构建的 8 个压缩 record 逐字节一致，确认
`sha256(token) xor build_seed` 的 scene seed 不受 shard 启动顺序影响。该小样本没有出现 GT-only
scene，因此不能估计全量困难场景比例；它只证明漏斗、真实 evaluator、train/val 覆盖元数据、
可复现性和 validator 调用链一致。

随后在固定的 128-scene 原始候选池上做了 proposal-source 配对。把“与 GT 的 SNSAD 模式分离”
前移到 Pareto dominance 之前后，基础池保留 `121/128` 个多模态 scene、254 条 non-GT mode；
加入未扩增的 DDV2/DrivoR 原始预测后为 `122/128`、262 条。外部候选在 24 个 scene 中各入选
一条，恢复 1 个 GT-only scene，丢失 0 个已有多模态 scene。基础池剩余 7 个 GT-only scene 的
漏斗原因为 `no_distinct_mode=1`、`no_pareto_candidate=6`；外部池分别为 1 和 5。这说明外部源有
有限但真实的全局增量，不足以也不需要逐 scene 补齐。

该配对还发现并修复了一个非单调选择 bug：旧顺序允许近 GT 的高指标候选先支配独立模式，
再因 FPS 距离不足被丢弃，曾出现加入外部源后恢复 2 个 scene、同时损失 1 个 scene。新契约
`observed_funnel_distinct_before_pareto_no_quota_v2` 消除了该路径。validator 现在从持久化的完整
selection-gate 配置重算 valid/quality、teacher eligibility 和 Pareto mask，而不是校验自报字段；
外部候选根目录保存 payload SHA256 与文件数。

当前代码的 128-scene 正式构建位于：

```text
outputs/sg_fps_v4_contract_v2_external_128_20260713/support_v4
```

其漏斗均值从每 scene 41.922 条 non-GT proposal 依次收敛到 40.586 条 evaluator-valid、39.258 条
trajectory-quality、36.477 条 trust-region、34.719 条具有至少 2/8 policy witness、15.133 条与 GT
足够不同、8.055 条 Pareto-eligible，最终选中 2.047 条。122/128 scene 有非 GT mode，6/128 为
GT-only；capacity-normalized coverage 为 1.0，Stage3 credit-ready scene ratio 为 42.97%。这些数字
说明难 scene 可以没有替代监督，但每一次候选消失都能定位到可重算的漏斗阶段。
