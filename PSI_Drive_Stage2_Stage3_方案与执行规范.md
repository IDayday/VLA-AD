# PSI-Drive：ReCogDrive Stage2–Stage3 高回报可行支持学习方案

> **定位**：面向 `IDayday/VLA-AD` 仓库 `feature/recogdrive-last-vla-v2` 分支的研究方案与工程执行规范。
> **基础选择**：Stage1 不改；Stage2 基于原版 ReCogDrive Stage2；Stage3 基于当前 SOTA 的 Core-Pareto GRPO v2。
> **核心目标**：缓解 narrow policy，使 DiT 在 Stage2 学到多个高分、可行的局部策略支持，并让 Stage3 在这些支持邻域中进行更可靠的相对优化。
> **原则**：核心创新保持单一；额外改动仅用于保证核心机制可训练、可诊断、可复现和可安全归档。

---

## 1. 核心命题

当前策略并非普遍缺少高分轨迹，而更接近“高分但局部化”的条件分布：多数场景的多次采样围绕少量 PDM 友好模式展开；少数困难场景中又缺少可被有限采样命中的高分替代轨迹。Stage3 后期的有效轨迹比例已经较高，主要训练信号逐渐转为有效轨迹内部的细粒度排序。

本方案将 Stage2 和 Stage3 统一为同一条主线：

```text
Stage2：获得高回报可行支持（support acquisition）
Stage3：在支持邻域内排序和吸收（support-relative absorption）
```

对场景 \(s\)，构建一个小型、自适应的支持集：

\[
\mathcal S_s=\{\tau_{s,1},\ldots,\tau_{s,M_s}\},\qquad 1\le M_s\le3.
\]

这些轨迹不是任意分散候选，而是位于同一场景的高质量可行区域。Stage2 学习由这些支持构成的目标分布；Stage3 仍只使用当前策略产生的 on-policy rollout，但根据其接近的场景支持原型划分局部邻域，在邻域内部进行主要相对优化，并保留一个支持集外的自由探索邻域。

方法由两个模块组成：

1. **Adaptive Pareto Support Distillation（APSD）**：Stage2 场景级自适应 Pareto 支持蒸馏；
2. **Support-Relative Pareto GRPO（SR-PGRPO）**：Stage3 支持邻域相对 Pareto GRPO。

统一名称可使用：

# **PSI-Drive**
## Pareto Support Iteration for Driving Diffusion Policies

---

## 2. 设计边界

### 2.1 本方案明确不改

- Stage1 VLM 训练、权重和缓存语义；
- 原版 ReCogDrive DiT 主体结构；
- 原版 diffusion scheduler、噪声预测目标和 DDIM/DDPM 推理接口；
- Stage2 official-aligned 的训练入口、train/val log split、fp32 模型权重、Lightning mixed precision、AdamW 与 WarmupCosLR 语义；
- Stage3 每场景 16 条 on-policy diffusion rollout；
- exact PDM/PDMS 评价口径；
- Core-Pareto v2 的 NC/DAC 可行性、DDC guard、EP floor、EP/TTC trade-off、Pareto front、slow positive cap、BC 和 reference KL 主干；
- navtest 的最终评估口径。

### 2.2 第一版不加入

- Support-ID/intent token；
- anchor-residual diffusion；
- trajectory selector；
- DPO、AWAC、IQL、self-imitation；
- 显式 pairwise trajectory repulsion；
- horizon-wise normalization；
- adaptive group size；
-在线更新 support bank；
- buffer-neighborhood bonus。

这些模块可作为后续消融或失败后的升级路径，不进入第一版核心实验。

---

## 3. 方法总览

```text
Stage1（不改）
    │
    ▼
原版 ReCogDrive Stage2 checkpoint
    │
    ├── GT
    ├── 原版 Stage2 deterministic/stochastic proposals
    ├── 现有 structured perturbations
    └── 可选：当前 Stage3 SOTA proposal teacher（独立消融）
    │
    ▼
navtrain exact PDM scoring
    │
    ▼
Core-safe + EP floor + quality band
    │
    ▼
每场景 1–3 条 Adaptive Pareto Supports
    │
    ▼
原版 DiT 多支持目标分布训练（APSD）
    │
    ▼
新的 Stage2 diffusion policy
    │
    ▼
Core-Pareto GRPO v2 每场景 G=16 on-policy rollout
    │
    ▼
按 scene support / free bucket 动态分组
    │
    ▼
valid+EP 邻域内排序 + 正向截断的邻域间奖励
    │
    ▼
SR-PGRPO 更新
```

---

# Part I：Stage2 — Adaptive Pareto Support Distillation

## 4. Stage2 基础训练保持原版

原版 Stage2 对当前目标轨迹 \(\tau\) 进行标准噪声预测：

\[
x_t=\sqrt{\bar\alpha_t}\tau+\sqrt{1-\bar\alpha_t}\epsilon,
\]

\[
\mathcal L_{diff}=\|\epsilon-\epsilon_\theta(x_t,t,s)\|_2^2.
\]

APSD 不改变上述损失，而是把每个场景的训练目标由单条轨迹变成一个小型分布：

\[
\tau\sim q_s(\tau),\qquad
\mathcal L_{S2}=\mathbb E_{\tau\sim q_s,t,\epsilon}\left[\mathcal L_{diff}\right].
\]

Stage2 的核心改动仅位于：

- support bank 构建；
- Dataset/Collate 对 support tensors 的加载；
- 训练态下目标轨迹的选择。

验证态和推理态仍使用原有单轨迹接口，不加载或依赖 support bank。

---

## 5. 支持候选来源

### 5.1 MVP 候选池

每个 navtrain 场景候选池建议包含：

| 来源 | 建议数量 | 作用 |
|---|---:|---|
| GT | 1 | 保留原始行为先验和回退目标 |
| 原版 Stage2 deterministic | 1 | 原始 IL 主模式 |
| 原版 Stage2 stochastic | 8 | 当前 IL 已有支持 |
| structured perturbations | 16–24 | 扩展 progress、lateral、timing 邻域 |

结构化扰动优先复用现有 `offline_action_explorer.py`：

- endpoint extension；
- speed scaling；
- time gamma；
- lateral offset；
- endpoint lateral offset；
- slow-first；
- delay；
- trajectory smoothing；
- heading-step guard；
- final-heading guard；
- forward monotonic repair。

### 5.2 可选 Bootstrap 版本

增加当前 `step21600` Stage3 SOTA 的 stochastic proposals，作为性能上限实验：

```text
Clean Support：GT + 原版 Stage2 + structured perturbations
Bootstrap Support：Clean Support + step21600 proposals
```

主实验必须保留 Clean Support，避免方法被解释为仅蒸馏当前 SOTA。

### 5.3 不允许的数据来源

- navtest 轨迹、navtest metric cache 或 navtest 评价结果；
- 测试集统计量；
- 由 navtest 挑选的场景或候选；
- dummy/smoke 数据作为真实训练支持。

---

## 6. 支持候选有效性和质量过滤

### 6.1 可行性定义

沿用当前 Core-Pareto v2 的训练语义：

\[
valid=NC\land DAC\land DDC_{guard},
\]

\[
valid^+=valid\land EP_{floor}.
\]

具体建议：

\[
NC=1,\qquad DAC=1,
\]

\[
DDC\ge\max(0.95,DDC_{ref}-0.01),
\]

\[
EP\ge EP_{ref}-0.02.
\]

第一版应与现有 Core-Pareto reference 语义保持一致，避免同时改变 reference 和 support 机制。coherent-reference 可作为独立消融。

### 6.2 质量带

对所有 `valid+EP` 候选计算现有 Core-Pareto score \(S_i\)，只保留：

\[
S_i\ge S_{best}-\epsilon_Q.
\]

建议：

```text
εQ = 0.02
消融：0.01 / 0.02 / 0.03
```

### 6.3 回退规则

- 场景存在 `valid+EP` 候选：从有效质量带中选支持；
- 不存在 `valid+EP` 候选：回退 GT；
- 若 GT 不可用或非有限：回退 deterministic IL；
- 不允许 invalid candidate 成为 Stage2 正目标；
- 不允许因候选不足而伪造重复支持，只保留真实数量和 mask。

---

## 7. 行为描述符

每条物理轨迹计算：

\[
\phi(\tau)=
[x_T,y_T,\psi_T,\bar v,v_T,r_{early}],
\]

其中：

- \(x_T\)：最终纵向进度；
- \(y_T\)：最终横向位置；
- \(\psi_T\)：最终航向；
- \(\bar v\)：平均速度；
- \(v_T\)：末端速度；
- \(r_{early}\)：前半段进度占总进度的比例，用于区分提前/延迟动作。

使用 navtrain 候选统计进行标准化：

\[
\tilde\phi_d=\frac{\phi_d-\mu_d}{\sigma_d+\epsilon}.
\]

距离：

\[
d_\phi(\tau_i,\tau_j)=\|\tilde\phi(\tau_i)-\tilde\phi(\tau_j)\|_2.
\]

必须把 `descriptor_mean`、`descriptor_std` 和版本写入 support index，Stage3 使用完全相同的统计量。

---

## 8. 自适应支持选择

每场景最多保留三条：

\[
M_{max}=3.
\]

选择流程：

1. 选择有效质量带中 score 最高的轨迹作为 `support-0`；
2. 从剩余候选中选择与已有支持最远的轨迹；
3. 若其到已有支持的最小描述符距离低于 `d_min`，停止；
4. 最多再执行一次，得到 `support-2`。

\[
\tau^*=\arg\max_{\tau\in\mathcal C_s}
\min_{\tau'\in\mathcal S_s}d_\phi(\tau,\tau').
\]

建议：

```text
Mmax = 3
d_min = 0.75
```

### 8.1 去重和 tie-break

候选分数接近时按以下顺序打破平局：

1. 更高 PDMS；
2. 更高 Core；
3. 更高 NC×DAC；
4. 更小与 reference 的异常几何距离；
5. 稳定的 source priority；
6. candidate index，保证可复现。

### 8.2 支持来源审计

support index 必须记录：

```text
source
candidate_index
pdms/core/components
valid/ep_floor/pareto flags
descriptor
nearest-support distance
selection rank
```

并输出：

- support_count=1/2/3 比例；
- source 分布；
- score gap；
- pairwise descriptor distance；
- invalid fallback 数；
- GT fallback 数；
- train token 数和唯一性；
- 与 val/navtest token 的交集必须为 0。

---

## 9. Stage2 目标分布

每个场景仍是一条 Dataset record，不把三条支持展开成三条重复记录。

推荐目标概率：

```text
best support：0.50
GT：0.20
other supports：合计 0.30
```

规则：

- GT 与 best support 重合时合并概率；
- 只有一个 support 时，剩余概率全部并入 best；
- GT 非有限或明确不可用时，把 GT 概率并入 best；
- other support 数为 1 或 2 时均分 0.30；
- 所有有效权重归一化后和为 1。

第一版使用 `torch.multinomial` 在训练 forward 中选目标即可。验证态始终使用原始 `trajectory`，避免 val loss 因随机 support target 波动。

### 9.1 为什么不均匀采样

均匀采样会使较弱的第三支持与 best support 获得相同权重，可能提高 oracle Best-of-K，却降低单次采样质量。上述混合保留 best 作为主要质量中心，同时给其他高质量区域非零概率。

### 9.2 为什么不加入 separation loss

第一版首先检验“高质量多目标监督”本身能否增加可采样支持。显式 repulsion 会同时改变数据分布和模型目标，并可能奖励几何发散，因此推迟到后续。

---

## 10. Stage2 数据格式

推荐 compact `.pt` 支持索引：

```python
{
    "version": 1,
    "branch": "feature/recogdrive-last-vla-v2",
    "git_commit": "...",
    "source_checkpoint": "...",
    "candidate_generator_config": {...},
    "filter_config": {...},
    "descriptor_names": [...],
    "descriptor_mean": Tensor[D],
    "descriptor_std": Tensor[D],
    "tokens": List[str],
    "token_to_row": Dict[str, int],
    "support_trajectories": Tensor[N, 3, 8, 3],
    "support_mask": BoolTensor[N, 3],
    "support_weights": Tensor[N, 3],
    "support_scores": Tensor[N, 3],
    "support_pdms": Tensor[N, 3],
    "support_core": Tensor[N, 3],
    "support_descriptors": Tensor[N, 3, D],
    "support_sources": List[List[str]],
    "support_metadata": ...,
    "summary": {...},
}
```

轨迹保持物理坐标 `[8,3]`，由 planner 使用现有 `norm_odo()` 归一化，避免在索引中混淆物理/归一化坐标。

---

## 11. Stage2 代码改动

### 11.1 新增

```text
navsim/agents/recogdrive/pareto_support.py
scripts/build_stage2_pareto_support_index.py
scripts/audit_stage2_pareto_support_index.py
tests/test_pareto_support_selection.py
tests/test_stage2_pareto_support_loader.py
tests/test_stage2_pareto_support_forward.py
```

### 11.2 修改 `run_training_recogdrive.py`

扩展：

```text
stage2_target_source:
  - gt
  - awac_elite_best_valid_above_gt_or_gt
  - pareto_support
```

`ChunkCacheDataset` 在 `pareto_support` 模式返回：

```python
targets = {
    "trajectory": gt_trajectory,
    "support_trajectories": Tensor[3, 8, 3],
    "support_mask": BoolTensor[3],
    "support_weights": Tensor[3],
    "support_scores": Tensor[3],
}
```

`custom_collate_fn` 只在字段存在时堆叠，GT 路径不受影响。

### 11.3 修改 `recogdrive_diffusion_planner.py`

新增：

```python
_select_stage2_pareto_support_target(...)
```

要求：

- 仅 `self.training` 且 support fields 存在时启用；
- 验证/推理忽略 support fields；
- disabled 模式与原版 forward 数值路径一致；
- 输出 target-source、support-index、support-count 等诊断。

---

## 12. Stage2 训练建议

### 12.1 开发运行

```text
初始化：与当前 Stage3 SOTA 对应的原版 ReCogDrive Stage2 checkpoint
epochs：60
LR：5e-5
optimizer：AdamW
weight decay：1e-4
warmup：2 epochs
scheduler：cosine
min LR：1e-6
effective batch：128
precision：16-mixed，模型权重 fp32
```

### 12.2 论文匹配运行

补充：

```text
GT single-target 200 epochs
Pareto-support 200 epochs
```

使用相同初始化、数据、batch、optimizer、scheduler 和评测预算。

---

# Part II：Stage3 — Support-Relative Pareto GRPO

## 13. Stage3 基础保持 Core-Pareto GRPO v2

保留：

- `G=16`；
- 当前策略 on-policy diffusion chain sampling；
- exact PDM scoring；
- NC/DAC hard validity；
- DDC guard；
- EP floor；
- EP/TTC trade-off penalty；
- EP/TTC/Comfort Pareto front；
- dominated positive cap；
- slow positive cap；
- all-invalid rescue；
- trajectory-level chain log-prob；
- old-policy BC；
- reference KL；
- 当前 SOTA 的 LR、batch、20-epoch scheduler 主配置。

支持轨迹不进入 GRPO group，不承担 off-policy policy-gradient 样本角色。

---

## 14. 场景支持邻域分配

Stage3 训练时根据 `tokens_list` 从 compact support bank 读取每场景的 1–3 条原型。

对每条 on-policy rollout 计算同一行为描述符：

\[
z_i=\arg\min_m d_\phi(\tau_i,\tau_{s,m}).
\]

若：

\[
\min_m d_\phi(\tau_i,\tau_{s,m})>d_{free},
\]

则进入 `free bucket`。

建议：

```text
d_free = 1.50
```

### 14.1 支持缺失

- token 在 support bank 中缺失：整组退化为原 Core-Pareto v2；
- support mask 为空：退化为原 Core-Pareto v2；
- 不允许因支持缺失中断训练；
- 记录 missing-support ratio。

### 14.2 训练/推理解耦

support bank 仅用于 Stage3 训练中的 advantage grouping：

- 不作为 DiT observation；
- 不改变 diffusion log-prob；
- 不在 navtest 推理中加载；
- 不增加部署推理成本。

---

## 15. Valid-only 邻域内排序

定义：

\[
V_i=valid_i\land EPFloor_i.
\]

对邻域 \(b\)，只有：

\[
n_b=\sum_i\mathbf 1[z_i=b\land V_i]\ge2
\]

并且：

\[
\Delta S_b=\max S_i-\min S_i\ge\delta_{rank}
\]

时，才计算主要邻域内优势：

\[
A_i^{intra}=\frac{S_i-\mu_b}{\max(\sigma_b,\sigma_{min})}.
\]

建议：

```text
δ_rank = 0.01
σ_min = 0.005
```

关键要求：

- `invalid` 和 `EP-fail` 不参与 \(\mu_b,\sigma_b\)；
- singleton 或低 gap 邻域的 `A_intra=0`；
- 不把极小绝对差值通过 z-score 强制放大为单位级更新。

---

## 16. 邻域间正向截断比较

对每个存在 `valid+EP` 候选的邻域，取：

\[
R_b=Top2Mean\{S_i:z_i=b,V_i=1\}.
\]

计算邻域代表的标准化值后，仅保留正向部分：

\[
A_b^{inter}=clip(\max(0,z(R_b)),0,c).
\]

最终：

\[
A_i^{valid}=A_i^{intra}+\beta A_{z_i}^{inter}+A_i^{reference}.
\]

建议：

```text
β = 0.15
c = 0.30
```

含义：

- 邻域内允许正负更新，提高同一策略局部的质量；
- 邻域间只奖励更优支持，不强惩罚其他已通过质量筛选的支持；
- 降低 Stage3 重新把多个高质量支持压成单一模式的风险。

---

## 17. Free bucket

free bucket 用于避免静态支持集封死新行为。

### 17.1 多样本 free bucket

若 free bucket 中至少两条 `valid+EP` 候选且 score span 达到 `δ_rank`，正常计算 `A_intra`。

### 17.2 单样本 free bucket

单个 free candidate 只有满足：

\[
S_i>S_{ref}+\delta_{novel}
\]

且属于 Pareto front 时，才允许获得小幅正更新。

建议：

```text
δ_novel = 0.01
novel_positive_cap = 0.20
```

否则只保留 reference/absolute 分支。

---

## 18. 绝对约束和最终 Advantage 语义

### 18.1 继续保留

- invalid：固定非正/负优势；
- slow fail：优势不得为正；
- dominated valid：正优势 cap；
- all-invalid rescue；
- group weight。

### 18.2 取消全 batch mean-centering

当前组内 cap 后再做全 batch z-score 可能改变零点和符号。建议：

```text
grpo_normalize_advantage_batch = false
```

若需要稳定尺度，使用 sign-preserving RMS scaling：

\[
A' = \frac{A}{\max(1,\sqrt{\mathbb E[A^2]})}.
\]

它不减均值、不改变 0 和正负号，只在整体尺度过大时缩小。

### 18.3 最终 re-cap

所有 advantage 组合、group weight 和 RMS scaling 后，重新执行：

```python
adv[invalid] = torch.minimum(adv[invalid], unsafe_cap)
adv[slow_fail] = torch.minimum(adv[slow_fail], 0)
adv[dominated_valid] = torch.minimum(adv[dominated_valid], 0)
```

新增最终日志：

```text
final_positive_invalid_ratio
final_positive_slow_fail_ratio
final_positive_dominated_ratio
final_advantage_mean/std/min/max/zero_ratio
```

---

## 19. Support-rankability group weight

定义邻域可排序：

```text
valid+EP count >= 2
score range >= 0.01
```

建议：

| 场景组状态 | 权重 |
|---|---:|
| 至少一个 rankable support/free bucket | 1.0 |
| 多个 valid+EP，但所有邻域低 gap | 0.25 |
| mixed valid/invalid | 沿用现有 0.5–1.0 逻辑 |
| all-invalid | 沿用当前 0.25 |
| all-slow | 沿用当前低权重 |

该设计减少训练后期大量 all-valid、低 gap 场景从 evaluator 数值噪声中持续制造更新。

---

## 20. Stage3 代码改动

### 20.1 配置新增

```yaml
grpo_use_support_relative: false
grpo_support_index_path: null
grpo_support_rank_margin: 0.01
grpo_support_std_floor: 0.005
grpo_support_free_distance: 1.5
grpo_support_novel_margin: 0.01
grpo_support_novel_positive_cap: 0.20
grpo_support_intra_weight: 1.0
grpo_support_inter_weight: 0.15
grpo_support_inter_clip: 0.30
grpo_support_positive_only_inter: true
grpo_support_low_rank_group_weight: 0.25
grpo_use_rms_advantage_scale: true
grpo_reapply_final_caps: true
```

默认全部关闭，保证旧配置兼容。

### 20.2 新增函数

```python
_load_pareto_support_index(...)
_lookup_scene_support_batch(tokens_list, ...)
_compute_trajectory_descriptors(...)
_assign_scene_support_buckets(...)
_compute_support_relative_pareto_advantages(...)
_sign_preserving_rms_scale(...)
_reapply_final_advantage_caps(...)
```

### 20.3 日志新增

```text
support_count_mean
support_missing_ratio
occupied_support_bucket_count
rankable_support_bucket_count
singleton_support_bucket_ratio
free_bucket_ratio
free_bucket_valid_ratio
within_support_score_std
between_support_rep_std
best_valid_minus_median_valid
best_valid_minus_reference
support_0/1/2_occupancy
support_concentration
frontier_gain
```

同时增加 `valid_count` 和 `valid+EP count` 的 0、1–4、5–8、9–15、16 直方图。

---

## 21. Stage3 训练建议

第一组 matched run 保持当前 SOTA 配置：

```text
初始化：最佳 Stage2 APSD checkpoint（由 val6000 选择）
G：16
LR：1e-4
effective scene batch：64
epochs：20
BC：0.10 -> 0.05，5 epochs
reference KL：0.02
EP floor：保留
slow positive cap：保留
TTC hard gate：关闭
DDC：只作 guard
```

只改变：

1. support-relative advantage；
2. RMS-only scaling 和 final re-cap；
3. 新日志。

若 matched run 显示明显 checkpoint volatility，再单独测试 `LR=5e-5`，不在首轮同时修改其他超参数。

---

# Part III：训练和评估协议

## 22. 固定 val6000

必须生成并长期复用固定 token 文件：

```text
artifacts/splits/navtrain_val6000_seed_<seed>.txt
artifacts/splits/navtrain_val6000_seed_<seed>.sha256
```

要求：

- 从 navtrain heldout/log-val 口径中确定；
- 固定排序和随机种子；
- 后续 Stage2/Stage3 所有实验使用同一文件；
- 优先使用 evaluator 的 `--sample-token-file`，不要只使用不稳定的 `--max-samples 6000`；
- 记录 token 数、唯一数和与 train/navtest 的交集；
- 交集检查失败立即终止。

所有正式评估使用 fp32。

---

## 23. Stage2 评估流程

1. 训练期间保存不可变 periodic/raw checkpoints；
2. 对每个候选 checkpoint 运行固定 val6000 exact PDM；
3. 按 val6000 PDMS 排名，tie-break：Core、NC×DAC、TTC、EP；
4. 保留并归档 **val6000 Top-5**；
5. 任何新 checkpoint 进入 val6000 Top-5 时，加入 full navtest 队列；
6. 对所有曾进入 val6000 Top-5 的 checkpoint 做 full navtest；
7. 在已完成 full navtest 的 checkpoint 中维护 **navtest Top-5**；
8. Stage3 初始化必须由 val6000 选择，不使用 navtest 反馈选择训练起点；
9. 最终报告 val6000-best、navtest-best、last 和 val-loss-best，不混淆口径。

---

## 24. Stage3 评估流程

1. 按当前 SOTA 节奏保存 step checkpoints，例如每 300 steps；
2. 所有稳定 checkpoint 先归档，再运行 val6000；
3. val6000 新 Top-5 entrant 进入 full navtest 队列；
4. 训练过程中只使用 val6000 观察，不以 navtest 结果触发 early stop、LR 或配置改变；
5. 训练结束后等待所有 val6000 Top-5 entrants 的 navtest 队列完成；
6. 维护 val6000 Top-5 和 navtest Top-5 两套权重与 manifest；
7. 报告：best、last、top5 mean/std、完整 checkpoint trend。

---

# Part IV：Checkpoint 不可破坏规范

## 25. 三层 checkpoint 结构

```text
RUN_ROOT/
  checkpoints/
    raw/                    # 所有 periodic/epoch/last 原始 checkpoint，永不自动删除
    val_loss_top5/          # Lightning val-loss Top-5，仅作辅助
  checkpoint_store/
    objects/<sha256>.ckpt   # 内容寻址、不可变归档对象
    inventory.tsv
  rankings/
    val6000/
      current_top5.json
      current_top5.tsv
      history/<timestamp>.json
    navtest/
      current_top5.json
      current_top5.tsv
      history/<timestamp>.json
  eval/
    val6000/<checkpoint_id>/...
    navtest/<checkpoint_id>/...
  state/
  logs/
```

### 25.1 Raw checkpoint

- periodic checkpoint callback 使用 `save_top_k=-1` 或自定义 append-only callback；
- `last.ckpt`、epoch checkpoint 和明确 step checkpoint全部保留；
- Lightning `val/loss top5` 必须放在独立目录，不能作为唯一 checkpoint 来源；
- 不运行会删除旧 checkpoint 的 cleanup 脚本；
- 磁盘不足时停止并报告，不自动清理。

### 25.2 内容寻址归档

每个待评估 checkpoint：

1. 等待文件大小稳定；
2. 计算 SHA256；
3. 写入临时文件 `objects/.<sha>.tmp`；
4. 优先 hardlink；跨文件系统则 `copy2`；
5. 重新计算目标 SHA256；
6. 一致后 atomic rename 为 `objects/<sha>.ckpt`；
7. 更新 `inventory.tsv`；
8. 原始 checkpoint 不移动、不删除。

### 25.3 Top-5 排名

- Top-5 只更新 manifest，不删除旧归档对象；
- 每次排名变化都追加到 `history/`；
- `current_top5.json` 引用内容寻址对象和原始 source path；
- 所有曾进入过 Top-5 的对象永久保留；
- 若完成评估不足 5 个，明确写 `complete=false`，不能伪造五条；
- 同一 checkpoint 以 SHA256 去重，不能因路径不同重复占名次。

### 25.4 禁止操作

正式脚本中禁止：

```text
rm -rf <run/checkpoints/output root>
git clean -fdx
git reset --hard
覆盖同名 checkpoint
移动原始 checkpoint 后不留副本
以 symlink 作为唯一权重保存方式
```

只允许删除 `.running/.failed` 等状态 marker；不得删除 `.ckpt` 对象。

---

## 26. Checkpoint inventory 必备字段

```text
checkpoint_id
sha256
source_path
object_path
size_bytes
mtime
stage
train_step
epoch
config_hash
git_commit
val6000_state
val6000_pdms/core/nc/dac/ttc/ep
navtest_state
navtest_pdms/core/nc/dac/ttc/ep
first_seen_at
archived_at
```

所有更新采用文件锁和 atomic replace，支持 watcher 重启和重复运行。

---

# Part V：实验矩阵

## 27. 最小主实验

| 实验 | Stage2 | Stage3 | 目的 |
|---|---|---|---|
| A | 原版 GT | Core-Pareto v2 | 当前基线 |
| B | Elite-1 | Core-Pareto v2 | 单目标质量基线 |
| C | APSD Clean | Core-Pareto v2 | 只验证 Stage2 支持扩展 |
| D | APSD Clean | SR-PGRPO | 完整核心方案 |
| E | APSD Bootstrap | SR-PGRPO | 使用当前 SOTA proposal 的上限 |
| F | APSD Clean | 固定 3×3 phenotype GRPO | scene support 与手工 bucket 对照 |
| G | D，无 free bucket | 验证支持外通道 |
| H | D，允许负 inter-support | 验证 positive-only inter |

关键比较：

- `C - A/B`：Stage2 高质量支持扩展；
- `D - C`：共享支持结构用于 Stage3 的额外价值；
- `E - D`：SOTA proposal teacher 的边际作用。

---

## 28. Stage2 指标

```text
MeanPDMS@1
P10/P50/P90 PDMS@1
BestValidPDMS@8/16
NearOptimalHit@16, δ=0.01/0.02
Valid@16
Valid+EP@16
SupportRecall@16
SupportOccupancyEntropy
MaxSupportOccupancyRatio
mean-pADE / mean-pFDE
minADE / minFDE
```

其中：

\[
SupportRecall@16=
\frac{\text{被至少一条有效采样覆盖的 support 数}}{M_s}.
\]

---

## 29. Stage3 指标

```text
valid_count / valid+EP_count histograms
rankable group ratio
rankable support bucket count
free bucket valid ratio
within-support score std
between-support representative std
best-valid minus median-valid
frontier gain
final positive invalid/slow/dominated ratios
reference KL
trajectory log-prob
checkpoint volatility
```

还应报告：

\[
Potential@16=BestValidPDMS@16-MeanPDMS@1,
\]

用于区分 Stage2 支持潜力与 Stage3 单次采样性能。

---

# Part VI：测试与验收

## 30. 单元测试

### 30.1 Support selection

- 只选择 `valid+EP`；
- 0 valid 时正确回退；
- support count 为 1–3；
- mask 和 weight 正确；
- 权重和为 1；
- quality band 生效；
- farthest selection 可复现；
- token 唯一；
- descriptor stats 有限。

### 30.2 Dataset/Collate

- 原 GT 模式输出不变；
- support tensors shape 正确；
- val loader 不随机替换 GT；
- DDP batch 可堆叠；
- 缺失 token 有回退。

### 30.3 Stage2 forward

- support disabled 与原版相同；
- training only target sampling；
- validation/inference 忽略 support；
- loss finite；
- checkpoint 兼容。

### 30.4 Stage3 advantage

- invalid/slow 不参与 valid-only统计；
- singleton 和低 gap 邻域 intra=0；
- positive-only inter 不产生负 inter；
- free bucket 规则正确；
- RMS 不改变符号；
- final re-cap 后违规正优势为 0；
- support 缺失退化原版；
- no NaN/Inf。

### 30.5 Checkpoint manager

- 原 checkpoint 不被删除；
- 相同 SHA 去重；
- 跨文件系统 copy fallback；
- hash mismatch 失败且不写 manifest；
- watcher 重启幂等；
- Top-5 变化只追加 history；
- 少于五个结果时标记不完整；
- 同名不同内容不覆盖。

---

## 31. Smoke 与回归

1. dummy/no-data 只验证计算流，不能报告 benchmark；
2. 小型真实缓存跑 1–2 train steps；
3. 原版 Stage2 disabled-path regression；
4. Stage3 Core-Pareto v2 disabled-path regression；
5. 固定少量 token 的 exact PDM smoke；
6. checkpoint archive/ranking dry-run；
7. full training 前生成完整 preflight report。

---

## 32. 研究风险与控制

| 风险 | 控制 |
|---|---|
| Best-of-K 提升但 Mean@1 下降 | best 0.5、GT 0.2、quality band；Mean@1 为必报指标 |
| 支持只是轻微坐标扰动 | 行为描述符与 `d_min` |
| 支持太多导致 bucket 稀疏 | `Mmax=3` |
| 静态支持限制新策略 | free bucket |
| 低 reward gap 被 z-score 放大 | `δ_rank` 和低 rankability 权重 |
| cap 被 batch norm 改变 | RMS-only + final re-cap |
| SOTA proposal 形成循环 | Clean/Bootstrap 分开 |
| 多目标 diffusion 仍 collapse | 先测 SupportRecall；失败后再考虑 Support Code |
| train high/test low | 固定 heldout、完整 navtest、family-level诊断 |
| checkpoint 丢失 | raw append-only + content-addressed store + history manifests |

---

## 33. 后续升级触发条件

### 33.1 引入 Support Code

仅当：

```text
离线 support 确实多样且高分；
Stage2 训练使用了多个 support；
但 SupportRecall@16 仍长期接近 1 个模式。
```

### 33.2 引入局部 buffer bonus

仅当：

```text
Stage3 能命中高分候选并正确排序；
但下一轮采样概率/Mean@1 不提高。
```

### 33.3 引入 Frontier Promotion

仅当 free bucket 中高分新模式跨多个 rollout/checkpoint 持续出现。此时可离线更新 support bank并短程蒸馏，不在第一版在线修改 bank。

---

## 34. 论文贡献表达

1. **Adaptive Pareto Support Distillation**：将单一 GT/Elite-1 监督扩展为每场景 1–3 条、质量约束且自适应的可行策略支持，不通过无约束几何分散扩大策略。
2. **Support-Relative Pareto GRPO**：利用相同支持原型对 on-policy 轨迹做局部划分，在支持内部进行主要相对优化，在支持之间采用正向截断比较，并保留支持外的新策略通道。
3. **Support acquisition–absorption diagnostics**：通过 BestValid@K、SupportRecall、rankability、frontier gain 和 Mean@1 区分 Stage2 的支持获取与 Stage3 的概率吸收。

统一核心：

\[
\boxed{\text{先建立高分可行支持，再在支持局部内持续改进}}
\]

---

## 35. 代码参考路径

```text
navsim/agents/recogdrive/recogdrive_diffusion_planner.py
navsim/agents/recogdrive/recogdrive_agent.py
navsim/agents/recogdrive/offline_action_explorer.py
navsim/agents/recogdrive/offline_rl_buffer.py
navsim/planning/script/run_training_recogdrive.py
navsim/planning/script/run_training_recogdrive_rl.py
navsim/planning/training/agent_lightning_module.py
scripts/last_vla_v2/two_expert_slot/build_stage2_elite_target_index.py
scripts/eval_recogdrive_expert_pdm.py
scripts/evaluation/watch_stage3_checkpoints_eval_8gpu.sh
scripts/evaluation/analyze_recogdrive_stage3_navtest_pdms.py
docs/OfficialAlignedBaselineGuardrails.md
```

---

## 36. 最终交付物

```text
代码：APSD + SR-PGRPO，默认关闭、向后兼容
配置：Stage2/Stage3 主实验与消融
数据：navtrain Pareto support index + audit
训练：Stage2 和 Stage3 完整 run
评估：固定 val6000 + full navtest
权重：Stage2/Stage3 的 val6000 Top-5 和 navtest Top-5
归档：不可变 checkpoint store、inventory、排名 history
报告：指标、消融、失败模式、checkpoint trend
```
