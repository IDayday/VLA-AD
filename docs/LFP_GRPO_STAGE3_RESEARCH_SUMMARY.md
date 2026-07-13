# LFP-GRPO Stage3 探究总结

更新时间：2026-07-13 04:05 UTC

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

### 2.4 v3 是高质量候选库，但未校准为多模态目标分布

- 全量 103,288 scene 的 selected support 均通过当前质量门控，`58.49%` 场景存在
  优于 GT 的候选；因此不支持“补充轨迹整体低质量”。
- reward 标签过度饱和：`75.81%` 场景 best support reward 等于 1，5,000-scene
  抽样中 `35.16%` 场景所有 support reward 都等于 1。
- support 有平均 `4.91` 个几何模式，但近邻 ADE 均值仅 `0.436m`，存在明显模式内冗余。
- 现有 adaptive beta 使用 source entropy，而 source entropy 与 trajectory mode capacity 的
  Spearman 相关为 `-0.263`。A5 名义采样 `3.821` 个 target，effective target count 仅
  `1.842`，最终 diffusion loss 的 GT mass 为 `87.29%`。
- 因此 Stage2 的基础性能是正向的，但“学到宽、support-aligned distribution”的监督目标没有
  真正成立。同一 v3 archive 上的 mode/density-balanced 对照已经完成：高 mode mass 能显著
  提高 recall 和 width，证明 archive 中存在可学习多样性；但 `beta>=0.45` 明确损害安全、
  PDMS 和 L1。当前可接受边界为 `beta=0.25`，其相对 legacy D0 的 SNSAD `+0.00589`、
  width ratio `+0.00925`，PDMS `-0.00143` 且 95% CI 跨 0。

### 2.5 Stage2 数据问题的准确定位

- v3 是有效候选库，不是自然校准的目标概率分布；archive record frequency 不能直接作为 mode mass。
- legacy sampler 使 diffusion loss 的实际 GT mass 达到约 87.3%，解释了 A5 的窄分布。
- 把 GT mass 一次降到约 56% 可以显著学宽，但会产生共享 DiT 的质量/安全干扰；延长到
  累计 1,000 step、降低 sampler variance 都不能消除该退化。
- `beta=0.15` 退回更高 GT mass，保住 PDMS 但没有新增多样性；`beta=0.25` 是首个同时具有
  显著 width/SNSAD 增益且 PDMS 差异未显著的点。
- 因此下一步不是立即重建 archive，也不是删除 FS-Norm/PTA，而是把 mode-balanced exposure
  放入 safety/quality trust budget，并做多 seed 复验。

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

对 v3 selected positive support 缓存 scene 固有容量：

```text
mode_capacity = 1 - 1 / reference_mode_count
coverage_ratio = clip(
    (policy_group_pairwise_ADE + 0.05)
    / (support_pairwise_ADE + 0.05),
    0, 1,
)
coverage_gap = mode_capacity * (1 - coverage_ratio)
frontier_bonus = 0.05 * coverage_gap * all_feasible
```

该 bonus 只进入下一 epoch sampler，不进入 advantage 或 policy loss。完整 cache 有 103,288 scenes，
support ADE 均值 `1.36082m`，mode count 均值 `4.9077`。

300-step 配对中，它使 frontier energy 与 mode capacity 的 Spearman 相关从 `0.0965` 提高到
`0.2327`。epoch1 实际采样的 multimodal ratio 提高 `0.49` 个百分点，没有 sampler collapse。

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

## 5. 当前原因排序

1. **首要：decoded trajectory trust 不足。** transition KL 不能阻止轨迹在物理空间显著漂移。
2. **首要：安全饱和后的 credit 偏向 progress。** Pareto gate 逻辑正确，但安全域内标量仍更容易
   奖励纵向激进轨迹；NAVSIM v2 comfort 不在 v1 reward 中。
3. **Stage2 多模态监督错位。** v3 轨迹库本身有质量和容量，但 reward 饱和、模式内冗余、
   source-entropy beta 和 anchor-heavy target sampling 使它没有变成 mode-balanced policy prior。
   单变量实验已确认 mode mass 存在安全边界：0.25 有温和正向多样性证据，0.45 以上干扰明显。
4. **次要：课程目标曾与核心多样性目标错位。** 旧 BPAE 更偏低 reference reward；容量课程已
   提供第一项正向因果证据，但效应温和。
5. **不是主因：优势使用 global std。** 严格对照已否定恢复 scene-group std。
6. **不是主因：Stage2 明显过拟合或补充轨迹整体低质量。** navtest 趋势、support audit 和
   A5 相对 Official 的 v1/v2 改善均不支持。

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

1. 在当前 v3 archive 上做 Stage2 D0/D1 短程单变量对照：当前 weighting vs
   trajectory-density/mode-balanced weighting。先分离训练目标问题，不立即重建数据。
2. 对 capacity frontier 复验至少一个 seed，并补 NAVSIM v2 EPDMS；在通过前保持默认关闭。
3. 单变量测试 reference-relative decoded trajectory trust，不能同时改 reward、KL 和 curriculum。
4. trust 有效后，再测试 v1/v2 分离的 quality objective；不能把 NAVSIM v2 与历史 Pareto GRPO v2
   混为一谈。
5. 最终候选必须同时报告 PDMS、EPDMS、NC/TTC/EC、L1 与 SNSAD，不以单一训练 reward 晋级。

本轮没有启动完整 Stage3 长训练，也没有继续占用 `training-rl-zt3`。容量课程的 v2 评估因该
资源已释放而未执行，不能从 v1 结果外推 v2 改善。
