# LFP-GRPO Stage3 R4 阶段性证据与结论

更新时间：2026-07-13 01:05 UTC

## 1. 范围与状态

| 项目 | 内容 |
|---|---|
| 仓库 | `/mnt/project/VLA-AD_last_vla_dev` |
| 分支 | `feature/recogdrive-last-vla-v2` |
| 当前代码 HEAD | `81a3020a10f4f5d1d6ec88f8d6bb1dd1b9d6f172` |
| Stage2 起点 | `stage2_pta_fs_dit_a5_full103k_standardv2_clean_20260710T091344Z/epoch_155.ckpt` |
| 正式 Stage3 | `stage3_lfp_grpo_v1_exact_b8g16_epoch155_r4_20260712T083012Z` |
| 正式训练配置 | 8 GPU，单卡 batch 8，accumulate 1，全局 scene batch 64，group size 16，LR `1e-5` |
| 本报告截止进度 | 约 step 4619，`epoch=2-step=4842` 尚未生成和评估 |
| 已取消任务 | 当前 Stage2 起点上的旧 Core-Pareto v2 300-step 排队探针 |
| 保留任务 | 正式 R4 训练及逐 epoch v1/v2 navtest watcher |

本报告只总结已经取得的证据。没有把尚未完成的 epoch 2 结果、未运行的
300-step S0 对照或未来假设写成实验结论。

## 2. 核心结论

1. **当前 LFP-GRPO 确实在学习，但尚未证明优于 Stage2。** 固定 seed、同一批
   predictions 的 v1 对照中，step 300、epoch 0、epoch 1 的平均 PDMS 均低于
   Stage2。epoch 1 已明显从 epoch 0 恢复，但对 Stage2 的差值置信区间仍跨 0，
   不能宣称提升。
2. **主要现象不是整体失败，而是小比例安全尾部抵消多数样本的 progress 收益。**
   step 300 的多数样本 PDMS 上升或持平，EP 显著上升；约 8.25% 出现
   NC/DAC/TTC 任一回退，这部分贡献约 `-0.0480`，抵消无安全回退样本的
   `+0.0422`。
3. **策略发生了明显的纵向激进偏移。** 相对 Stage2，step 300 的终点 x 平均
   增加约 3.09 m，epoch 0 增加约 4.07 m，epoch 1 增加约 3.88 m。EP 随之提高，
   但 TTC/NC 尾部风险增加。
4. **Frontier curriculum 不是当前退化的充分解释。** step 300 发生在 curriculum
   生效前；curriculum 生效后的 epoch 1 反而比 epoch 0 恢复。采样分布没有坍缩，
   但 train delta 的提高部分来自采到更低 reference-score 的场景，不能直接视为
   policy improvement。
5. **“TTC/progress 失败样本大多被置零、缺少负梯度”的假设已被短探针反证。**
   TTC-fail 平均 advantage 为 `-1.260`，零 credit 仅约 `0.96%`；progress-fail
   平均 advantage 为 `-0.652`。因此不应基于该假设增加新的 penalty。
6. **batch 不是新旧结果差异的直接原因。** 当前 R4 与历史 Core-Pareto v2 的
   全局 scene batch 都是 64。两者 micro-batch/accumulation 不同，但现有证据不能
   将效果差异归因于 batch。
7. **不能直接恢复旧版 v2 的完整配方。** 在当前 PTA+FS Stage2 上，旧配方仅训练
   10 step 就使固定 seed v1 PDMS 降至 `0.84929`。这证明旧版的 `LR=1e-4 + BC +
   KL + Core credit` 组合不能原样迁移，但不能区分其中各单项作用。
8. **目前最可信、但尚未因果验证的不足是分布锚定。** 当前 LFP 只有在 current-policy
   chains 上计算的 exact transition `KL(current || Stage2)`；旧版 BC 本质上提供了
   Stage2-chain 覆盖方向的近似 forward cross-entropy。该差异与观察到的输出漂移一致，
   但在没有单变量对照前只能列为待验证假设。
9. **NAVSIM v2 下降是确定的 cross-protocol 风险，但不是 v1 实现错误的直接证据。**
   当前训练 reward 是 NAVSIM v1 PDMS，不包含 v2 的 extended comfort、lane keeping
   和 TLC 完整目标。固定 seed v2 EPDMS 的显著下降主要伴随 extended comfort 下跌。

## 3. 可复现评估协议

稳定配对评估目录：

```text
outputs/stage3_lfp_grpo_v1_exact_b8g16_epoch155_r4_20260712T083012Z/
  navtest_stable_seed0_paired_20260712/
```

协议要点：

- 所有 checkpoint 使用相同 `initial_noise_seed=0`；
- 32 shards；
- 相同 navtest cache 和 agent 配置；
- v1 与 v2 由同一组 trajectory predictions 派生；
- v1 有 12,138 个有效样本、1,203 个 scene token；
- 置信区间按 scene token 做 cluster bootstrap，避免把同一 scene 的样本当成独立样本。

正式 watcher 的旧结果没有固定 initial-noise seed。它适合持续选权重，但不用于本报告的
因果比较。

## 4. 固定 Seed NAVSIM v1 结果

| Checkpoint | PDMS | 相对 Stage2 | EP | NC | DAC | TTC | DDC | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2 epoch 155 | 0.873111 | 0 | 0.820586 | 0.980804 | 0.953782 | 0.942907 | 0.980269 | 0.283344 |
| R4 step 300 | 0.867241 | -0.005870 | 0.865294 | 0.960702 | 0.950239 | 0.895205 | 0.968735 | 0.679971 |
| R4 epoch 0 | 0.856414 | -0.016697 | 0.865501 | 0.950362 | 0.944307 | 0.879222 | 0.967664 | 0.895203 |
| R4 epoch 1 | 0.870006 | -0.003105 | 0.875852 | 0.953328 | 0.953452 | 0.888861 | 0.971494 | 0.855051 |

Scene-cluster bootstrap 的 PDMS 差值 95% CI：

| Checkpoint | 95% CI | 解释 |
|---|---:|---|
| step 300 vs Stage2 | `[-0.00960, +0.00280]` | 未显著优于或劣于 |
| epoch 0 vs Stage2 | `[-0.02041, -0.00617]` | 明确下降 |
| epoch 1 vs Stage2 | `[-0.01023, +0.00372]` | 已恢复，但未证明提升 |
| epoch 1 vs epoch 0 | 约 `[+0.00532, +0.01451]` | 恢复明确 |

step 300 的样本级 PDMS delta 中位数为 `+0.01682`，up/tie/down 数量分别为
6,980 / 4,145 / 1,013。这说明均值下降不能概括整个分布，问题集中在负尾部。

## 5. 安全尾部归因

| Checkpoint | 任一 NC/DAC/TTC 回退比例 | 回退子集对总均值贡献 | 无回退子集贡献 |
|---|---:|---:|---:|
| step 300 | 8.25% | -0.04803 | +0.04216 |
| epoch 0 | 10.94% | -0.06491 | +0.04821 |
| epoch 1 | 9.45% | -0.05512 | +0.05202 |

step 300 的 repair/regress 计数：

| 指标 | repair | regress |
|---|---:|---:|
| NC | 24 | 271 |
| DAC | 178 | 221 |
| TTC | 57 | 636 |

step 300 最差 5% 样本对总 PDMS delta 贡献约 `-0.03657`，占所有负质量约
75.7%。因此当前问题应表述为 **tail-risk/generalization regression**，而不是
“所有场景都没有学到 reward”。

## 6. 固定 Seed NAVSIM v2 结果

| Checkpoint | EPDMS | 相对 Stage2 | EP | NC | DAC | TTC | LK | HC | EC | TLC | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage2 epoch 155 | 0.868010 | 0 | 0.881822 | 0.980776 | 0.953812 | 0.971678 | 0.966820 | 0.978594 | 0.821016 | 0.997777 | 0.994031 |
| R4 step 300 | 0.826458 | -0.041552 | 0.949516 | 0.960728 | 0.950272 | 0.944673 | 0.954800 | 0.967232 | 0.543625 | 0.993084 | 0.985921 |
| R4 epoch 0 | 0.795077 | -0.072932 | 0.964756 | 0.950395 | 0.944344 | 0.932406 | 0.952906 | 0.945908 | 0.350896 | 0.991273 | 0.986374 |
| R4 epoch 1 | 0.811487 | -0.056522 | 0.964193 | 0.953359 | 0.953483 | 0.937675 | 0.957517 | 0.965338 | 0.375598 | 0.991520 | 0.987815 |

按 136 个 log 聚类的 EPDMS delta 95% CI：

- step 300：`[-0.04910, -0.03159]`
- epoch 0：`[-0.08288, -0.06318]`
- epoch 1：`[-0.06775, -0.04846]`

v2 下降在当前协议下是统计明确的。最大单项变化是 extended comfort，同时 EP 大幅提高，
符合“更激进纵向策略”的轨迹漂移证据。

## 7. 训练内部证据

| 区间 | rollout scalar | reference scalar | delta | EP | TTC | policy loss | exact KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 前 300 step | 0.94728 | 0.93892 | +0.00837 | 0.93917 | 0.96182 | 0.09219 | 0.27512 |
| epoch 0 后段 | 0.95835 | 0.93959 | +0.01876 | - | - | 0.05250 | 0.22155 |
| epoch 1 | 0.95162 | 0.92831 | +0.02331 | - | - | 0.05437 | 0.27154 |
| epoch 2 截止 step 4340 | 0.95731 | 0.92811 | +0.02921 | - | - | 0.04164 | 0.28756 |

这些 train 指标不能直接和 navtest 等价：

- curriculum 使 epoch 0 到 epoch 1 的 sampled reference scalar 从约 `0.9389`
  降到 `0.9283`；
- reference source code 均值由约 `1.288` 变为 `1.359`；
- 因而 train delta 增长包含场景分布变化；
- navtrain rollout reward 高不保证 navtest 的安全尾部不退化。

Frontier 在 epoch 1 后的状态：

| 指标 | 值 |
|---|---:|
| sampler entropy | 11.257，最大可能值约 11.545 |
| effective sample size | 60,247 / 103,288，约 58.3% |
| max/median priority | 约 5.9 |
| 实际 uniform mixture | 约 19.92% |
| unseen ratio | 0 |

这不支持 sampler collapse 假设。

## 8. Credit 与梯度诊断

不改变算法行为的 10-step 探针：

```text
outputs/stage3_lfp_grpo_credit_diag10_logged_20260712T1618Z
```

| 诊断 | 10-step 均值 |
|---|---:|
| advantage mean / std | -0.11757 / 0.68825 |
| positive / negative / zero ratio | 23.96% / 49.88% / 26.15% |
| positive-eligible advantage mean | +0.27316 |
| TTC-fail ratio | 2.31% |
| TTC-fail advantage mean | -1.26014 |
| TTC-fail zero-credit ratio | 0.96% |
| progress-fail ratio | 6.47% |
| progress-fail advantage mean | -0.65177 |
| progress-fail zero-credit ratio | 7.41% |
| total policy gradient norm | 0.68354 |
| Planning Adapter gradient norm | 0.03752 |

未裁剪总梯度范数的 10 个值中，前两步为 1.248 和 1.060，之后降到 1 以下。
`max_grad_norm=1` 只影响了早期少数 step，不支持“训练一直被 gradient clipping
压死”的解释。

当前正式 R4 进程在梯度日志 hook 修复前已经启动，其旧日志是 AMP-scaled norm，不能用于
上述判断。这里使用的是修复后短探针的 unscaled norm。

## 9. 与历史 Core-Pareto v2 的可比结论

历史配置：

| 项目 | 历史 v2 |
|---|---:|
| LR | `1e-4` |
| epochs | 20 |
| per-rank batch / accumulation | 2 / 4 |
| 全局 scene batch | 64 |
| group size | 16 |
| BC | `0.10 -> 0.05`，5 epochs |
| reverse KL | 0.02 |
| advantage normalization | scene-local z-score |
| reward/credit | Core EP/TTC/comfort、reference EP floor、DDC guard、Pareto、slow/unsafe floor |

历史 run 的最佳报告点为 step 21600，PDMS `0.910274`。但该 run 使用旧 Stage2、旧表示和
非固定 initial-noise 的评估记录，不能与当前 fixed-seed 数字直接作差。历史 run 早期也存在
“EP 上升而 NC/DAC/TTC 回退”的振荡，例如 step 3000 PDMS `0.890736`，step 3600 降到
`0.881710`。因此旧算法并没有消除同类风险，只是在更长训练中出现过更高权重。

当前 Stage2 上的旧配方 10-step smoke：

| 指标 | Stage2 | S0 10-step | 差值 |
|---|---:|---:|---:|
| v1 PDMS | 0.873111 | 0.849294 | -0.023817 |
| v1 EP | 0.820586 | 0.819496 | -0.001090 |
| v1 NC | 0.980804 | 0.963627 | -0.017177 |
| v1 DAC | 0.953782 | 0.942495 | -0.011287 |
| v1 TTC | 0.942907 | 0.905586 | -0.037321 |
| v2 EPDMS | 0.868010 | 0.834137 | -0.033872 |

该 smoke 只证明完整旧配方在当前模型上初始更新过猛。它不能证明 BC、Core credit 或 KL
中的任一单项无效。

## 10. 已排除、未排除与已确认

| 假设 | 当前判断 | 证据 |
|---|---|---|
| LFP 没有实际梯度 | 排除 | DiT/adapter 均有 finite gradient |
| TTC/progress fail 大多被置零 | 排除 | 失败样本平均 advantage 明确为负 |
| Frontier sampler 已坍缩 | 排除 | entropy、ESS、uniform ratio 均正常 |
| Frontier 导致 step 300 退化 | 排除 | step 300 在 active curriculum 前 |
| 全局 batch 太小或不同 | 不支持 | 新旧均为 64 |
| 当前 LR 单独导致失败 | 未证明 | 当前仅 `1e-5`；旧 `1e-4` 全配方在当前模型上更差 |
| 训练优化了 progress，安全尾部泛化不足 | 确认 | EP、终点 x、tail attribution 一致 |
| v1 reward 与 v2 EPDMS 不完全对齐 | 确认 | v1 不含完整 EC/LK/TLC 目标，v2 EC 明显下降 |
| reverse-only trust region 锚定不足 | 合理但未验证 | 输出漂移、KL 方向差异支持，缺少单变量实验 |
| 旧 BC 的 forward-coverage 原理有价值 | 合理但未验证 | 数学上是 reference-chain cross-entropy，旧配方存在混杂变量 |

## 11. Scheduler 语义

R4 配置写了 `grpo_scheduler_warmup_epochs=1`，但当前 `WarmupCosLR` 是 epoch-level：

```python
lr = base_lr * (last_epoch + 1) / warmup_epochs
```

当 warmup epochs 为 1 时，第一个 epoch 就是完整 `1e-5`，因此实际上没有低 LR
warmup。TensorBoard 也显示 step 0 为 `1e-5`。这是配置语义与通常“从较小 LR
升到目标 LR”的理解不一致，但现有实验没有证明它是性能主因。

## 12. 当前不应做出的结论

- 不能因为 train scalar/reference delta 持续增大就宣称策略在 navtest 提升；
- 不能因为 v2 下降就断言 v1 evaluator 或 LFP 实现错误；
- 不能因为旧 v2 历史 top 更高就直接恢复完整旧配方；
- 不能把 epoch 1 的恢复归因于 frontier，缺少 curriculum-off 对照；
- 不能按单次随机 navtest 的点估计排序算法，必须使用固定噪声和配对统计；
- 不能从 loss 数值大小推断各项梯度主导程度，KL 与 policy loss 的梯度尺度尚未分解。

## 13. 若未来恢复探究，最小因果实验

本轮已经停止，不再启动下列实验。若以后恢复，应保持同一 Stage2、同一训练 token 顺序、
同一 seed、同一全局 batch 和固定 seed navtest，只改变一个变量：

| 条件 | 目的 |
|---|---|
| A：当前 LFP | 控制组 |
| B：仅提高 exact reverse KL | 检验 current-chain trust-region 强度 |
| C：A + exact Stage2-chain forward transition KL | 检验旧 BC 中的 distribution coverage 原理，避免旧 std mismatch |
| D：C + 保守的 TTC tail credit 改动 | 只有 C 仍出现 tail regression 时才检验 credit shaping |

每个条件先比较 300-step 参数漂移、终点 x 漂移、固定 seed v1/v2 和安全尾部；只有方向正确
才运行一个完整 epoch。不要同时更改 LR、credit、KL 和 curriculum。

## 14. 最终阶段性判断

LFP-GRPO 当前实现不是“没有工作”，它在 navtrain 上稳定产生正负 credit，并显著提高多数
rollout 的 progress。问题是这种更新在 navtest 上形成了较小但代价很高的安全负尾部，且
对 v2 extended comfort 的泛化明显变差。现有证据支持保留 LFP 的 on-policy、coherent
reference、global moments 和 frontier 结构作为研究基础，但不支持把当前 R4 宣称为已优于
Stage2，也不支持原样退回历史 Core-Pareto v2。

截至本报告，真正未决的核心问题是：**如何在保留 reward-driven policy improvement 的同时，
用与 diffusion transition 一致的 reference-distribution coverage 约束输出漂移。** 这一点需要
单变量 trust-region 对照才能下结论，本轮按要求停止继续探究。

## 15. 2026-07-12 16:56 UTC 恢复探究

### 15.1 R4 最新正式 watcher 结果

正式逐 epoch watcher 已完成 `epoch=2-step=4842`：

| Epoch | Step | v1 PDMS | v2 EPDMS |
|---:|---:|---:|---:|
| 0 | 1614 | 0.854536 | 0.794952 |
| 1 | 3228 | 0.868949 | 0.812215 |
| 2 | 4842 | 0.873864 | 0.815552 |

epoch 2 的 v1 子项为：NC `0.957448`、DAC `0.952463`、TTC `0.896688`、
EP `0.877827`、DDC `0.970794`、trajectory L1 `0.804498`。v2 子项为：
NC `0.957393`、DAC `0.952495`、TTC `0.940474`、EP `0.963599`、LK
`0.957599`、HC `0.970196`、EC `0.392829`、TLC `0.991849`、DDC
`0.987239`。

该 watcher 没有固定 initial-noise seed，因此不能把 epoch 2 的 `0.873864` 与固定 seed
Stage2 的 `0.873111` 直接作差。可以确认的是，R4 从 epoch 0 到 epoch 2 连续恢复，已经
回到 Stage2 点估计附近，并非持续单调退化。epoch 2 的 seed0 配对评估已启动，完成后再更新
统计显著性和安全尾部。

### 15.1.1 Epoch 2 固定 Seed 配对结果

epoch 2 的 seed0 配对评估随后完成：

| Checkpoint | v1 PDMS | EP | NC | DAC | TTC | DDC | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage2 epoch 155 | 0.873111 | 0.820586 | 0.980804 | 0.953782 | 0.942907 | 0.980269 | 0.283344 |
| R4 epoch 2 | 0.876641 | 0.880954 | 0.958230 | 0.954441 | 0.896935 | 0.970918 | 0.803424 |
| Delta | +0.003530 | +0.060368 | -0.022574 | +0.000659 | -0.045971 | -0.009351 | +0.520127 |

12,138 个配对样本中，PDMS up/tie/down 为 7,056 / 4,034 / 1,048，中位数 delta
为 `+0.022775`。按 1,203 个 scene token 做 sample-weighted cluster bootstrap，PDMS
delta 的 95% CI 为 `[-0.00308, +0.00999]`，bootstrap 中 delta 不大于 0 的比例约
14.5%。因此 epoch 2 已取得正点估计，但仍未达到常用 95% 统计显著性。

安全负尾部仍然决定均值：

| 项目 | Epoch 2 |
|---|---:|
| 任一 NC/DAC/TTC 回退比例 | 8.64% |
| 安全回退子集对总均值贡献 | -0.05008 |
| 无安全回退子集贡献 | +0.05361 |
| 安全回退子集平均 PDMS delta | -0.57948 |
| 无安全回退子集平均 PDMS delta | +0.05868 |
| 最差 5% 对总均值贡献 | -0.03801 |
| 最差 5% 占全部负质量 | 75.1% |

NC/DAC/TTC 的满分 repair/regress 数分别为 43/318、236/228、102/660。相对 epoch 1，
epoch 2 的负尾部比例从约 9.45% 降至 8.64%，clean subset 收益继续增加，解释了总 PDMS
恢复；但 NC/TTC regression 仍显著多于 repair。

同一组 predictions 的 v2 EPDMS 为 `0.819029`，相对 Stage2 `0.868010` 仍低
`-0.048981`。按 136 个 log 聚类 bootstrap 的 95% CI 为
`[-0.05672, -0.04143]`，明确为负。v2 EC 仅 `0.393028`，仍是最主要的跨协议短板。

### 15.2 原版 Core-Pareto v2 的早期历史

原版 v2 run 初始化于官方发布的 ReCogDrive Stage2 checkpoint：

```text
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/
  ReCogDrive_Diffusion_Planner_2B_IL.ckpt
```

该官方 Base 2B IL 在现有 full-navtest 记录中的 PDMS 为 `0.858575`，本次 seed0 重评为
`0.859505`。另一个 official-aligned A0 Stage2 checkpoint `step_00100000` 的 PDMS 为
`0.864891`，它是后续完成的本地复现实验，不是 Pareto-GRPO v2 的初始化权重。原版 v2 的
早期训练增益必须以官方 Base 2B IL 为基线，不能以 A0 代替。

原版 v2 的早期记录：

| Checkpoint | PDMS | NC | DAC | TTC | EP | DDC |
|---|---:|---:|---:|---:|---:|---:|
| step 300 | 0.887645 | 0.985706 | 0.961031 | 0.956747 | 0.830495 | 0.974172 |
| step 600 | 0.888936 | 0.985088 | 0.965398 | 0.954193 | 0.830769 | 0.979033 |
| epoch 0, step 1330 | 0.892644 | 0.983852 | 0.966881 | 0.955924 | 0.836308 | 0.968034 |
| epoch 1, step 2660 | 0.893123 | 0.984717 | 0.969682 | 0.957242 | 0.831955 | 0.972483 |
| epoch 2, step 3990 | 0.894053 | 0.981875 | 0.970588 | 0.953534 | 0.838887 | 0.972154 |

按点估计，原版 v2 从 step 300 起就高于 Base 2B IL `0.858575`，前三个完整 epoch 也没有
先低于 Stage2 再恢复。训练内部存在显著振荡，例如 step 1500 为 `0.881128`、step 1800
恢复到 `0.892090`、step 3600 降到 `0.881710`，但其 epoch 级早期结果仍明显高于旧
Stage2。

历史评估没有固定 initial-noise seed，也没有 scene-cluster bootstrap，因此不能精确量化早期
增益的置信区间。不过 `+0.028` 到 `+0.034` 的 epoch 级差距远大于当前 R4 相对 Stage2 的
千分位差距，足以说明两条训练曲线的早期形态确实不同。

### 15.2.1 旧 v2 Step 300 的固定 Seed 复核

本次重评首先复现官方 Stage2 seed0 基线 PDMS `0.859505`，随后得到旧 v2 step 300：

| Checkpoint | PDMS | NC | DAC | TTC | EP | Comfort | DDC | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Official Stage2 | 0.859505 | 0.979486 | 0.943895 | 0.937304 | 0.806867 | 0.999094 | 0.979857 | 0.259543 |
| Old v2 step 300 | 0.873191 | 0.980886 | 0.952381 | 0.947685 | 0.819039 | 0.989702 | 0.971700 | 0.413870 |
| Delta | +0.013686 | +0.001401 | +0.008486 | +0.010381 | +0.012171 | -0.009392 | -0.008156 | +0.154334 |

12,138 个配对样本的 PDMS up/tie/down 为 5,079 / 3,775 / 3,284。按 scene token 做
cluster bootstrap，PDMS delta 的 95% CI 为 `[+0.00743,+0.02004]`，20,000 次抽样中
delta 不大于 0 的比例约 `0.005%`。

因此可以确认：**旧 Pareto-GRPO v2 在官方 Stage2 上从 step 300 就已产生统计明确的提升，
并且 NC、DAC、TTC、EP 同时改善。** 它不是先经历类似 R4 epoch 0 的明显退化再恢复。
Comfort 和 DDC 有回退，但没有抵消其他核心项的收益。

这使后续 matched control 具有明确判别力：若完全相同的旧 v2 配方在 A5 Stage2 上 step 300
仍退化，主要矛盾更接近 Stage2 prior/FS-PTA 接口与 Stage3 更新尺度不匹配；若它在 A5 上也
提升，则当前 LFP credit/trust-region 设计是主要差异。该 matched control 正在排队执行。

固定 seed trajectories 还显示出两种早期更新方向不同：

| 300-step 更新 | mean abs x diff | endpoint signed x | endpoint signed y | L1 增量 |
|---|---:|---:|---:|---:|
| Old v2 / Official Stage2 | 0.609 m | -0.179 m | -0.166 m | +0.154 |
| R4 LFP / A5 Stage2 | 1.493 m | +3.087 m | +0.209 m | +0.397 |

旧 v2 在没有整体增加终点纵向距离的情况下同时提高 EP、NC、DAC 和 TTC；当前 LFP 的早期
收益方向则明显由纵向 progress shift 主导。由于两者 Stage2 架构和表示不同，这张表不能
单独作为算法因果结论，但它排除了“旧 v2 也只是同样向前平移，只是随机评估更幸运”的解释。

### 15.2.2 旧 v2 前三个 Epoch 的固定 Seed 完整曲线

| Checkpoint | PDMS | Delta vs Official Stage2 | Scene-cluster 95% CI |
|---|---:|---:|---:|
| Official Stage2 | 0.859505 | 0 | - |
| Old v2 step 300 | 0.873191 | +0.013686 | `[+0.00743,+0.02004]` |
| Old v2 epoch 0 | 0.887078 | +0.027573 | `[+0.02036,+0.03476]` |
| Old v2 epoch 1 | 0.891738 | +0.032233 | `[+0.02545,+0.03925]` |
| Old v2 epoch 2 | 0.894714 | +0.035209 | `[+0.02867,+0.04164]` |

固定 seed 和 scene-cluster bootstrap 下，旧 v2 从 step 300 到 epoch 2 的增益全部严格为正，
点估计持续提高。历史训练中单个中间 step 确实会振荡，但“头几个完整 epoch 先倒退”不成立。
这与当前 R4 的固定-seed 曲线 `-0.00587/-0.01670/-0.00310/+0.00353`
（step300/epoch0/epoch1/epoch2）形成清晰差异；R4 到 epoch2 的 CI 仍跨 0。

### 15.3 这是否说明当前 Stage2 有问题

目前不能直接下结论“Stage2 训练错误”，只能确认 **当前 PTA+FS Stage2 对 Stage3 更新的
响应与旧 Stage2 不同**：

- 新 Stage2 本身固定 seed v1 PDMS 为 `0.873111`，高于旧 Base 2B IL 和 A0 Stage2；
- 新 Stage2 的 L1 为 `0.283344`，输出质量并未表现为基础 checkpoint 失效；
- 当前 LFP 在新 Stage2 上前两轮退化、第三轮恢复；
- 旧 v2 完整配方在新 Stage2 上的 10-step smoke 已降到 PDMS `0.849294`，说明旧配方
  也不能无条件稳定该 checkpoint；
- 但 10-step smoke 同时改变了 LR、credit、BC 和 KL，不能区分是 Stage2 表示敏感，还是
  旧优化步过强。

严格区分需要同一新 Stage2 起点的原版 v2 中程曲线。当前状态：10-step 已完成，300-step
及 epoch 级对照尚未完成。epoch 2 seed0 评估结束后，将恢复这个单一对照，不同时修改其他
算法参数。

### 15.4 Stage2 质量与“宽分布前提”不是同一问题

已有 SNSAD v2 评估在固定 1,024 个 navtrain/v3-support 场景、每场景 32 次 DDIM 采样下，
给出了以下结果：

| Stage2 | Precision AUC | Recall AUC | KEMR AUC | Width ratio | SNSAD | Pairwise ADE |
|---|---:|---:|---:|---:|---:|---:|
| Official Stage2 | 0.905771 | 0.617563 | 0.504625 | 0.302735 | 0.485336 | 0.142481 m |
| A5 epoch 155 | 0.984085 | 0.639157 | 0.498695 | 0.221943 | 0.464131 | 0.087649 m |

A5 相比 Official Stage2 的 support precision 和 recall 更高，但 KEMR、相对宽度、SNSAD 和
pairwise ADE 更低。分层结果也显示 A5 在单模态场景更好，在 2+ 模式场景更窄。

因此需要区分两个命题：

1. **Stage2 checkpoint 是否是低质量策略：否。** A5 的 PDMS、support precision、GT ADE
   和 support recall 都较强。
2. **Stage2 是否提供了 LFP 最初假设的宽、支持集覆盖充分的 policy prior：否。** 现有实测
   表明 A5 更接近高精度窄分布拟合器。

这会直接影响 Stage3 解释。若同一 scene 的 G=16 rollout 主要落在同一窄模式附近，LFP
不是在多个可行策略模式间重分配概率，而更可能沿局部 reward 梯度整体平移当前模式。R4
观察到的终点 x 增加 3 到 4 m、EP 普遍提高和少量安全尾部回退，与这种“局部模式平移”一致。
但这是机制一致性证据，不是最终因果证明；还需要直接比较 Stage2/R4 的 group-level 物理
轨迹 spread 与支持模式覆盖。

阶段性判断应改为：**当前问题不是 Stage2 基础性能训练失败，而是 Stage2 没有交付 LFP
算法假定的宽 prior，Stage2 与 Stage3 的设计前提存在错位。** 原版 v2 是否能在同一 A5
起点上避免早期回退，仍由正在进行的 matched control 决定。

### 15.5 A5 Stage2 的 NAVSIM v2 基础质量

官方 Stage2 和 A5 epoch155 的 seed0 predictions 已使用相同 NAVSIM v2 one-stage EPDMS
scorer 重评：

| Stage2 | EPDMS | NC | DAC | TTC | EP | LK | HC | EC | TLC | DDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official Stage2 | 0.860313 | 0.979376 | 0.943932 | 0.969373 | 0.877169 | 0.968549 | 0.981887 | 0.855976 | 0.997777 | 0.993413 |
| A5 epoch155 | 0.868010 | 0.980776 | 0.953812 | 0.971678 | 0.881822 | 0.966820 | 0.978594 | 0.821016 | 0.997777 | 0.994031 |
| Delta | +0.007697 | +0.001400 | +0.009880 | +0.002305 | +0.004653 | -0.001729 | -0.003293 | -0.034960 | 0 | +0.000617 |

按 136 个 log 做 cluster bootstrap，EPDMS delta 的 95% CI 为
`[+0.00284,+0.01231]`，20,000 次抽样中 delta 不大于 0 的比例约 `0.1%`。

因此 A5 Stage2 在 v1 PDMS 和 v2 EPDMS 上都相对官方 Stage2 明确正向，不能解释为基础
checkpoint 训练失败。其 v2 改进主要来自 DAC、EP、NC 和 TTC；extended comfort 明显下降，
说明 Stage2 已存在舒适性 trade-off，但尚未抵消总 EPDMS。R4 Stage3 将 EC 从 A5 的
`0.821016` 进一步降到 epoch2 的 `0.393028`，这才是当前 v2 大幅退化的主要阶段。

补充的 v1 seed0 配对结果为：Official Stage2 `0.859505`，A5 epoch155 `0.873111`，
delta `+0.013606`，scene-cluster 95% CI `[+0.00836,+0.01893]`。因此 A5 的 v1/v2
基础质量改进在当前统一协议下都具有统计支持。

### 15.6 当前退化原因的证据分层

截至 epoch 2，A5 Stage2 epoch155 与 R4 LFP-GRPO 使用固定 `seed=0` 的同协议 v2
结果如下：

| Checkpoint | EPDMS | EP | NC | DAC | TTC | EC |
|---|---:|---:|---:|---:|---:|---:|
| A5 Stage2 epoch155 | 0.868010 | 0.881822 | 0.980776 | 0.953812 | 0.971678 | 0.821016 |
| R4 Stage3 epoch2 | 0.819029 | 0.963482 | 0.958299 | 0.954471 | 0.941874 | 0.393028 |
| Delta | -0.048981 | +0.081660 | -0.022477 | +0.000659 | -0.029804 | -0.427988 |

按 136 个 log 聚类 bootstrap，EPDMS delta 的 95% CI 为
`[-0.05672,-0.04143]`，所以这不是 initial-noise 或有限样本造成的表面退化。

目前可以分三层判断：

1. **已确认的直接原因：** R4 把策略明显推向更高 progress，但同时破坏安全尾部和 v2
   extended comfort。当前训练使用 `benchmark=navsim_v1`，v2 EPDMS、EC、LK、TLC
   不是训练目标；因此 v1 上能够由 EP 收益抵消的更新，在 v2 上没有对应保护。
2. **有较强证据的机制：** A5 Stage2 的 on-policy 分布比官方 Stage2 更窄，G=16 rollout
   更可能集中在单一局部模式。LFP 的相对 credit 因而表现为整体移动该模式，而不是在多条
   已有可行模式间重新分配概率。step 300 已观察到 endpoint x 平均增加 `3.087 m`，与
   EP 上升、TTC/NC 负尾部和 EC 崩落方向一致。
3. **仍未完成唯一归因的部分：** `reference_kl_coeff=0.005` 且无 BC 对 FS/PTA policy
   是否约束不足，以及 LFP credit 本身相对旧 Core-Pareto v2 的贡献，还需要同一 A5
   epoch155 起点的旧 v2 配方 300-step matched control。该对照已启动，在结果完成前不能把
   根因只归于 LFP 或只归于 Stage2 表示。

因此当前结论不是“A5 Stage2 质量差”，而是：**A5 Stage2 本身在 v1/v2 均优于官方
Stage2；当前 v1-LFP Stage3 的 progress-biased 更新破坏了它的 v2 优势，而 A5 的窄分布和
偏弱的 reference coverage 约束很可能放大了这种退化。**

### 15.7 旧 Core-Pareto v2 配方在 A5 起点上的 300-step 对照

固定 `seed=0` 的 matched control 已完成。旧 Core-Pareto v2 的 credit、BC、KL、batch、
G=16、LR=`1e-4` 和 300-step 配置保持不变，只将初始化权重替换为 A5 epoch155：

| 起点与更新 | v1 PDMS | Delta | Scene-cluster 95% CI |
|---|---:|---:|---:|
| Official Stage2 | 0.859505 | - | - |
| Official + old v2, step300 | 0.873191 | +0.013686 | `[+0.00743,+0.02004]` |
| A5 Stage2 epoch155 | 0.873111 | - | - |
| A5 + old v2, step300 | 0.863133 | -0.009978 | `[-0.01473,-0.00510]` |

同一 A5 control 的 v2 EPDMS 从 `0.868010` 降到 `0.832124`，delta 为
`-0.035885`，136-log cluster bootstrap 95% CI 为 `[-0.04156,-0.02999]`。
因此 v1/v2 的退化都不是评估噪声。

A5 control 的 v1 分项变化为：EP `+0.021329`、NC `-0.013964`、DAC
`-0.004861`、TTC `-0.034273`、DDC `-0.004614`、trajectory L1
`+0.226452`。NC/DAC/TTC 的 repair/regress 分别为 `24/193`、`161/220`、
`73/489`。v2 EC 从 `0.821016` 降到 `0.581275`。

固定 seed 轨迹相对 A5 起点的平均绝对 x/y/heading 漂移为
`0.891 m / 0.154 m / 0.0119 rad`，终点 signed x 为 `+1.740 m`。作为对照，旧
v2 配方在 Official Stage2 上的终点 signed x 是 `-0.179 m`。相同 reward recipe 在两个
Stage2 起点上产生了相反的局部更新方向。

这一结果排除了“当前退化完全由 LFP 新 credit 导致，恢复旧 v2 即可解决”的解释。更准确的
阶段性结论是：**旧 v2 的有效优化尺度和局部 reward geometry 没有迁移到 A5 FS/PTA
policy。** A5 本身的基础分数更高但 rollout 更窄；两种 Stage3 算法都把其局部模式推向更高
progress，并产生安全与舒适性负尾部。尚不能区分这是 LR 对 FS/PTA 过大，还是窄 prior 下
relative policy gradient 的结构性失配。

### 15.8 R4 Epoch 3 与“自行恢复”假设

正式 watcher 的 epoch3 非固定种子结果为：v1 PDMS `0.865092`、v2 EPDMS
`0.801399`、v2 EC `0.334562`；均低于 epoch2 的 `0.873864 / 0.815552 /
0.392829`。由于 watcher 没有固定 initial-noise seed，不能把该差值作为严格配对因果量，
但它至少否定了“epoch2 之后已经形成稳定单调恢复”的乐观判断。epoch3 的固定 seed 配对评估
尚未完成。

### 15.9 已排除的实现风险与下一对照

静态调用链复核排除了两个直接 correctness 怀疑：

- `grpo_denoised_clip_value=1.0` 和 `grpo_final_action_clip_value=1.0` 在 FS-Norm
  `stats_bounds` 模式下仅是 legacy fallback；`p_mean_variance()`、`sample_chain()` 和最终
  输出实际都调用 `_bound_output_representation()` 并应用 FS 统计上下界，没有错误裁到
  `[-1,1]`。
- 旧 v2 的 BC 不是直接对 `action_input.action` 使用 legacy `norm_odo()`；它从同表示的
  frozen old policy 采 teacher chain，再计算当前 policy log-prob，没有发现 GT/FS 编码旁路。

已在 `training-rl-zt3` 启动严格单变量 control：旧 v2 配方与 A5 起点保持不变，只将 LR
从 `1e-4` 调为 `1e-5`，仍训练 300 steps 并做固定 `seed=0` 的 v1/v2 navtest。若该对照
恢复或接近 A5，主因是更新尺度不匹配；若仍显著下降，则需要把重点转向 rollout coverage、
reference-relative physical trust region 和 credit 的 support-aware 约束，而不是继续调 LR。

### 15.10 FS-Norm 对 Stage3 探索尺度的定量影响

旧 Core-Pareto v2 的采样与 log-prob 超参数是在 legacy absolute-trajectory normalization
下设定的，没有按 FS-Norm 的逐步 delta 表示重新标定。legacy decoder 对一个 normalized
unit 的物理尺度为 `33.37 m / 21.0 m / 1.765 rad`。当前 FS v2 标准差通过 8 个 delta
累积到终点后，一个各 step 独立 normalized unit 的终点 RMS 尺度为
`4.616 m / 1.957 m / 0.173 rad`。

因此相同的最终 transition noise floor `sigma=0.04` 对应：

| Representation | Endpoint x RMS | Endpoint y RMS | Heading RMS |
|---|---:|---:|---:|
| Legacy absolute trajectory | 1.335 m | 0.840 m | 0.0706 rad |
| FS step delta | 0.185 m | 0.078 m | 0.0069 rad |

FS 下的物理探索宽度分别缩小约 `7.23x / 10.73x / 10.21x`。若只按终点 RMS 等效，
legacy `sigma=0.04` 对应的 FS normalized sigma 约为
`0.289 / 0.429 / 0.408`，而不是统一的 `0.04`。这不是建议直接把训练 sigma 提到该值；
多步 reverse process 仍需用真实 rollout spread 做标定，但它证明旧超参数在两种表示下不是
同一个物理策略分布。

这个尺度差异给出了目前最具体的机制解释：A5 的 G=16 group 在物理轨迹空间可能缺少旧 v2
所依赖的候选跨度，相对 advantage 更像是在几乎同一模式内放大微小 reward 差异，随后整体
移动 policy mean，而不是在多个可行候选间重分配概率。固定系数 KL 和 BC 也不能直接沿用，
因为其 transition distribution、score-function 梯度和物理输出漂移的对应关系已经改变。

PTA-FS-DiT 设计中原本定义了 A1=FS-only、A2=PTA-only、A3=FS+PTA 的干净消融，但当前
磁盘没有这些配置的完整训练 checkpoint，只有 A5 完整权重。因此现有证据把 FS-Norm 提升为
首要解释变量，但仍不能把 PTA 和 Stage2 窄分布的贡献数学上完全剥离。

### 15.11 Stage2 窄分布的实际训练权重证据

A5 的 TensorBoard 全 200 epoch 诊断显示，support archive 本身并不窄：每 scene 平均有
`10.279` 条 selected support，support 间平均物理距离为 `1.581 m`。训练也名义上每次采
`3.821` 条 target，但权重分布高度集中：

| 诊断 | 全程 epoch 均值 |
|---|---:|
| adaptive Pareto beta | 0.2709 |
| pre-sampling GT weight ratio | 0.7350 |
| pre-sampling external weight ratio | 0.2091 |
| pre-sampling effective target count | 1.842 |
| sampled target count | 3.821 |
| force-anchor hit ratio | 0.9897 |
| force-best hit ratio | 1.0000 |
| final diffusion-loss GT weight ratio | 0.8729 |

最后一行来自 `selected_target_is_gt_ratio`，是经过 target sampling 和重归一化后真正进入
epsilon/trajectory/feasibility losses 的权重。也就是说，A5 虽然读取了宽 support set，最终
约 `87.3%` 的训练质量仍分配给 GT，名义上的四目标监督只有约 `1.84` 条有效 target。

这些数字直接证明 support-set 知识没有充分转化为训练概率质量，但**不能单独解释 A5 为何
比 Official Stage2 更窄**：Official Stage2 使用 100% GT 训练，却具有更大的采样宽度。
因此 final GT mass `87.3%` 只能用于否定“当前 A5 已充分学习 support mixture”，不能作为
“GT 监督导致相对 Official 收缩”的因果证据。trajectory auxiliary 对每条 selected target
使用独立 noise 和一致 target，并没有跨 mode 直接做均值回归，目前也没有证据把
`trajectory_aux_weight=0.05` 认定为首要收缩原因。

### 15.12 是否重训 Stage2 与整体修改顺序

当前不应删除 FS-Norm，也不应立即从头重训 200 epoch：A5 epoch155 在 v1/v2 都显著优于
Official Stage2，说明表示对基础性能有效。优先顺序应为：

1. 先在 A5 epoch155 上修复 Stage3 的 FS-aware exploration、log-prob covariance、物理
   reference trust region 和 optimizer parameter groups，用 300-step probe 验证；
2. 若 Stage3 可以稳定保持 v1/v2，则保留当前 Stage2 作为性能主线；
3. 若论文仍要主张“Stage2 学到宽 support-aligned prior”，从 epoch155 做短程
   diversity-balanced fine-tune，而不是立即全量重训；
4. 只有短程 fine-tune 无法在保持 PDMS/EPDMS 的同时提高 SNSAD，才启动完整 Stage2
   重训和 A1/A2/A3 因果消融。

建议的 Stage2 fine-tune 不增加新的 imitation loss，而是修改现有 target distribution：限制
GT/anchor mass、为 multimodal scene 设置 Pareto mass floor、对 selected support 混入
scene-adaptive uniform/diversity mass，并把 effective target count 提到至少约 `3`。单模态 scene
不强制变宽。成功门槛应同时约束 PDMS/EPDMS、precision、recall、KEMR、width ratio 和
SNSAD，不能只看 pairwise ADE。

建议的 Stage3 首版保持简单：采样和 log-prob 使用同一 FS-aware 对角 covariance；加入以
decoded raw trajectory 计量的 frozen-Stage2 x0 trust penalty/deadband；低 spread group 不做
global z-score 放大；安全/comfort 相对 reference 回退时禁止正 credit；PTA/gates 首 epoch
冻结或使用 DiT LR 的 `0.1x`。frontier curriculum 在 300-step 稳定性通过前关闭。

### 15.13 Official GT-only 反例带来的因果修正

Official Stage2 的 GT-only 训练是重要反例。它在 SNSAD 评估中比 A5 更宽，但并不意味着它
学到了更好的 support modes：

| Stage2 | Precision AUC | Recall AUC | KEMR AUC | Width ratio | Pairwise ADE |
|---|---:|---:|---:|---:|---:|
| Official GT-only | 0.905771 | 0.617563 | 0.504625 | 0.302735 | 0.142481 m |
| A5 | 0.984085 | 0.639157 | 0.498695 | 0.221943 | 0.087649 m |

Official 更宽，但 precision 更低、recall 也更低，KEMR 只略高。这说明其额外宽度至少有一部分
可能是 legacy representation/sampler 产生的 off-support stochastic spread，而不是有语义的
多策略覆盖。旧 v2 Stage3 可能恰好利用了这种较大的物理探索来获得 reward variance；这和
“Stage2 已学到 Pareto 多模态 policy”不是同一个命题。

因此需要把两个问题分开：

1. **Stage3 是否有足够探索：** 可以通过 FS-aware on-policy covariance 在 Stage3 注入，
   不要求先重训 Stage2；
2. **Stage2 是否显式建模 support distribution：** 当前 A5 没有充分做到，若论文需要该主张，
   才需要 diversity-balanced Stage2 fine-tune。

最先做的因果实验应保持 A5 epoch155 权重不变，只扫描 FS sampling covariance，测量 raw-space
group spread、reward std、feasible/Pareto 数、SNSAD precision-recall-width 曲线以及 v1/v2。
若存在一个 covariance 区间能提高 Stage3 candidate quality 而不破坏 A5 基线，就不应重训
Stage2。只有调宽 sampler 后仍然无法覆盖 support modes，才说明 model score field 本身已
收缩，需要修改 target distribution 并 fine-tune Stage2。

### 15.14 FS-aware Stage3 最小修复与实验拆分

已按上述因果顺序完成一版默认关闭的最小代码修改，没有改变当前 R4 进程和 legacy GRPO：

1. 新增物理终点方差到 FS normalized step-delta covariance 的映射。配置给定
   `x/y/heading` 的终点标准差后，每个 step 使用等物理 delta 方差，再除以 FS 的逐 step、
   逐轴 scale。标量 floor 仍作为下限。
2. `sample_chain()` 与 `_chain_transition_distribution()` 调用同一个 floor helper；因此采样、
   trajectory log-prob 和 frozen Stage2 exact KL 使用完全一致的对角 Gaussian，不引入
   off-policy likelihood mismatch。
3. 对 decoded raw rollout 计算 group pairwise ADE、终点 x/y 标准差和 circular heading
   标准差。所有日志都是 scalar。
4. 新增可选 low-information credit gate。只有 group pairwise ADE 和 official scalar span 都
   达到阈值时，该 scene 的 relative credit 才有效；随后可选 advantage deadband。gate 在
   REINFORCE 和 BPAE/frontier energy 之前生效，避免 curriculum 继续放大无信息 scene。

所有新字段默认值为关闭：

```text
fs_transition_std_enabled=false
fs_endpoint_std_x_m=0
fs_endpoint_std_y_m=0
fs_endpoint_std_heading_rad=0
min_group_pairwise_ade_m=0
min_group_scalar_span=0
advantage_deadband=0
```

当前 A5 FS stats 下，原 `0.04` floor 的物理终点 RMS 已由代码复算为
`0.1846 m / 0.0783 m / 0.00692 rad`。第一轮 probe 不直接跳到 legacy-equivalent，而采用
逐级、单变量扫描：

| Profile | x | y | heading | credit gate |
|---|---:|---:|---:|---|
| `baseline` | current scalar floor | current | current | off |
| `cov_c2` | 0.40 m | 0.16 m | 0.014 rad | off |
| `cov_c4` | 0.80 m | 0.32 m | 0.028 rad | off |
| `gate_only` | current | current | current | ADE 0.03 m, span 0.005, deadband 0.05 |
| `cov_c2_gate` | 0.40 m | 0.16 m | 0.014 rad | same gate |

`cov_c2` 和 `cov_c4` 约为当前 floor 的 2x/4x 物理尺度；它们不是论文最终超参数，只用于
定位“探索不足”和“credit 噪声放大”各自的贡献。新增 probe launcher 为
`scripts/training/sg_fps/run_train_lfp_grpo_fs_probe.sh`。

`min_group_pairwise_ade_m=0.03` 也不是用来强迫策略变宽。A5 当前 1024-scene、每 scene
32 samples 的 raw pairwise ADE 分布中，p10/p50 分别约 `0.041/0.057 m`，低于 `0.03 m`
的 scene 仅 `1.86%`。因此该阈值只过滤近乎完全 collapse 的 group；若 1-step Stage3
calibration 显示实际过滤比例明显更高，说明训练时 G=16 reverse chain 比离线 diversity
评估更窄，届时再按实测分位数调整，不能直接固定该值。

静态检查和 LFP 全套定向测试已通过：67 个原有/扩展测试通过；新增的 transition integration
测试进一步确认 2-step DDIM sampling 与 log-prob distribution 使用同一 FS floor，且
FS-aware REINFORCE 梯度可以回传到 DiT/PTA、frozen Stage2 reference 无梯度。长训练尚未
基于新代码启动，等待当前低 LR old-v2 matched control 释放远端资源后，先运行短 probe。

### 15.15 2026-07-12 20:38 UTC 运行状态

R4 训练最新 TensorBoard step 为 `7149`，epoch4 checkpoint 尚未生成。最近一个完整、自动
NAVTEST 结果仍为 epoch3：v1 PDMS `0.865092`，v2 EPDMS `0.801399`，v2 EC
`0.334562`。epoch2 仍是当前 watcher 的 v1/v2 双口径 top1：`0.873864 / 0.815552`。

训练内 metric 从 epoch0 到 epoch3 为 `0.955874, 0.952433, 0.958837, 0.961474`，而
exact KL 为 `0.231954, 0.271635, 0.291028, 0.291166`。训练 reward 上升但 NAVTEST 在
epoch3 回落，进一步说明不能把 online evaluator reward 的单调变化等同于泛化提升。

旧 Core-Pareto v2 + A5 的 `LR=1e-5` 单变量 300-step 对照已在 `training-rl-zt3` 使用 8 卡
启动。GPU 有实际负载，但首 batch 超过 30 分钟尚未写出进度；该路径的旧 evaluator 明显比
LFP batched evaluator 慢。当前不把“未出 step”解释为训练结果，并继续监控进程与 checkpoint。

### 15.16 梯度范数日志的复核

TensorBoard 的四个 `lfp_policy_gradient_norm_epoch` 都是 `inf`，但旧实现是在
`on_after_backward` 记录，因此数值仍乘有 AMP GradScaler。step 可见值 `2656--30344` 和 PTA
`259--2873` 不是实际 pre-clip norm。epoch reduction 同时还会受到旧 float32
`sum(grad^2)` 聚合溢出的影响，所以这些日志不能用于判断 clipping 饱和或梯度爆炸。

optimizer state 给出了更精确的答案：epoch0--3 的 `global_step` 分别为
`1614/3228/4842/6456`，对应 Adam step 为 `1613/3226/4839/6452`，即累计只跳过
`1/2/3/4` 次 update，约 `0.062%`。AMP scale 在前三个 checkpoint 为 `32768`，epoch3 为
`16384`。因此确有极少量 non-finite batch，足以让 epoch mean 变成 `inf`，但远不足以解释
持续 NAVTEST 退化。按各 checkpoint 保存的 GradScaler 反缩放后，policy norm 的 epoch0--3
均值分别为 `0.413/0.404/0.381/0.445`，p90 为 `0.602/0.590/0.546/0.711`。前三个 epoch
的可见 step 没有超过 clip `1.0`，epoch3 仅约 `4.35%` 超过，最大 `1.423`。因此 R4 并非
长期处于 clipping 饱和区，不能用“方向归一化更新”解释退化。

诊断代码已移到 `on_before_optimizer_step`，此时 Lightning 已完成 AMP unscale、尚未 clipping；
同时改用 max-rescaled L2 accumulation并记录 policy/PTA finite ratio。该修改只影响日志，
不改变反向传播或实际 clipping。新 probe 的真实 pre-clip norm 约为 `0.27--0.61`，与上述
反缩放结果一致。

### 15.17 旧 v2 配方低 LR matched control

严格单变量 control 已完成：A5 epoch155 起点、旧 Core-Pareto v2 的 credit/BC/KL/G=16/
batch/300 steps 全部不变，只把 LR 从 `1e-4` 降到 `1e-5`。固定 seed0 结果：

| 配置 | v1 PDMS | Delta vs A5 | v2 EPDMS | Delta vs A5 |
|---|---:|---:|---:|---:|
| A5 Stage2 | 0.873111 | - | 0.868010 | - |
| old v2, LR=1e-4 | 0.863133 | -0.009978 | 0.832124 | -0.035885 |
| old v2, LR=1e-5 | 0.872955 | -0.000156 | 0.847939 | -0.020071 |

低 LR 的 v1 scene-cluster paired delta 为 `-0.000134`，95% CI
`[-0.00336,+0.00302]`，不能拒绝与 A5 持平。相同 predictions 的 v2 136-log cluster
delta 为 `-0.020071`，95% CI `[-0.02453,-0.01580]`，仍然是明确退化。

低 LR v2 相对 A5 的 component delta 为：EP `+0.03389`、NC `-0.01009`、DAC
`+0.00074`、TTC `-0.01153`、LK `-0.00593`、HC `-0.01029`、EC `-0.16474`、TLC
`-0.00313`、DDC `-0.00202`。v1 同样表现为 EP `+0.02472`，但 NC/TTC 分别下降约
`0.01018/0.02332`，只是 v1 乘积总分恰好抵消。

这个 control 将根因拆成了两个独立部分：

1. **更新尺度不匹配已被确认。** 旧 v2 的 `1e-4` 对 A5 FS/PTA 过大；降为 `1e-5` 后，
   300-step v1 总分退化基本消失。
2. **训练目标缺失仍然存在。** 即使低 LR、旧 BC 和 `KL=0.02` 同时存在，v1 reward 仍会
   用 progress 换取 v2 extended comfort 和部分安全，EPDMS 仍显著下降。只调 LR 或恢复旧
   v2 不能得到 v1/v2 同时改进。

因此下一版不能把“扩大 FS exploration”单独作为完整解法。covariance calibration 仍有必要，
因为它决定 relative credit 是否有可辨识候选；但正式 300-step 配置还必须同时加强 frozen
reference trust 或直接使用 v2 objective/guard。否则更宽 exploration 只会更快找到高 EP、低
EC 的方向。

### 15.18 FS on-policy covariance calibration

在同一 A5 epoch155、同一 seed0、同一首个 global batch（64 scenes、每 scene 16 rollouts）
上完成了四组严格对照。结果文件位于
`outputs/lfp_fs_calibration_matrix_seed0_r2_20260712T2116Z/summary.tsv`。

| Profile | pairwise ADE | scalar span | scalar | feasible | positive adv | policy grad |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.3834 m | 0.1251 | 0.9226 | 99.22% | 41.31% | 1.2480 |
| cov_c2 | 0.6760 m | 0.2267 | 0.8998 | 96.48% | 10.25% | 1.0332 |
| cov_c4 | 1.3234 m | 0.3938 | 0.8724 | 92.87% | 26.86% | 0.3794 |
| gate_only | 0.3834 m | 0.1251 | 0.9226 | 99.22% | 6.05% | 1.0769 |

结论与此前“FS Stage2 太窄，所以 Stage3 没有探索”的假设不一致：Stage2 普通 fixed-seed
sampling 的 pairwise ADE 约为 `0.0877 m`，但当前 Stage3 reverse transition floor 已把训练
group 扩到 `0.3834 m`，同时 scalar span 达到 `0.1251`。因此当前 on-policy relative credit
有足够候选差异，Stage3 退化不能归因于 rollout collapse。

继续扩大 covariance 会以近似单调方式降低候选质量：C2/C4 相对 baseline 的 scalar 分别下降
`0.0227/0.0502`，可行率下降 `2.73/6.35` 个百分点；C4 的 TTC 也降到 `0.9707`。更宽并不
等于更有用，主方案应保留当前 transition floor。新增 FS-aware covariance 能力保持默认关闭，
作为后续消融工具，不进入下一轮正式配置。

`gate_only` 的 trajectory 和 metric 与 baseline 完全一致，因而是纯 credit 对照。当前阈值将
active group ratio 降到 `68.75%`，并把正优势比例从 `41.31%` 压到 `6.05%`；这不是修复低
信息 scene，而是过度丢弃学习信号。gate 也保持默认关闭。下一项单变量实验改为 baseline
covariance、无 gate，只将 frozen Stage2 exact-KL 系数从 `0.005` 提高到 `0.02`，在 step300
用相同 seed0 同时评估 v1/v2，以检验 reference trust 是否能阻止早期安全与 comfort 漂移。

### 15.19 Frontier curriculum 与真实多策略结构的对齐度

新增可复现分析脚本 `scripts/evaluation/analyze_lfp_frontier_alignment.py`，读取 checkpoint 中的
103,288 个 scene frontier EMA、coherent reference cache，并与 1,024-scene 的 v3 support
diversity 评估逐 token 对齐。完整结果为
`reports/recogdrive_stage3/r4_frontier_alignment_epoch3.json`。

R4 epoch3 的 fast BPAE 有 `45.36%` scene 为 0，均值 `0.03233`。它与 reference scalar 的
Spearman 相关为 `-0.515`，与 GT/Stage2 scalar 绝对分歧为 `+0.485`，说明现有 sampler
确实在主动转向 policy/reference 有分歧的场景，而不是 uniform sampling。sampler ESS 为
`60,902 / 103,288`，entropy `11.259`，也没有离散 collapse。

但它与真正的多策略结构对齐很弱：

| v3 support / policy 指标 | Spearman(fast BPAE, metric) |
|---|---:|
| support dispersion | -0.028 |
| reference mode count | -0.111 |
| kernel effective-mode coverage AUC | +0.087 |
| policy raw pairwise ADE | +0.258 |
| support reward mean | -0.559 |

因此当前 BPAE curriculum 更接近“低 reward / teacher-disagreement hard mining”，尚不能声称是
面向多策略 Pareto 冲突的主动学习。为把课程机制与核心创新对齐，新增了默认关闭的
`frontier_tradeoff_weight`：对每个 scene 统计可行 rollout pair 在 EP/TTC/quality 上同时存在
gain 和 loss 的比例，并用该强度调制 BPAE。共享排序的候选不被视作冲突，多目标排序相反的
候选才提高相对课程权重。默认 0 完全恢复原 BPAE；第一轮只做 calibration，不直接进入长训练。

同时新增两个默认关闭的正信用约束：

1. `reference_pareto_gate_enabled`：把同一条 coherent reference 作为虚拟 Pareto 候选，被其在
   EP/TTC/quality 同时支配的 rollout 不得正信用；
2. `quality_positive_credit_guard`：quality 低于 reference 超过容差时不得正信用，仍允许容差内
   Pareto trade-off。

新增诊断会直接记录正优势中低于 reference scalar/quality、以及被 coherent reference 支配的
比例。所有行为开关默认关闭，legacy output 不变；当前 LFP 定向测试为 `72 passed`。

### 15.20 Frozen-reference KL=0.02 matched control

完成 A5 epoch155、LR `1e-5`、G=16、global batch64、无 curriculum 的 300-step 单变量
control；相对原 R4 step300 只把 exact transition KL 从 `0.005` 提到 `0.02`。训练前 300
step 的 exact KL 均值从 `0.2751` 降到 `0.0974`，TTC 从 `0.9618` 提到 `0.9758`，EP 从
`0.9392` 降到 `0.9285`，scalar 只变化 `-0.00088`。这证明 stronger frozen-reference trust
能在几乎不损失 online scalar 的情况下抑制激进纵向漂移。

固定 seed0 NAVTEST：

| 配置 | v1 PDMS | v2 EPDMS | v2 EC |
|---|---:|---:|---:|
| A5 Stage2 | 0.873111 | 0.868010 | 0.821016 |
| R4 LFP, KL=0.005, step300 | 0.867241 | 0.826458 | 0.543625 |
| LFP, KL=0.02, step300 | 0.868668 | 0.836434 | 0.599801 |

相对 R4，KL=0.02 的 v1 PDMS delta 为 `+0.001427`，scene-cluster bootstrap 95% CI
`[-0.00157,+0.00448]`；aggregate 提升尚不显著，但 NC `+0.00449`、TTC `+0.00840`、
DDC `+0.00391` 和 L1 `-0.09667` 的 CI 均不跨 0。v2 EPDMS delta 为 `+0.009976`，
136-log cluster CI `[+0.00654,+0.01360]`，统计明确；其中 NC `+0.00449`、TTC
`+0.00840`、EC `+0.05618`，EP `-0.01252`。

但相对 A5，KL=0.02 的 v1 delta 仍为 `-0.004443`，CI
`[-0.00925,+0.00032]`；v2 delta 为 `-0.031576`，CI
`[-0.03745,-0.02570]`。NC/TTC/DDC 与 v2 EC 回退均显著，EP 则显著上升。因此：

1. `KL=0.005` 对 FS/PTA policy 明显偏弱，后续候选配置应以 `0.02` 为 trust 基线；
2. KL 只能约束更新距离，不能判断哪个 rollout 应获得正信用；
3. 即使 policy 靠近 Stage2，当前 group-relative Pareto credit 仍会用安全和 comfort 换 progress；
4. 下一项因果实验必须作用于 coherent-reference positive eligibility，而不是继续增大 KL 或噪声。

### 15.21 R4 epoch4 与停止决定

正式 R4 epoch4（step8070）NAVTEST 为 v1 PDMS `0.870327`、v2 EPDMS `0.808869`。它比
epoch3 回升，但两种口径均未超过 epoch2 的 `0.873864 / 0.815552`，也仍明显低于固定
seed0 A5 的 `0.873111 / 0.868010`，尤其 v2 不存在继续训练可解释的改善趋势。

训练在 epoch5 step8189 后以 graceful SIGTERM 停止；GPU 进程已退出，epoch0--4、step8100
及各 epoch predictions 均保留。当前 top3 为：

| Rank | v1 epoch / PDMS | v2 epoch / EPDMS |
|---:|---|---|
| 1 | epoch2 / 0.873864 | epoch2 / 0.815552 |
| 2 | epoch4 / 0.870327 | epoch1 / 0.812215 |
| 3 | epoch1 / 0.868949 | epoch4 / 0.808869 |

停止依据不是“暂时没涨”这一点估计，而是三条独立证据一致：KL=0.02 matched control 已证明
旧 trust 太弱；FS covariance control 已排除 exploration collapse；frontier alignment 已证明
现有课程主要挖低 reward/teacher disagreement，而不是多策略 Pareto conflict。继续跑旧
`KL=0.005 + group-relative credit` 不会检验任何剩余假设，故不再消耗后五个 epoch。

### 15.22 安全优先语义复核与 safety-first frontier smoke

术语统一如下：本文的 `NAVSIM v1` 和 `NAVSIM v2` 分别指 PDMS 与 EPDMS 评估口径；历史
Stage3 算法一律写作 `Pareto GRPO v2`，它与 NAVSIM v2 评估口径没有对应关系。

代码复核确认 LFP rollout credit 保留了 Pareto GRPO v2 的词典序安全结构。候选先通过：

```text
NC == 1
DAC == 1
DDC_candidate >= GT_DDC - 0.01
NAVSIM v2 额外要求 TLC == 1
```

只有 feasible 候选可进入 EP/TTC/quality Pareto front；mixed group 中 infeasible rollout 固定
负优势，all-infeasible rescue 也只允许非正优势。因此此前把“首批 Pareto tradeoff intensity
为 0”解释成 Pareto 本身无效是不充分的。Pareto 同时承担安全域内的 non-dominated screening，
不要求每个 batch 都存在大量互相冲突的 objective pair。

本机 8 卡、10 个全局 batch 的诊断覆盖约 640 scenes / 10,240 rollouts：

| Metric | Mean | Range |
|---|---:|---:|
| feasible ratio | 0.98848 | 0.97461--0.99902 |
| NC | 0.99932 | 0.99805--1.00000 |
| DAC | 0.98994 | 0.97559--0.99902 |
| DDC | 0.98896 | 0.96338--1.00000 |
| Pareto-front ratio | 0.46436 | 0.40723--0.52539 |
| dominated ratio | 0.43691 | 0.35059--0.50000 |
| tradeoff-pair intensity | 0.00773 | 0--0.02850 |
| unsafe advantage mean | -0.38750 | -0.62500---0.12500 |

10 个 batch 均无 all-infeasible group，说明当前 Stage2 rollout 的 train-time safety 已较高；
主要失效来自 DAC/DDC，NC 基本饱和。真实多目标冲突存在但稀疏，因此课程不能只按 tradeoff
intensity 加权。远端同 seed 首批矩阵也表明：quality guard 对 credit 完全无影响；coherent
reference Pareto gate 只移除 0.25% 正优势；纯 tradeoff curriculum 会在首批把 frontier
energy 压到 0。这三项均不进入主配置。

更重要的缺口在 curriculum：原 BPAE 对 all-negative/all-infeasible group 给 0 energy，与“先
修安全、全安全后再优化 Pareto”的词典序目标不一致。新增默认关闭、可消融的
`frontier_safety_first`：

```text
all-feasible group:  energy = BPAE
mixed group:         energy = 4 * safe_fraction * unsafe_fraction * mean(abs(A))
all-infeasible:      energy = mean(abs(A))
```

policy advantage、hard guard、global normalization 和 exact KL 均不变；只改变 epoch-end
frontier state 和下一 epoch sampler weights。dataclass/Hydra 默认仍为 false，保持旧 checkpoint/
config 行为；新的 V1/V2 LFP 训练入口显式默认 true。训练入口的 reference KL 同时按 matched
control 结论从 `0.005` 调整为 `0.02`，dataclass 旧默认仍保留。

使用同一 A5、seed0、G=16、global batch64、KL=0.02 做了旧 BPAE 与 safety-first 的
2 epoch x 10 batch 配对 smoke。epoch0 用 uniform sampler 生成 frontier state，epoch1 才使用
新权重。safety-first 相对旧 BPAE 的 epoch1 rollout aggregate 为：feasible `+0.00049`、NC
`+0.00098`、DAC `+0.00078`、DDC `+0.00322`、TTC `+0.00000`、EP `+0.00596`、scalar
`+0.00393`。smoke 规模不足以宣称 navtest 提升，但确认 sampler 能更新、20% uniform mixture
保留、数值/梯度 finite，并且没有出现“安全优先必然压低 progress”。

本轮验证为 `75 passed, 16 warnings`，另通过 py_compile、bash syntax 和 `git diff --check`。
完整训练和 navtest 尚未基于 safety-first 版本启动；下一步应先做 300-step matched control，
再决定是否投入完整 Stage3，不能从 20 个 train batch 外推最终 PDMS/EPDMS。

### 15.23 组内标准化与 DDP-global 标定的严格对照

这里的 `NAVSIM v1/v2` 仍只表示 PDMS/EPDMS 评估口径；本节所说的旧 v2 是历史
`Pareto GRPO v2` 算法。当前 LFP 并没有用跨 scene 的 batch mean 给 rollout 排名。两种
方案都先在每个 scene 内减去 feasible rollout 的均值，唯一变量是 centered score 的尺度：

```text
global_std:       A_i,g = centered_i,g / std_DDP(all feasible centered scores)
scene_group_std:  A_i,g = centered_i,g / max(std_g(feasible centered_i), 0.05)
```

因此这项实验检验的是优势尺度，不是把不同 scene 的 reward 直接互相排序。两组均从 A5
epoch155 开始，使用相同 seed、LR=`1e-5`、G=16、global scene batch=64、KL=`0.02`、
300 steps，并关闭 curriculum；只改变上述 normalization。权重与完整评估位于：

```text
outputs/lfp_norm300_global_seed0_20260712T2304Z
outputs/lfp_norm300_scene_group_seed0_20260712T2304Z
```

固定 seed0 的完整 NAVTEST 结果：

| 配置 | v1 PDMS | v1 NC | v1 TTC | v1 L1 | v2 EPDMS | v2 NC | v2 TTC | v2 EC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 Stage2 | 0.873111 | 0.980804 | 0.942907 | 0.283344 | 0.868010 | 0.980776 | 0.971678 | 0.821016 |
| global std | 0.868840 | 0.965192 | 0.904927 | 0.582774 | 0.836221 | 0.965215 | 0.954059 | 0.598606 |
| scene-group std | 0.863489 | 0.961279 | 0.892898 | 0.619939 | 0.828873 | 0.961304 | 0.947061 | 0.568227 |

组内标准化相对 global std 的 paired delta 为：v1 PDMS `-0.005351`、NC `-0.003913`、
TTC `-0.012028`、L1 `+0.037164`；按 1,204 个 scene cluster bootstrap，95% CI 分别为
`[-0.00740,-0.00330]`、`[-0.00543,-0.00254]`、`[-0.01450,-0.00972]`、
`[+0.03495,+0.03928]`。v2 EPDMS delta 为 `-0.007348`，136-log cluster CI
`[-0.00963,-0.00520]`；EC `-0.030378`、NC `-0.003911`、TTC `-0.006998` 的 CI 也均
不跨 0。两种口径一致否定“恢复旧组内标准化即可修复当前退化”。

首批 credit 诊断解释了为何组内标准化反而更差：它把 advantage std 从约 `0.743` 降到
`0.611`，但也把低 spread scene 按自身很小的 std 放大到与高信息 scene 同等尺度。该操作
消除了 scene reward spread 所携带的可靠性信息，并不能只用“梯度总体更小”来判断更安全。
此外，global 模式在 exact tie 附近存在 float rounding 产生的极小正值，影响 BPAE 的符号计数；
这些值对 policy gradient 的实际贡献极小，且 scene-group 方案修正该计数后 navtest 反而更差，
所以它不是当前策略退化的主因。后续可单独给 frontier sign 加 deadband，但不应借此把
scene-group std 设为主方案。

Stage2 过拟合假设同样缺少证据。A5 的 v1 epoch100--129/130--154/155--179/180--200
平均 PDMS 分别为 `0.86667/0.86942/0.87123/0.87103`，对应 L1 均值持续从 `0.29790`
降至 `0.28864`。epoch155 是平台期最好点，后续 navtest 没有持续下滑；它相对官方 Stage2
在 v1 和 v2 上还分别有统计明确的 `+0.01361/+0.00770`。A5 的 policy prior 确实比官方
Stage2 更窄，但当前 Stage3 rollout floor 已产生约 `0.383 m` pairwise ADE 和 `0.125`
scalar span，因而“完全没有可比较候选”也已被 calibration 排除。

当前退化应分为三个已被数据支持的因素：

1. v1 on-policy credit 持续偏向 progress；300-step global std 相对 A5 的 EP `+0.05460`，
   但 v2 EC `-0.21404`、NC `-0.01725`、TTC `-0.02025`。
2. hard safety gate 只约束当前 navtrain sampled rollout；训练 feasible 约 99%，安全失败信号
   稀疏，不能保证共享 DiT 更新后 held-out deterministic DDIM 的安全尾部不变。
3. exact transition KL=`0.02` 已明显优于 `0.005`，但没有物理 trajectory trust 语义；两种
   normalization 的 L1 都从 `0.283` 增至 `0.58+`，说明 frozen Stage2 附近约束仍不足。

阶段性算法决定：保留 **scene mean centering + DDP-global std**，不恢复纯 scene-group std；
正式方案继续以 KL=`0.02` 为最低 trust 基线，并优先验证 reference-relative physical
trajectory trust / v2 comfort-aware objective，而不是继续调 advantage normalization。若要兼顾
不同 scene 的 reward spread，可后续测试 global/scene variance shrinkage 或有界 scale ratio，
但必须作为独立消融，不能与 reward、KL、curriculum 同时改变。

### 15.24 支持集容量归一化 frontier curriculum

#### 动机与定义

对 R4 epoch3 frontier state 的静态审计显示，旧 BPAE priority 与支持集 dispersion、effective-mode
coverage 的 Spearman 相关仅为 `-0.028/+0.087`，却与 reference scalar、support reward mean
呈 `-0.515/-0.559`。因此旧课程更接近“挖低回报/师生分歧场景”，没有直接表达本文希望的
“安全可行前提下，主动学习尚未覆盖的多策略容量”。

新增默认关闭的 capacity-normalized frontier。对 scene `s`：

```text
mode_capacity_s = 1 - 1 / max(reference_mode_count_s, 1)
coverage_ratio_s = clip(
    (policy_group_pairwise_ADE_s + 0.05)
    / (support_pairwise_ADE_s + 0.05),
    0, 1,
)
coverage_gap_s = mode_capacity_s * (1 - coverage_ratio_s)
bonus_s = 0.05 * coverage_gap_s * 1[all rollouts feasible]
frontier_energy_s = safety_first_energy_s + bonus_s
```

`support_pairwise_ADE` 使用 v3 selected positive support 的 density-balanced 原始 XY pairwise ADE，
单位为米；不读取 hard negative。只有 all-feasible group 获得 diversity bonus，mixed/all-unsafe
scene 仍完全由 safety-first energy 排序。该信号只更新下一 epoch sampler priority，不进入
reward、advantage 或 policy loss，因而没有把 support archive 变成 imitation loss。

完整 v3 cache 含 `103,288` scenes：support pairwise ADE 均值 `1.36082m`，p10/p50/p90 为
`0.64282/1.23913/2.36509m`；reference mode count 均值 `4.9077`，多模态 scene 比例
`92.62%`。缓存格式为 version 2，并记录 archive fingerprint 和 metric config。此前试制的
version 1 把无量纲 SNSAD distance 与米制 policy ADE 相除，存在单位错误，已禁止加载且未用于训练。

#### 配对协议

两组均从 A5 epoch155 开始，seed0、LR=`1e-5`、G=16、KL=`0.02`、scene mean centering +
DDP-global std、safety-first frontier；各占本机 4 GPU，单卡 batch 8、accumulate 1，global
scene batch 32。训练 `2 epochs x 150 batches = 300 steps`：epoch0 使用相同 uniform sampler，
epoch1 才使用 frontier weights。唯一变量为 capacity bonus：

```text
outputs/lfp_capacity_pair_safety_seed0_20260713T001926Z
outputs/lfp_capacity_pair_diversity_seed0_20260713T001926Z
```

epoch0 的 4,800 个共同 scene 上，frontier energy 与 support ADE 的相关从 `0.2731` 提升到
`0.3818`，与 mode capacity 的相关从 `0.0965` 提升到 `0.2327`；新增 energy delta 与 mode
capacity 的相关为 `0.4260`。这证明新增项确实改变了课程方向，而不是仅提高 entropy。

epoch1 的实际 sampled unique scenes 对比：

| 配置 | unique | support ADE mean | mode count mean | mode capacity | multimodal ratio |
|---|---:|---:|---:|---:|---:|
| safety-only | 4,686 | 1.36992 | 4.9110 | 0.72465 | 0.92851 |
| diversity-capacity | 4,686 | 1.38101 | 4.9315 | 0.72834 | 0.93342 |

采样方向正确但幅度温和：support ADE `+0.01109m`、mode capacity `+0.00369`、multimodal
ratio `+0.49` 个百分点。20% uniform branch 实际为 `19.918%`；priority max/median 从
safety-only 的 `3.19` 降到 `2.81`，没有形成少数热点坍缩。

#### 训练与固定 seed0 NAVTEST

epoch1 train aggregate：

| 配置 | scalar | feasible | EP | TTC | group ADE | exact KL |
|---|---:|---:|---:|---:|---:|---:|
| safety-only | 0.94509 | 0.98070 | 0.92707 | 0.97417 | 0.81726m | 0.09789 |
| diversity-capacity | 0.94391 | 0.97977 | 0.92589 | 0.97444 | 0.78505m | 0.10017 |

train scalar 和 group ADE 没有因课程变大；容量 scene 本身更难，不能用 train reward 判断
active sampling 是否有效。固定 `initial_noise_seed=0`、32 shards、12,138 个有效样本的 v1
NAVTEST 为：

| 配置 | PDMS | NC | DAC | TTC | EP | DDC | comfort | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 Stage2 | 0.873111 | 0.980804 | 0.953782 | 0.942907 | 0.820586 | 0.980269 | 0.999670 | 0.283344 |
| safety-only | 0.866870 | 0.966345 | 0.951557 | 0.908634 | 0.846880 | 0.973842 | 0.999423 | 0.464194 |
| diversity-capacity | 0.871628 | 0.969764 | 0.952381 | 0.916955 | 0.847800 | 0.975367 | 0.999670 | 0.444137 |

capacity 相对 safety-only 的 paired delta 与 scene-cluster bootstrap 95% CI：

| Metric | Delta | 95% CI |
|---|---:|---:|
| PDMS | +0.004758 | `[+0.002472,+0.007083]` |
| NC | +0.003419 | `[+0.002007,+0.004998]` |
| DAC | +0.000824 | `[-0.001232,+0.002873]` |
| TTC | +0.008321 | `[+0.006262,+0.010448]` |
| EP | +0.000920 | `[-0.001393,+0.003280]` |
| DDC | +0.001524 | `[+0.000750,+0.002366]` |
| L1 | -0.020057 | `[-0.021574,-0.018519]` |

PDMS/NC/TTC/DDC/L1 改善均有统计支持，说明“容量感知主动采样能缓解 safety-only Stage3
退化”得到第一项因果证据。但 capacity 相对 A5 的 PDMS 仍为 `-0.001483`，95% CI
`[-0.005117,+0.002252]`；NC `-0.01104`、TTC `-0.02595`、L1 `+0.16079` 均明确更差，
而 EP `+0.02721`。因此课程修复了数据选择方向，却没有解决 progress-biased credit 和缺少
physical trajectory trust 的主问题。

阶段决定：保留该实现和消融入口，默认继续关闭；需要至少另一个 seed 和 NAVSIM v2
EPDMS 复核后才能进入主配置。`training-rl-zt3` 已按要求释放，本轮未继续占用它做 v2 scoring。
评估过程中另发现同 basename checkpoint 会碰撞输出目录，批量脚本已改为 basename 加规范路径
SHA256 前 10 位，避免第二个权重被静默跳过。

### 15.25 KL 剂量对照：约束物理漂移，但不解决 credit trade-off

在 capacity curriculum 其他配置不变的条件下，将 frozen Stage2 exact transition KL 从
`0.02` 扫描到 `0.05/0.10`。三组均从 A5 epoch155 开始，seed0、G=16、LR=`1e-5`、
2x150 steps，并使用固定 `initial_noise_seed=0` 的 32-shard 全量 v1 NAVTEST。

| KL | train exact KL | train group ADE | PDMS | NC | TTC | EP | L1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.10017 | 0.78505m | 0.871628 | 0.969766 | 0.916958 | 0.847799 | 0.444004 |
| 0.05 | 0.03911 | 0.69828m | 0.870997 | 0.971579 | 0.921653 | 0.839347 | 0.386443 |
| 0.10 | 0.01822 | 0.60062m | 0.869298 | 0.971950 | 0.925526 | 0.833024 | 0.360407 |
| A5 Stage2 | - | - | 0.873112 | 0.980806 | 0.942910 | 0.820586 | 0.283275 |

存在清晰的剂量反应：KL 越强，train KL/group ADE 和 held-out L1 越小，NC/TTC 越高；
但 EP 持续下降，PDMS 并未提高。KL=0.05 相对 KL=0.02 的 L1 差为 `-0.05756`，
95% scene-cluster CI `[-0.06240,-0.05273]`，TTC 差为 `+0.00470`，CI
`[+0.00262,+0.00684]`；PDMS 差 `-0.00063`，CI 跨 0。KL=0.10 相对 A5 的 PDMS
差为 `-0.00381`，CI `[-0.00651,-0.00113]`。

因此 transition KL 不是无效，它确实是 trust 约束；但增大标量 KL 只能在进度和
安全之间移动 trade-off，不会自动修正 credit 或 Stage2 的多模态先验。下一个 trust 实验
若继续，应使用 decoded physical trajectory deadband，不再继续提高全局 KL。

### 15.26 Stage2 v3 数据审计：高质量候选库，但不是校准的多模态分布

需要区分两个问题：入选轨迹是否低质量，以及它们是否给出了正确的模式概率监督。

1. **单轨迹质量并不差。** 全量 archive 的 selected-valid ratio 为 1，`58.49%` 场景
   存在优于 GT 的 support。固定协议下 A5 也比 Official Stage2 提高 v1 PDMS
   `+0.01361` 和 v2 EPDMS `+0.00770`。
2. **reward 标签高度饱和。** 103,288 scene 中，support reward mean 为 `0.96363`，
   `46.26%` 场景的 mean 不低于 `0.99`；`75.81%` 场景的 best support reward 等于
   1。固定 seed 抽样 5,000 scene 中，`35.16%` 场景的所有 support reward 都等于 1。
   这些标签能过滤坏轨迹，但不能稳定区分安全域内的行为模式。
3. **候选有容量，但局部冗余明显。** support pairwise ADE 均值为 `1.36082m`，
   reference mode count 均值为 `4.9077`；同时 5,000-scene 抽样的 support 近邻 ADE
   均值只有 `0.43619m`，`39.32%` 的 support 存在 `<0.3m` 近邻。因此“总宽度大”
   不等于“每个有效模式获得均衡概率”。
4. **当前 DPSI 把来源多样性误当行为多样性。** 抽样 support 的来源大类平均只有
   `3.006`，source entropy 均值 `0.722`，其与 trajectory mode capacity 的 Spearman
   相关为 `-0.263`。现有 adaptive beta 同时使用 support count、轨迹距离和 source entropy，
   最终 beta 与 mode capacity 的相关仍只有约 `0.206`。
5. **真正进入 loss 的分布仍然锚定 GT。** A5 全 200 epoch 中每 scene 平均有
   `10.279` 条 selected support，名义采样 `3.821` 条 target，但 pre-sampling effective
   target count 只有 `1.842`，adaptive beta 为 `0.2709`，最终 diffusion loss 的 GT mass
   为 `87.29%`。

Official Stage2 的 GT-only 训练能获得很好的 deterministic PDMS，说明 NAVSIM 单轨迹
评估并不要求学会宽分布。Official 在 SNSAD 中的采样宽度更大，但 precision 更低；
那部分宽度可能是 legacy representation/sampler 噪声，不是被证明的语义模式。
因此不能用“Official 不用补充轨迹也很好”推出 v3 轨迹低质量；可以推出的是：
**当前 Stage2 对 deterministic planner 有效，但 v3 selection + DPSI weighting 没有实现论文
所需的 density/mode-balanced conditional distribution。**

下一步先做同 archive 的单变量因果实验，不立即重建全量数据：

- D0：当前 v3 + 当前 target distribution；
- D1：当前 v3 + trajectory-density/mode-balanced target distribution，去掉 source entropy 对 beta 的作用；
- D2：对 v3 做模式内去重后 + 同一 mode-balanced distribution。

先从 A5 epoch155 做短程 fine-tune，同时检查 PDMS/EPDMS 和 SNSAD precision/recall/KEMR/
width。只有 D1 无效而 D2 有效，才能把主因归结为 archive 构建；如果 D1 已有效，
主因是训练目标分布而不是轨迹库本身。
