# SG-FPS v5 Evidence Frontier Contract

更新时间：2026-07-13 13:23 UTC

## 1. 核心结论

Stage2 数据集不需要、也不应该要求每个 scene 都产生固定数量的替代轨迹。不同 scene 在冻结的
proposal generator、Stage2 初始化策略和 evaluator 下具有不同的**观测生成难度**。没有可靠
替代模式时，record 只保留 GT 是合法结果；不能用低质量、重复或不可学习轨迹填满配额。

v5 将三个问题分开处理：

1. synthetic/template proposal 只是假设，不是行为模式存在的证据；
2. 独立证据决定某条非 GT 轨迹能否成为 teacher；
3. 跨 scene 的采样决定稀有 frontier scene 获得多少训练曝光。

因此 `GT-only` 只表示“当前冻结生成协议下没有独立证据”，不表示 scene 天生单模态，也不会被
写成永久容量标签。生成器或初始化策略改变后必须重建 archive 和 scene index。

## 2. Variable-cardinality archive

每个 scene 的监督集合为：

```text
S_i = {GT anchor} union {0..K independently evidenced non-GT modes}
```

没有最小 `K`，只有 `support_top_m` 上限。每条非 GT mode 必须先通过 evaluator validity、轨迹
质量、GT trust region、reward drop、policy reachability 和 GT SNSAD separation。默认独立模式
阈值为 `SNSAD >= 0.40`，与支持集对齐评估中的模式定义一致。

通过 hard gate 后，至少满足一种证据：

```text
policy evidence:
  >= 2 current-policy rollouts are closer to the candidate than GT
  by margin 0.01 and lie within SNSAD radius 0.35

cross-source evidence:
  >= 2 direct source families among policy / DDV2 / DriveOR
  independently support the same local mode
```

候选随后在 scene-relative performance、GT SNSAD diversity 和 independent learnability 三维上做
epsilon-Pareto，再用 SNSAD FPS 选互异模式。candidate count、template count 和 quota tag 都不进入
teacher probability 或 scene priority。

独立 validator 从持久化轨迹、components、source 和 selection config 重算 evidence、front 和
selected set。可读 scene index 使用以下语义：

```text
capacity_semantics = observed_selected_support_not_scene_intrinsic
no_per_scene_candidate_quota = true
priority_semantics = independent_evidence_confidence_not_candidate_count
```

## 3. 128-scene fixed-pool audit

正式固定池位于：

```text
outputs/sg_fps_v5_mode040_128_20260713/support_v5
outputs/sg_fps_v5_mode040_128_20260713/validation.json
outputs/sg_fps_v5_mode040_128_20260713/stage2_frontier_scene_index.json
```

validator 结果为 128/128 records、0 contract error、0 missing token。只有 `13/128` scene 在
`0.40` 模式阈值下有 independently evidenced non-GT mode；其余 115 个合法 GT-only scene 中，
12 个没有 distinct hypothesis，103 个有 hypothesis 但没有独立证据。13 个 selected mode 的
reward delta 均值为 `+0.03503`，policy reachability SNSAD 均值为 `0.16594`。

这组比例只描述当前 128-scene 固定池，不能外推为 scene 的固有容量，也不是全量 archive 的
晋级配额。旧 v4 的 `122/128` multi-mode 来自“模板通过 broad policy radius 即视为可学习”的
错误语义；它混淆了 proposal reachability 与当前策略对该模式的实际支持。

## 4. Global frontier sampling

均匀 scene sampling 会让 115 个 GT-only scene 的 continuation gradient 淹没 13 个新模式。v5
使用真正的 DataLoader sampler，而不是 loss 伪加权：

```text
P(i) = u / N + (1-u) * I[i is evidenced] * priority_i / sum(priority)
```

默认 `u=0.50`。因此所有 GT-only scene 始终有非零概率，约一半 index stream 来自全数据 uniform
replay；另一半聚焦 independently evidenced scene。priority 来自 evidence confidence，不来自
候选数量。128-scene 8-GPU smoke 的 expected frontier ratio 为 `0.5508`，实际为 `0.5703`，实际
uniform branch ratio 为 `0.4922`。

## 5. Continuation loss contract

v5 是从冻结 A5 Stage2 checkpoint 开始的短 continuation，不是从随机初始化重新训练。对有 active
mode 的 scene：

```text
q_i = (1-beta) * delta_GT + beta * Uniform(explicit modes)
beta = 0.45
```

同一 scene 的 GT/mode 使用 paired timestep 和 diffusion noise。residual-gradient proxy 将 non-GT
贡献限制在每 scene `0.45`，但不改变 GT anchor。对没有 active mode 的 scene，target 仍为 GT，
scene loss mass 乘 `0.25`；它继续提供锚定，但不再与包含新信息的 frontier scene 等权。

该权重默认只在 `learning_frontier_v5` 中启用。通用配置默认值仍为 `1.0`，所以旧 checkpoint、
旧 Stage2 target distribution 和 flag-off 路径不变。

## 6. Causal training result

全部对照使用相同 A5 epoch155 初始化、128-scene cache、seed 0、global batch 32、LR `1e-5`、
FS-Norm/PTA、无 x0 trajectory/geometry auxiliary，并用每 scene 32 次 DDIM 采样评估。下表只统计
13 个有 evidence-backed mode 的 scene：

| configuration | Recall | F1 | KEMR | Norm modes | Pairwise ADE | SNSAD | Mean GT ADE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 start | 0.86884 | 0.91570 | 0.73743 | 0.53314 | 0.07684 | 0.56546 | 0.13199 |
| uniform scenes, 100-step long scheduler | 0.84359 | 0.90379 | 0.73040 | 0.50000 | 0.05303 | 0.54209 | 0.11803 |
| frontier sampler, beta 0.25, 25-epoch scheduler | 0.86858 | 0.92081 | 0.73553 | 0.50758 | 0.07112 | 0.56216 | 0.08863 |
| **v5 main: sampler + GT-only 0.25 + beta 0.45 + 25 epochs** | **0.87465** | **0.92361** | **0.73728** | **0.50758** | **0.07697** | **0.56727** | **0.09534** |

v5 main 相对 A5 的 mode-stratum delta：Recall `+0.00580`、F1 `+0.00792`、Pairwise ADE
`+0.00013m`、SNSAD `+0.00181`，best/mean GT ADE 分别改善 `0.03390m/0.03665m`。128-scene
全量上，best/mean GT ADE、precision、recall、F1 和 SNSAD 也均小幅改善或持平。

`Norm modes` 仍低 `0.02556`，但它是 13-scene 上的离散阈值统计；连续 KEMR、raw dispersion 和
SNSAD 均与 A5 持平，bootstrap 区间跨零。该项保留为全量晋级约束，不能据此宣称已解决泛化。

继续维持较高 LR 到 50/75 data epochs 会再次收缩：相同 `beta=0.45` 的 step200 mode SNSAD 降至
`0.54971`、Pairwise ADE 降至 `0.05932m`。因此停止条件是算法的一部分，不能把 25-epoch
continuation 扩成另一个 200-epoch Stage2 全训。

完整结果：

```text
outputs/stage2_frontier_v5_mode040_fixed300_diversity_k32_20260713/aggregate_final/summary.json
```

## 7. Main launcher

全量 8-GPU archive 构建：

```bash
POLICY_CHECKPOINT=/path/to/a5_epoch_155.ckpt \
FS_NORM_STATS_PATH=/path/to/a5_fs_stats_v2.npz \
OUTPUT_PATH=/path/to/support_v5 \
RUN_ROOT=/path/to/build_state \
EXPECTED_TOKEN_SOURCE=/path/to/full_token_archive_or_manifest \
EXTERNAL_CANDIDATE_ROOTS='ddv2=/path/to/ddv2,driveor=/path/to/driveor' \
bash scripts/training/sg_fps/run_build_sg_fps_support_v5_8gpu.sh
```

训练 continuation：

```bash
CHECKPOINT_PATH=/path/to/a5_epoch_155.ckpt \
SUPPORT_ARCHIVE_PATH=/path/to/validated/support_v5 \
DPSI_FRONTIER_SCENE_INDEX_PATH=/path/to/stage2_frontier_scene_index.json \
RECOGDRIVE_VLM_PATH=/path/to/vlm \
CACHE_PATH=/path/to/cache \
FS_NORM_STATS_PATH=/path/to/a5_fs_stats_v2.npz \
TRAIN_TEST_SPLIT=navtrain \
OUTPUT_DIR=/path/to/output \
MASTER_PORT=29500 \
bash scripts/training/sg_fps/run_train_pta_fs_dit_frontier_v5.sh
```

主默认值：

```text
learning_frontier_v5
beta_max=0.45
gt_only_scene_weight=0.25
non_gt_residual_mass_cap=0.45
frontier_scene_uniform_ratio=0.50
LR=1e-5
max_epochs=25
scheduler_epochs=25
trajectory_aux_weight=0
feasibility_aux_weight=0
```

## 8. Promotion boundary

全量 v5 archive 已完成并通过静态晋级门：

```text
outputs/sg_fps_v5_full_evidence_20260713T1329Z/support_v5
outputs/sg_fps_v5_full_evidence_20260713T1329Z/build/validation.json
outputs/sg_fps_v5_full_evidence_20260713T1329Z/build/stage2_frontier_scene_index.json
```

完整域为 `103,288/103,288` unique token，missing/extra/duplicate 均为 0，contract error 为 0，
`static_promotion_pass=true`。其中 `11,855` 个 scene 有 independently evidenced frontier，
`91,433` 个 scene 合法保持 GT-only；共选出 `11,945` 条非 GT mode。GT-only 原因为：

| reason | scenes |
|---|---:|
| no independent mode evidence | 79,600 |
| no distinct mode hypothesis | 10,746 |
| no policy-reachable candidate | 803 |
| no trajectory-quality candidate | 276 |
| no trust-region candidate | 7 |
| no evaluator-valid candidate | 1 |

`77.07%` scene 有 hypothesis 但没有独立 evidence，说明当前瓶颈主要是证据不足，而不是应当通过
逐 scene 配额补齐。selected mode 的 reward delta 均值为 `+0.04912`（p50 `+0.03609`），policy
reachability SNSAD 均值为 `0.20303`。这些都是固定 A5 policy、proposal sources、evaluator 和 gate
下的观测统计，不能解释成 scene 的固有多模态比例。

第一次全量验证曾报告 5/103,288 个边界错误。逐条审计确认 archive 未损坏：validator 为
ADE/FDE/reward gate 额外添加了构建器不存在的 `1e-6` 宽松区，并把 float32 policy distance 提升到
float64 后再做阈值比较。validator 已统一为落盘 float32 契约并增加 reward/policy-radius 边界测试；
修复后 82 项相关回归测试及全量独立重算均通过。

该结果足以启动预注册的 **25-epoch Stage2 continuation**，但不足以直接批准最终 checkpoint 或
Stage3。训练完成后必须：

1. 在固定 held-out panel 上比较 A5 与 5/10/15/20/25 epoch checkpoint；
2. 同时检查 NAVSIM v1/v2、GT precision、mode-stratum KEMR/SNSAD 和 Stage3 reward spread；
3. 只保留同时满足质量与多样性约束的 checkpoint，不按 train loss 选模型；
4. 若任一 checkpoint 未通过质量/安全约束，则停止在 Stage2，不启动 Stage3。
