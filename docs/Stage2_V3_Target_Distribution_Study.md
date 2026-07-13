# Stage2 v3 Target Distribution Study

更新时间：2026-07-13 04:05 UTC

## 1. 问题

本轮检验两个容易混淆的假设：

1. v3 support archive 中的补充轨迹本身质量差；
2. 候选轨迹有效，但 Stage2 把它们组织成训练目标分布的方式不合理。

结论支持第二项，不支持第一项。原版 ReCogDrive Stage2 的 GT-only 监督是一个低条件熵、
易拟合的任务；它能取得较好 PDMS，并不能证明把任意数量的安全候选等概率加入同一条件分布
也会更好。多样性训练同时要求候选有效、模式去重、概率质量受控，以及模型能保持模式而不是
在模式之间产生不安全插值。

## 2. v3 Archive 审计

全量 archive 包含 103,288 个 scene：

- 每个 scene 都有 valid candidate；
- selected non-GT 中未发现低 reward、首点错位、局部折角或语义错配；
- 58.486% scene 的 selected set 含优于 GT 的候选；
- 96.078% scene 的 selected set 来自至少两个 source；
- support 平均有 4.91 个几何模式，但模式内存在明显冗余；
- 75.81% scene 的 best support reward 为 1，reward 对有效模式的排序能力有限。

因此 v3 适合作为候选库，但不能直接把 archive 中的记录频率解释为目标概率。轨迹安全通过
筛选，也不等于它在共享 DiT 中可以获得任意大的训练质量而不干扰 GT anchor。

审计来源：

```text
outputs/sg_fps_support_semantic_full_20260705T032521Z/
  reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/
  audit_full_20260705T190919Z/full_audit.json
```

## 3. 因果对照

所有短训都从同一个 A5 epoch155 checkpoint 开始，固定 seed 0、全局 batch 64、LR `1e-5`，
训练 300 optimizer steps。D0 保留 legacy target distribution；D1 只改变 scene 内目标分布，
模型、archive、FS-Norm、PTA 和 loss 不变。

Mode-balanced 分布使用 SNSAD 固定轨迹距离计算 kernel density，以 inverse-density 去除重复候选
的额外票数。非 anchor 质量上限为：

```text
beta(scene) = beta_max * (1 - 1 / effective_mode_count(scene))
```

目标使用 systematic resampling；每个 draw 权重相等，保持原目标分布的无偏期望，并降低
multinomial count variance。`legacy` 路径完全不调用 mode helper。

### 3.1 实际训练质量

| 配置 | beta max | sampled GT mass | target ESS | sampled unique ratio |
|---|---:|---:|---:|---:|
| D0 legacy | legacy | 0.8735 | 1.856 | legacy sampler |
| mode low | 0.15 | 0.8851 | 1.245 | 0.353 |
| mode selected | 0.25 | 0.8180 | 1.465 | 0.420 |
| mode medium | 0.45 | about 0.700 | about 2.00 | about 0.54 |
| mode full | 0.75 | about 0.56 | about 3.20 | higher |

`beta=0.15` 的 GT mass 比 D0 更高，因此它实际上退回更窄的 anchor-dominant 分布。它不能
作为多样性方案的正证据。

### 3.2 固定 seed NAVSIM v1

| 配置 | PDMS | NC | DAC | TTC | EP | DDC | L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A5 epoch155 | 0.873111 | 0.980804 | 0.953782 | 0.942907 | 0.820586 | 0.980269 | 0.283344 |
| D0 legacy, step300 | 0.870513 | 0.978909 | 0.952134 | 0.939117 | 0.820878 | 0.979527 | 0.292797 |
| mode 0.15, step300 | 0.869893 | 0.980269 | 0.950816 | 0.942000 | 0.817506 | 0.979980 | 0.287590 |
| mode 0.25, step300 | 0.869079 | 0.978827 | 0.951145 | 0.938540 | 0.819112 | 0.979692 | 0.305483 |
| mode 0.45, step300 | 0.865694 | - | - | - | - | - | 0.338254 |
| mode 0.75, step300 | 0.862283 | 0.973019 | 0.948509 | 0.926018 | 0.820043 | 0.976726 | 0.376257 |

`beta=0.25` 相对 D0 的 PDMS delta 为 `-0.001434`，scene-cluster bootstrap 95% CI
为 `[-0.003244,+0.000340]`。当前证据不能判定其 PDMS 与 D0 不同。`beta>=0.45` 则随
mode mass 增加而单调退化。把 D0 和 0.45 继续训练到累计 1,000 step 后差距仍存在，排除了
“仅仅 300 step 未收敛”。

### 3.3 Support-aligned diversity

协议为固定 1,024 个 navtrain scene、每 scene 32 次 5-step DDIM、相同 seed 和 v3 support。

| 配置 | Precision | Recall | KEMR AUC | Width ratio | Pairwise ADE m | SNSAD v2 |
|---|---:|---:|---:|---:|---:|---:|
| A5 epoch155 | 0.984085 | 0.639157 | 0.498695 | 0.221943 | 0.087649 | 0.464131 |
| D0 legacy | 0.984083 | 0.652396 | 0.499506 | 0.227664 | 0.093886 | 0.469284 |
| mode 0.15 | 0.985411 | 0.631220 | 0.496592 | 0.212320 | 0.076959 | 0.458341 |
| mode 0.25 | 0.984095 | 0.658400 | 0.501027 | 0.236909 | 0.103358 | 0.475173 |
| mode 0.75 | 0.972458 | 0.765235 | 0.534785 | 0.373021 | 0.270971 | 0.559025 |

`beta=0.25` 相对 D0 的配对变化：

| 指标 | 均值差 | 95% CI |
|---|---:|---:|
| Recall | +0.006003 | [+0.003825,+0.008281] |
| KEMR AUC | +0.001521 | [+0.000617,+0.002198] |
| Width ratio | +0.009246 | [+0.005533,+0.013541] |
| Pairwise ADE | +0.009473 m | [+0.005067,+0.012716] |
| SNSAD | +0.005890 | [+0.004467,+0.007251] |

`beta=0.75` 证明 v3 中确实有模型能学到的多样性，但单样本 precision、安全和 L1 代价过大。
systematic 与 multinomial 在 `beta=0.45` 上结果几乎相同，因此 sampler count variance 不是
退化主因。

## 4. 对 FS-Norm 和 PTA 的判断

A5 相比 official Stage2 的固定协议 v1 PDMS 高约 0.0136、v2 EPDMS 高约 0.0077，且轨迹
更准确；因此现有结果不支持“FS-Norm 或 PTA 让 Stage2 整体学坏”。但是 A5 的 width ratio
和 SNSAD 低于 official Stage2，说明这些结构与表示改进主要提高了单峰精度，没有自动解决
多模态概率学习。

目前没有完成 official representation、FS-Norm、PTA 的全因子长训，不能把多样性收缩单独
归因给任一模块。已经由单变量实验确认的问题是 target distribution：legacy 实际 GT mass
约 87%，而大幅降低 anchor mass 又造成共享模型的质量/安全干扰。

## 5. 设计决定

1. 全局默认保持 `dpsi_target_distribution=legacy`，保证旧配置和 checkpoint 行为不变。
2. mode-balanced 实验入口默认 `beta_max=0.25`，不再使用已被否定的 0.45。
3. 不重建 v3 archive；先把它视为候选集合，而不是经验概率分布。
4. 下一轮完整 Stage2 前先复验 0.25 的多 seed/更长训练，并同时监控 PDMS、L1 和 SNSAD。
5. 若 0.25 复验成立，进一步把 diversity mass 做成受 safety/quality trust 约束的课程变量；
   scene mode capacity 决定“最多需要多少多样性”，held-out safety 决定“当前允许释放多少”。
6. 不让 diversity priority 再乘入 loss；它只控制 scene/target exposure，Pareto quality gate 继续
   决定哪些候选有资格进入分布。

推荐实验入口：

```bash
DPSI_TARGET_DISTRIBUTION=mode_balanced \
DPSI_MODE_DENSITY_BANDWIDTH=0.40 \
DPSI_BETA_MAX=0.25 \
bash scripts/training/sg_fps/run_train_pta_fs_dit_mode_balanced.sh
```

本结论仍受单 seed、300-step 主剂量实验限制。它足以否定 `beta>=0.45` 作为主配置，也足以
证明 archive 包含可学习多样性；尚不足以启动一次新的 200-epoch Stage2 完整训练。
