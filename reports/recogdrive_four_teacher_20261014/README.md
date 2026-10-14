**ReCogDrive 四教师与 DrivoR Scorer：实验设计、数据和结果汇总**

汇总日期：2026-10-14。本文整理已经完成的历史实验，未启动新训练、推理或官方评分。分数表使用 **0–100 PDMS**；机器可读证据保留 **0–1**。Oracle 是逐场景用真实官方分数选最高者的离线候选上界，实际 Scorer 结果单列。

实验的主要结果是：保留官方 RL，再加入三个针对性训练的独立动作教师，能够显著增加候选覆盖；外部 DrivoR Scorer 只实现了部分 Oracle 收益。后续提高候选 Oracle，没有带来同等幅度的实际选轨提升。不能将这些结果表述为单模型达到 95/98 分，也不能用它证明当前 IQE 从零训练的内置 Scorer 已有效。

**代码与证据身份。** 历史实现位于 `IDayday/VLA-AD` 的本地分支 `analysis/multi-teacher-opd-20261004`，工作区 `/mnt/project/VLA-AD-worktrees/multi-teacher-opd-20261004`，核查 HEAD 为 `82b753299b0df0302b2d767a3169d4635ce1219d`。

| 历史提交 | 内容 |
|---|---|
| `3160e18` | 同源专科教师训练及训练侧 Oracle 审计 |
| `9ec2b80` | 专科分工、数据边界和固定端点审计 |
| `c8abf79` | 完整教师 Navtest 候选银行及结果分析 |
| `26999f930745cd3b0da15932d7a0f72866c762c9` | 原生 DrivoR SimScale Scorer 选择固定四教师 |
| `f1cde42` | 四教师、Oracle 与实际选择汇总 |
| `7a3475d` | 强化教师完整训练集 Oracle 与新候选 Scorer 迁移结果 |

截至本次查询，`origin` 没有同名历史远端分支。本汇总发布在独立文档分支 `docs/recogdrive-four-teacher-summary-20261014`，从远端 `main` 的 `7cbb1e3eb6e9a32a173edf1d03e05e8d988acc09` 创建，只新增本文及小型结果/来源证据，不将未发布的历史实现和其他实验一并推送。历史工作区的未跟踪修改未纳入本文提交。代码路径、内容 SHA256、原始运行记录及结果位置见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。历史短提交可在原本地仓库用 `git show` 查看，不保证仅克隆本汇总分支即可取得它们。

**同一个 Base 与两版固定四候选。** Base 为官方 `ReCogDrive-2B-RL`，动作权重为：

```text
/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-RL/ReCogDrive_Diffusion_Planner_2B_RL.ckpt
SHA256: 23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20
```

原 ReCogDrive VLM、动作条件投影、normalizer 及原生轨迹采样契约保留。动作模型是 DiT 扩散规划器；每个部署候选输出一条 8×3 轨迹，4 秒 horizon、dt=0.5 秒、局部 rear-axle 坐标。它与当前 IQE 的单 Query 动作分支不是同一架构。

每版主候选集合都严格为四条：`official_rl`、交互教师 EMA、道路/方向教师 EMA、安全进度教师 EMA。GT 不在部署集合中，raw 与 EMA 不混池，不混多个训练 seed 扩大候选数。

| 方法 | 完整 Navtrain PDMS | 完整 Navtest PDMS |
|---|---:|---:|
| 官方 RL Base | 94.9484 | 90.5514 |
| 首版：交互教师 EMA10k，单独使用 | 89.4398 | 85.3271 |
| 首版：道路/方向教师 EMA10k，单独使用 | 91.1481 | 85.5678 |
| 首版：安全进度教师 EMA10k，单独使用 | 91.1950 | 84.6513 |
| 首版四候选 Oracle@4 | **97.4171** | **95.1680** |
| 首版四候选 + 冻结 DrivoR 实际选择 | **95.9166** | **92.1649** |
| 强化版：交互教师 EMA4k，单独使用 | 88.5648 | 83.9184 |
| 强化版：道路/方向教师 EMA4k，单独使用 | 89.2002 | 82.4856 |
| 强化版：安全进度教师 EMA4k，单独使用 | 89.3322 | 81.9212 |
| 强化版四候选 Oracle@4 | **98.2216** | **95.8931** |
| 强化版四候选 + 同一冻结 DrivoR 实际选择 | 本汇总未核验 | **92.2993** |

完整 Navtrain 为 **103,288 场景 / 1,192 logs**；完整 Navtest 为 **12,146 场景 / 136 logs**。均为场景等权。Navtrain 含模型训练成员，只能作为训练集描述。完整精度、分母和来源在 [RESULTS.csv](RESULTS.csv)；空缺不填成 0 或外推值。

**首版教师如何构建。** 三个教师均从同一个官方 RL 独立初始化，各自训练一套 DiT LoRA。16 个 small-DiT block 中，每层适配 attention Q/V/输出和两个 FFN 投影，共 80 个线性层、160 个 LoRA A/B 张量。`rank=128, alpha=256`，每个教师实际可训练参数 **8,650,752**。LoRA B 零初始化，原基础权重、bias、VLM 和投影保持冻结。训练完成后导出合并 LoRA 的原生动作头，保留 raw 和 EMA 两种端点。

教师名表示训练数据中观测到的评分失败类型，不表示通过人工标注找到了行人横穿、转弯等物理成因，也不是按语义聚类或部署硬路由划分。各自的真实参数更新为监督学习；从 RL checkpoint 初始化不等于再次运行 RL。

**首版训练数据。** 先继承此前质量认证的 Navtrain 纠错池，再加入 `MrShallow/navsim-generator` 的合法 HF 补充记录；后者与其他历史 33k SimScale 库不是同一批数据。

Navtrain 上游纠错池要求监督 PDMS≥0.9、规定分项满分、曲线合法，且监督比 `min(official_il, official_rl)` 高超过 0.02；排除已选 replay 成员后，专科准备阶段读取了 **11,967** 条已认证纠错候选。它没有在当轮重新对所有潜在原始 Navtrain 标签做无限扩库。HF 从 3,613 条记录中保留 **2,525** 条质量合格候选，仅移除早期“至少一个父本低于 50 分”的难度限制，没有放宽监督质量门槛。

随后依据官方 RL 的实际分项，使用以下规则分组：

| 教师 | Base 失败/改进规则 | Navtrain | HF 补充 | 训练场景数 |
|---|---|---:|---:|---:|
| `interaction` | NC<1 或 TTC<1 | 1,268 | 1,102 | **2,370** |
| `road_direction` | DAC<1 或 DDC<1 | 541 | 158 | **699** |
| `progress_comfort` | Comfort<1；或 NC/DAC/TTC/Comfort/DDC 全为1、PDMS<0.9、监督比 Base 高>0.02 | 1,523 | 65 | **1,588** |

NC 为有责碰撞合规，DAC 为可驾驶区域合规，TTC 为碰撞时间边界，DDC 为行驶方向合规，EP 为进度。交互与道路组重叠 **153** 条，其他组之间无重叠，去重合计 **4,504** 条。`progress_comfort` 实际没有 Comfort 失败样本，因此本轮准确称为“安全进度教师”。[COHORTS.csv](COHORTS.csv) 给出实际数量、原 Base/监督均分和 manifest 路径。

所有最终监督都要求 PDMS≥0.9，NC/DAC/TTC/Comfort/DDC 全为1，同时检查时间戳、原生输出范围和曲线。既有阈值为 jerk≤5 m/s³、横向加速度≤3 m/s²、航向加速度≤30 deg/s²，目标间隔为0.5秒、容差0.025秒。监督由原日志生成并与评分收据、metric cache 和冻结特征绑定；未来目标只作监督，不进入规划输入。

没有统一要求每条 GT 比 RL 高5分。道路组有 **53** 条监督名义 PDMS 低于 RL，来自方向纠正；该评分配置的 DDC 权重为0。方向合规改善不能直接写成 PDMS 提高。这个历史数据选择细节保留在证据中，不能用当前 IQE 的新规则改写旧实验。

三个专科训练器只消费各自专项集合，按固定种子逐 epoch 洗牌循环。**没有普通 anchor/replay 混采，也没有全场景 Base 蒸馏约束。** 虽然准备目录包含16,384条 `anchor_replay`，专科 trainer 没有读取该清单。HF 样本权重2作用于损失，Navtrain权重1，不是把HF重采样两次。

**首版训练配方与真实端点。** 损失为 DDPM epsilon MSE，加上原生5步随机 DDIM 输出到监督轨迹的 normalized SmoothL1（旧报告称 Huber，beta=0.05），两项系数均为1。PDMS用于筛选、监督审计和评估，没有作为可微奖励直接优化。

| 配方 | 实际设置 |
|---|---|
| 每教师训练预算 | 10,000 optimizer updates |
| 全局 batch / 每 rank microbatch | 64 / 8 |
| 最终训练 world size | 8；早期从4卡step500状态迁移，保留global batch64 |
| 优化器 | AdamW，lr=1e-4，weight decay=0 |
| 学习率 | 100步warmup，乘余弦衰减，最低为峰值0.1倍 |
| 裁剪 / EMA | global grad norm 1 / 0.999 |
| seed / 保存频率 | 20261004 / 每500步 |
| 主结果 | 预先固定step10000 EMA；raw仅作另列诊断 |

三个 `TRAINING_COMPLETE.json` 均确认10,000步，每个教师对应640,000次样本呈现，不是640,000个独立场景。训练未用 Navtest 选择checkpoint，历史配置没有另设 validation split。4卡到8卡迁移涉及随机流和浮点归约变化，历史文档明确不称跨world-size逐位继续；本汇总没有重新执行恢复测试。

**后续强化版：按整个旧候选池的缺口再次训练。** 主结果98.2216/95.8931对应 `teacher_oracle_complementarity_v5` 的三个固定专科，不是并行sMCL实验，也不是后来的IL教师实验。

从各自首版 EMA10k 动作头初始化，再添加可训练 LoRA；冻结共享观测链与该初始化的基础权重。此次新建 AdamW 优化状态，4000步是新一轮局部步数，不是从官方RL重新训练4000步，也不是沿用首轮10k优化器状态。

训练只用完整 Navtrain，在读取新教师结果前冻结以下条件：

1. 计算旧 `RL + 三专科 EMA10k` 的逐场景 Oracle。
2. 从已冻结的离线优化轨迹快照取监督，显式标记 `frozen_offline_optimized_trajectory_NOT_original_GT`。它不是原始GT，也不是直接把DrivoR预测分数当真值。标签快照来自此前离线优化任务，进入本轮前通过官方评分身份与几何质量审计；本汇总不重新运行该优化过程。
3. 要求监督质量资格通过、strict geometry通过、PDMS≥0.95，并比旧K4 Oracle至少高0.01。
4. 按旧Oracle赢家的失败分项互斥分配：NC失败优先交互，其次DAC/DDC失败分给道路，再次TTC失败分给交互，其余进度/舒适缺口分给进度组。

| 强化教师 | 新一轮 Navtrain 场景 | 初始化 | 新增步数 |
|---|---:|---|---:|
| 交互 | **53** | 对应交互EMA10k | 4,000 |
| 道路/方向 | **102** | 对应道路EMA10k | 4,000 |
| 安全进度 | **17,892** | 对应进度EMA10k | 4,000 |
| 去重并集 | **18,047** | 三组互斥 | — |

这是旧池残余缺口的分布，不意味着完整训练集只存在53个交互困难场景。强化版仅使用原生5步DDIM normalized SmoothL1监督，**移除了首版epsilon MSE项**；仍为LoRA r128/alpha256、lr1e-4、AdamW wd0、100步warmup加余弦、batch64、microbatch4、2卡、clip1、EMA0.999。没有GD、replay、旧行为约束或跨教师WTA；固定专科模式每个任务只有一个训练成员。所有主结果仍使用预先固定4000步EMA。

三个训练收据确认达到4000步；交互和道路经历了step2500恢复，最终一次进程的 `actual_optimizer_updates=1500` 是该恢复段长度，不能误读为总共仅训练1500步。

首版到强化版同时改变了监督来源、缺口定义、数据规模/构成和损失。因此它展示的是整套强化配方的结果，不能单独归因于更长训练、某一损失或某个分组算法。

**DrivoR Scorer 的实际接线。** 两版此处的主选轨结果都使用同一个外部预训练DrivoR权重：

```text
nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth
SHA256: 617e22c5ebcf8b24c542d42a470514b09c3cadce1a2630071d49f2d422d76672
DrivoR source commit: f02665403df799c1b4ddd8b0d34e073f0555c13a
```

该模型以DrivoR自身的四视图F0/B0/L0/R0、4帧ego状态、DINOv2及其公开LoRA提取场景表征，再对ReCogDrive生成的四条轨迹评分。它额外使用自己的视觉编码器，没有复用ReCogDrive VLM来从零训练评分头，也没有通过真实PDMS执行部署选择。

原生 `pos_embed → scorer_attention → scorer` 预测NC/DAC/DDC/TTC/EP/Comfort，按原权重1/1/0/5/5/2的log聚合排序，argmax选择；精确平局优先固定顺序中的RL。真实官方分数在GPU选择之后用于CPU评测。网络保留跨候选self-attention，公开训练K=64、此次推理K=4；因此**预测分数可依赖候选集合**，不能宣称单候选函数不变。官方单轨迹真实标签使用固定reference上下文，与这个学习评分器的候选注意力是两个不同问题。

历史实现保存原生接口数值对齐及官方轨迹评分复用证据，但没有匹配该134k权重的发布版完整CSV，不能把作者公开系统成绩当作本地全量复现。主评测中没有重新训练DrivoR；后续旧银行适配的其他Scorer实验不混入本文主结果。

**互补性、实际收益和风险。** 首版Navtest相对RL，Oracle增益 **+4.6166分**，实际Scorer增益 **+1.6135分**，实现约 **34.95%** 的Oracle增量。实际增益的配对log-cluster 95%区间为 **[+1.2448,+2.0004]分**。逐场景提高/下降/持平为3,144/1,528/7,474。

强化版相对相同RL，Oracle增益 **+5.3417分**，实际选轨增益 **+1.7479分**，实现约 **32.72%** 的Oracle增量。两个比例均是 `(selected_mean-base_mean)/(oracle_mean-base_mean)` 的派生点估计，未为该比值新增置信区间。

| 首版→强化版的同场景变化 | 点估计 | 历史配对95% log区间 |
|---|---:|---|
| Navtrain Oracle | +0.8044分 | [+0.7610,+0.8467] |
| Navtest Oracle | +0.7252分 | [+0.4819,+0.9661] |
| Navtest同一DrivoR实际选轨 | +0.1345分 | [−0.1689,+0.4358] |
| Navtest regret增加 | +0.5907分 | [+0.3624,+0.8277] |

首版单独使用任一专科都弱于RL，强化版单教师全量均分进一步下降，但Oracle更高。这支持候选互补性的观察，也说明保留Base和正确选轨不可省略；不能只展示Oracle而隐藏单策略退化。

首版RL→Scorer零分场景由495降至331，零分率4.0754%→2.7252%；仍有130个“原安全→失败”和329个反向场景，安全在该审计中定义为NC=DAC=TTC=1。总体改善不是逐场景安全保证。

强化版相对首版，四候选共同零分在Navtrain由111增至157，在Navtest由161增至170；Scorer实际选轨零分由331增至357，NC失败137→165、DAC失败210→207、TTC失败445→524。选轨零分率变化的95%区间跨零，不额外宣称显著恶化。Oracle均分提高没有消除尾部风险。

统计采用场景等权、以log为簇的5000次配对bootstrap；Scorer主汇总bootstrap seed为20261005。首轮训练为一个主seed，固定checkpoint每场景一次原生采样。全Navtest没有RL独立采样4条的完整匹配预算对照；四教师对单候选的全部收益不能都归因于专科能力。历史四对四微调池是另一个对照，不与本文四专科池混名。

**与当前IQE研究的关系。** 可复用的证据是“冻结原策略，专项监督形成额外候选，独立评价Oracle与实际选择”。需要独立验证的部分是当前Query动作结构、同次VLA前向的W/A评分表征、从随机初始化训练的内置Scorer、无跨候选attention，以及不同数据隔离/采样目标。历史使用外部DINOv2+预训练评分权重，不能替代这些实验。

**原始实验资产与本次核验范围。** 所有源路径和SHA256在 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，聚合证据、训练设置、父权重和六份EMA教师注册信息在 [EVIDENCE.json](EVIDENCE.json)。大权重hash引用归档注册表，本次只检查教师文件存在和大小，没有重新扫描大权重。

| 资产 | 项目共享存储位置（相对 `/mnt/project/VLA-AD/outputs/multi_teacher_opd_20261004/`） |
|---|---|
| 首版教师清单及监督绑定 | `expert_cohorts_v2/`、`teacher_training_inputs_v2/` |
| 首版训练状态、raw/EMA权重 | `teacher_campaign_8gpu_v2_20261004/teachers/{interaction,road_direction,progress_comfort}/` |
| 首版候选、DrivoR选择及逐场景数据 | `research_iteration_v2/drivor_simscale134k_four_teacher_full_v1/analysis/{navtrain,navtest}/` |
| 离线优化监督快照 | `context_clusters_snapshot_v1/` |
| 强化版清单及训练状态 | `research_iteration_v2/teacher_oracle_complementarity_v5/{inputs,teachers}/` |
| 强化版完整Oracle结果 | `research_iteration_v2/teacher_oracle_complementarity_v5/evaluation_specialists_v2/analysis/{navtrain,navtest}/` |
| 强化版DrivoR实际选择 | `research_iteration_v2/new_specialist_scorer_transfer_v3/analysis/` |

本次从首版完整Navtrain和Navtest CSV重新核对唯一场景/log数、四候选真实分数、原生预测log-score的argmax与平局顺序、选择分数和逐场景Oracle，再与结果JSON及完成receipt校验。强化版核对归档结果/完成记录、Base及旧银行一致性、训练manifest条件和真实完成步数；没有新增模型调用、真实评分或bootstrap重算。逐项结果见 [VERIFICATION.json](VERIFICATION.json)。

本次只提交聚合数据、设计说明和hash索引；不提交原图、metric cache、权重、完整逐场景银行或任何动态视觉缓存工程。原训练、缓存生产、评测计划、正式ACTIVE和历史实验文件均未修改。
