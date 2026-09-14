# PSI-Drive：官方IL → 候选SFT → SR-PGRPO历史训练链

核查日期：2026-09-14。此次没有权重更新或新PDMS评测。历史分数统一转换为0–100 point。原始V1/V2/V3和PSI文件均未修改。

## 核心结论

这条训练链确实存在：官方IL → 2026-06-24 APSD/Pareto-support SFT → epoch_011（87.6724）→ 2026-06-27 SR-PGRPO → step_00021000（90.8940）。后续训练的初始化路径和reference policy路径均指向SFT epoch11的SHA256 `051ac3312a848b36eeeaccc3935e03582f9acc89737a99672349a5887faa0aff`。

历史上保存并评测了88个SR-PGRPO step checkpoint，步数300至26,400，间隔300。**这是历史保存事实；当前权重实体可用性要另行判断。** 截至本次核查，原始raw/store/backup目录及原远程评测机对应目录内未找到这些GRPO权重。不能把现存索引当成可加载checkpoint。SFT的30个实存权重则已在前次核查中完整hash并成功读取。

## 一、候选SFT的实际设置

Run：`outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/`。

| 项目 | 实际设置与证据 |
|---|---|
| 初始化 | 官方 `ReCogDrive_Diffusion_Planner_2B_IL.ckpt`；启动日志确认加载347个keys |
| 观察编码 | 官方2B VLM hidden-state cache；冻结VLM，不训练backbone |
| 更新参数 | action head可训练参数34,329,219；专家、JEPA、VGGT、Last-VLA关闭 |
| 训练/验证 | 85,109 / 18,179条记录；训练target为pareto_support，验证target为GT |
| support档案 | `outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt`，103,288 tokens |
| target表示 | 每个场景最多3条 `[8,3]` 轨迹，XY及heading，4秒/0.5秒采样 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight decay=1e-4 |
| 学习率 | 初始5e-5；WarmupCosLR，warmup 3 epochs，调度长度200 epochs，min_lr=1e-6 |
| 实际训练长度 | 60 epochs；epoch11对应7,315 optimizer steps |
| batch | 8 GPU × 每卡16场景 × accumulation 1 = effective batch 128 |
| 精度与梯度 | Lightning DDP，16-mixed，gradient clip norm=1.0 |
| seed | 0 |

训练只运行60 epochs，**调度器却按200 epochs设置**。读取保留的optimizer/scheduler状态确认，epoch11学习率约4.9801e-5，epoch59仍约4.0863e-5。因此不能把这轮实验理解为已经完整退火到1e-6的60-epoch微调。

此前加载日志中的208个missing keys也已进一步核查：保存模型有555个state keys，其中208个全部属于CoT分支，其余347个与官方IL键集合完全一致，shape无差异。CoT分支的11,828,752个参数均冻结，`use_last_vla=false`；可恢复历史代码在没有CoT tokens时跳过该分支。这些缺失键不能作为“基础IL未加载”的证据。本次没有执行该历史模型的完整前向parity。

## 二、候选怎么构建

原料来自训练elite buffer：GT、IL/policy rollout，以及endpoint/lateral/progress/timing等结构化扰动。构建模式为`clean`。没有证据表明这份档案使用后来DDV2/DrivOR候选。

从可恢复PSI实现看，选择过程是：

1. 以GT与确定性IL中PDMS较高者作为reference。
2. 要求候选原始valid，NC≥1、DAC≥1；DDC≥0.95或不低于reference DDC−0.01；EP不低于reference EP−0.02。
3. 使用Core-Pareto复合分数，而非仅按裸PDMS排序。其中 `core=(5EP+5TTC+2Comfort)/12`；复合分数包含PDMS、相对reference的core margin、慢行/EP-TTC tradeoff惩罚，以及Pareto front奖励0.2。该Pareto目标为 **EP、TTC、Comfort**，不要与后来V2/V3的目标定义混淆。
4. 保留距最高复合分数0.02以内的候选；这里的0.02是复合分数带宽，不能直接叫作“PDMS相差2分”。
5. 先取最高分候选，再按标准化物理描述子做最远点选择，最多3条，最小描述子距离0.75。描述子为终点X/Y/heading、平均速度、末速度、前半段进度比例。
6. 无合格候选时有GT等fallback；实际训练档案不能笼统称为每场景都有多条高分候选。

直接重读support档案，排除18,179条无原elite buffer的验证GT fallback后，恰好剩85,109条训练记录：

| 每场景有效support数量 | 训练场景数 |
|---|---:|
| 1 | 46,838 |
| 2 | 31,232 |
| 3 | 7,039 |

平均1.5324条/训练场景；约55.0%的训练场景只有一条target。

## 三、具体怎样SFT，损失是什么

每次forward针对每个场景，使用`torch.multinomial`，按support_weights从有效候选中抽取**一条**监督轨迹。再次遇到同一场景时可以抽到其他候选，因此是在训练过程中学习一个加权target分布。

默认权重预算为：最高分候选0.5，已入选GT 0.2，其他候选共0.3。若GT未入选，则0.2加给最高分候选；若没有other，则0.3也加给最高分候选；只有一条时权重为1。**GT并非强制进入support，0.2也不是每个场景额外施加的GT-retention loss。**

抽取target后进行原生`norm_odo`归一化，对diffusion timestep均匀采样，并加标准高斯噪声。损失为原生epsilon预测MSE：

\[
j\sim\mathrm{Categorical}(w_s),\quad x_0=\mathrm{norm}(\tau_{s,j}),\quad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\]

\[
L_{\mathrm{SFT}}=\mathbb E_{s,j,t,\epsilon}\left[\|\epsilon_\theta(x_t,t,o_s)-\epsilon\|_2^2\right].
\]

MSE在batch、8个时刻及XY/heading维度取平均；diffusion loss权重1.0，GRPO/KD及相关辅助目标关闭。PDMS影响离线候选选择，不是这一步的在线强化学习reward。

实际训练support中，只有41.19%的场景包含GT。按已保存weights计算的**期望抽样占比**为：GT 31.93%，IL 8.94%，policy 29.13%，结构化扰动合计30.00%。这是由权重计算的期望比例，不是逐次抽样日志中的实测比例。

## 四、后续SR-PGRPO训练

Run：`outputs/psi_drive_stage3_sr_pgrpo_epoch011_b2acc4_20e_20260627T135941Z/`。

| 项目 | 实际设置 |
|---|---|
| 初始化及冻结reference | SFT epoch11，同一SHA256 `051ac331…` |
| 算法 | Core-Pareto GRPO v2上开启support-relative advantage，即SR-PGRPO |
| 每场景rollout group | 16 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight decay=1e-4 |
| LR/scheduler | 1e-4，20-epoch cosine，warmup=0，min_lr=1e-5 |
| 训练预算 | max_epochs=20；现有索引覆盖300至26,400步 |
| batch | 8 GPU × 每卡2场景 × accumulation 4 = effective batch 64 |
| regularization | reference_kl_coeff=0.02；BC coefficient在5 epochs内由0.10降到0.05 |
| BC target | 冻结SFT reference生成的diffusion chains；不能把这个BC项描述为本轮额外GT epsilon-MSE |
| 数据 | 同一85,109训练/18,179验证；数据报告的train/val token交集为0 |
| support | 换用GT-supplemented support档案；不是SFT时原文件原封不动地复用 |
| 保存 | 每300 step；88个step checkpoint有历史评测结果 |

Support-relative的作用：按物理描述子把当前rollout分配到support附近的bucket；在bucket内进行质量标准化比较，附加跨bucket的正向奖励、free bucket新颖性限制、保持符号的RMS advantage缩放，并重新施加invalid/slow/dominated等正advantage上限。它改变了advantage定义，因此**不能当作与90.41原版GRPO算法完全相同的初始化对照**。

## 五、这条历史链有没有提升

以下为保存的历史Navtest评测记录，12138条有效场景记录；评测CSV包含汇总行时总行数12139。本次没有重新推理。

| 阶段 | checkpoint | PDMS | 相对SFT epoch11 |
|---|---|---:|---:|
| 候选SFT | epoch_011 | 87.6724 | 0 |
| SR-PGRPO | step_00000300 | 88.4576 | +0.7852 |
| SR-PGRPO | step_00000600 | 88.8015 | +1.1291 |
| SR-PGRPO | step_00003000 | 89.5028 | +1.8304 |
| SR-PGRPO | step_00012000 | 90.2021 | +2.5296 |
| SR-PGRPO，历史最高 | step_00021000 | 90.8940 | +3.2215 |
| SR-PGRPO，最后现存索引 | step_00026400 | 90.6311 | +2.9587 |

历史最高点相对SFT：EP +3.7922个百分点，DAC +2.5540个百分点，TTC +0.7332个百分点，NC −0.1359个百分点。性能改善并不意味着所有安全指标都改善。

因此，这条记录反对“候选SFT之后GRPO必然退化”的泛化结论。但它不能单独证明候选SFT优于GT-SFT初始化，因为同时改变了强化学习advantage规则，也没有相同recipe、相同预算的GT-IL起点对照。

另一个限制是checkpoint选择：存在`stage2_navtest_selected_top1.tsv`，明确记录按历史Navtest选择epoch11。Stage3的90.8940也是沿训练过程反复Navtest评测后的最高点。这些可用于内部历史机制分析，不能包装为untouched Navtest上的无偏模型选择。

## 六、权重现在是否可用

SFT epoch11实体已找到，见前次`IL_INITIALIZED_CANDIDATE_SFT_CHECKPOINT_LOOKUP_20260914.md`。

SR-PGRPO step21000历史SHA256：

`f865bdc6a70500b49106fc39919afecfe83f04d9b7cac7e9f204e19bddca090d`

历史上同时登记过raw、immutable store、Top1 backup三处路径；当前这三处均没有可读实体。原远程评测机`training-rl-zt3`可访问，其对应run存在，但权重目录也未找到实体。扩大检索检查了当前/旧outputs、checkpoint目录及container backup中的219,552个匹配扩展名文件，依据88个原始checkpoint的精确文件大小缩小范围，并hash了2个大小匹配对象，均不是这条GRPO链的权重。精确搜索范围和排除目录另存`checkpoint_search_audit.json`；这不是对所有存储的穷尽证明。

**当前结论：GRPO训练与历史checkpoint保存是CONFIRMED；本次查找范围内GRPO权重实体为MISSING，尚不能直接开展该权重的新rollout实验。** 未查找到不等于证明所有其他存储都已删除。本轮没有改名其他GRPO权重来替代，也没有重新训练冒充旧权重。

## 证据与可复核性

本次新增小型记录位于`outputs/five_checkpoint_training_distribution/psi_training_chain_audit/`：

- `training_settings.json`：两阶段精确配置字段、参数档案hash。
- `support_distribution.json`：直接读取support后的数量及期望抽样权重。
- `stage3_historical_checkpoints.csv`：全部88个checkpoint的hash、历史路径、分数及当前实体检查。
- `audit.json`：初始化链、state key核查、代码来源与限制。
- `checkpoint_search_audit.json`：扩大查找范围与实体hash核验结果。

历史日志/配置记录的代码commit为`d2a09e03…`，但该commit不包含当时未提交的PSI新增实现。此次从后续`870d7167b17889dbdc89d1517f428cd8d8ee083d`恢复PSI源码，文件hash已记录；它不是已证明逐字节一致的当时运行时源码快照。损失/选择实现的解释来自这一可恢复版本，并由真实配置、support文件和训练日志相互印证；优化器与SFT scheduler还直接读了实存checkpoint状态。
