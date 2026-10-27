# 本机 ReCogDrive 蒸馏：代码与开训快照

本分支保存本机实验代码和 2026-10-27 UTC 的真实开训记录。训练正在进行，尚无完整学生 Validation/Test 成绩。

## 当前实验范围

- 当前保留 **20261027** 一个 training seed，四组：H_full、H_gain1、H_mu_teacher、H_mu_student。
- 20261028 的四个重复 run 已按用户要求停止，已有权重、日志及实际停止证明保留，释放 16 张 GPU 的任务分配。单 seed 调度控制器已接续；四个保留训练进程继续运行，PID、启动时间、命令和原始attempt均未改变。
- 每个保留 run 仍训练全部 85,813 场景、40 epochs，global batch=64，world_size=4、microbatch=1、累积16次；每 epoch 1,341 次更新，共53,640次。
- 用户已取消五类额外资格验收，采用显式 `USER_AUTHORIZED_BASIC_NORMAL_START`。未执行的验收没有标为通过。保留实际训练的有限 loss/梯度检查、梯度裁剪和正常 checkpoint 保存/加载。

## 源码身份与入口

实际运行的训练代码提交为 **`0a58bd8e6271e26103512add2c5cb7816d7eb8f1`**。本分支在该提交上新增发布资料；运行中的源码 HEAD 保持原值，以兼容已有 checkpoint 的代码身份。

- [学生代码](student_distillation/)
- [有效基础配置](configs/student_distillation/)
- [本机教师完整重建代码](local_teacher_rebuild/native_candidates_20261027/)
- [教师缓存无损读取](scripts/student_distillation/read_teacher_cache.py)
- [原始实验指南](docs/recogdrive_no_routing_distillation_round1_guide.md)
- [保留四组的实际 effective_config 与资产身份](reports/round1_initial_20261027/active_seed20261027/)

冻结证据基线为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`；完整原研究方法源码证据为 `f85d4e557b12a7d05f5deb8d1640834cacd94d32`。本轮教师使用本机完整重拟合结果，PPU 参数及分数仅作历史证据。

冻结训练入口原先注册两个 seed。当前单 seed 计划以本分支保存的用户指令和实际控制器接续记录为准；重跑时应使用配套单 seed 操作入口，避免直接套用原双 seed 默认值。

[单 seed 调度源码](local_teacher_rebuild/native_candidates_20261027/engineering/single_seed_resource_saving_20261027/)在独立进程中缩减注册、选择锁、评测子进程和报告范围，训练命令保持原样。[14:58真实接续证明](reports/round1_initial_20261027/resource_transition/ACTUAL_LIVE_ADOPTION.json)显示四个保留run分别已到A2538、B1159、C3352、D2785步，均沿用唯一原始训练attempt。源码中的机器路径和OWNER_SPEC对应当前平台，迁机需配置当地路径。

## 教师与学生方法

本机教师仅做 **D／full384先聚类／共享D48／K8／增强double**。使用本机 Base 重筛、GT 接口反传配对、子空间及专家重新拟合，再进行各场景固定分支选择。冻结教师及完整 Train/Test 评测已经完成。

| 完整人群 | 场景数 | 本机 Base PDMS | 本机 full 教师 PDMS |
|---|---:|---:|---:|
| Train | 85,813 | 95.07208906 | 97.52980208 |
| Test | 12,146 | 90.67922438 | 96.55443602 |

以上是离线选择教师上限，不是学生成绩。Train gain1 教师为97.47006368，严格增益>1点保留36,359个修复，其余49,454个目标为Base。

学生冻结 VLM、feature_encoder、原状态编码器及 RL DiT。共享连续适配器使用4个 learned queries、hidden256、两层 cross-attention、4heads、FFN expansion4、dropout0，读取有效 post384 token 与24维普通状态，输出48维系数，经固定D48解码为共享384维修复量。最终层零初始化，保留完整padding和原生动作链；没有专家分类、gate、MoE或候选选择。

| 组别 | 表征目标 | 功能损失状态来源 |
|---|---|---|
| A / H_full | full | 无 |
| B / H_gain1 | gain1 | 无 |
| C / H_mu_teacher | gain1 | 冻结教师链 |
| D / H_mu_student | gain1 | 当前学生接口detach后重新生成 |

四组沿用AdamW LR1e-4、指定不衰减组、clip1、warmup1,341步及cosine至1e-5。C/D前4个epoch只训练H，接着2个epoch将功能权重升至0.5，随后34个epoch联合训练。FP32适配器/动作头，TF32关闭。固定损失尺度只来自Train，已实际计算完成，见[定标数据](reports/round1_initial_20261027/protocol/CALIBRATION.json)。

## 实际开训与初期 loss

14:22:51 UTC 的记录证明原先8个run均已发生真实优化更新，不能将其理解为仅提交脚本。随后用户要求节省资源，20261028的四个run已于14:49:52–54实际停止，保留20261027。

第一次完整epoch统计已回填：

| 保留 run | 完整epoch1场景加权平均loss |
|---|---:|
| H_full_seed20261027 | 0.3914273065 |
| H_gain1_seed20261027 | 当时仍为partial，未伪记完整均值 |
| H_mu_teacher_seed20261027 | 0.3147765080 |
| H_mu_student_seed20261027 | 0.3147765080 |

完整epoch均核对1,341次更新、85,813个真实样本及checkpoint实际SHA。C/D此时按课程只使用H，因此同seed结果一致。

初期快照中，各run前100步与最近100步的loss均值下降19.8%–30.5%，平滑曲线近期趋于平台；逐batch有波动。该观察不能替代PDMS评测。

![初期loss曲线](reports/round1_initial_20261027/loss/EARLY_TRAINING_LOSS.png)

每步loss持续写入训练日志；独立CPU汇总保存每epoch JSON/CSV，按真实样本数加权，尾批53例按53计入，零/非零目标H按各自数量统计，原始μ和实际加权功能项分别记录。读取按每run预计epoch完成时间加缓冲调度。

## 后续正式评价

每个epoch保存checkpoint。完整Validation为17,475场景×S0–S4，在epoch8/16/24/32/40执行并选择best，另保留epoch40 last。当前四个保留run均完成训练和规定验证、锁定best/last SHA之后，才进行完整Test12,146×S0–S4。Train行为用锁定8,192场景面板，完整Train表征误差另报。best另执行Validation S5–S9。

## 文件与限制

- [发布文件SHA清单](reports/round1_initial_20261027/PUBLICATION_MANIFEST.json)
- [实际开训快照](reports/round1_initial_20261027/startup/STARTUP_OBSERVATION.1422Z.json)
- [停止重复seed的真实证明](reports/round1_initial_20261027/resource_transition/ALL_FOUR_SEED28_STOPPED.json)
- [单seed用户指令](reports/round1_initial_20261027/protocol/SINGLE_SEED_USER_INSTRUCTION.json)
- [历史20261028配置与统计](reports/round1_initial_20261027/cancelled_seed20261028/)

本次发布包含源码、配置、指纹、初始日志快照与统计，没有打包RL/VLM大权重、NAVSIM原始数据、505GB Base缓存、全量教师目标或训练checkpoint。JSON中的本机路径是实际执行溯源信息，迁机需映射并按SHA核对。完整实验报告待训练、验证和测试实际完成后另行提交。
