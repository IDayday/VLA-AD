# 当前四组蒸馏方法与区别

本说明按本机正在运行的四份有效配置与实际训练代码整理。训练代码冻结于 `0a58bd8e6271e26103512add2c5cb7816d7eb8f1`；教师是本机完整重建的 **D／先聚类／共享 D48／K8／增强 double**，不是直接沿用旧 PPU 专家参数。当前只保留训练 seed **20261027**。

## 四组比较

| 组别 | 实际实验名 | 表征目标 | 功能损失 | 功能比较所用的动作状态 | 主要比较问题 |
|---|---|---|---|---|---|
| A | `H_full` | 本机冻结教师的 full 修复接口 | 无；全程仅 H | 不需要构造 | 只模仿完整教师表征，能吸收多少修复能力？ |
| B | `H_gain1` | 教师相对 Base 五噪声平均 PDMS 增益严格大于 1 点时保留修复，否则目标为 Base 接口 | 无；全程仅 H | 不需要构造 | 去掉小收益修复目标，能否减少原有成功能力的损伤？ |
| C | `H_mu_teacher` | 与 B 完全相同的 gain1 目标 | H + 课程权重 × 去噪转移均值损失 μ | 冻结教师接口生成的原生去噪链状态 | 在表征模仿之外，约束动作网络的实际响应，是否优于仅 H？ |
| D | `H_mu_student` | 与 B/C 完全相同的 gain1 目标 | 与 C 相同 | 当前学生接口生成的原生去噪链状态；每次访问刷新 | 在学生实际会遇到的动作状态上对齐教师，是否优于只在教师状态上对齐？ |

这四组不是之前的 H／T／HT 三组。当前没有一组“仅轨迹监督 T”；C/D 的 μ 是**原生 DDIM 一步条件转移均值监督**，不直接用 GT 轨迹作回归目标，也不是把最终五步轨迹当作 loss 反传。

配置依据：[A 配置](/mnt/project/recogdrive-no-routing-round1-results-20261026/H_full_seed20261027/effective_config.yaml:180)、[B 配置](/mnt/project/recogdrive-no-routing-round1-results-20261026/H_gain1_seed20261027/effective_config.yaml:180)、[C 配置](/mnt/project/recogdrive-no-routing-round1-results-20261026/H_mu_teacher_seed20261027/effective_config.yaml:180)、[D 配置](/mnt/project/recogdrive-no-routing-round1-results-20261026/H_mu_student_seed20261027/effective_config.yaml:180)。

## 共同的学生结构与可训练参数

四组都从同一 ReCogDrive RL Base 出发，冻结原 VLM、`feature_encoder`、RL DiT、原历史轨迹／车辆状态编码器。训练使用本机冻结 Base 的原生 FP32 `post384` 缓存，保留完整 token 布局和 padding 值。

只训练新加入的同一个连续适配器，**2,029,872 个参数，42 个参数张量**。四组 `TRAINABLE_PARAMETERS.json` 的 SHA256 相同：`1a8e44f063fc389a7b3ba2e1a067da82d2a94198cf3ca513eb4cb1a06593c103`。这是新增适配器的全参数训练，不是对原 RL/VLM 插入 LoRA。

适配器读取有效 image/text token 与 24 维普通状态（history12 + status8 + command4），使用 4 个 learned queries、hidden256、2 层 cross-attention、4 heads、FFN expansion4、dropout0。输出 48 维连续系数，经同一固定 D48 basis 解码为一个共享 384 维修复向量，对本场景所有有效 image/text token 加相同残差。适配器中的新状态 MLP 可训练；原动作头的状态编码器保持冻结，二者不是同一个模块。

最终输出层零初始化，因此开训时修复量为零。幅度上限来自本机冻结教师，`cap_in_d0=4`；实际 FP32 相加后再次检查原生 RMS 上限。padding 不被修改，也不从下游动作链中裁掉。

所有场景都执行同一个网络，没有 expert 分类、gate、MoE、argmax 或多候选推理。expert 身份只用于离线目标来源记录和事后统计，不进入网络输入、不改变样本权重。

源码依据：[适配器结构与零初始化](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/adapter.py:96)、[连续系数解码和实际限幅](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/adapter.py:118)、[普通状态布局](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/feature_cache.py:31)、[动作头冻结](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/train.py:270)、[实际可训练参数清单](/mnt/project/recogdrive-no-routing-round1-results-20261026/H_full_seed20261027/TRAINABLE_PARAMETERS.json)。

## 教师目标：full 和 gain1

记冻结 Base 接口为 \(H_0\)，有效 image/text mask 为 \(M\)，本地冻结教师的实际共享修复量为 \(\Delta_T\)。目标接口为：

\[
H_T=\operatorname{where}(M,H_0+\Delta_T,H_0).
\]

目标来自独立冻结 Base 上的 `actual_role_shifts`。导出要求实际 image/text shift 逐值相同，并检查用共享 shift 重建完整教师接口及 padding；不以 requested shift 替代实际执行结果。

full 使用本机教师已冻结的逐场景选择，包含 Base 分支和所有实际选择的修复。gain1 在 full 的基础上，只有教师比 Base 的 S0–S4 平均 PDMS **严格高于 1 点**才保留原修复量，否则置为零；它不会另外再选一套专家。

| Train 目标 | 全部场景 | 非零修复目标 | 零修复／Base 目标 |
|---|---:|---:|---:|
| full（A） | 85,813 | 46,573 | 39,240 |
| gain1（B/C/D） | 85,813 | 36,359 | 49,454 |

零目标场景仍正常参与 H 训练，使学生学习在这些场景保留 Base 接口。没有额外叠加一项与教师目标冲突的 Base 保持 loss。gain1 的 1 点门槛只用于预先冻结离线训练目标；部署时不读取教师评分、不判断增益门槛、不按门槛选择是否修复。

依据：[逐场景五噪声选择与 gain>1](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/build_targets.py:14)、[actual shift 重建检查](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/build_targets.py:130)、[full/gain1 目标构造](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/build_targets.py:199)、[本机 Train 目标清单](/mnt/project/recogdrive-no-routing-round1-results-20261026/targets/train/MANIFEST.json)。

## H：表征监督

设当前学生接口为 \(H_S\)。每场景先对有效 token 和 384 通道计算均方误差，再除以冻结的 Train 定标量：

\[
L_H=\frac{\sum_{i:M_i=1}\|H_{S,i}-H_{T,i}\|_2^2}
{384\,|M|\,s_H}.
\]

四组共用 `s_H=0.8812627424525735`。该数来自 full Train 中 46,573 个非零实际教师 shift 的场景等权能量统计；不由测试成绩调整，也不随训练变化。A/B 的总 loss 就是 H；C/D 的 H 定义相同，但采用 gain1 目标。

因此 A 和 B/C/D 的绝对 H 对应不同监督目标，不能直接以“谁的 H 更小”判断跨组能力更强。

依据：[实际 H loss](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/losses.py:146)、[H 定标定义](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/calibration.py:108)、[冻结定标结果](/mnt/project/recogdrive-no-routing-round1-results-20261026/calibration/CALIBRATION.json)。

## μ：动作网络功能监督

原动作网络参数冻结，但其输出对输入接口的梯度保留。记原生一步 DDIM 转移均值为 \(\mu(a_j,t_j;H,c)\)，其中 \(c\) 是原状态编码器产生的普通动作上下文。C/D 每次为场景安排一个明确整数步号 \(j\in\{0,\ldots,4\}\) 与一个噪声 bank。教师与学生在**同一个动作状态 \(a_j\)、同一个步号／时刻 \(t_j\)**上比较：

\[
L_\mu=\frac{\operatorname{MSE}\!\left(
\mu(a_j,t_j;H_S,c),
\operatorname{stopgrad}\mu(a_j,t_j;H_T,c)
\right)}{s_{\mu,j}},\qquad
L=L_H+\lambda_\mu(u)L_\mu.
\]

`s_mu[j]` 是预先固定的五个 Train 标尺，来自同一批 gain1 修复场景中，冻结 Base 链状态上教师和 Base 的条件均值差异。两组共用同一份定标，数值约为 `[2.4553e-4, 1.7319e-4, 2.8329e-4, 5.1608e-4, 1.2517e-4]`。

- **C 的状态**：用冻结教师接口 \(H_T\) 和该场景噪声运行原生去噪链前缀得到 \(a_j\)。教师状态和教师均值不随学生变化，所以允许复用已经核验的纯函数查询缓存。
- **D 的状态**：用当前 \(H_S.detach()\) 和同一噪声生成 \(a_j\)，每次访问都重算。状态生成前缀及教师均值不反传；学生均值查询用未 detach 的 \(H_S\)，因此梯度经冻结 DiT 回到适配器。D 没有沿“学生状态生成前缀”做完整反传，这是当前明确的半梯度设计。

两组都忠实使用原生 DDIM 五步 `[80,60,40,20,0]`，`eta=1`、`deterministic=False`、更新噪声 clip±1、每步标准差下限1e-4，最后一步也保留最小噪声。`p_mean_variance` 直接接受 post384，不再次调用 `feature_encoder`。训练每次访问只监督安排到的一步，并在场景／epoch间轮换五个步号和五个噪声 bank；S0–S4 不是五个独立训练 seed。

依据：[同状态教师／学生 μ 查询和梯度边界](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/losses.py:187)、[原生均值查询保留输入梯度](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/native_bridge.py:72)、[原生去噪前缀](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/native_bridge.py:89)、[步号／bank 轮换](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/noise.py:39)、[μ 定标](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/calibration.py:125)。

## 训练课程和预算

A/B 显式关闭 `enable_transition_loss`，40轮全程只有 H。C/D 前4轮只有 H；第5–6轮按 optimizer update 将 μ 权重从0线性升至0.5；第7–40轮保持0.5。对应零起始 update 公式为：

\[
\lambda_\mu(u)=0.5\,\operatorname{clip}\!\left(\frac{u-5364}{2682},0,1\right).
\]

四组各训练85,813个 Train场景、40 epochs、global batch64、每轮1,341次更新，总53,640次更新；当前每组4卡、microbatch1、累积16。尾部53个真实场景按实际数量归一化，补齐slot权重为零。自然场景等权，不按专家类别平衡采样。

四组 AdamW 均为 LR1e-4、betas(0.9,0.999)、eps1e-8、weight decay0.01、全局梯度裁剪1.0；bias、norm、learned queries和最终输出层不衰减。首1,341次更新 warmup，之后 cosine 降至1e-5。适配器／动作头／loss FP32，关闭 TF32 和 autocast。

每份不可变 effective_config 里仍保留原始两seed列表，这是原协议文本；实际 `training_seed=20261027`，20261028四个run已按后续用户指令取消，当前只有单seed四组，不能据旧列表重复启动。

依据：[课程](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/losses.py:101)、[真实场景归一化与 optimizer step](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/train.py:380)、[优化器排除衰减项](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/train.py:207)、[实际单seed指令](/mnt/project/VLA-AD/experiments/recogdrive_dc8_native_candidates_20261027/engineering/single_seed_20261027/USER_INSTRUCTION.json)。

## 本次 A 组 loss 与测试 PDMS 相关性如何解释

A 组在全部轮次中目标和权重不变，总 loss=H，因此适合先做这个诊断。需要区分两类数值：

1. **该 epoch 的在线训练平均 H**：整轮85,813场景等权，反映这一轮不断更新的模型经历过的误差；它不是轮末保存 checkpoint 在全人群上的固定模型误差。
2. **固定 checkpoint 的重新前向 H**：所有 checkpoint 在同一个预先固定的 Train面板上重新前向，采用同一 full 教师、有效mask和冻结定标。它更直接对应被拿去评测 Test 的那一份参数。若用8,192场景面板，就明确报告“Train8192面板H”，不称完整Train误差。

建议将上述两列分别与同一完整Test12,146场景×S0–S4的场景等权平均PDMS做散点图，并报告 Pearson（线性关系）和 Spearman（排序关系）。同时列原始MSE、标准化H、PDMS和相对同条件Base的增益。H只是固定正数缩放，原始MSE和标准化H的相关系数应一致。

“loss越小分数越高”对应负相关；“loss下降但分数也下降”对应正相关。少量checkpoint来自同一次训练轨迹，彼此有关联，不能把它们当作多个独立训练seed；相关性不能证明因果，也不能仅凭很少的点宣称普遍规律。

本任务已按用户最新要求以Test PDMS比较／选择权重。Test因此参与模型选择，报告需保留这个身份；Validation仅作诊断，不再按Validation分数决定优劣。Test场景／教师目标依然不进入学生梯度训练或loss定标。

固定模型 H 计算依据：[实际 FP32 目标与表征误差](/mnt/project/recogdrive-no-routing-round1-20261026/student_distillation/evaluate.py:174)。
