# ReCogDrive OPD 研究设计

本研究使用原生 ReCogDrive-2B：冻结的 InternVL/Qwen2（1536 维）加 small DiT（384 维），联合输出八个半秒轨迹点，每点为后轴坐标 x、y 和弧度航向。只有 34,329,219 个动作规划参数训练。专家、Last-RD、BIT、risk-VLA、额外对齐目标均关闭。原 baseline 的默认代码未修改。

IL 与 RL 的动作头结构相同。RL checkpoint 同时包含 active action_head 与固定 IL old_policy；严格加载器只选择 active head，并列出全部被忽略的非活动参数。历史 checkpoint 没有 VLM 参数，不能仅由它们证明历史编码器完全相同。本实验显式使用同一冻结 VLM，在线与缓存一致性另行验证。

固定官方源码版本为 `5831ee83c7a6a40bf6919715b96b0e9ba2b48335`。当前仓库曾 trim 系统提示词边界换行并改为动态padding，仅关闭专家开关不能恢复标准观测。OPD专用兼容层显式恢复边界换行和left padding到max_length=2800，保留全部hidden states；原仓库默认行为保持不变。真实8场景与隔离导入的官方Agent特征逐元素相同，动作头输出相同。旧v1批次为INVALID，受影响训练、gate和评测统一重做，不混合其曲线或结论。

## 三种蒸馏的区别

E1 用教师最终物理轨迹作伪标签，调用原生 diffusion IL：norm_odo、随机训练 timestep、加噪与噪声预测 MSE。E2 使用教师生成的五步状态，在固定状态上匹配原生 DDIM 转移均值。E3 在每个 optimizer step 用当前学生无梯度重新生成状态，随后教师无梯度查询、学生带梯度重新预测。状态已 detach，不进行全链反传。其 on-policy 含义限于生成链；固定日志 NAVSIM 不提供持续环境交互。

这对应 [DiffusionOPD](https://arxiv.org/abs/2605.15055) 的学生状态分布监督思想。其[官方训练代码](https://github.com/ali-vilab/DiffusionOPD/blob/main/scripts/train_sd3_opd.py)的转移比较原则被适配到原生 ReCogDrive，未移植图像数据逻辑。[多教师 Flow VLA](https://arxiv.org/abs/2610.21522)讨论的环境层 on-policy 不属于本实验。[PIVOT-Q](https://arxiv.org/abs/2610.05745)为脆弱状态选择与行为锚定提供相关思路；本研究不是量化 VLA 的直接复现。

## DDIM 的实际语义

原生五步时间为 `[80,60,40,20,0]`。配置 `ddim_eta=0` 不代表实际随机方差为零；活动路径使用 EtaFixed。随机 get_action 的标准差下界为 1e-4，噪声截断到 [-1,1]，最终 latent 也裁剪。adapter 复用原生 p_mean_variance 并保留 RNG 调用顺序，通过逐步状态/均值和最终轨迹一致性检查。DiffGRPO 另有 0.04 的随机标准差下界及不同噪声裁剪，完整保留。

主损失是FP32原生proposal均值MSE，均匀step权重。这里的μ明确指p_mean_variance返回的最终动作clamp之前的proposal mean，不宣称是clamp后输出核的严格期望；最终clamp保留在实际rollout、逐步parity和PDMS评测中。它并非两个确定性核的有限Gaussian KL。可选共享方差 KL 只允许明确选择的未截断高斯变体；原生截断核、不同方差与极小/零方差不能冒称满足解析 Gaussian KL 假设。

## 保留行为与选择性监督

E4-A 在相同学生状态上加入冻结 IL anchor 均值监督；E4-B 加原始 GT diffusion；E4-C 同时加入。E5 静态 gate 由训练集 IL/RL 五次配对评测构造：优势 LCB 超过 margin，且 NC、DAC、TTC 不退化。仅 A 类（IL<0.8、RL≥0.8）且 gate 有效时接受教师 OPD，B/C 类使用 IL anchor，所有类别可用训练 GT，D 类不假定固定教师提供新能力。mask 按有效数归一化，全零 mask 的教师项为零。

辅助权重先在固定训练输入上按梯度尺度校准，再锁定。每组有相同的两个学习率槽位与相同更新预算。E5 也从原始 IL 出发。正式主结果固定取 1000 步，不能按开发/测试得分选其他时点。协议文件及外部 lock hash 是最终依据。

## DiffGRPO 扩展

原生 rollout、奖励、组内 advantage、diffusion logprob、BC=0.1 均保留。零 OPD 权重直接调用原函数；原生DiffGRPO的截断采样和带标准差下界的logprob实现均保留，不能把原有logprob近似当成OPD解析高斯KL成立的依据；非零权重复用学生原生组 rollout 状态，没有其他策略的伪 on-policy 样本。固定教师只参与训练，推理仍只有学生。已有 BC 不再叠加等价的 IL anchor。正式 E6 受预注册证据和资源门禁约束，小规模集成不等于超过教师。

## 信息与统计边界

固定 train/cal/dev 为 4115/518/1696 场景，完整 log 与父驾驶记录互斥，均来自官方 navtrain，不能称底模从未见过。navtest 仅在锁定选择后评分。VLM 缓存仅含原生推理观测；GT、未来占用、评分和 gate 都与编码输入隔离。PDMS 使用正式逐候选 reference 归一化，保留分项及首次违规日志；首次 TTC 检查失败时间不是物理剩余碰撞时间。

置信区间联合保留训练种子、完整 log 和配对推理种子的层级结构。主要差异包括平均 PDMS、五次场景均值估计的下降幅度 M_minus、独立筛选的 IL 高分集合保持率与新增零分。三个训练种子与五个推理种子不能互相替代。允许否定结果与证据不足，不按结果添加模块或扩展预算。
