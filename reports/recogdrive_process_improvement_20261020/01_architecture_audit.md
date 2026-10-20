# 架构与数值路径审计

模型是标准 ReCogDrive-2B 的基础动作规划器：共享冻结 InternVL hidden dim 1536，planner dim 384，small DiT，8×3 轨迹，DDIM5 的实际 timestep 为 `[80,60,40,20,0]`。完整基础 planner 参与训练，VLM 不参与反向传播。

统一严格 loader 关闭 expert、last-RD、BIT、risk-VLA、action-aware 辅助、policy-KD、JEPA/VGGT alignment 和未来目标输入。加载不是依赖目录存在或静默随机初始化。输入为当前前视图、导航指令、自车状态及四帧历史；缓存是动作头真正使用的原始 padded VLM tokens，既不删 padding 也不缓存可训练 feature projection。

原生 `p_mean_variance` 是唯一转移公式来源。`ddim_eta=0` 配置与实际路径有差异：原生 EtaFixed 仍返回 eta=1，随机推理 std 约为 `[0.73775488,0.51396751,0.29510862,0.02506145,0.0001]`，原始正态噪声截断在 ±1。GRPO 则使用噪声 ±5、采样 std 最小 0.04，logprob std 最小 0.1，随后概率值裁剪。两者明确区分。

归一化保留原生规则：x 使用偏置 1.57、跨度 66.74；y 使用偏置 19.68、跨度 42；heading 使用偏置 1.67、跨度 3.53。输出为自车后轴坐标系，x 前、y 左、航向弧度，时间点 0.5–4 秒。整条 8 点轨迹是一项联合动作。

新 adapter 保留最后一次转移的 raw state，只有部署输出执行 final clamp/denorm；suffix 从 next_index 开始，不重复当前转移。确定性分支仍按原生调用顺序处理噪声，但 actual std 为零。

真实数值验证覆盖 8 个场景、IL/RL/COPY、确定/随机两种分支，共 48 组：中间状态、最终轨迹与原生 get_action 完全一致；每个 cut=0…5 的 suffix 也完全一致。在线 VLM 与缓存 exact equality 通过。学生 feature_encoder 梯度非零，教师/Anchor 参数和 buffers 不变，相同师生的过程 MSE 为零，更新后新学生 chain 改变。

同世界规模 save/resume 的参数与优化器状态完全一致；单 GPU 与双 GPU 全局均值归约后的参数最大差为 `1.1920928955078125e-7`。原生 batch 4 与逐样本累积的梯度差有单独记录。完整证据在 tables 中。

一次原型失败来自非 GRPO 纯推理 head 没有显式挂载 eval clipping 属性；修复为读取原生 get_action 相同的缺省值，失败日志保留，之后全部重新跑数值门禁。该失败未进入正式训练。
